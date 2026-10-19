"""Low-level QNN implementation delegating first-order evaluation to qc_executor."""

import copy
from typing import Callable, Union

import numpy as np

from qc_executor import Parameters
from qc_executor.parameters import Parameter

from ...util.operator_utils import square_operator

from ...observables.observable_base import ObservableBase
from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...encoding_circuit.layered_encoding_circuit import LayeredEncodingCircuit
from ...util import Executor
from ...util.data_preprocessing import adjust_features, adjust_parameters, to_tuple

from .evaluation_classes import eval_var, eval_dvardx, eval_dvardp, eval_dvardop, get_eval_laplace

# Frameworks for which Executor.expectation_value/expectation_value_derivatives
# (and therefore evaluation) are available.
_NATIVE_FRAMEWORKS = frozenset({"qiskit", "pennylane", "qulacs"})

# Frameworks whose qc_executor backend cannot bind a shared "p_op" vector across a
# List[QuantumOperatorBase] where each entry only owns a different slice of it: they require
# the array passed under a vector-style key to match that *specific* observable's own
# parameter count exactly, not the length of the full, shared vector. Qiskit's backend
# handles the list natively and is left on the faster, single-call path.
_LIST_OBSERVABLE_NEEDS_PER_OBSERVABLE_CALLS = frozenset({"pennylane", "qulacs"})

# Maps every native key to the observable attribute it is measured against and the
# qc_executor derivative parameter it needs ("x"/"p"/"p_op", or None for the plain
# expectation value). The "fcc"* keys are expectation values of the *squared* observable
# <O^2> - the "var" family below is expressed entirely in terms of these plus the plain
# "f"/"dfdx"/"dfdp"/"dfdop" values, so it needs no derivative order beyond 1.
_NATIVE_KEY_INFO = {
    "f": ("_native_observable", None),
    "dfdx": ("_native_observable", "x"),
    "dfdp": ("_native_observable", "p"),
    "dfdop": ("_native_observable", "p_op"),
    "fcc": ("_native_observable_squared", None),
    "dfccdx": ("_native_observable_squared", "x"),
    "dfccdp": ("_native_observable_squared", "p"),
    "dfccdop": ("_native_observable_squared", "p_op"),
}

# Trailing-shape dimension for each derivative key, as an attribute name resolved on the
# instance (num_features/num_parameters/num_parameters_observable). "f"/"fcc" have no
# derivative axis and are handled separately in _trailing_shape.
_NATIVE_KEY_TRAILING_DIM_ATTR = {
    "dfdx": "num_features",
    "dfccdx": "num_features",
    "dfdp": "num_parameters",
    "dfccdp": "num_parameters",
    "dfdop": "num_parameters_observable",
    "dfccdop": "num_parameters_observable",
}

# "var"/"varf" and their gradients are pure post-processing of expectation values of the
# observable and its square: var = <O^2> - <O>^2. Maps each requested key to the native
# keys it needs and the post-processing function that combines them.
_VAR_FAMILY = {
    "var": (("f", "fcc"), eval_var),
    "varf": (("f", "fcc"), eval_var),
    "dvardx": (("f", "dfccdx", "dfdx"), eval_dvardx),
    "dvarfdx": (("f", "dfccdx", "dfdx"), eval_dvardx),
    "dvardp": (("f", "dfccdp", "dfdp"), eval_dvardp),
    "dvarfdp": (("f", "dfccdp", "dfdp"), eval_dvardp),
    "dvardop": (("f", "dfccdop", "dfdop"), eval_dvardop),
    "dvarfdop": (("f", "dfccdop", "dfdop"), eval_dvardop),
}

# Higher-order and mixed (chained) derivatives - native for Qiskit and PennyLane:
# qc_executor's Qiskit backend supports chained (tuple-form) derivatives -
# expectation_value_derivatives(circuit, observable, ("x", "x"), ...) - via
# OpTree differentiation applied once per side (circuit, observable) and
# combined once, exact because circuit and observable parameters are
# disjoint. PennyLane's qc_executor backend supports the same tuple form
# natively too, via chained qml.jacobian (max_diff=len(todo)). Qulacs's
# qc_executor backend has no equivalent, so this stays gated to
# {"qiskit", "pennylane"} (see _evaluate).
# Maps each key to (observable_attr, derivative_param_tuple), the same shape
# as _NATIVE_KEY_INFO above, generalized from a single "x"/"p"/"p_op" string
# to a tuple of them (one entry per differentiation, in order).
_NATIVE_KEY_INFO_CHAINED = {
    "dfdxdx": ("_native_observable", ("x", "x")),
    "dfdpdp": ("_native_observable", ("p", "p")),
    "dfdopdp": ("_native_observable", ("p_op", "p")),
    "dfdpdop": ("_native_observable", ("p", "p_op")),
    "dfdopdop": ("_native_observable", ("p_op", "p_op")),
    "dfdpdx": ("_native_observable", ("p", "x")),
    "dfdxdp": ("_native_observable", ("x", "p")),
    "dfdopdx": ("_native_observable", ("p_op", "x")),
    "dfdopdxdx": ("_native_observable", ("p_op", "x", "x")),
    "dfccdxdx": ("_native_observable_squared", ("x", "x")),
    "dfccdpdp": ("_native_observable_squared", ("p", "p")),
    "dfccdopdx": ("_native_observable_squared", ("p_op", "x")),
    "dfccdopdop": ("_native_observable_squared", ("p_op", "p_op")),
    # Order-3, pure circuit-side permutations.
    "dfdxdxdp": ("_native_observable", ("x", "x", "p")),
    "dfdxdpdx": ("_native_observable", ("x", "p", "x")),
    "dfdpdxdx": ("_native_observable", ("p", "x", "x")),
}
_NATIVE_KEYS_CHAINED = frozenset(_NATIVE_KEY_INFO_CHAINED)

# Trailing-shape dimensions for the keys above: one attribute name per tuple
# position, in the same order as the derivative tuple - generalizes
# _NATIVE_KEY_TRAILING_DIM_ATTR's single-attribute form.
_NATIVE_KEY_TRAILING_DIM_ATTR_CHAINED = {
    "dfdxdx": ("num_features", "num_features"),
    "dfdpdp": ("num_parameters", "num_parameters"),
    "dfdopdp": ("num_parameters_observable", "num_parameters"),
    "dfdpdop": ("num_parameters", "num_parameters_observable"),
    "dfdopdop": ("num_parameters_observable", "num_parameters_observable"),
    "dfdpdx": ("num_parameters", "num_features"),
    "dfdxdp": ("num_features", "num_parameters"),
    "dfdopdx": ("num_parameters_observable", "num_features"),
    "dfdopdxdx": ("num_parameters_observable", "num_features", "num_features"),
    "dfccdxdx": ("num_features", "num_features"),
    "dfccdpdp": ("num_parameters", "num_parameters"),
    "dfccdopdx": ("num_parameters_observable", "num_features"),
    "dfccdopdop": ("num_parameters_observable", "num_parameters_observable"),
    "dfdxdxdp": ("num_features", "num_features", "num_parameters"),
    "dfdxdpdx": ("num_features", "num_parameters", "num_features"),
    "dfdpdxdx": ("num_parameters", "num_features", "num_features"),
    # laplace_dp/laplace_dop (below) keep a single trailing axis - the x,x
    # axes of their dfdpdxdx/dfdopdxdx dependency are traced away by
    # get_eval_laplace, not carried through to the key's own shape.
    "laplace_dp": "num_parameters",
    "laplace_dop": "num_parameters_observable",
}

# laplace/laplace_dp/laplace_dop are pure post-processing (a trace over the
# feature-Hessian's diagonal) of a chained higher-order key above - same
# pattern as _VAR_FAMILY.
_LAPLACE_FAMILY = {
    "laplace": (("dfdxdx",), get_eval_laplace("dfdxdx")),
    "laplace_dp": (("dfdpdxdx",), get_eval_laplace("dfdpdxdx")),
    "laplace_dop": (("dfdopdxdx",), get_eval_laplace("dfdopdxdx")),
}

# Merged lookups used at runtime: the qiskit-only keys are only ever looked
# up here after the framework gate in _evaluate has already restricted them
# to qiskit, so a single merged dict (instead of two conditionally-consulted
# ones) keeps _compute_native/_compute_native_per_observable/_trailing_shape
# simple - no key collides between the two source dicts.
_NATIVE_KEY_INFO_ALL = {**_NATIVE_KEY_INFO, **_NATIVE_KEY_INFO_CHAINED}
_NATIVE_KEY_TRAILING_DIM_ATTR_ALL = {
    **_NATIVE_KEY_TRAILING_DIM_ATTR,
    **_NATIVE_KEY_TRAILING_DIM_ATTR_CHAINED,
}


class LowLevelQNN:
    """Low-level, framework-agnostic evaluation engine for a parameterized quantum circuit
    together with one or more observables.

    Evaluates the expectation value (``f``) and its derivatives with respect to the circuit's
    input features (``x``), its trainable parameters (``p``), and the observable's trainable
    parameters (``p_op``), as well as the corresponding values of the squared observable
    (``fcc``/``dfccdx``/``dfccdp``/``dfccdop``) and the variance family derived from those
    (``var``/``dvardx``/``dvardp``/``dvardop``). For Qiskit and PennyLane, arbitrary chained
    higher-order derivatives (e.g. ``dfdxdx``, ``laplace``) are supported as well; for Qiskit,
    derivatives can additionally be requested directly by parameter identity (e.g.
    ``llqnn.parameters[0]``). Qulacs supports the base key set without chained derivatives.
    Any unsupported key combination raises ``NotImplementedError``.

    Args:
        pqc (EncodingCircuitBase): The parameterized quantum circuit.
        observable (Union[ObservableBase, list]): The observable(s) to measure.
        executor (Executor): The executor for the quantum circuit.
        num_features (int): Dimension of the input features.
        post_processing (Callable): Optional post processing function operating on the result
            dict after evaluate.
        caching (bool): Caching of the result for each `x`, `param`, `param_op` combination
            (default = True)
    """

    _NATIVE_KEYS = frozenset(_NATIVE_KEY_INFO)

    def __init__(
        self,
        parameterized_quantum_circuit: EncodingCircuitBase,
        observable: Union[ObservableBase, list],
        executor: Executor,
        num_features: int,
        post_processing: Callable = None,
        caching=True,
    ) -> None:
        self._num_features = num_features
        self.caching = caching
        self._framework = executor.quantum_framework

        if self._framework == "qiskit" and not executor.backend_chosen:
            executor.select_backend(parameterized_quantum_circuit, num_features)

        self._pqc = copy.copy(parameterized_quantum_circuit)
        self._observable = copy.copy(observable)
        self._executor = executor
        self._post_processing = post_processing

        if isinstance(self._pqc, LayeredEncodingCircuit):
            self._pqc._build_layered_pqc(num_features)

        self._x = Parameters("x", num_features)
        self._p = Parameters("p", self._pqc.num_parameters)

        # qc_executor.QiskitExecutor performs its own ISA transpilation lazily at execution
        # time, so the observable is kept in the pqc's own (untransposed) qubit numbering.
        if isinstance(self._observable, list):
            num_qubits_operator = 0
            n_op_params = 0
            for obs in self._observable:
                num_qubits_operator = max(num_qubits_operator, obs.num_qubits)
                n_op_params += obs.num_parameters
        else:
            num_qubits_operator = self._observable.num_qubits
            n_op_params = self._observable.num_parameters

        if self._pqc.num_qubits != num_qubits_operator:
            raise ValueError("Number of Qubits are not the same!")
        self._num_qubits = self._pqc.num_qubits

        self._p_op = Parameters("p_op", n_op_params)

        self._native_circuit = self._pqc.get_circuit(self._x, self._p)
        if isinstance(self._observable, list):
            native_observables = []
            # (offset, length) of each observable's own slice within the shared "p_op"
            # vector - needed because qc_executor's list-observable derivative collapse
            # cannot handle observables that own different numbers of "p_op" elements.
            self._observable_p_op_slices = []
            ioff = 0
            for obs in self._observable:
                native_observables.append(obs.get_operator(self._p_op[ioff:]))
                self._observable_p_op_slices.append((ioff, obs.num_parameters))
                ioff += obs.num_parameters
            self._native_observable = native_observables
            # <O^2> for the "var" family (see _VAR_FAMILY). Composing an observable with
            # itself reuses the same p_op Parameter objects on both sides, so squaring
            # cannot introduce new parameter symbols beyond the ones already in
            # _observable_p_op_slices
            self._native_observable_squared = [
                square_operator(obs) for obs in self._native_observable
            ]
        else:
            self._native_observable = self._observable.get_operator(self._p_op)
            self._native_observable_squared = square_operator(self._native_observable)

        self.result_container = {}

    def evaluate(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
        *values: Union[
            str,
            Parameter,
            Parameters,
            tuple,
        ],
    ) -> dict:
        """General function for evaluating the output of derivatives of the QNN.

        Evaluation works for given combination of
        input features `x` and parameters `param` and `param_op`.
        The function includes caching of results

        If `x`, `param`, and/or `param_op` are given as a nested list
        (for example multiple sets of parameters),
        the values are returned in a nested list.

        Args:
            x (np.ndarray): Values of the input feature data.
            param (np.ndarray): Parameter values of the PQC parameters
            param_op (np.ndarray): Parameter values of the operator parameters
            values : Derivatives (or values) of the QNN that are evaluated. Higher order
                derivatives are given as tuples of parameters or vectors.

        Results:
            Returns a dictionary with the computed values.
            The keys of the dictionary are given by the entries in the values tuple

        """
        result_dict = self._evaluate(x, param, param_obs, *values)
        if self._post_processing:
            result_dict = self._post_processing(result_dict)
        return result_dict

    def gradient(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ):
        """Return the gradient wrt. trainable parameters of the QNN.

        Args:
            x (Union[float, np.ndarray]): Input data.
            param (Union[float, np.ndarray]): Parameters of the PQC.
            param_obs (Union[float, np.ndarray]): Parameters of the observable.

        Returns:
            np.ndarray: Gradient of the QNN.

        """
        return np.concatenate(
            (
                self.evaluate(x, param, param_obs, "dfdp")["dfdp"],
                self.evaluate(x, param, param_obs, "dfdop")["dfdop"],
            ),
            axis=None,
        )

    def __call__(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ):
        """
        Function for evaluating the QNN.

        Args:
            x (Union[float, np.ndarray]): Input data.
            param (Union[float, np.ndarray]): Parameters of the PQC.
            param_obs (Union[float, np.ndarray]): Parameters of the observable.

        Returns:
            np.ndarray: The output of the QNN.

        """
        return self.evaluate(x, param, param_obs, "f")["f"]

    def _unsupported_key_error(self, keys) -> NotImplementedError:
        """Build the error raised when one or more requested evaluation keys are not
        supported for the current framework, listing the keys that are.
        """
        supported = sorted(self._NATIVE_KEYS | set(_VAR_FAMILY))
        if self._framework in ("qiskit", "pennylane"):
            supported += sorted(_NATIVE_KEYS_CHAINED | set(_LAPLACE_FAMILY))
        extra = (
            " plus generic identity-based derivative keys (e.g. llqnn.parameters[0])"
            if self._framework == "qiskit"
            else ""
        )
        return NotImplementedError(
            f"Unsupported evaluation key(s) {list(keys)!r} for framework "
            f"{self._framework!r}: only the evaluation keys {supported}{extra} are supported."
        )

    def get_params(self, deep: bool = True) -> dict:
        """Returns the dictionary of the hyper-parameters of the QNN.

        In case of multiple outputs, the hyper-parameters of the operator are prefixed
        with ``op0__``, ``op1__``, etc.
        """
        params = dict(num_qubits=self.num_qubits)

        if deep:
            params.update(self._pqc.get_params())
            if isinstance(self._observable, list):
                for i, oper in enumerate(self._observable):
                    for key, value in oper.get_params().items():
                        if key != "num_qubits":
                            params["op" + str(i) + "__" + key] = value
            else:
                params.update(self._observable.get_params())
        return params

    def set_params(self, **params) -> None:
        """Sets the hyper-parameters of the QNN.

        In case of multiple outputs, the hyper-parameters of the operator are prefixed
        with ``op0__``, ``op1__``, etc.
        """
        valid_params = self.get_params(deep=True)
        for key in params:
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. Valid parameters are {sorted(valid_params)!r}."
                )

        dict_pqc = {key: value for key, value in params.items() if key in self._pqc.get_params()}
        if dict_pqc:
            self._pqc.set_params(**dict_pqc)

        if isinstance(self._observable, list):
            for i, oper in enumerate(self._observable):
                prefix = "op" + str(i) + "__"
                dict_operator = {
                    key.split("__", 1)[1]: value
                    for key, value in params.items()
                    if key.startswith(prefix)
                }
                if dict_operator:
                    oper.set_params(**dict_operator)
        else:
            dict_operator = {
                key: value for key, value in params.items() if key in self._observable.get_params()
            }
            if dict_operator:
                self._observable.set_params(**dict_operator)

    def set_shots(self, num_shots: int) -> None:
        """Sets the number of shots for the next evaluations."""
        self._executor.set_shots(num_shots)

    def get_shots(self) -> int:
        """Getter for the number of shots."""
        return self._executor.get_shots()

    def reset_shots(self) -> None:
        """Resets the number of shots to the initial ones."""
        self._executor.reset_shots()

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN."""
        return self._num_qubits

    @property
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC."""
        return self._num_features

    @property
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC."""
        return self._pqc.num_parameters

    @property
    def num_operator(self) -> int:
        """Return the number of outputs."""
        return len(self._observable) if isinstance(self._observable, list) else 1

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator."""
        return len(self._p_op)

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used."""
        return isinstance(self._observable, list)

    # For Qiskit, qc_executor's tuple-form expectation_value_derivatives (see
    # _build_derivative_arg/_evaluate below) resolves a raw Parameter/ParameterVector
    # taken from here by object identity against self._native_circuit/self._native_observable,
    # which these vectors were built from - so a key like `llqnn.parameters[0]` is evaluated
    # natively. PennyLane/Qulacs have no identity-based tuple mechanism, so these vectors are
    # only useful there for introspection (e.g. `len(llqnn.parameters)`), not as raw
    # derivative keys.
    @property
    def parameters(self) -> Parameters:
        """Return the parameter vector of the PQC."""
        return self._p

    @property
    def features(self) -> Parameters:
        """Return the feature vector of the PQC."""
        return self._x

    @property
    def parameters_operator(self) -> Parameters:
        """Return the parameter vector of the cost operator."""
        return self._p_op

    def _build_derivative_arg(self, val):
        """Translate a raw derivative key (Parameter/Parameters/tuple of those, e.g.
        `llqnn.parameters[0]` or `(llqnn.parameters[0], llqnn.parameters[1])`) into the
        (str | Parameter | tuple) form qc_executor's expectation_value_derivatives expects.
        A whole Parameters vector becomes its name string (the same convention "x"/"p"/"p_op"
        already use); a bare Parameter is wrapped in a 1-tuple so qc_executor returns an
        array-shaped (not scalar) result.
        """
        if isinstance(val, tuple):
            return tuple(self._translate_derivative_entry(e) for e in val)
        if isinstance(val, Parameters):
            return val.name
        if isinstance(val, Parameter):
            return (val,)
        raise ValueError(f"Unsupported derivative key: {val!r}")

    def _translate_derivative_entry(self, entry):
        if isinstance(entry, Parameters):
            return entry.name
        if isinstance(entry, Parameter):
            return entry
        raise ValueError(f"Unsupported derivative key element: {entry!r}")

    def _trailing_shape_raw(self, val) -> tuple:
        """Trailing shape for a raw derivative key - one axis per entry (a bare, non-tuple
        key counts as a single entry), sized 1 for a Parameter or the vector's length for a
        whole Parameters."""

        def entry_dim(e):
            if isinstance(e, Parameters):
                return len(e)
            if isinstance(e, Parameter):
                return 1
            raise ValueError(f"Unsupported derivative key element: {e!r}")

        entries = val if isinstance(val, tuple) else (val,)
        dims = tuple(entry_dim(e) for e in entries)
        return (self.num_operator,) + dims if self.multiple_output else dims

    def _is_p_op_entry(self, entry) -> bool:
        if isinstance(entry, str):
            return entry == "p_op"
        if isinstance(entry, Parameter):
            return entry in self._p_op
        return False

    def _p_op_entry_excludes_observable(self, entry, ioff: int, n: int) -> bool:
        """Whether observable i's own p_op slice [ioff, ioff+n) makes this p_op-involving
        tuple entry contribute nothing for that observable. For the "p_op" string this is
        the existing "owns no p_op parameters at all" check; for a raw p_op Parameter (only
        possible via _build_derivative_arg, i.e. a `llqnn.parameters_operator[i]`-style key)
        it is a by-index membership check instead - unlike the string form, qc_executor's
        per-call resolution of a raw element does not verify observable membership itself, so
        that must happen here before the call is made.
        """
        if isinstance(entry, str):
            return n == 0
        return not (ioff <= entry.index < ioff + n)

    def _trailing_shape(self, key) -> tuple:
        """Shape of a single evaluation's result for `key`, excluding the x/param/param_op
        batch axes (output axis before the derivative axis)."""
        if not isinstance(key, str):
            return self._trailing_shape_raw(key)
        multi = self.multiple_output
        n_op = self.num_operator
        if key in ("f", "fcc", "laplace"):
            # laplace traces away both feature-Hessian axes of its dfdxdx
            # dependency entirely, leaving the same (empty) trailing shape
            # as a plain expectation value.
            return (n_op,) if multi else ()
        try:
            attr = _NATIVE_KEY_TRAILING_DIM_ATTR_ALL[key]
        except KeyError:
            raise ValueError(f"Unknown native key: {key}") from None
        dims = (
            tuple(getattr(self, a) for a in attr)
            if isinstance(attr, tuple)
            else (getattr(self, attr),)
        )
        return (n_op,) + dims if multi else dims

    def _compute_native(self, key, parameters: dict):
        if isinstance(key, str):
            observable_attr, derivative_param = _NATIVE_KEY_INFO_ALL[key]
        else:
            # A raw identity-based key (see _build_derivative_arg) always targets the plain
            # observable "O" - there is no squared-observable form for these.
            observable_attr, derivative_param = "_native_observable", self._build_derivative_arg(
                key
            )
        observable = getattr(self, observable_attr)
        has_p_op = self._is_p_op_entry(derivative_param) or (
            isinstance(derivative_param, tuple)
            and any(self._is_p_op_entry(e) for e in derivative_param)
        )
        if self.multiple_output and (
            # A "*dop"/"*dopdx"/... key always needs it: each observable only
            # contributes a gradient for its own "p_op" slice, and
            # qc_executor's list-observable collapse assumes every list entry
            # produces a result of the same shape - it can't zero-pad the rest.
            has_p_op
            # The other keys need it only for backends that can't bind a shared "p_op"
            # vector across list entries owning different-sized slices of it.
            or self._framework in _LIST_OBSERVABLE_NEEDS_PER_OBSERVABLE_CALLS
        ):
            return self._compute_native_per_observable(key, parameters)
        if derivative_param is None:
            return self._executor.expectation_value(self._native_circuit, observable, **parameters)
        return self._executor.expectation_value_derivatives(
            self._native_circuit, observable, derivative_param, **parameters
        )

    def _compute_native_per_observable(self, key, parameters: dict) -> np.ndarray:
        """Evaluate ``key`` for each observable individually."""
        if isinstance(key, str):
            observable_attr, derivative_param = _NATIVE_KEY_INFO_ALL[key]
        else:
            observable_attr, derivative_param = "_native_observable", self._build_derivative_arg(
                key
            )
        observable_list = getattr(self, observable_attr)
        needs_local_p_op_slice = self._framework in _LIST_OBSERVABLE_NEEDS_PER_OBSERVABLE_CALLS
        # Positions of "p_op" within the derivative tuple/shape (every position for a tuple
        # containing it, [0] for a bare "p_op" string or raw p_op Parameter, [] if it isn't
        # involved at all). A key like "dfdopdop" has "p_op" at *two* positions - each must be
        # sliced down to the observable's own (ioff, n) range independently, since qc_executor
        # resolves each tuple element against the same observable-local parameter set.
        if isinstance(derivative_param, tuple):
            p_op_axes = [ax for ax, e in enumerate(derivative_param) if self._is_p_op_entry(e)]
        elif self._is_p_op_entry(derivative_param):
            p_op_axes = [0]
        else:
            p_op_axes = []
        out = np.zeros(self._trailing_shape(key), dtype=float)
        for i, (obs, (ioff, n)) in enumerate(zip(observable_list, self._observable_p_op_slices)):
            if p_op_axes and any(
                self._p_op_entry_excludes_observable(
                    (
                        derivative_param[ax]
                        if isinstance(derivative_param, tuple)
                        else derivative_param
                    ),
                    ioff,
                    n,
                )
                for ax in p_op_axes
            ):
                # Observable i's own p_op slice excludes at least one requested p_op entry -
                # df_i/d(...) stays 0 for that entry, including a raw entry belonging to a
                # *different* observable's slice entirely (see _p_op_entry_excludes_observable).
                continue
            obs_parameters = dict(parameters)
            if needs_local_p_op_slice:
                obs_parameters["p_op"] = parameters["p_op"][ioff : ioff + n]
            elif "p_op" in obs_parameters:
                obs_parameters["p_op"] = self._p_op_prefix(obs, parameters["p_op"])
            if derivative_param is None:
                out[i] = self._executor.expectation_value(
                    self._native_circuit, obs, **obs_parameters
                )
            elif p_op_axes:
                value = self._executor.expectation_value_derivatives(
                    self._native_circuit, obs, derivative_param, **obs_parameters
                )
                dest_shape = out[i].shape

                def _local_size_and_slice(ax):
                    entry = (
                        derivative_param[ax]
                        if isinstance(derivative_param, tuple)
                        else derivative_param
                    )
                    if isinstance(entry, str):
                        return n, slice(ioff, ioff + n)
                    return 1, slice(entry.index, entry.index + 1)

                local_sizes = []
                global_slices = []
                for ax in range(len(dest_shape)):
                    if ax in p_op_axes:
                        size, gslice = _local_size_and_slice(ax)
                    else:
                        size, gslice = dest_shape[ax], slice(None)
                    local_sizes.append(size)
                    global_slices.append(gslice)
                value = np.asarray(value, dtype=float).reshape(tuple(local_sizes))
                # df_i/dp_op_j is 0 by construction for every j outside observable i's own
                # slice - only that slice of row i (at the "p_op" axes) is filled in.
                out[(i,) + tuple(global_slices)] = value
            else:
                value = self._executor.expectation_value_derivatives(
                    self._native_circuit, obs, derivative_param, **obs_parameters
                )
                out[i] = np.asarray(value, dtype=float).reshape(-1)
        return out

    @staticmethod
    def _p_op_prefix(observable, p_op) -> np.ndarray:
        """Cut the shared ``p_op`` vector down to what ``observable`` actually references.

        qc_executor's Qiskit executor resolves a parameter by its global index and expects
        exactly ``highest referenced index + 1`` values, so an observable that only uses
        the first entries of the shared vector must not be handed the whole of it.
        """
        referenced = [
            parameter.index
            for parameter in observable.parameters
            if parameter.vector_name == "p_op"
        ]
        needed = max(referenced) + 1 if referenced else 0
        p_op = np.asarray(p_op)
        return p_op[:, :needed] if p_op.ndim == 2 else p_op[:needed]

    def _compute_native_batched(self, key, parameters: dict, batch_size: int) -> np.ndarray:
        """Same contract as :meth:`_compute_native`, but exactly one of
        ``parameters["x"]``/``["p"]``/``["p_op"]`` is a 2D batch (shape
        ``(batch_size, dim)``) instead of a single 1D vector - qc_executor batches
        that axis internally in a single call instead of one call per
        point. Always returns an array of shape ``(batch_size,) + self._trailing_shape(key)``,
        batch axis first, regardless of where qc_executor's own call places it.
        """
        if isinstance(key, str):
            observable_attr, derivative_param = _NATIVE_KEY_INFO_ALL[key]
        else:
            observable_attr, derivative_param = "_native_observable", self._build_derivative_arg(
                key
            )
        observable = getattr(self, observable_attr)
        has_p_op = self._is_p_op_entry(derivative_param) or (
            isinstance(derivative_param, tuple)
            and any(self._is_p_op_entry(e) for e in derivative_param)
        )
        # A batched "p_op" additionally forces the per-observable path even when this
        # key's own derivative doesn't involve p_op at all (has_p_op False): qc_executor's
        # combined list-observable call cannot resolve a shared "p_op" batch against a
        # list whose entries reference different-sized (including zero-sized) slices of
        # it.
        p_op_is_batched = np.ndim(parameters.get("p_op")) == 2
        if self.multiple_output and (
            has_p_op
            or p_op_is_batched
            or self._framework in _LIST_OBSERVABLE_NEEDS_PER_OBSERVABLE_CALLS
        ):
            return self._compute_native_per_observable_batched(key, parameters, batch_size)
        if derivative_param is None:
            raw = self._executor.expectation_value(self._native_circuit, observable, **parameters)
        else:
            raw = self._executor.expectation_value_derivatives(
                self._native_circuit, observable, derivative_param, **parameters
            )
        raw = np.asarray(raw, dtype=float)
        if self.multiple_output and derivative_param is None:
            # observable is a list: qc_executor's expectation_value returns (n_op, batch) -
            # move batch to the front to match this method's contract. Its derivatives
            # already come as (batch, n_op, ...), so they must be left alone.
            raw = np.moveaxis(raw, 1, 0)
        # A multi-element derivative_param may still carry a structurally size-1 axis
        # from OpTree's differentiation machinery - reshape absorbs it.
        return raw.reshape((batch_size,) + self._trailing_shape(key))

    def _compute_native_per_observable_batched(
        self, key, parameters: dict, batch_size: int
    ) -> np.ndarray:
        """Batched counterpart of :meth:`_compute_native_per_observable` - see
        :meth:`_compute_native_batched` for the shared batching contract."""
        if isinstance(key, str):
            observable_attr, derivative_param = _NATIVE_KEY_INFO_ALL[key]
        else:
            observable_attr, derivative_param = "_native_observable", self._build_derivative_arg(
                key
            )
        observable_list = getattr(self, observable_attr)
        needs_local_p_op_slice = self._framework in _LIST_OBSERVABLE_NEEDS_PER_OBSERVABLE_CALLS
        if isinstance(derivative_param, tuple):
            p_op_axes = [ax for ax, e in enumerate(derivative_param) if self._is_p_op_entry(e)]
        elif self._is_p_op_entry(derivative_param):
            p_op_axes = [0]
        else:
            p_op_axes = []
        out = np.zeros((batch_size,) + self._trailing_shape(key), dtype=float)
        for i, (obs, (ioff, n)) in enumerate(zip(observable_list, self._observable_p_op_slices)):
            if p_op_axes and any(
                self._p_op_entry_excludes_observable(
                    (
                        derivative_param[ax]
                        if isinstance(derivative_param, tuple)
                        else derivative_param
                    ),
                    ioff,
                    n,
                )
                for ax in p_op_axes
            ):
                continue
            obs_parameters = dict(parameters)
            if needs_local_p_op_slice:
                p_op_val = np.asarray(parameters["p_op"])
                obs_parameters["p_op"] = (
                    p_op_val[:, ioff : ioff + n]
                    if p_op_val.ndim == 2
                    else p_op_val[ioff : ioff + n]
                )
            elif "p_op" in obs_parameters:
                obs_parameters["p_op"] = self._p_op_prefix(obs, parameters["p_op"])
            dest_shape = out[:, i].shape  # (batch_size,) + dims
            if derivative_param is None:
                raw = self._executor.expectation_value(self._native_circuit, obs, **obs_parameters)
                out[:, i] = self._broadcast_batched_result(raw, batch_size, dest_shape[1:])
            elif p_op_axes:
                raw = self._executor.expectation_value_derivatives(
                    self._native_circuit, obs, derivative_param, **obs_parameters
                )

                def _local_size_and_slice(ax):
                    entry = (
                        derivative_param[ax]
                        if isinstance(derivative_param, tuple)
                        else derivative_param
                    )
                    if isinstance(entry, str):
                        return n, slice(ioff, ioff + n)
                    return 1, slice(entry.index, entry.index + 1)

                # dims axes start at position 1 in dest_shape (position 0 is the batch
                # axis prepended for batching) - offset p_op_axes (indices into dims,
                # the unbatched numbering) by one accordingly.
                local_sizes = [batch_size]
                global_slices = [slice(None), i]
                for ax in range(1, len(dest_shape)):
                    if (ax - 1) in p_op_axes:
                        size, gslice = _local_size_and_slice(ax - 1)
                    else:
                        size, gslice = dest_shape[ax], slice(None)
                    local_sizes.append(size)
                    global_slices.append(gslice)
                out[tuple(global_slices)] = np.asarray(raw, dtype=float).reshape(
                    tuple(local_sizes)
                )
            else:
                raw = self._executor.expectation_value_derivatives(
                    self._native_circuit, obs, derivative_param, **obs_parameters
                )
                out[:, i] = self._broadcast_batched_result(raw, batch_size, dest_shape[1:])
        return out

    @staticmethod
    def _broadcast_batched_result(raw, batch_size: int, dims: tuple) -> np.ndarray:
        """Reshape a per-observable batched-call result to ``(batch_size,) + dims``.

        If the batched axis is "p_op" and this specific observable owns none of the
        p_op parameters it carries (n=0), nothing relevant to this observable's own
        circuit/operator actually varies across the batch - qc_executor then collapses
        the call to a single unbatched result instead of one per batch row.
        Detected by a size mismatch and handled by broadcasting that one
        result across the batch axis: the value is identical for every row since the
        observable doesn't depend on the batched parameter at all.
        """
        raw = np.asarray(raw, dtype=float)
        expected = batch_size * int(np.prod(dims, dtype=int)) if dims else batch_size
        if raw.size == expected:
            return raw.reshape((batch_size,) + dims)
        single = raw.reshape(dims)
        return np.broadcast_to(single, (batch_size,) + dims).copy()

    def _fill_native_array(
        self,
        arr: np.ndarray,
        key,
        x_inp: np.ndarray,
        param_inp: np.ndarray,
        param_op_inp: np.ndarray,
        batch_axis: Union[str, None],
    ) -> None:
        """Fill ``arr[ix, ip, iop, ...]`` for every (x, param, param_op) combination.
        Whichever axis actually varies (``batch_axis``, chosen by the caller: "x" is
        preferred - the common "many training points, one current parameter vector"
        case - then "p", then "p_op") is batched through a single qc_executor call
        instead of one call per point; the other two axes (typically
        singleton) are still looped in plain Python, since qc_executor batches only
        one shared axis per call.
        """
        n_x, n_p, n_pop = len(x_inp), len(param_inp), len(param_op_inp)
        if batch_axis == "x":
            for ip, p_vec in enumerate(param_inp):
                for iop, p_op_vec in enumerate(param_op_inp):
                    arr[:, ip, iop, ...] = self._compute_native_batched(
                        key, {"x": x_inp, "p": p_vec, "p_op": p_op_vec}, n_x
                    )
        elif batch_axis == "p":
            for iop, p_op_vec in enumerate(param_op_inp):
                arr[0, :, iop, ...] = self._compute_native_batched(
                    key, {"x": x_inp[0], "p": param_inp, "p_op": p_op_vec}, n_p
                )
        elif batch_axis == "p_op":
            arr[0, 0, :, ...] = self._compute_native_batched(
                key, {"x": x_inp[0], "p": param_inp[0], "p_op": param_op_inp}, n_pop
            )
        else:
            value = self._compute_native(
                key, {"x": x_inp[0], "p": param_inp[0], "p_op": param_op_inp[0]}
            )
            arr[0, 0, 0, ...] = np.asarray(value, dtype=float).reshape(arr.shape[3:])

    def _evaluate_native(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
        keys: list,
    ) -> dict:
        x_inp, multi_x = adjust_features(x, self.num_features)
        param_inp, multi_param = adjust_parameters(param, self.num_parameters)
        param_op_inp, multi_param_op = adjust_parameters(param_op, self.num_parameters_observable)
        n_x, n_p, n_pop = len(x_inp), len(param_inp), len(param_op_inp)

        if n_x > 1:
            batch_axis = "x"
        elif n_p > 1:
            batch_axis = "p"
        elif n_pop > 1:
            batch_axis = "p_op"
        else:
            batch_axis = None

        caching_tuple = None
        cached = {}
        if self.caching:
            caching_tuple = (
                to_tuple(x),
                to_tuple(param),
                to_tuple(param_op),
                (self._executor.shots is None),
            )
            cached = self.result_container.get(caching_tuple, {})

        out = {}
        for key in keys:
            if key in cached:
                out[key] = cached[key]
                continue

            trailing = self._trailing_shape(key)
            arr = np.zeros((n_x, n_p, n_pop) + trailing, dtype=float)
            if 0 not in trailing:
                # A zero-sized trailing axis (e.g. dfdop with no observable parameters at
                # all) has nothing to fill in - qc_executor has no "empty" result to return.
                self._fill_native_array(arr, key, x_inp, param_inp, param_op_inp, batch_axis)

            final_shape = []
            if multi_x:
                final_shape.append(n_x)
            if multi_param:
                final_shape.append(n_p)
            if multi_param_op:
                final_shape.append(n_pop)
            final_shape += list(trailing)

            out[key] = arr.reshape(final_shape) if final_shape else float(arr.reshape(()))
            cached[key] = out[key]

        if self.caching:
            self.result_container[caching_tuple] = cached

        return out

    def _evaluate(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
        *values,
    ) -> dict:
        if self._framework in _NATIVE_FRAMEWORKS:
            allowed_native_keys = self._NATIVE_KEYS
            allowed_post_keys = dict(_VAR_FAMILY)
            # The chained-derivative keys (dfdxdx, ...) and the laplace family built on
            # top of them are native for Qiskit and PennyLane (see
            # _NATIVE_KEY_INFO_CHAINED's docstring) - Qulacs's qc_executor backend has no
            # equivalent yet.
            if self._framework in ("qiskit", "pennylane"):
                allowed_native_keys = allowed_native_keys | _NATIVE_KEYS_CHAINED
                allowed_post_keys.update(_LAPLACE_FAMILY)
            native_keys = [v for v in values if isinstance(v, str) and v in allowed_native_keys]
            var_keys = [v for v in values if isinstance(v, str) and v in allowed_post_keys]
            # Generic identity-based derivative keys (Parameter/Parameters/tuple of those,
            # e.g. `llqnn.parameters[0]`) are native only for Qiskit - qc_executor's tuple
            # mechanism resolves them by object identity against
            # self._native_circuit/self._native_observable (see _build_derivative_arg).
            # PennyLane/Qulacs have no such identity-based tuple mechanism, so such a key
            # is unsupported for them - it falls into unsupported_keys below.
            if self._framework == "qiskit":
                raw_keys = [
                    v
                    for v in values
                    if not isinstance(v, str) and isinstance(v, (tuple, Parameters, Parameter))
                ]
            else:
                raw_keys = []
        else:
            allowed_post_keys = {}
            native_keys = []
            var_keys = []
            raw_keys = []
        unsupported_keys = [
            v for v in values if v not in native_keys and v not in var_keys and v not in raw_keys
        ]
        if unsupported_keys:
            raise self._unsupported_key_error(unsupported_keys)

        # "var" and its gradients need nothing beyond the plain expectation value and its
        # first derivatives (see _VAR_FAMILY); "laplace" and its gradients need nothing
        # beyond a single chained higher-order key (see _LAPLACE_FAMILY) - fold their
        # dependencies into the same native batch call instead of evaluating them
        # separately.
        underlying = {dep for key in var_keys for dep in allowed_post_keys[key][0]}
        combined_native_keys = sorted(set(native_keys) | underlying) + list(
            dict.fromkeys(raw_keys)
        )

        result = {}
        if combined_native_keys:
            result.update(self._evaluate_native(x, param, param_op, combined_native_keys))
        for key in var_keys:
            _, evaluation_function = allowed_post_keys[key]
            result[key] = evaluation_function(result)

        result["x"] = x
        result["param"] = param
        result["param_op"] = param_op
        return result
