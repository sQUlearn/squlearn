"""Unified low-level QNN implementation delegating first-order evaluation to qc_executor."""

from typing import Callable, Union
from warnings import warn

import numpy as np

from qc_executor import Parameters

from ...observables.observable_base import ObservableBase
from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...util import Executor
from ...util.data_preprocessing import adjust_features, adjust_parameters, to_tuple

from .lowlevel_qnn_base import LowLevelQNNBase
from .lowlevel_qnn_qiskit import LowLevelQNNQiskit
from .lowlevel_qnn_pennylane import LowLevelQNNPennyLane

# Frameworks for which Executor.as_qc_executor (and therefore the native f/dfdx/dfdp/dfdop
# path) is available. Frameworks not in this set always go through the fallback engine.
_NATIVE_FRAMEWORKS = frozenset({"qiskit", "pennylane"})


class LowLevelQNNUnified(LowLevelQNNBase):
    """Low-level QNN that evaluates ``f``/``dfdx``/``dfdp``/``dfdop`` directly through
    ``qc_executor`` (no ``OpTree`` construction) for frameworks it supports, and falls back to
    the legacy framework-specific engine (:class:`LowLevelQNNQiskit`, :class:`LowLevelQNNPennyLane`,
    ...) for every other requested derivative (``dfdxdx``, ``laplace``, ``var``, ...) as well as
    for frameworks qc_executor does not yet cover.

    Args:
        pqc (EncodingCircuitBase): The parameterized quantum circuit.
        observable (Union[ObservableBase, list]): The observable(s) to measure.
        executor (Executor): The executor for the quantum circuit.
        num_features (int): Dimension of the input features.
        post_processing (Callable): Optional post processing function operating on the result
            dict after evaluate.
        caching (bool): Caching of the result for each `x`, `param`, `param_op` combination
            (default = True)
        primitive (str): Qiskit-only primitive selection. Only forwarded to the fallback engine
            (used for derivatives beyond ``f``/``dfdx``/``dfdp``/``dfdop``), since ``qc_executor``
            has no equivalent per-call primitive selection. Ignored (with a warning) for
            frameworks that don't support it, matching the old per-framework classes.
    """

    _NATIVE_KEYS = frozenset({"f", "dfdx", "dfdp", "dfdop"})

    def __init__(
        self,
        parameterized_quantum_circuit: EncodingCircuitBase,
        observable: Union[ObservableBase, list],
        executor: Executor,
        num_features: int,
        post_processing: Callable = None,
        caching=True,
        primitive: Union[str, None] = None,
    ) -> None:
        self._num_features = num_features
        self.caching = caching
        self._fallback_engine = None
        self._framework = executor.quantum_framework

        if self._framework == "pennylane" and primitive is not None:
            warn("Primitive argument is not supported for PennyLane. Ignoring...")
            primitive = None
        self._primitive = primitive

        if self._framework == "qiskit" and not executor.backend_chosen:
            executor.select_backend(parameterized_quantum_circuit, num_features)

        super().__init__(parameterized_quantum_circuit, observable, executor, post_processing)

        self._x = Parameters("x", num_features)
        self._p = Parameters("p", self._pqc.num_parameters)

        # No TranspiledEncodingCircuit/set_map involved here (unlike LowLevelQNNQiskit):
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
        else:
            self._native_observable = self._observable.get_operator(self._p_op)

        self.result_container = {}

    @property
    def _fallback(self) -> Union[LowLevelQNNQiskit, LowLevelQNNPennyLane]:
        """Lazily-constructed legacy, framework-specific engine. Used for every derivative
        order/kind not covered by the native qc_executor path (``dfdxdx``, ``laplace``, ``var``,
        ...), and for every key at all when the framework has no qc_executor bridge yet."""
        if self._fallback_engine is None:
            if self._framework == "qiskit":
                self._fallback_engine = LowLevelQNNQiskit(
                    self._pqc,
                    self._observable,
                    self._executor,
                    self._num_features,
                    post_processing=None,
                    caching=self.caching,
                    primitive=self._primitive,
                )
            elif self._framework == "pennylane":
                self._fallback_engine = LowLevelQNNPennyLane(
                    self._pqc,
                    self._observable,
                    self._executor,
                    self._num_features,
                    post_processing=None,
                    caching=self.caching,
                )
            else:
                raise RuntimeError(f"Unsupported quantum framework: {self._framework}")
        return self._fallback_engine

    def get_params(self, deep: bool = True) -> dict:
        """Returns the dictionary of the hyper-parameters of the QNN.

        In case of multiple outputs, the hyper-parameters of the operator are prefixed
        with ``op0__``, ``op1__``, etc.
        """
        params = dict(num_qubits=self.num_qubits)
        params["primitive"] = self._primitive

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

        if "primitive" in params:
            self._primitive = params["primitive"]
            if self._fallback_engine is not None:
                self._fallback_engine.set_params(primitive=params["primitive"])
            params.pop("primitive")

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

    # NOTE: these intentionally return the *fallback* engine's parameter vectors, not the
    # native self._x/self._p/self._p_op used for the qc_executor path. Tuple-based derivative
    # specs built from these (e.g. `llqnn.parameters[0]`) are only ever evaluated through the
    # fallback (native keys are plain strings, see _evaluate), and the fallback's OpTree
    # differentiation matches parameters by object identity - it must see its own vectors.
    @property
    def parameters(self) -> Parameters:
        """Return the parameter vector of the PQC."""
        return self._fallback.parameters

    @property
    def features(self) -> Parameters:
        """Return the feature vector of the PQC."""
        return self._fallback.features

    @property
    def parameters_operator(self) -> Parameters:
        """Return the parameter vector of the cost operator."""
        return self._fallback.parameters_operator

    def _trailing_shape(self, key: str) -> tuple:
        """Shape of a single evaluation's result for `key`, excluding the x/param/param_op
        batch axes (matches the axis convention of :class:`LowLevelQNNQiskit`: output axis
        before the derivative axis)."""
        multi = self.multiple_output
        n_op = self.num_operator
        if key == "f":
            return (n_op,) if multi else ()
        if key == "dfdx":
            d = self.num_features
        elif key == "dfdp":
            d = self.num_parameters
        elif key == "dfdop":
            d = self.num_parameters_observable
        else:
            raise ValueError(f"Unknown native key: {key}")
        return (n_op, d) if multi else (d,)

    def _compute_native(self, key: str, parameters: dict):
        qc_exec = self._executor.as_qc_executor
        if key == "f":
            return qc_exec.expectation_value(
                self._native_circuit, self._native_observable, **parameters
            )
        if key == "dfdop" and self.multiple_output:
            # Each observable in the list only owns a slice of the shared "p_op" vector.
            # qc_executor's automatic list-observable collapse assumes every list entry
            # produces a result of the same shape, which fails here since the per-observable
            # "p_op" slices have different lengths. Loop per observable instead and place
            # each result into its slice; everywhere else df_i/dp_op_j is zero by construction.
            full = np.zeros((self.num_operator, self.num_parameters_observable))
            for i, (obs, (ioff, n)) in enumerate(
                zip(self._native_observable, self._observable_p_op_slices)
            ):
                if n == 0:
                    continue
                value = qc_exec.expectation_value_derivatives(
                    self._native_circuit, obs, "p_op", **parameters
                )
                full[i, ioff : ioff + n] = np.asarray(value, dtype=float).reshape(n)
            return full
        derivative_param = {"dfdx": "x", "dfdp": "p", "dfdop": "p_op"}[key]
        return qc_exec.expectation_value_derivatives(
            self._native_circuit, self._native_observable, derivative_param, **parameters
        )

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
            arr = np.zeros((len(x_inp), len(param_inp), len(param_op_inp)) + trailing, dtype=float)
            if 0 not in trailing:
                # A zero-sized trailing axis (e.g. dfdop with no observable parameters at
                # all) has nothing to fill in - qc_executor has no "empty" result to return.
                for ix, x_vec in enumerate(x_inp):
                    for ip, p_vec in enumerate(param_inp):
                        for iop, p_op_vec in enumerate(param_op_inp):
                            value = self._compute_native(
                                key, {"x": x_vec, "p": p_vec, "p_op": p_op_vec}
                            )
                            arr[ix, ip, iop, ...] = np.asarray(value, dtype=float).reshape(
                                trailing
                            )

            final_shape = []
            if multi_x:
                final_shape.append(len(x_inp))
            if multi_param:
                final_shape.append(len(param_inp))
            if multi_param_op:
                final_shape.append(len(param_op_inp))
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
            native_keys = [v for v in values if isinstance(v, str) and v in self._NATIVE_KEYS]
        else:
            native_keys = []
        fallback_keys = [v for v in values if v not in native_keys]

        result = {}
        if native_keys:
            result.update(self._evaluate_native(x, param, param_op, native_keys))
        if fallback_keys:
            result.update(self._fallback._evaluate(x, param, param_op, *fallback_keys))

        result["x"] = x
        result["param"] = param
        result["param_op"] = param_op
        return result
