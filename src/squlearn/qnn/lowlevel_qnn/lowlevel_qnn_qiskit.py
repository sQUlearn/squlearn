import numpy as np
from typing import Union

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement

from ...observables.observable_base import ObservableBase
from ...observables.observable_derivatives import ObservableDerivatives

from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...encoding_circuit.encoding_circuit_derivatives import EncodingCircuitDerivatives
from ...encoding_circuit.transpiled_encoding_circuit import TranspiledEncodingCircuit

from ...util.data_preprocessing import adjust_features, adjust_parameters, to_tuple
from ...util import Executor

from ...util.optree.optree import OpTreeList, OpTreeCircuit, OpTree

from .lowlevel_qnn_base import LowLevelQNNBase

import copy


class Expec:
    """Data structure that holds the set-up of derivative of the expectation value.

    Args:
        wave_function (Union[str, tuple, ParameterVectorElement]): Describes the wave function or
            its derivative. If tuple or ParameterVectorElement the differentiation with respect to
            the parameters in the tuple or with respect to the ParameterVectorElement is considered
        observable (str): String for the expectation value observable (``"O"``, ``"OO"``,
            ``"dop"``, ``"dopdop"``, ``"var"``).
        label (str): Label that is used for displaying or in the value dict of the QNN class.

    """

    def __init__(
        self,
        wave_function: Union[str, tuple, ParameterVectorElement],
        observable: str,
        label: str = "",
    ):
        self.wave_function = wave_function
        self.operator = observable
        self.label = label

    def __var_to_str(self, val: Union[str, tuple, ParameterExpression, ParameterVector]) -> str:
        """Converter for variables to string.

        Args:
            val (Union[str, tuple, ParameterExpression, ParameterVector]): Input that is converted
                to string

        Returns:
            String that contains the converted val variable

        """
        if isinstance(val, ParameterExpression):
            out_str = str(val.name)
        elif isinstance(val, ParameterVector):
            out_str = str(val.name)
        elif isinstance(val, tuple):
            out_str = "("
            for x in val:
                out_str += self.__var_to_str(x) + ","
            out_str += ")"
        elif isinstance(val, str):
            out_str = val
        else:
            out_str = str(val)
        return out_str

    def __repr__(self) -> str:
        """Build-in string conversion for Expec class."""
        return self.__str__()

    def __str__(self) -> str:
        """Build-in string conversion for Expec class."""
        return (
            "Expec("
            + self.__var_to_str(self.wave_function)
            + ","
            + self.__var_to_str(self.operator)
            + ","
            + self.__var_to_str(self.label)
            + ")"
        )

    def __len__(self) -> int:
        """Build-in length of Expec class (return 1)."""
        return 1

    def __eq__(self, other) -> bool:
        """Build-in comparison of two Expec class objects."""
        return (
            isinstance(other, self.__class__)
            and self.wave_function == other.wave_function
            and self.operator == other.operator
        )

    def __hash__(self) -> int:
        """Build-in hash function for Expec class."""
        return hash((self.wave_function, self.operator))

    @classmethod
    def from_string(cls, val: str):
        """Converts an input string to the Expec data structure.

        Args:
            String that defines the expectation value derivative

        Returns:
            Associated Expec object

        """

        if isinstance(val, str):
            if val == "f":
                return cls("I", "O", "f")
            elif val == "dfdx":
                return cls("dx", "O", "dfdx")
            elif val == "dfdxdx":
                return cls("dxdx", "O", "dfdxdx")
            elif val == "laplace":
                return cls("laplace", "O", "laplace")
            elif val == "laplace_dp":
                return cls("laplace_dp", "O", "laplace_dp")
            elif val == "laplace_dop":
                return cls("laplace", "dop", "laplace_dop")
            elif val == "dfdp":
                return cls("dp", "O", "dfdp")
            elif val == "dfdpdp":
                return cls("dpdp", "O", "dfdpdp")
            elif val == "dfdopdp":
                return cls("dp", "dop", "dfdop")
            elif val == "dfdpdop":
                raise ValueError("Not implemented, please use dfdopdp instead and transpose!")
            elif val == "dfdop":
                return cls("I", "dop", "dfdop")
            elif val == "dfdopdop":
                return cls("I", "dopdop", "dfdop")
            elif val == "dfdpdx":
                return cls("dpdx", "O", "dfdpdx")
            elif val == "dfdopdx":
                return cls("dx", "dop", "dfdopdx")
            elif val == "dfdopdxdx":
                return cls("dxdx", "dop", "dfdopdxdx")
            elif val == "fcc":
                return cls("I", "OO", "fcc")
            elif val == "dfccdx":
                return cls("dx", "OO", "dfccdx")
            elif val == "dfccdxdx":
                return cls("dxdx", "OO", "dfccdxdx")
            elif val == "dfccdp":
                return cls("dp", "OO", "dfccdp")
            elif val == "dfccdpdp":
                return cls("dpdp", "OO", "dfccdpdp")
            elif val == "dfccdopdx":
                return cls("dx", "OOdop", "dfccdpdp")
            elif val == "dfccdop":
                return cls("I", "OOdop", "dfccdop")
            elif val == "dfccdopdop":
                return cls("I", "OOdopdop", "dfccdopdop")
            elif val in ("var", "varf"):
                return cls("I", "var", val)
            elif val in ("dvardx", "dvarfdx"):
                return cls("dx", "var", val)
            elif val in ("dvardp", "dvarfdp"):
                return cls("dp", "var", val)
            elif val in ("dvardop", "dvarfdop"):
                return cls("I", "dvardop", val)
            elif val == "fischer":
                return cls("I", "fischer", val)
            else:
                raise ValueError("Unknown input string:", val)
        else:
            raise TypeError("String expected, found type:", type(val))

    @classmethod
    def from_tuple(cls, val: tuple, operator: str = "O", label=None):
        """Creates an Expec object from an input tuple

        Args:
            val (tuple): Tuple for the differentiation of the wave function.
            operator (str): String for the operator, default='O'.

        Returns
            Associated Expec object
        """

        circuit_tuple = tuple()  # Derivatives affecting the circuit
        observable_tuple = tuple()  # Derivatives affecting the observable

        # Split the derivative tuple into the circuit and observable part
        for i in val:
            if isinstance(i, ParameterVector) or isinstance(i, ParameterVectorElement):
                if "p_op" in i.name:
                    observable_tuple += (i,)
                elif "p" in i.name:
                    circuit_tuple += (i,)
                elif "x" in i.name:
                    circuit_tuple += (i,)
                else:
                    raise ValueError("Unknown parameter name:", i.name)
            else:
                raise ValueError("Unknown type:", type(i))

        if len(circuit_tuple) == 0:
            circuit_tuple = "I"
        if len(observable_tuple) == 0:
            observable_tuple = operator

        if label is None:
            label = val

        return cls(circuit_tuple, observable_tuple, label)

    @classmethod
    def from_parameter_vector(cls, val: ParameterVectorElement, operator: str = "O"):
        """Creates an Expec object from an inputted parameter

        Args:
            val (ParameterVectorElement): Parameter that is used in the differentiation.
            operator (str): String for the operator, default='O'.

        Returns
            Associated Expec object
        """
        return cls.from_tuple((val,), operator, val)

    @classmethod
    def from_parameter(cls, val: ParameterVectorElement, operator: str = "O"):
        """Creates an Expec object from an inputted parameter

        Args:
            val (ParameterVectorElement): Parameter that is used in the differentiation.
            operator (str): String for the operator, default='O'.

        Returns
            Associated Expec object
        """
        return cls.from_tuple((val,), operator, val)

    @classmethod
    def from_variable(cls, val):
        """Creates an Expec object from an inputted value

        Args:
            val (Union[Expec,str,tuple,ParameterVectorElement]): value that defines the derivative

        Returns
            Associated Expec object
        """

        if isinstance(val, Expec):
            return val
        elif isinstance(val, str):
            return cls.from_string(val)
        elif isinstance(val, tuple):
            return cls.from_tuple(val)
        elif isinstance(val, ParameterVector):
            return cls.from_parameter_vector(val)
        elif isinstance(val, ParameterVectorElement):
            return cls.from_parameter(val)
        else:
            raise TypeError("Unsupported type:", type(val))


class LowLevelQNNQiskit(LowLevelQNNBase):
    """Low level implementation of QNNs and its derivatives based on Qiskit.

    Args:
        pqc (EncodingCircuitBase) : parameterized quantum circuit in encoding circuit format
        operator (Union[ObservableBase,list]): Operator that is used in the expectation
            value of the QNN. Can be a list for multiple outputs.
        executor (Executor) : Executor that is used for the evaluation of the QNN
        caching : Caching of the result for each `x`, `param`, `param_op` combination
            (default = True)
        primitive (str): Primitive that is used for the evaluation of the QNN. Possible values are
            ``"estimator"`` or ``"sampler"``. If None, the primitive is set according to the
            executor. (default = None)

    Attributes:
    -----------

    Attributes:
        num_qubits (int): Number of qubits of the QNN
        num_features (int): Dimension of features of the PQC
        num_parameters (int): Number of trainable parameters of the PQC
        num_operator (int): Number of outputs
        num_parameters_observable (int): Number of trainable parameters of the expectation value operator
        multiple_output (bool): True if multiple outputs are used
        parameters (ParameterVector): Parameter vector of the PQC
        features (ParameterVector): Feature vector of the PQC
        parameters_operator (ParameterVector): Parameter vector of the cost operator

    Methods:
    --------
    """

    def __init__(
        self,
        parameterized_quantum_circuit: EncodingCircuitBase,
        operator: Union[ObservableBase, list],
        executor: Executor,
        caching=True,
        primitive: Union[str, None] = None,
    ) -> None:

        self.caching = caching

        if executor.backend_chosen:
            # Skip transpilation for parallel qpu execution
            if not executor.qpu_parallelization:
                parameterized_quantum_circuit = TranspiledEncodingCircuit(
                    parameterized_quantum_circuit, executor.backend
                )
        else:
            # Automatically select backend (also returns a TranspiledEncodingCircuit except
            # for parallel qpu execution)
            parameterized_quantum_circuit, _ = executor.select_backend(
                parameterized_quantum_circuit
            )

        super().__init__(parameterized_quantum_circuit, operator, executor)

        self.operator = copy.deepcopy(operator)

        # Set-Up Executor
        self._set_primitive(primitive)

        # Initialize derivative classes
        self._initilize_derivative()

    def _set_primitive(self, primitive: Union[str, None] = None) -> None:
        """
        Sets the primitive for evaluating the of the QNN.

        Args:
            primitive (str): Primitive that is used for the evaluation of the QNN.
                Can be ``"estimator"`` or ``"sampler"``. If None, the primitive is set
                according to the executor. (default = None)

        """
        if primitive is None:
            if self._executor.optree_executor == "estimator":
                self._estimator = self._executor.get_estimator()
                self._sampler = None
                self._primitive = "estimator"
            else:
                self._sampler = self._executor.get_sampler()
                self._estimator = None
                self._primitive = "sampler"
        else:
            if primitive.lower() == "estimator":
                self._estimator = self._executor.get_estimator()
                self._sampler = None
                self._primitive = "estimator"
            elif primitive.lower() == "sampler":
                self._sampler = self._executor.get_sampler()
                self._estimator = None
                self._primitive = "sampler"
            else:
                raise ValueError("Unknown primitive:", primitive)

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
                    oper_dict = oper.get_params()
                    for key, value in oper_dict.items():
                        if key != "num_qubits":
                            params["op" + str(i) + "__" + key] = value
            else:
                params.update(self._observable.get_params())
        return params

    def set_params(self, **params) -> None:
        """Sets the hyper-parameters of the QNN

        In case of multiple outputs, the hyper-parameters of the operator are prefixed
        with ``op0__``, ``op1__``, etc.

        Args:
            params: Hyper-parameters that are adjusted, e.g. ``num_qubits=4``

        """

        # Check if all parameters are valid
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        if "primitive" in params:
            self._set_primitive(params["primitive"])
            params.pop("primitive")

        # Set parameters of the PQC
        dict_pqc = {}
        for key, value in params.items():
            if key in self._pqc.get_params():
                dict_pqc[key] = value
        if len(dict_pqc) > 0:
            self._pqc.set_params(**dict_pqc)

        # Set parameters of the operator
        if isinstance(self._observable, list):
            for i, oper in enumerate(self._observable):
                dict_operator = {}
                for key, value in params.items():
                    if key == "num_qubits":
                        dict_operator[key] = value
                    else:
                        if key.startswith("op" + str(i) + "__"):
                            dict_operator[key.split("__", 1)[1]] = value
                if len(dict_operator) > 0:
                    oper.set_params(**dict_operator)
        else:
            dict_operator = {}
            for key, value in params.items():
                if key in self._observable.get_params():
                    dict_operator[key] = value
            if len(dict_operator) > 0:
                self._observable.set_params(**dict_operator)

        self._initilize_derivative()

    def _initilize_derivative(self):
        """Initializes the derivative classes"""

        num_qubits_operator = 0
        if isinstance(self._observable, list):
            for i in range(len(self._observable)):
                self._observable[i].set_map(self._pqc.qubit_map, self._pqc.num_physical_qubits)
                num_qubits_operator = max(num_qubits_operator, self._observable[i].num_qubits)
        else:
            self._observable.set_map(self._pqc.qubit_map, self._pqc.num_physical_qubits)
            num_qubits_operator = self._observable.num_qubits

        self.operator_derivatives = ObservableDerivatives(self._observable)
        self.pqc_derivatives = EncodingCircuitDerivatives(self._pqc)

        if self._pqc.num_virtual_qubits != num_qubits_operator:
            raise ValueError("Number of Qubits are not the same!")
        else:
            self._num_qubits = self._pqc.num_virtual_qubits

        if self._executor.optree_executor == "sampler":
            # In case of the sampler primitive, X and Y Pauli matrices have to be treated extra
            # This can be very inefficient!
            operator_string = str(self._observable)
            if "X" in operator_string or "Y" in operator_string:
                self._split_paulis = True
                print(
                    "The observable includes X and Y gates, consider switching"
                    + " to the Estimator primitive for a faster performance!"
                )
            else:
                self._split_paulis = False
        else:
            self._split_paulis = False

        # Initialize result cache
        self.result_container = {}

    def set_shots(self, num_shots: int) -> None:
        """Sets the number shots for the next evaluations.

        Args:
            num_shots (int): Number of shots that are set
        """

        self._executor.set_shots(num_shots)

    def get_shots(self) -> int:
        """Getter for the number of shots.

        Returns:
            Returns the number of shots that are used for the current evaluation."""
        return self._executor.get_shots()

    def reset_shots(self) -> None:
        """Function for resetting the number of shots to the initial ones"""
        self._executor.reset_shots()

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN"""
        return self._num_qubits

    @property
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC"""
        return self.pqc_derivatives.num_features

    @property
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC"""
        return self.pqc_derivatives.num_parameters

    @property
    def num_operator(self) -> int:
        """Return the number outputs"""
        return self.operator_derivatives.num_operators

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        return self.operator_derivatives.num_parameters

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        return self.operator_derivatives.multiple_output

    @property
    def parameters(self) -> ParameterVector:
        """Return the parameter vector of the PQC."""
        return self.pqc_derivatives.parameter_vector

    @property
    def features(self) -> ParameterVector:
        """Return the feature vector of the PQC."""
        return self.pqc_derivatives.feature_vector

    @property
    def parameters_operator(self) -> ParameterVector:
        """Return the parameter vector of the cost operator."""
        return self.operator_derivatives.parameter_vector

    def get_optree_from_string(self, input_string: str):
        """Return the OpTree expression of the given PQC

        Args:
            input_string (str): String from which the OpTree is obtained.

        Returns:
            OpTree structure created from the string.
        """
        return self.get_optree_from_expec(Expec.from_string(input_string))

    def get_optree_from_expec(self, input_expec: Expec):
        """Returns the OpTree expression for the given :class:`Expec` object.

        Args:
            input_expec (Expec): :class:`Expec` object from which the OpTree is obtained

        Returns:
            OpTree structure created from the :class:`Expec` object.
        """
        return OpTree.gen_expectation_tree(
            self.pqc_derivatives.get_derivative(input_expec.wave_function),
            self.operator_derivatives.get_derivative(input_expec.operator),
        )

    def evaluate_diff_tuple(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
        diff_tuple,
    ) -> Union[float, np.ndarray]:
        """Evaluate the given tuple of derivatives of the PQC.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator
            diff_tuple: Tuple with parameters used in the differentiation

        Returns:
            Differentiated values of the QNN
        """
        return self.evaluate(x, param, param_op, diff_tuple)[diff_tuple]

    def evaluate_from_string(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
        input_string: str,
    ) -> Union[float, np.ndarray]:
        """Evaluate the given PQC from an input string

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator
            input_string (str): Input string that determines the evaluated value(s)

        Returns:
            Values from the QNN defined by the string
        """
        return self.evaluate(x, param, param_op, input_string)[input_string]

    def evaluate_f(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Values from the QNN
        """
        return self.evaluate_from_string(x, param, param_op, "f")

    def evaluate_dfdx(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates derivatives of the QNN with respect to `x`.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated derivatives of the the QNN with respect to `x`
        """
        return self.evaluate_from_string(x, param, param_op, "dfdx")

    def evaluate_dfdxdx(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates second order derivatives of the QNN with respect to `x`.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated second order derivatives of the the QNN with respect to `x`
        """
        return self.evaluate_from_string(x, param, param_op, "dfdxdx")

    def evaluate_laplace(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates Laplacian of the QNN for `x`.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated Laplacian of the the QNN for `x`
        """
        return self.evaluate_from_string(x, param, param_op, "laplace")

    def evaluate_laplace_dp(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Evaluates the derivative of the Laplacian with respect to the PQC's parameters.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated derivative of the Laplacian with respect to the PQC's parameters
        """
        return self.evaluate_from_string(x, param, param_op, "laplace_dp")

    def evaluate_laplace_dop(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates the derivative of the Laplacian with respect to the operator's parameters.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated derivative of the Laplacian with respect to the operator's parameters
        """
        return self.evaluate_from_string(x, param, param_op, "laplace_dop")

    def evaluate_dfdp(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates the derivative of the QNN with respect to the PQC's parameters.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated derivative of the the QNN with respect to the PQC's parameters.
        """
        return self.evaluate_from_string(x, param, param_op, "dfdp")

    def evaluate_dfdop(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates the derivative of the QNN with respect to the operator's parameters.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated derivative of the the QNN with respect to the operator's parameters.
        """
        return self.evaluate_from_string(x, param, param_op, "dfdop")

    def evaluate_dfdpdx(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates the derivative of the QNN with respect to the PQC's parameters and `x`.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated derivative of the QNN with respect to the PQC's parameters and `x`
        """
        return self.evaluate_from_string(x, param, param_op, "dfdpdx")

    def evaluate_dfdopdx(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates the derivative of the QNN with respect to the operator's parameters and `x`.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated derivative of the QNN with respect to the operator's parameters and `x`
        """
        return self.evaluate_from_string(x, param, param_op, "dfdopdx")

    def evaluate_variance(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates the variance (<OO>-<O>^2) of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated variance of the the QNN
        """
        return self.evaluate_from_string(x, param, param_op, "var")

    def evaluate_var(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates the variance (<OO>-<O>^2) of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated variance of the the QNN
        """
        return self.evaluate_variance(x, param, param_op)

    def evaluate_probabilities(self, x: Union[float, np.ndarray], param: Union[float, np.ndarray]):
        """Evaluate the probabilities of the encoding circuit / PQC.

        The function only works with the QuantumInstance executer.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC

        Returns:
            List of probabilities stored in the SparseVectorStateFn format.
            (dictionary can be obtained by `.to_dict_fn()` or `to_dict_fn().primitive`)
        """

        optree = self.pqc_derivatives.get_derivative("I")
        dictionary = dict(zip(self.parameters, param))
        dictionary.update(zip(self.features, x))
        OpTree.assign_parameters(optree, dictionary, inplace=True)

        if isinstance(optree, QuantumCircuit):
            circuit = optree
        elif isinstance(optree, OpTreeCircuit):
            circuit = optree.circuit
        else:
            raise TypeError("Unsported optree type:", type(optree))

        if circuit.num_clbits == 0:
            circuit.measure_all()

        sampler = self._executor.get_sampler()
        result = sampler.run(circuit).result()
        return result.quasi_dists[0].binary_probabilities()

    def evaluate(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
        *values: Union[
            str,
            Expec,
            ParameterVector,
            ParameterVectorElement,
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

        def generate_real_todo_dic(values, value_dict):
            """Converts the input values into a sorted dictionary
            of of Expec items"""

            # helper function for adding elemets to the real todo dict
            def add_to_real_todo_dic(item: Expec, real_todo_dic, value_dict):
                if item not in value_dict:
                    if item.wave_function in real_todo_dic:
                        #  check if i is already in the real todo list
                        if item not in real_todo_dic[item.wave_function]:
                            real_todo_dic[item.wave_function].append(item)
                    else:
                        real_todo_dic[item.wave_function] = [item]
                return real_todo_dic

            # labels can be overwritten
            try:
                expec_list = [Expec.from_variable(i) for i in values]
            except TypeError:
                expec_list = [Expec.from_variable(values)]
            # build dictionary for later use
            real_todo_dic = {}
            for i in expec_list:
                # special cases of variance computation for post-processing:
                if i.operator == "var" and i.wave_function == "I":
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("I", "OO"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("I", "O"), real_todo_dic, value_dict
                    )
                elif i.operator == "var" and i.wave_function == "dx":
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("dx", "OO"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("I", "O"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("dx", "O"), real_todo_dic, value_dict
                    )
                elif i.operator == "var" and i.wave_function == "dp":
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("dp", "OO"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("I", "O"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("dp", "O"), real_todo_dic, value_dict
                    )
                elif i.operator == "dvardop" and i.wave_function == "I":
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("I", "OOdop"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("I", "O"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        Expec("I", "dop"), real_todo_dic, value_dict
                    )
                else:
                    real_todo_dic = add_to_real_todo_dic(i, real_todo_dic, value_dict)
            return real_todo_dic

        # Done with the helper functions, start of the evaluate function

        # input adjustments for x, param, param_op to get correct stacking of values
        x_inp, multi_x = adjust_features(x, self.num_features)
        param_inp, multi_param = adjust_parameters(param, self.num_parameters)
        param_op_inp, multi_param_op = adjust_parameters(param_op, self.num_parameters_observable)

        # build dictionary for later use
        dict_encoding_circuit = []
        for x_inp_ in x_inp:
            dd = dict(zip(self.pqc_derivatives.feature_vector, x_inp_))
            for param_inp_ in param_inp:
                ddd = dd.copy()
                ddd.update(zip(self.pqc_derivatives.parameter_vector, param_inp_))
                dict_encoding_circuit.append(ddd)
        dict_operator = [
            dict(zip(self.operator_derivatives.parameter_vector, p)) for p in param_op_inp
        ]

        # Sort the values, more complicated because values can be tuples of ParameterVectors
        indices = np.argsort([str(t) for t in values])
        values = tuple([values[i] for i in indices])

        # return dictionary for input data, it will be empty
        # if the combination of x,param,param_op is touched the first time
        if self.caching == True:
            caching_tuple = (
                to_tuple(x),
                to_tuple(param),
                to_tuple(param_op),
                (self._executor.shots == None),
            )
            value_dict = self.result_container.get(caching_tuple, {})
        else:
            value_dict = {}

        # create dictionary sorted w.r.t. the circuits
        # expectation values with the same circuits are evaluated only once
        # variance set-up is created here
        real_todo_dic = generate_real_todo_dic(values, value_dict)

        for key, op_list in real_todo_dic.items():
            # Obtained the derivative from the operator module

            operators = OpTreeList(
                [self.operator_derivatives.get_derivative(expec_.operator) for expec_ in op_list]
            )

            # get the circuits of the PQC derivatives from the encoding circuit module
            pqc_optree = self.pqc_derivatives.get_derivative(key)
            num_nested = OpTree.get_num_nested_lists(pqc_optree)

            if self._sampler is not None:
                val = OpTree.evaluate.evaluate_with_sampler(
                    pqc_optree, operators, dict_encoding_circuit, dict_operator, self._sampler
                )
            elif self._estimator is not None:
                val = OpTree.evaluate.evaluate_with_estimator(
                    pqc_optree, operators, dict_encoding_circuit, dict_operator, self._estimator
                )
            else:
                raise ValueError("No execution is set!")

            is_val_empty = False
            if val.size == 0:
                is_val_empty = True

            # val has the following order:
            # 0 -> x_inp and param_inp combined (has to be separated later)
            # 1 -> param_op_inp (often just shape 1)
            # 2:2+num_nested -> circuit derivatives of parameter shift (if needed)
            #                   (num_nested = order of derivatives)
            # 2+num_nested ->  different observables (op_list - todos for same circuit)
            # 3+num_nested:last-1 -> observable derivatives (if needed)
            # last : multi_output

            # Swapp results into the following final order:
            # 1. different observables (op_list - todo for same circuit)
            # 2. different input data/ encoding circuit parameters (x_inp,params) -> separated later
            # 3. different operator parameters (param_op_inp)
            # 4. different output values (multi_output)
            # 5. If there, lists of the operators (e.g. operator derivatives)
            # 6. if there, lists of the circuits (e.g. array for gradient)
            # swapping is done in two stages, first the op_list are swapped to the first position
            # afterward the order of each element in the op_list is adjusted

            # FIRST SWAP: different observables to the first place of the array
            # swap i=2+num_nested to position 0, keep the rest in order
            if not is_val_empty:
                index_list = list(range(len(val.shape)))
                swapp_list = [index_list[2 + num_nested]]
                swapp_list += [index_list[0]] + [index_list[1]]
                if num_nested > 0:
                    swapp_list += index_list[2 : 2 + num_nested]
                swapp_list += index_list[3 + num_nested :]
                val = np.transpose(val, axes=swapp_list)

            # store results in value_dict
            for iexpec, expec_ in enumerate(op_list):
                if is_val_empty:
                    value_dict[expec_] = np.array([])
                else:
                    # take array associated with op_list, convert to numpy if needed
                    if isinstance(val[iexpec], object):
                        # tolist() is needed, since numpy array conversion is otherwise hanging
                        val_final = np.array(val[iexpec].tolist(), dtype=float)
                    else:
                        val_final = val[iexpec]

                    # SECOND SWAP
                    # swap the multiple outputs to third position
                    # swap the observable derivatives to the second position
                    # swap the circuit derivatives to the last position
                    index_list = list(range(len(val_final.shape)))
                    swapp_list = [0, 1]
                    if self.multiple_output:
                        swapp_list += [index_list.pop()]
                    if len(index_list) > 2 + num_nested:
                        swapp_list += index_list[2 + num_nested :]
                    if num_nested > 0:
                        swapp_list += index_list[2 : 2 + num_nested]
                    val_final = np.transpose(val_final, axes=swapp_list)

                    # Reshape for the final output, adjust x and param if needed
                    reshape_list = []
                    shape = val_final.shape
                    if multi_x:
                        reshape_list.append(len(x))
                    if multi_param:
                        reshape_list.append(len(param))
                    if multi_param_op:
                        reshape_list.append(shape[1])
                    if self.multiple_output:
                        reshape_list.append(shape[2])
                    if self.multiple_output:
                        if len(shape) > 3:
                            reshape_list += list(shape[3:])
                    else:
                        if len(shape) > 2:
                            reshape_list += list(shape[2:])
                    if len(reshape_list) == 0:
                        value_dict[expec_] = val_final.reshape(-1)[0]
                    else:
                        value_dict[expec_] = val_final.reshape(reshape_list)

        # Set-up lables from the input list
        for todo in values:
            todo_expec = Expec.from_variable(todo)

            # post-processing of the variance
            # variance
            if todo_expec.operator == "var" and todo_expec.wave_function == "I":
                value_dict[todo_expec] = value_dict[Expec("I", "OO")] - np.square(
                    value_dict[Expec("I", "O")]
                )
            # d/dx variance
            elif todo_expec.operator == "var" and todo_expec.wave_function == "dx":
                if self.num_features == 1:
                    value_dict[todo_expec] = value_dict[Expec("dx", "OO")] - 2.0 * (
                        np.multiply(value_dict[Expec("dx", "O")], value_dict[Expec("I", "O")])
                    )
                else:
                    value_dict[todo_expec] = np.zeros(value_dict[Expec("dx", "OO")].shape)
                    for i in range(value_dict[Expec("dx", "OO")].shape[-1]):
                        value_dict[todo_expec][..., i] = value_dict[Expec("dx", "OO")][
                            ..., i
                        ] - 2.0 * (
                            np.multiply(
                                value_dict[Expec("dx", "O")][..., i],
                                value_dict[Expec("I", "O")],
                            )
                        )
            # d/dp variance
            elif todo_expec.operator == "var" and todo_expec.wave_function == "dp":
                value_dict[todo_expec] = np.zeros(value_dict[Expec("dp", "OO")].shape)
                for i in range(value_dict[Expec("dp", "OO")].shape[-1]):
                    value_dict[todo_expec][..., i] = value_dict[Expec("dp", "OO")][
                        ..., i
                    ] - 2.0 * (
                        np.multiply(
                            value_dict[Expec("dp", "O")][..., i],
                            value_dict[Expec("I", "O")],
                        )
                    )
            # d/dop variance
            elif todo_expec.operator == "dvardop" and todo_expec.wave_function == "I":
                value_dict[todo_expec] = np.zeros(value_dict[Expec("I", "OOdop")].shape)
                for i in range(value_dict[Expec("I", "OOdop")].shape[-1]):
                    value_dict[todo_expec][..., i] = value_dict[Expec("I", "OOdop")][
                        ..., i
                    ] - 2.0 * (
                        np.multiply(
                            value_dict[Expec("I", "dop")][..., i],
                            value_dict[Expec("I", "O")],
                        )
                    )

            # assign values to the label of the expectation value
            value_dict[todo] = value_dict[todo_expec]
            if isinstance(todo, Expec) and todo.label != "":
                value_dict[todo.label] = value_dict[todo_expec]

        # Add x, param, and param_op to the dictionary as default
        value_dict["x"] = x
        value_dict["param"] = param
        value_dict["param_op"] = param_op

        # Store the updated dictionary for the theta value
        if self.caching:
            self.result_container[caching_tuple] = value_dict

        return value_dict
