from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.opflow import OperatorBase, CircuitStateFn, OperatorStateFn
from qiskit.opflow import ListOp, SummedOp, ComposedOp, TensoredOp
from qiskit.opflow import Zero, One
from qiskit.opflow.list_ops.list_op import ListOp as real_ListOp
from qiskit.opflow.expectations import PauliExpectation
from qiskit.utils import QuantumInstance

from typing import Union
import numpy as np

from ..expectation_operator.expectation_operator_base import ExpectationOperatorBase
from ..expectation_operator.expectation_operator_derivatives import ExpectationOperatorDerivatives

from ..feature_map.feature_map_base import FeatureMapBase
from ..feature_map.feature_map_derivatives import (
    FeatureMapDerivatives,
    measure_feature_map_derivative,
)
from ..feature_map.transpiled_feature_map import TranspiledFeatureMap

from ..util.data_preprocessing import adjust_input

from ..util import Executor


class expec:
    """Data structure that holds the set-up of derivative of the expectation value.

    Args:
        wavefunction (Union[str, tuple, ParameterVectorElement]): Describes the wavefuction or its
            derivative. If tuple or ParameterVectorElement the differentiation with respect to the
            parameters in the tuple or with respect to the ParameterVectorElement is considered
        operator (str): String for the expectation value operator (O, OO, dop, dopdop, var).
        label (str): Label that is used for displaying or in the value dict of the QNN class.

    """

    def __init__(
        self,
        wavefunction: Union[str, tuple, ParameterVectorElement],
        operator: str,
        label: str = "",
    ):
        self.wavefunction = wavefunction
        self.operator = operator
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
        """Build-in string conversion for expec class."""
        return self.__str__()

    def __str__(self) -> str:
        """Build-in string conversion for expec class."""
        return (
            "expec("
            + self.__var_to_str(self.wavefunction)
            + ","
            + self.__var_to_str(self.operator)
            + ","
            + self.__var_to_str(self.label)
            + ")"
        )

    def __len__(self) -> int:
        """Build-in length of expec class (return 1)."""
        return 1

    def __eq__(self, other) -> bool:
        """Build-in comparison of two expec class objects."""
        return (
            isinstance(other, self.__class__)
            and self.wavefunction == other.wavefunction
            and self.operator == other.operator
        )

    def __hash__(self) -> int:
        """Build-in hash function for expec class."""
        return hash((self.wavefunction, self.operator))

    @classmethod
    def from_string(cls, val: str):
        """Converts an input string to the expec data structure.

        Args:
            String that defines the expectation value derivative

        Returns:
            Associated expec object

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
    def from_tuple(cls, val: tuple, operator: str = "O"):
        """Creates an expec object from an input tuple

        Args:
            val (tuple): Tuple for the differentiation of the wavefunction.
            operator (str): String for the operator, default='O'.

        Returns
            Associated expec object
        """
        return cls(val, operator, val)

    @classmethod
    def from_parameter(cls, val: ParameterVectorElement, operator: str = "O"):
        """Creates an expec object from an inputted parameter

        Args:
            val (ParameterVectorElement): Parameter that is used in the differentiation.
            operator (str): String for the operator, default='O'.

        Returns
            Associated expec object
        """
        return cls((val,), operator, (val,))

    @classmethod
    def from_variable(cls, val):
        """Creates an expec object from an inputted value

        Args:
            val (Union[expec,str,tuple,ParameterVectorElement]): value that defines the derivative

        Returns
            Associated expec object
        """

        if isinstance(val, expec):
            return val
        elif isinstance(val, str):
            return cls.from_string(val)
        elif isinstance(val, tuple):
            return cls.from_tuple(val)
        elif isinstance(val, ParameterVectorElement):
            return cls.from_parameter(val)
        else:
            raise TypeError("Unsupported type:", type(val))


class QNN:
    """A class for working with QNNs and its derivatives

    Args:
        pqc (FeatureMapBase) : parameterized quantum circuit in feature map format
        operator (Union[ExpectationOperatorBase,list]): Operator that are used in the expectation value
            of the QNN. Can be a list for multiple outputs.
        executor (Executor) : Executor that is used for the evaluation of the QNN
        opflow_caching : Caching of the opflow expressions (default = True recommended)
        result_caching : Caching of the result for each x,param,param_op combination
            (default = True)
    """

    def __init__(
        self,
        pqc: FeatureMapBase,
        operator: Union[ExpectationOperatorBase, list],
        executor: Executor,
        opflow_caching=True,
        result_caching=True,
    ) -> None:
        # Potential TODO: move executor into extra util class
        # Executer set-up
        self.executor = executor
        self.backend = self.executor.backend

        # Storing the input data # TODO: custom transpilation function
        self.pqc = TranspiledFeatureMap(pqc, self.backend)
        self.operator = operator
        num_qubits_operator = 0
        if isinstance(self.operator, list):
            for i in range(len(self.operator)):
                self.operator[i].set_map(self.pqc.qubit_map, self.pqc.num_physical_qubits)
                num_qubits_operator = max(num_qubits_operator, self.operator[i].num_qubits)
        else:
            self.operator.set_map(self.pqc.qubit_map, self.pqc.num_physical_qubits)
            num_qubits_operator = self.operator.num_qubits

        self.operator_derivatives = ExpectationOperatorDerivatives(self.operator, opflow_caching)
        self.pqc_derivatives = FeatureMapDerivatives(self.pqc, opflow_caching)

        if self.pqc.num_virtual_qubits != num_qubits_operator:
            raise ValueError("Number of Qubits are not the same!")
        else:
            self._num_qubits = self.pqc.num_virtual_qubits

        if self.executor.get_opflow_executor() in ("sampler", "quantum_instance"):
            # For Quantum Instance or the Sampler primitive, X and Y Pauli matrices have to be treated extra
            # This is very inefficient!
            operator_string = str(self.operator)
            if "X" in operator_string or "Y" in operator_string:
                self.split_paulis = True
                print(
                    "The expectation operator includes X and Y gates, consider switching"
                    + " to the Estimator primitive for a faster performance!"
                )
            else:
                self.split_paulis = False
        else:
            self.split_paulis = False

        # Set-up shots from backend
        self._inital_shots = self.executor.get_shots()

        # Initialize result cache
        self._result_caching = result_caching
        self.result_container = {}

    def set_shots(self, num_shots: int) -> None:
        """Sets the number shots for the next evaluations.

        Args:
            num_shots (int): Number of shots that are set
        """

        self.executor.set_shots(num_shots)

    def get_shots(self) -> int:
        """Getter for the number of shots.

        Returns:
            Returns the number of shots that are used for the current evaluation."""
        return self.executor.get_shots()

    def reset_shots(self) -> None:
        """Function for resetting the number of shots to the inital ones"""
        self.executor.reset_shots()

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
    def num_parameters_operator(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        return self.operator_derivatives.num_parameters

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        return self.operator_derivatives.multiple_output

    @property
    def parameters(self):
        """Return the parameter vector of the PQC."""
        return self.pqc_derivatives._p

    @property
    def features(self):
        """Return the feature vector of the PQC."""
        return self.pqc_derivatives._x

    @property
    def parameters_operator(self):
        """Return the parameter vector of the cost operator."""
        return self.operator_derivatives._parameter_vector

    def get_opflow_from_string(self, input_string: str):
        """Return the opflow expression of the given PQC

        Args:
            input_string (str): String from which the opflow is obtained.

        Returns:
            Opflow structure created from the string.
        """
        return self.get_opflow_from_expec(expec.from_string(input_string))

    def get_opflow_from_expec(self, input_expec: expec):
        """Returns the opflow expression for the given expec object.

        Args:
            input_expec (expec): Expec pbject from which the opflow is obtained

        Returns:
            Opflow structure created from the expec object.
        """
        return measure_feature_map_derivative(
            self.pqc_derivatives.get_derivative(input_expec.wavefunction),
            self.operator_derivatives.get_derivative(input_expec.operator),
        )

    def evaluate_diff_tuple(
        self,
        diff_tuple,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluate the given tuple of derivatives of the PQC.

        Args:
            diff_tuple: Tuple with parameters used in the differentiation
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Differentiated values of the QNN
        """
        return self.evaluate((diff_tuple,), x, param, param_op)[diff_tuple]

    def evaluate_from_string(
        self,
        input_string: str,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluate the given PQC from an input string

        Args:
            input_string (str): Input string that determines the evaluated value(s)
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Values from the QNN defined by the string
        """
        return self.evaluate(input_string, x, param, param_op)[input_string]

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
        return self.evaluate_from_string("f", x, param, param_op)

    def evaluate_dfdx(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates d/dx of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated d/dx of the the QNN
        """
        return self.evaluate_from_string("dfdx", x, param, param_op)

    def evaluate_dfdxdx(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates d^2/dxdx of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated d^2/dxdx of the the QNN
        """
        return self.evaluate_from_string("dfdxdx", x, param, param_op)

    def evaluate_laplace(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates laplace(x) of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated laplace(x) of the the QNN
        """
        return self.evaluate_from_string("laplace", x, param, param_op)

    def evaluate_laplace_dp(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates d laplace(x)/dp of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated d laplace(x)/dp of the the QNN
        """
        return self.evaluate_from_string("laplace_dp", x, param, param_op)

    def evaluate_laplace_dop(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates d laplace(x)/dop of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated d laplace(x)/dop of the the QNN
        """
        return self.evaluate_from_string("laplace_dop", x, param, param_op)

    def evaluate_dfdp(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates d/dp of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated d/dp of the the QNN
        """
        return self.evaluate_from_string("dfdp", x, param, param_op)

    def evaluate_dfdop(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates d/dop of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated d/dop of the the QNN
        """
        return self.evaluate_from_string("dfdop", x, param, param_op)

    def evaluate_dfdpdx(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates d^2/dpdx of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated d^2/dpdx of the the QNN
        """
        return self.evaluate_from_string("dfdpdx", x, param, param_op)

    def evaluate_dfdopdx(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Evaluates d^2/dopdx of the QNN

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC
            param_op (Union[float,np.ndarray]): Parameter values of the operator

        Returns:
            Evaluated d^2/dopdx of the the QNN
        """
        return self.evaluate_from_string("dfdopdx", x, param, param_op)

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
        return self.evaluate_from_string("var", x, param, param_op)

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
        """Evaluate the probabilities of the feature map / PQC.

        The function only works with the QuantumInstance executer.

        Args:
            x (Union[float,np.ndarray]): Input data values
            param (Union[float,np.ndarray]): Parameter values of the PQC

        Returns:
            List of probabilities stored in the SparseVectorStateFn format.
            (dictionary can be obtained by .to_dict_fn() or to_dict_fn().primitive)
        """
        # TODO: Implementation with Sampler
        opflow = self.get_opflow_from_string("f")
        opflow_with_param = self.pqc_derivatives.assign_parameters(opflow, x, param)
        if self.quantum_instance is not None:
            return evaluate_opflow_qi(self.quantum_instance, opflow_with_param)
        else:
            raise RuntimeError("Probabilities are only implemented for Quantum Instances!")

    def evaluate(
        self,
        values,  # TODO: data type definition missing Union[str,expec,tuple,...]
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_op: Union[float, np.ndarray],
    ) -> dict:
        """General function for evaluating the output of derivatives of the QNN.

        Evaluation works for given combination of
        input features x and parameters param and param_op.
        The function includes caching of results

        If x, param, and/or param_op are given as a nested list
        (for example multiple sets of parameters),
        the values are returned in a nested list.

        Args:
            values : list of what values and derivatives of the qnn are eveluated.
                Multiple inputs have to be a tuple.
            x (np.ndarray): Values of the input feature data.
            param (np.ndarray): Parameter values of the PQC parameters
            param_op (np.ndarray): Parameter values of the operator parameters


        Results:
            Returns a dictionary with the computed values.
            The keys of the dictionary are given by the entries in the values tuple

        """

        def generate_real_todo_dic(values, value_dict):
            """Converts the input values into a sorted dictionary
            of of expec items"""

            # helper function for adding elemets to the real todo dict
            def add_to_real_todo_dic(item: expec, real_todo_dic, value_dict):
                if item not in value_dict:
                    if item.wavefunction in real_todo_dic:
                        #  check if i is already in the real todo list
                        if item not in real_todo_dic[item.wavefunction]:
                            real_todo_dic[item.wavefunction].append(item)
                    else:
                        real_todo_dic[item.wavefunction] = [item]
                return real_todo_dic

            # labels can be overwritten
            try:
                expec_list = [expec.from_variable(i) for i in values]
            except TypeError:
                expec_list = [expec.from_variable(values)]
            # build dictionary for later use
            real_todo_dic = {}
            for i in expec_list:
                # special cases of variance computation for post-processing:
                if i.operator == "var" and i.wavefunction == "I":
                    real_todo_dic = add_to_real_todo_dic(
                        expec("I", "OO"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        expec("I", "O"), real_todo_dic, value_dict
                    )
                elif i.operator == "var" and i.wavefunction == "dx":
                    real_todo_dic = add_to_real_todo_dic(
                        expec("dx", "OO"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        expec("I", "O"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        expec("dx", "O"), real_todo_dic, value_dict
                    )
                elif i.operator == "var" and i.wavefunction == "dp":
                    real_todo_dic = add_to_real_todo_dic(
                        expec("dp", "OO"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        expec("I", "O"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        expec("dp", "O"), real_todo_dic, value_dict
                    )
                elif i.operator == "dvardop" and i.wavefunction == "I":
                    real_todo_dic = add_to_real_todo_dic(
                        expec("I", "OOdop"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        expec("I", "O"), real_todo_dic, value_dict
                    )
                    real_todo_dic = add_to_real_todo_dic(
                        expec("I", "dop"), real_todo_dic, value_dict
                    )
                else:
                    real_todo_dic = add_to_real_todo_dic(i, real_todo_dic, value_dict)
            return real_todo_dic

        def to_tuple(x):
            """helper function for converting data into hashable tuples"""

            def flatten(container):
                for i in container:
                    if isinstance(i, (list, tuple, np.ndarray)):
                        for j in flatten(i):
                            yield j
                    else:
                        yield i

            if isinstance(x, float):
                return tuple([x])
            elif len(np.shape(x)) == 1:
                return tuple(list(x))
            else:
                return tuple(flatten(x))

        def measure_to_list(measure_op, offset=0):
            """
            Creates a list the numbers multpile measuremnt operators for later resorting
            """

            measure_list = []
            global countervar
            countervar = offset - 1

            def build_circuit_list(operator: OperatorBase):
                if isinstance(operator, CircuitStateFn):
                    raise RuntimeError("No circuits are allows at this point!")
                elif isinstance(operator, OperatorStateFn):
                    global countervar
                    countervar = countervar + 1
                    measure_list.append(operator)
                    return countervar
                elif isinstance(operator, ListOp):
                    list = []
                    for op in operator.oplist:
                        list.append(build_circuit_list(op))
                    return list

            index_list = build_circuit_list(measure_op)

            return index_list, measure_list

        def sort_back_to_nparray(index_list, val):
            """
            Uses the list generated by measure_to_list to sort back the values into the nested
            output list structure
            """

            def sort_back_val(index_list):
                if isinstance(index_list, list):
                    return [sort_back_val(op) for op in index_list]
                elif isinstance(index_list, int):
                    return val[index_list]
                else:
                    raise RuntimeError("Wrong format of inputed index list")

            return sort_back_val(index_list)

        # Done with the helper functions, start of the evaluate function

        # input adjustments for x, param, param_op to get correct stacking of values
        x_inp, multi_x = adjust_input(x, self.num_features)
        param_inp, multi_param = adjust_input(param, self.num_parameters)
        param_op_inp, multi_param_op = adjust_input(param_op, self.num_parameters_operator)

        # If values is not a tuple, convert it
        if not isinstance(values, tuple):
            values = (values,)

        # return dictionary for input data, it will be empty
        # if the combination of x,param,param_op is touched the first time
        if self._result_caching == True:
            caching_tuple = (to_tuple(x), to_tuple(param), to_tuple(param_op))
            value_dict = self.result_container.get(caching_tuple, {})
        else:
            value_dict = {}

        # create dictionary sorted w.r.t. the circuits
        # expectation values with the same circuits are evaluated only once
        # variance set-up is created here
        real_todo_dic = generate_real_todo_dic(values, value_dict)

        for key, op_list in real_todo_dic.items():
            # creates list of operators that can be evaluated for the same circuit
            measure_list = []  #  listoff all considered measure operators
            index_list = []  # list for index counting
            offset = 0
            for expec_ in op_list:
                # Obtained the derivative from the operator module
                operator = self.operator_derivatives.get_derivative(expec_.operator)
                # Assign parameters and convert to sparse Pauli representation
                op_with_param = self.operator_derivatives.assign_parameters(operator, param_op_inp)

                # flatten the measurement operator list,
                # but keep the nested list structure in index_list for later reconstruction
                index_list_op, measure_list_op = measure_to_list(op_with_param, offset)
                offset = offset + len(measure_list_op)
                measure_list = measure_list + measure_list_op
                index_list.append(index_list_op)

            # get the circuits of the PQC derivatives from the feature map module
            pqc_opflow = self.pqc_derivatives.get_derivative(key)

            # check for multiple circuits (e.g. gradient)
            if isinstance(pqc_opflow, ListOp):
                array_circ = True
            else:
                array_circ = False

            # add operator measurments to the circuits
            opflow_measured = measure_feature_map_derivative(pqc_opflow, ListOp(measure_list))

            if self.split_paulis:
                # If necessary, split the measurements containing X and Y operators
                # into extra measurements
                opflow_measured = _split_paulis(opflow_measured, len(measure_list) == 1)

            # assign parameters of the circuit
            opflow_with_param = self.pqc_derivatives.assign_parameters(
                opflow_measured, x_inp, param_inp
            )

            # evaluate the list of opflows
            val = self.executor.opflow_exec(opflow_with_param)

            # In case of multiple circuits, swapp measurment operator to index 2 for a clearer nesting
            if array_circ:
                swapp_list = list(np.arange(len(val.shape), dtype=int))
                swapp_list = [swapp_list[0]] + [swapp_list[1]] + swapp_list[-1:] + swapp_list[2:-1]
                val = np.transpose(val, axes=swapp_list)

            # store results in value_dict
            # if get rid of unncessary arrays to fit the input vector nesting
            ioff = 0
            for iexpec, expec_ in enumerate(op_list):
                val_x = []
                for i in range(len(val)):
                    val_param = []
                    for j in range(len(val[i])):
                        val_param.append(
                            np.array(sort_back_to_nparray(index_list[iexpec], val[i][j]))
                        )
                    val_x.append(np.array(val_param))
                val_final = np.array(val_x)

                reshape_list = []
                shape = val_final.shape
                if multi_x:
                    reshape_list.append(shape[0])
                if multi_param:
                    reshape_list.append(shape[1])
                if multi_param_op:
                    reshape_list.append(shape[2])
                if len(shape) == 4:
                    reshape_list.append(shape[3])
                if len(shape) > 4:
                    reshape_list += list(shape[3:])
                if len(reshape_list) == 0:
                    value_dict[expec_] = val_final.reshape(-1)[0]
                else:
                    value_dict[expec_] = val_final.reshape(reshape_list)
                ioff = ioff + 1

        # Set-up lables from the input list
        for todo in values:
            todo_expec = expec.from_variable(todo)

            # post-processing of the variance
            # variance
            if todo_expec.operator == "var" and todo_expec.wavefunction == "I":
                value_dict[todo_expec] = value_dict[expec("I", "OO")] - np.square(
                    value_dict[expec("I", "O")]
                )
            # d/dx variance
            elif todo_expec.operator == "var" and todo_expec.wavefunction == "dx":
                if self.num_features == 1:
                    value_dict[todo_expec] = value_dict[expec("dx", "OO")] - 2.0 * (
                        np.multiply(value_dict[expec("dx", "O")], value_dict[expec("I", "O")])
                    )
                else:
                    value_dict[todo_expec] = np.zeros(value_dict[expec("dx", "OO")].shape)
                    for i in range(value_dict[expec("dx", "OO")].shape[-1]):
                        value_dict[todo_expec][..., i] = value_dict[expec("dx", "OO")][
                            ..., i
                        ] - 2.0 * (
                            np.multiply(
                                value_dict[expec("dx", "O")][..., i],
                                value_dict[expec("I", "O")],
                            )
                        )
            # d/dp variance
            elif todo_expec.operator == "var" and todo_expec.wavefunction == "dp":
                if self.num_parameters == 1:
                    value_dict[todo_expec] = value_dict[expec("dp", "OO")] - 2.0 * (
                        np.multiply(value_dict[expec("dp", "O")], value_dict[expec("I", "O")])
                    )
                else:
                    value_dict[todo_expec] = np.zeros(value_dict[expec("dp", "OO")].shape)
                    for i in range(value_dict[expec("dp", "OO")].shape[-1]):
                        value_dict[todo_expec][..., i] = value_dict[expec("dp", "OO")][
                            ..., i
                        ] - 2.0 * (
                            np.multiply(
                                value_dict[expec("dp", "O")][..., i],
                                value_dict[expec("I", "O")],
                            )
                        )
            # d/dop variance
            elif todo_expec.operator == "dvardop" and todo_expec.wavefunction == "I":
                if self.num_parameters_operator == 1:
                    value_dict[todo_expec] = value_dict[expec("I", "OOdop")] - 2.0 * (
                        np.multiply(value_dict[expec("I", "dop")], value_dict[expec("I", "O")])
                    )
                else:
                    value_dict[todo_expec] = np.zeros(value_dict[expec("I", "OOdop")].shape)
                    for i in range(value_dict[expec("I", "OOdop")].shape[-1]):
                        value_dict[todo_expec][..., i] = value_dict[expec("I", "OOdop")][
                            ..., i
                        ] - 2.0 * (
                            np.multiply(
                                value_dict[expec("I", "dop")][..., i],
                                value_dict[expec("I", "O")],
                            )
                        )

            # assign values to the label of the expectation value
            value_dict[todo] = value_dict[todo_expec]
            if isinstance(todo, expec) and todo.label != "":
                value_dict[todo.label] = value_dict[todo_expec]

        # Add x, param, and param_op to the dictionary as default
        value_dict["x"] = x
        value_dict["param"] = param
        value_dict["param_op"] = param_op

        # Store the updated dictionary for the theta value
        if self._result_caching:
            self.result_container[caching_tuple] = value_dict

        return value_dict


def _split_paulis(operator: OperatorBase, single_measure: bool) -> OperatorBase:
    """
    Split the opflow such that it can be measured in Z basis.

    Splits up the opflow expression such that the meaurement operators can be evaluated
    in the Z-basis. The routine is needed, if Y and X operators are present in the measurement

    Args:
        operator (OperatorBase) : Opflow expression that is checked
        single_measure (bool): Is true if there is only a single
            measurement in the opflow structure, otherwise a ListOp is considered
    Returns:
        New opflow expression where measurements are correctly split
        into seprate circuits
    """
    # We reached a ComposedOp term -> seprate the different measurment operators
    # using qiskit's PauliExpectation
    if isinstance(operator, ComposedOp):
        coeff = operator.coeff
        operator_splitted = PauliExpectation().convert(operator)
        operator_splitted._coeff = coeff
        if single_measure == True:
            return ListOp([operator_splitted])
        else:
            return operator_splitted

    # We reached a CircuitStateFn term that should not be present in this routine
    elif isinstance(operator, CircuitStateFn):
        raise ValueError("CircuitStateFn type is not allowed!")

    # We reached a list of opflow terms, recursive call of
    # the function to all children in the list
    elif isinstance(operator, ListOp):
        op_list = [_split_paulis(op, single_measure) for op in operator.oplist]
        if isinstance(operator, SummedOp):
            return SummedOp(
                oplist=[op for op in op_list if op != ~Zero @ One], coeff=operator.coeff
            )
        elif isinstance(operator, TensoredOp):
            return TensoredOp(
                oplist=[op for op in op_list if op != ~Zero @ One], coeff=operator.coeff
            )
        elif isinstance(operator, real_ListOp):
            return ListOp(
                oplist=[op for op in op_list if op != ~Zero @ One],
                coeff=operator.coeff,
                combo_fn=operator.combo_fn,
            )
        else:
            raise ValueError("Unknown ListOp type in _split_paulis:", type(operator))
    else:
        raise ValueError("Unknown type in _split_paulis:", type(operator))
