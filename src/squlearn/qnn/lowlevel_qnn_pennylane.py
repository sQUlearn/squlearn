from typing import Union, List
import numpy as np
import copy

from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

import pennylane as qml
import pennylane.numpy as pnp

from .lowlevel_qnn_base import LowLevelQNNBase

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase

from ..util import Executor
from ..util.data_preprocessing import adjust_features, adjust_parameters, to_tuple
from ..util.pennylane.pennylane_circuit import PennyLaneCircuit
from ..util.decompose_to_std import decompose_to_std


class DirectEvaluation:
    """Container class for evaluation of derivatives of the QNN

    Args:
        key (Union[str, tuple, ParameterVector, ParameterVectorElement]): Key of the derivative for
                                                                          the value dictionary
        order (int): Order of the derivative
        argnum (Union[None, List[int]]): List of which arguments (x,param,param_obs) to
                                         differentiate
        return_grad_param (bool): Calculate the gradient with respect to the parameters
        return_grad_param_obs (bool): Calculate the gradient with respect to the observable
                                      parameters
        return_grad_x (bool): Calculate the gradient with respect to the input
        squared (bool): Calculate the squared value of the derivative
    """

    def __init__(
        self,
        key: Union[str, tuple, ParameterVector, ParameterVectorElement],
        order: int = 0,
        argnum: Union[None, List[int]] = None,
        return_grad_param: bool = False,
        return_grad_param_obs: bool = False,
        return_grad_x: bool = False,
        squared: bool = False,
    ):
        self.key = key
        self.order = order
        self.argnum = argnum
        self.return_grad_param = return_grad_param
        self.return_grad_param_obs = return_grad_param_obs
        self.return_grad_x = return_grad_x
        self.squared = squared


class PostProcessingEvaluation:
    """Container class for the post processing evaluation of derivatives of the QNN

    Args:
        key (Union[str, tuple, ParameterVector, ParameterVectorElement]): Key of the derivative
                                                                          for the value dictionary
        evaluation_tuple (tuple): Tuple of direct_evaluation objects that are used for evaluating
                                  the values that are used in the post processing
        evaluation_function (callable): Function to evaluate the post processing
    """

    def __init__(
        self,
        key: Union[str, tuple, ParameterVector, ParameterVectorElement],
        evaluation_tuple: tuple,
        evaluation_function: callable,
    ):
        self.key = key
        self.evaluation_tuple = evaluation_tuple
        self.evaluation_function = evaluation_function


class LowLevelQNNPennyLane(LowLevelQNNBase):
    """
    Low level implementation of QNNs and its derivatives based on PennyLane.

    Args:
        pqc (EncodingCircuitBase) : parameterized quantum circuit in encoding circuit format
        operator (Union[ObservableBase,list]): Operator that is used in the expectation
            value of the QNN. Can be a list for multiple outputs.
        executor (Executor) : Executor that is used for the evaluation of the QNN
        caching : Caching of the result for each `x`, `param`, `param_op` combination
            (default = True)

    Attributes:
    -----------

    Attributes:
        num_qubits (int): Number of qubits of the QNN
        num_features (int): Dimension of the features of the PQC
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
        observable: Union[ObservableBase, list],
        executor: Executor,
        caching: bool = True,
    ) -> None:

        super().__init__(parameterized_quantum_circuit, observable, executor)

        # Initialize result cache
        self.caching = caching
        self.result_container = {}

        self._initialize_pennylane_circuit()

    def _initialize_pennylane_circuit(self):
        """Function to initialize the PennyLane circuit function of the QNN"""

        # Parameter vectors for the PQC and the observable
        self._x = ParameterVector("x", self._pqc.num_features)
        self._param = ParameterVector("param", self._pqc.num_parameters)
        self._qiskit_circuit = decompose_to_std(self._pqc.get_circuit(self._x, self._param))

        # Pre-process the observable
        if isinstance(self._observable, ObservableBase):
            # Single output, single observable
            self._multiple_output = False
            self._num_operators = 1
            self._num_parameters_observable = self._observable.num_parameters
            self._param_obs = ParameterVector("param_obs", self._num_parameters_observable)
            self._qiskit_observable = self._observable.get_operator(self._param_obs)
            self._qiskit_observable_squared = self._qiskit_observable.power(2).simplify()
        elif isinstance(self._observable, list):
            # Multiple outputs, multiple observables
            self._multiple_output = True
            self._num_operators = len(self._observable)
            self._num_parameters_observable = 0
            for obs in self._observable:
                self._num_parameters_observable += obs.num_parameters
            self._param_obs = ParameterVector("param_obs", self._num_parameters_observable)
            self._qiskit_observable = []
            self._qiskit_observable_squared = []
            ioff = 0
            for obs in self._observable:
                self._qiskit_observable.append(obs.get_operator(self._param_obs[ioff:]))
                self._qiskit_observable_squared.append(
                    self._qiskit_observable[-1].power(2).simplify()
                )
                ioff = ioff + obs.num_parameters
        else:
            raise ValueError("Observable must be of type ObservableBase or list")

        # PennyLane Circuit function of the QNN
        self._pennylane_circuit = PennyLaneCircuit(
            self._qiskit_circuit, self._qiskit_observable, self._executor
        )
        # PennyLane Circuit function with a squared observable
        self._pennylane_circuit_squared = PennyLaneCircuit(
            self._qiskit_circuit, self._qiskit_observable_squared, self._executor
        )

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

        self._initialize_pennylane_circuit()

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
                    oper_dict = oper.get_params()
                    for key, value in oper_dict.items():
                        if key != "num_qubits":
                            params["op" + str(i) + "__" + key] = value
            else:
                params.update(self._observable.get_params())
        return params

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN"""
        return self._pennylane_circuit._num_qubits

    @property
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC"""
        return self._pqc.num_features

    @property
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC"""
        return self._pqc.num_parameters

    @property
    def num_operator(self) -> int:
        """Return the number outputs"""
        return self._num_operators

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        return self._num_parameters_observable

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        return self._multiple_output

    @property
    def parameters(self) -> ParameterVector:
        """Return the parameter vector of the PQC."""
        return self._param

    @property
    def features(self) -> ParameterVector:
        """Return the feature vector of the PQC."""
        return self._x

    @property
    def parameters_operator(self) -> ParameterVector:
        """Return the parameter vector of the cost operator."""
        return self._param_obs

    def evaluate(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
        *values: Union[
            str,
            DirectEvaluation,
            PostProcessingEvaluation,
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

        # Pre-process the input data to the format [[x1],[x2]]
        x_inp, multi_x = adjust_features(x, self._pqc.num_features)
        x_inpT = np.transpose(x_inp)
        param_inp, multi_param = adjust_parameters(param, self._pqc.num_parameters)
        param_obs_inp, multi_param_op = adjust_parameters(
            param_obs, self._num_parameters_observable
        )

        # Check if the order of the circuit arguments is correct
        # !! DIFFERENT TO THE INPUT ORDER OF THE EVALUATION FUNCTION !!
        compare_list = []
        if self.num_parameters > 0:
            compare_list.append("param")
        if self.num_features > 0:
            compare_list.append("x")
        if self.num_parameters_observable > 0:
            compare_list.append("param_obs")
        if self._pennylane_circuit.circuit_arguments != compare_list:
            raise NotImplementedError("Wrong order of circuit arguments!")

        # return dictionary for input data, it will be empty
        # if the combination of x,param,param_op is touched the first time
        if self.caching == True:
            caching_tuple = (
                to_tuple(x),
                to_tuple(param),
                to_tuple(param_obs),
                (self._executor.shots == None),
            )
            value_dict = self.result_container.get(caching_tuple, {})
        else:
            value_dict = {}

        # Store input data in the dictionary
        value_dict["x"] = x
        value_dict["param"] = param
        value_dict["param_op"] = param_obs

        post_processing_values = []
        values = list(values)  # Convert to list to be able to append
        # Sort the values, more complicated because values can be tuples of ParameterVectors
        indices = np.argsort([str(t) for t in values])
        values = [values[i] for i in indices]
        for todo in values:

            todo_class = _get_class_from_string(todo)

            if todo_class.key in value_dict:
                # Skip if the value is already calculated
                continue

            if isinstance(todo_class, PostProcessingEvaluation):
                # In case of post processing, the evaluation function is called later
                # Add necessary evaluations to the values list
                for sub_todo in todo_class.evaluation_tuple:
                    if sub_todo not in values:
                        values.append(sub_todo)
                # Create a list of post processing evaluations
                post_processing_values.append(todo_class)
            else:

                # Direct evaluation of the QNN
                if (todo_class.return_grad_x and todo_class.order > 1) or (
                    todo_class.order >= 1 and self._executor.shots is not None
                ):  # TODO: Should be removed if PennyLane bug 4462 is fixed
                    # evaluate every single x, param, param_op combination separately
                    # faster evaluation for higher-order derivatives w.r.t. x
                    output = [
                        self._evaluate_todo_single_x(
                            todo_class, x_inp_, param_inp_, param_obs_inp_
                        )
                        for x_inp_ in x_inp
                        for param_inp_ in param_inp
                        for param_obs_inp_ in param_obs_inp
                    ]
                    output = np.array(output)

                else:
                    # evaluate only param, param_op combination separately and all x together
                    # Faster evaluation for lower-order derivatives
                    output = [
                        self._evaluate_todo_all_x(todo_class, x_inpT, param_inp_, param_obs_inp_)
                        for param_inp_ in param_inp
                        for param_obs_inp_ in param_obs_inp
                    ]
                    # Restore order of _evaluate_todo_single_x
                    output = np.array(output)

                    # Capture some strange edge cases
                    if self._executor.shots is not None and sum(np.shape(x)) == self.num_features:
                        output = output.reshape(output.shape + (1,))
                    if len(x) == 0 and self._executor.shots is None:
                        output = output.reshape((1,) + output.shape)

                    index_list = list(range(len(output.shape)))
                    if self.multiple_output:
                        swap_list = [2, 0, 1] + index_list[3:]
                    else:
                        swap_list = [1, 0] + index_list[2:]

                    output = output.transpose(swap_list)
                    output = output.reshape(
                        (output.shape[0] * output.shape[1],) + tuple(output.shape[2:])
                    )

                # Swap higher order derivatives into correct order
                index_list = list(range(len(output.shape)))
                if self.multiple_output:
                    swap_list = index_list[0:2] + list(reversed(index_list[2:]))
                else:
                    swap_list = index_list[0:1] + list(reversed(index_list[1:]))

                output = output.transpose(swap_list)

                # Reshape to correct format
                reshape_list = []
                shape = output.shape
                if multi_x:
                    reshape_list.append(len(x))
                if multi_param:
                    reshape_list.append(len(param))
                if multi_param_op:
                    reshape_list.append(len(param_obs))
                if self.multiple_output:
                    reshape_list.append(shape[1])
                if self.multiple_output:
                    reshape_list += list(shape[2:])
                else:
                    reshape_list += list(shape[1:])

                if len(reshape_list) == 0:
                    value_dict[todo_class.key] = output.reshape(-1)[0]
                else:
                    value_dict[todo_class.key] = output.reshape(reshape_list)

        # Do the post processing of the derivatives
        # Calculate the variance of the QNN output, the Laplace operation, or pick single elements
        for post in post_processing_values:
            value_dict[post.key] = post.evaluation_function(value_dict)

        # Store the updated dictionary for the theta value
        if self.caching:
            self.result_container[caching_tuple] = value_dict

        return value_dict

    def _evaluate_todo_single_x(
        self, todo_class: DirectEvaluation, x: np.array, param: np.array, param_obs: np.array
    ) -> np.array:
        """
        Helper for evaluating the derivative of the QNN with respect to a single x value

        Args:
            todo_class (direct_evaluation): Class that defines the derivative to evaluate
            x (np.array): X value (single entry)
            param (np.array): Parameter values (single entry)
            param_obs (np.array): Observable parameter values (single entry)

        Returns:
            Evaluated value of the derivative
        """
        if todo_class.squared:
            hash_func = self._pennylane_circuit_squared.hash
            if todo_class.order <= 1:
                func = self._pennylane_circuit_squared
            else:
                func = self._pennylane_circuit_squared.build_pennylane_circuit(
                    max_diff=todo_class.order
                )
        else:
            hash_func = self._pennylane_circuit.hash
            if todo_class.order <= 1:
                func = self._pennylane_circuit
            else:
                func = self._pennylane_circuit.build_pennylane_circuit(max_diff=todo_class.order)

        # Convert input to PennyLane arrays and requested gradients
        param_ = pnp.array(param, requires_grad=todo_class.return_grad_param)
        param_obs_ = pnp.array(param_obs, requires_grad=todo_class.return_grad_param_obs)
        x_ = pnp.array(x, requires_grad=todo_class.return_grad_x)

        eval_tuple = tuple()
        argnum_dict = {}
        ioff = 0
        if len(param_) > 0:
            eval_tuple += (param_,)
            argnum_dict[0] = ioff
            ioff += 1
        else:
            argnum_dict[0] = None
        if len(x_) > 0:
            eval_tuple += (x_,)
            argnum_dict[1] = ioff
            ioff += 1
        else:
            argnum_dict[1] = None
        if len(param_obs_) > 0:
            eval_tuple += (param_obs_,)
            argnum_dict[2] = ioff
            ioff += 1
        else:
            argnum_dict[2] = None

        if todo_class.order == 0:
            # Plain function evaluation
            # value = func(*(eval_tuple))
            value = self._executor.pennylane_execute(func, *(eval_tuple))
        elif todo_class.order > 0:
            # Generate iterative derivative function for higher order derivatives
            order = todo_class.order - 1
            argnum = copy.copy(todo_class.argnum)
            arg_index = argnum_dict[argnum.pop()]
            if arg_index is None:
                return np.array([])
            else:
                deriv = qml.jacobian(func, argnum=arg_index)
            while order > 0:
                order -= 1
                arg_index = argnum_dict[argnum.pop()]
                if arg_index is None:
                    return np.array([])
                else:
                    deriv = qml.jacobian(deriv, argnum=arg_index)

            deriv.hash = hash_func + str(todo_class.argnum)
            value = self._executor.pennylane_execute(deriv, *(eval_tuple))

        # Convert back to numpy array
        return np.real_if_close(np.array(value))

    def _evaluate_todo_all_x(
        self, todo_class: DirectEvaluation, x: np.array, param: np.array, param_obs: np.array
    ):
        """
        Helper for evaluating the derivative of the QNN with respect to all x values at once

        Args:
            todo_class (direct_evaluation): Class that defines the derivative to evaluate
            x (np.array): X values (all entries together)
            param (np.array): Parameter values (single entry)
            param_obs (np.array): Observable parameter values (single entry)

        Returns:
            Evaluated value of the derivative
        """
        if todo_class.squared:
            hash_func = self._pennylane_circuit_squared.hash
            if todo_class.order <= 1:
                func = self._pennylane_circuit_squared
            else:
                func = self._pennylane_circuit_squared.build_pennylane_circuit(
                    max_diff=todo_class.order
                )
        else:
            hash_func = self._pennylane_circuit.hash
            if todo_class.order <= 1:
                func = self._pennylane_circuit
            else:
                func = self._pennylane_circuit.build_pennylane_circuit(max_diff=todo_class.order)

        # Convert input to PennyLane arrays and requested gradients
        param_ = pnp.array(param, requires_grad=todo_class.return_grad_param)
        param_obs_ = pnp.array(param_obs, requires_grad=todo_class.return_grad_param_obs)
        x_ = pnp.array(x, requires_grad=todo_class.return_grad_x)

        eval_tuple = tuple()
        argnum_dict = {}
        ioff = 0
        if len(param_) > 0:
            eval_tuple += (param_,)
            argnum_dict[0] = ioff
            ioff += 1
        else:
            argnum_dict[0] = None
        if len(x_) > 0:
            eval_tuple += (x_,)
            argnum_dict[1] = ioff
            ioff += 1
        else:
            argnum_dict[1] = None
        if len(param_obs_) > 0:
            eval_tuple += (param_obs_,)
            argnum_dict[2] = ioff
            ioff += 1
        else:
            argnum_dict[2] = None

        if todo_class.order == 0:
            # Plain function evaluation
            # value = func(*eval_tuple)
            value = self._executor.pennylane_execute(func, *(eval_tuple))
        elif todo_class.order > 0:
            # Generate iterative derivative function for higher order derivatives
            order = todo_class.order - 1
            argnum = copy.copy(todo_class.argnum)
            arg_index = argnum_dict[argnum.pop()]
            if arg_index is None:
                return np.array([[]])
            else:
                deriv = qml.jacobian(func, argnum=arg_index)
            while order > 0:
                order -= 1
                arg_index = argnum_dict[argnum.pop()]
                if arg_index is None:
                    return np.array([[]])
                else:
                    deriv = qml.jacobian(deriv, argnum=arg_index)

            deriv.hash = hash_func + str(todo_class.argnum)
            value = self._executor.pennylane_execute(deriv, *(eval_tuple))

        # Convert back to numpy format
        values = np.array(value)
        # sum over zero values entries due to dx differentiation
        if todo_class.return_grad_x:
            sum_t = tuple()
            ioff = 0
            if self.multiple_output:
                ioff = 1
            for var in reversed(todo_class.argnum):
                if var == 1:  # dx differentiation
                    sum_t += (ioff + 2,)
                    ioff = ioff + 2
                else:
                    ioff = ioff + 1
            values = values.sum(axis=sum_t)
        return np.real_if_close(values)


def _get_class_from_string(
    val: Union[
        str,
        tuple,
        DirectEvaluation,
        PostProcessingEvaluation,
        ParameterVector,
        ParameterVectorElement,
    ]
) -> Union[DirectEvaluation, PostProcessingEvaluation]:
    """Converts an input string to the direct or post processing evaluation object

    Args:
        val (Union[str, tuple, direct_evaluation, post_processing_evaluation,
                ParameterVector, ParameterVectorElement]): Input string or evaluation object

    Returns:
        Associated Expec object

    """

    # Create the evaluation object from the input variables
    if isinstance(val, str):
        if val == "f":
            return DirectEvaluation("f")
        elif val == "dfdx":
            return DirectEvaluation("dfdx", 1, argnum=[1], return_grad_x=True)
        elif val == "dfdxdx":
            return DirectEvaluation("dfdxdx", 2, argnum=[1, 1], return_grad_x=True)
        elif val == "laplace":
            return PostProcessingEvaluation(val, ("dfdxdx",), _get_eval_laplace("dfdxdx"))
        elif val == "laplace_dp":
            return PostProcessingEvaluation(val, ("dfdpdxdx",), _get_eval_laplace("dfdpdxdx"))
        elif val == "laplace_dop":
            return PostProcessingEvaluation(val, ("dfdopdxdx",), _get_eval_laplace("dfdopdxdx"))
        elif val == "dfdp":
            return DirectEvaluation("dfdp", 1, argnum=[0], return_grad_param=True)
        elif val == "dfdpdp":
            return DirectEvaluation("dfdpdp", 2, argnum=[0, 0], return_grad_param=True)
        elif val == "dfdopdp":
            return DirectEvaluation(
                "dfdopdp", 2, argnum=[2, 0], return_grad_param=True, return_grad_param_obs=True
            )
        elif val == "dfdpdop":
            return DirectEvaluation(
                "dfdpdop", 2, argnum=[0, 2], return_grad_param=True, return_grad_param_obs=True
            )
        elif val == "dfdop":
            return DirectEvaluation("dfdop", 1, argnum=[2], return_grad_param_obs=True)
        elif val == "dfdopdop":
            return DirectEvaluation("dfdopdop", 2, argnum=[2, 2], return_grad_param_obs=True)
        elif val == "dfdpdx":
            return DirectEvaluation(
                "dfdpdx", 2, argnum=[0, 1], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdxdp":
            return DirectEvaluation(
                "dfdxdp", 2, argnum=[1, 0], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdxdxdp":
            return DirectEvaluation(
                "dfdxdxdp", 3, argnum=[1, 1, 0], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdxdpdx":
            return DirectEvaluation(
                "dfdxdpdx", 3, argnum=[1, 0, 1], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdpdxdx":
            return DirectEvaluation(
                "dfdpdxdx", 3, argnum=[0, 1, 1], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdopdx":
            return DirectEvaluation(
                "dfdopdx", 2, argnum=[2, 1], return_grad_param_obs=True, return_grad_x=True
            )
        elif val == "dfdopdxdx":
            return DirectEvaluation(
                "dfdopdxdx",
                3,
                argnum=[2, 1, 1],
                return_grad_param_obs=True,
                return_grad_x=True,
            )
        elif val == "fcc":
            return DirectEvaluation("fcc", squared=True)
        elif val == "dfccdx":
            return DirectEvaluation("dfccdx", 1, argnum=[1], return_grad_x=True, squared=True)
        elif val == "dfccdxdx":
            return DirectEvaluation("dfccdxdx", 2, argnum=[1, 1], return_grad_x=True, squared=True)
        elif val == "dfccdp":
            return DirectEvaluation("dfccdp", 1, argnum=[0], return_grad_param=True, squared=True)
        elif val == "dfccdpdp":
            return DirectEvaluation(
                "dfccdpdp", 2, argnum=[0, 0], return_grad_param=True, squared=True
            )
        elif val == "dfccdopdx":
            return DirectEvaluation(
                "dfccdopdx",
                2,
                argnum=[2, 1],
                return_grad_param_obs=True,
                return_grad_x=True,
                squared=True,
            )
        elif val == "dfccdop":
            return DirectEvaluation(
                "dfccdop", 1, argnum=[2], return_grad_param_obs=True, squared=True
            )
        elif val == "dfccdopdop":
            return DirectEvaluation(
                "dfccdopdop", 2, argnum=[2, 2], return_grad_param_obs=True, squared=True
            )
        elif val in ("var", "varf"):
            return PostProcessingEvaluation(val, ("f", "fcc"), _eval_var)
        elif val in ("dvardx", "dvarfdx"):
            return PostProcessingEvaluation(val, ("f", "dfccdx", "dfdx"), _eval_dvardx)
        elif val in ("dvardp", "dvarfdp"):
            return PostProcessingEvaluation(val, ("f", "dfccdp", "dfdp"), _eval_dvardp)
        elif val in ("dvardop", "dvarfdop"):
            return PostProcessingEvaluation(val, ("f", "dfccdop", "dfdop"), _eval_dvardop)
        elif val == "fischer":
            return None
        else:
            raise ValueError("Unknown input string:", val)
    elif isinstance(val, tuple):
        return _get_direct_evaluation_from_tuple(val)
    elif isinstance(val, ParameterVectorElement) or isinstance(val, ParameterVector):
        evaluation = _get_direct_evaluation_from_tuple((val,))
        evaluation.key = val  # Replace it to remove tuple structure
        return evaluation
    elif isinstance(val, DirectEvaluation):
        return val  # Nothing to do
    elif isinstance(val, PostProcessingEvaluation):
        return val  # Nothing to do
    else:
        raise TypeError("String expected, found type:", type(val))


def _eval_var(value_dict: dict) -> np.ndarray:
    """Evaluate the variance of the QNN output

    Args:
        value_dict (dict): Dictionary of QNN derivatives values

    Returns:
        Variance of the QNN output
    """
    return value_dict["fcc"] - np.square(value_dict["f"])


def _eval_dvardx(value_dict: dict) -> np.ndarray:
    """Evaluate the derivative of the variance with respect to the input

    Args:
        value_dict (dict): Dictionary of QNN derivatives values

    Returns:
        Derivative of the variance with respect to the input
    """
    return_val = np.zeros(value_dict["dfccdx"].shape)
    for i in range(value_dict["dfccdx"].shape[-1]):
        return_val[..., i] = value_dict["dfccdx"][..., i] - 2.0 * (
            np.multiply(
                value_dict["dfdx"][..., i],
                value_dict["f"],
            )
        )
    return return_val


def _eval_dvardp(value_dict: dict) -> np.ndarray:
    """Evaluate the derivative of the variance with respect to the parameters

    Args:
        value_dict (dict): Dictionary of QNN derivatives values

    Returns:
        Derivative of the variance with respect to the parameters
    """
    return_val = np.zeros(value_dict["dfccdp"].shape)
    for i in range(value_dict["dfccdp"].shape[-1]):
        return_val[..., i] = value_dict["dfccdp"][..., i] - 2.0 * (
            np.multiply(
                value_dict["dfdp"][..., i],
                value_dict["f"],
            )
        )
    return return_val


def _eval_dvardop(value_dict: dict) -> np.ndarray:
    """Evaluate the derivative of the variance with respect to the observable parameters

    Args:
        value_dict (dict): Dictionary of QNN derivatives values

    Returns:
        Derivative of the variance with respect to the observable parameters
    """
    return_val = np.zeros(value_dict["dfccdop"].shape)
    for i in range(value_dict["dfccdop"].shape[-1]):
        return_val[..., i] = value_dict["dfccdop"][..., i] - 2.0 * (
            np.multiply(
                value_dict["dfdop"][..., i],
                value_dict["f"],
            )
        )
    return return_val


def _get_eval_laplace(todo: str) -> callable:
    """Evaluate of the Laplace Operation on the QNN output.

    Args:
        todo (str): String that defines the type of Laplace operation

    Returns:
        Function to evaluate the Laplace operation
    """

    def eval_laplace(value_dict: dict) -> np.ndarray:
        """Summing the diagonal of the feature Hessian matrix

        Args:
            value_dict (dict): Dictionary of QNN derivatives values

        Returns:
            Sum of the diagonal of the feature Hessian matrix
        """
        return_val = np.zeros(value_dict[todo].shape[:-2])
        for i in range(value_dict[todo].shape[-1]):
            return_val += value_dict[todo][..., i, i]
        return return_val

    return eval_laplace


def _get_pick_elements(pick_indices: list, label: tuple) -> callable:
    """
    Function for generating the post-processing function for general derivatives

    Args:
        pick_indices (list): List of indices to pick single elements of derivatives
                                integer indices for specific parameters or "all" for all elements
        label (str): Label of the derivative

    Returns:
        Function to pick single elements of derivatives
    """

    def pick_elements(value_dict: dict) -> np.ndarray:
        """
        Function to pick single elements of general derivatives

        Args:
            value_dict (dict): Dictionary of QNN derivatives values

        Returns:
            Picked (single) elements of the derivatives
        """
        val = value_dict[label]
        if all(i == "all" for i in pick_indices):
            # If only ParameterVectors are used, the return the full array
            return val
        else:
            # If single elements occur, pick them
            slice_tuple = tuple()  # -> tuple that is used for picking the elements
            for i in pick_indices:
                if i == "all":
                    # Keep all elements
                    slice_tuple += (slice(None),)
                else:
                    # Pick single element
                    slice_tuple += (slice(i, i + 1, 1),)
            # Dimension at the beginning of the array
            extra_dim = len(val.shape) - len(slice_tuple)
            if extra_dim > 0:
                slice_tuple = (slice(None),) * extra_dim + slice_tuple

            # Return the picked elements
            return val[slice_tuple]

    return pick_elements


def _get_direct_evaluation_from_tuple(
    todo: tuple,
) -> Union[DirectEvaluation, PostProcessingEvaluation]:
    """
    Function for generating the evaluation object from a tuple

    Handles derivatives of the QNN output with respect to a tuple of
    ParameterVector and ParameterVectorElement

    Args:
        todo (tuple): Tuple of ParameterVector and ParameterVectorElement entries that
                        define the derivative

    Returns:
        Evaluation object for the derivative, either direct_evaluation or
        post_processing_evaluation
    """

    # Generate direct evaluation object for the derivative
    # in case of a single ParameterVectorElement, post processing is needed
    return_grad_param = False
    return_grad_param_obs = False
    return_grad_x = False
    argnum = []
    order = 0
    val_tuple = tuple()
    pick_list = []

    for sub_val in todo:
        if isinstance(sub_val, ParameterVector) or isinstance(sub_val, ParameterVectorElement):
            # Increase order of derivative
            order += 1
            # Generate list for picking the single elements in post processing
            # and the tuple for the direct evaluation
            if isinstance(sub_val, ParameterVectorElement):
                val_tuple += (sub_val.vector,)
                pick_list.append(sub_val.index)
            else:
                val_tuple += (sub_val,)
                pick_list.append("all")
            # Generate argnum list and set return_grad flags
            if "param_obs" in sub_val.name:
                return_grad_param_obs = True
                argnum.append(2)
            elif "param" in sub_val.name:
                return_grad_param = True
                argnum.append(0)
            elif "x" in sub_val.name:
                return_grad_x = True
                argnum.append(1)
            else:
                raise ValueError("Unsupported parameter name:", sub_val.name)
        else:
            raise ValueError("Unsupported input type:", type(sub_val))

    # Generate the direct evaluation object
    evaluation = DirectEvaluation(
        val_tuple, order, argnum, return_grad_param, return_grad_param_obs, return_grad_x
    )

    if all(i == "all" for i in pick_list):
        # In case of only ParameterVectors direct evaluation is enough
        return evaluation
    else:
        # In case of minimal one ParameterVectorsElement,
        # post processing is needed for picking
        return PostProcessingEvaluation(
            todo, (evaluation,), _get_pick_elements(pick_list, val_tuple)
        )
