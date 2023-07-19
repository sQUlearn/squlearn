from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.opflow import OperatorBase
from qiskit.opflow import StateFn, CircuitStateFn, OperatorStateFn
from qiskit.opflow import SummedOp, ComposedOp, TensoredOp
from qiskit.opflow import ListOp, PauliOp
from qiskit.opflow import Zero, One
from qiskit.compiler import transpile
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliOp
from qiskit.opflow.gradients import Gradient
from qiskit.opflow.list_ops.list_op import ListOp as real_ListOp
import numpy as np
from typing import Union

from .feature_map_base import FeatureMapBase


class FeatureMapDerivatives:
    r"""
    Class for automatic differentiation of feature maps.

    This class allows to compute derivatives of a feature map with respect to its parameters
    by utilizing the parameter-shift rule.
    The derivatives can be obtained by the method :meth:`get_derivative`.
    The type of derivative can be specified by either a string (see table below)
    or a ParameterVector or (a list) of ParameterElements that can be accessed
    via :meth:`feature_vector` or :meth:`parameter_vector`, respectively.

    .. list-table:: Strings that are recognized by the :meth:`get_derivative` method
       :widths: 25 75
       :header-rows: 1

       * - String
         - Derivative
       * - ``"I"``
         - Identity Operation (returns the feature map circuit)
       * - ``"dx"``
         - Gradient with respect to feature :math:`x`:
           :math:`\nabla_x = \big( \frac{\partial}{\partial x_1},\ldots,
           \frac{\partial}{\partial x_n} \big)`
       * - ``"dp"``
         - Gradient with respect to parameter :math:`p`:
           :math:`\nabla_p = \big( \frac{\partial}{\partial p_1},\ldots,
           \frac{\partial}{\partial p_m} \big)`
       * - ``"dxdx"``
         - Hessian with respect to feature :math:`x`:
           :math:`H^x_{ij} = \frac{\partial^2}{\partial x_i \partial x_j}`
       * - ``"dpdxdx"``
         - Derivative of the feature Hessian with respect to parameter :math:`p`:
           :math:`\nabla_p H^x_{ij} = \big( \frac{\partial H^x_{ij}}{\partial p_1},\ldots,
           \frac{\partial H^x_{ij}}{\partial p_m} \big)`
       * - ``laplace``
         - Laplace operator with respect to :math:`x`:
           :math:`\Delta = \nabla^2 = \sum_i \frac{\partial^2}{\partial x^2_i}`
       * - ``laplace_dp``
         - Derivative of the laplace operator with respect to parameter :math:`p`:
           :math:`\nabla_p \circ \Delta = \big( \frac{\partial }{\partial p_1}\Delta,\ldots,
           \frac{\partial}{\partial p_m} \Delta \big)`
       * - ``"dpdp"``
         - Hessian with respect to parameter :math:`p`:
           :math:`H^p_{ij} = \frac{\partial^2}{\partial p_i \partial p_j}`
       * - ``"dxdp"`` (or ``"dxdp"``)
         - Mixed Hessian with respect to feature :math:`x` and parameter :math:`p`:
           :math:`H^{xp}_{ij} = \frac{\partial^2}{\partial x_i \partial p_j}`

    **Example: Feature Map gradient with respect to the trainable parameters**

    .. code-block:: python

       from squlearn.feature_map import QEKFeatureMap, FeatureMapDerivatives
       fm = QEKFeatureMap(num_qubits=2, num_features=2, num_layers=2)
       fm_deriv = FeatureMapDerivatives(fm)
       grad = fm_deriv.get_derivative("dp")

    **Example: Derivative with respect to only the first trainable parameter**

    .. code-block:: python

       from squlearn.feature_map import QEKFeatureMap, FeatureMapDerivatives
       fm = QEKFeatureMap(num_qubits=2, num_features=2, num_layers=2)
       fm_deriv = FeatureMapDerivatives(fm)
       dp0 = fm_deriv.get_derivative((fm_deriv.parameter_vector[0],))


    Args:
        feature_map (FeatureMapBase): Feature map to differentiate
        opflow_caching (bool): If True, the opflow expressions are cached for faster
                               evaluation. (default: True)
    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
        opflow_caching: bool = True,
    ):
        self.feature_map = feature_map

        self._x = ParameterVector("x", self.feature_map.num_features)
        self._p = ParameterVector("p", self.feature_map.num_parameters)

        circuit = feature_map.get_circuit(self._x, self._p)
        self.circuit_opflow = CircuitStateFn(primitive=circuit, coeff=1.0)
        self.num_qubits = self.circuit_opflow.num_qubits

        identity_measurement = StateFn(PauliOp(Pauli("I" * self.num_qubits)), is_measurement=True)
        identity_measurement.replace = True

        self.opflow = identity_measurement @ self.circuit_opflow

        self.opflow_cache = {}
        self.opflow_caching = opflow_caching
        if self.opflow_caching:
            self.opflow_cache["f"] = self.opflow

        # get the instruction gates from the inital circuit for transpiling the circuits
        # back to this basis
        self.instruction_set = set()
        for instruction in circuit.data:
            self.instruction_set.add(instruction.operation.name)
        self.instruction_set = list(self.instruction_set)

    def get_derivative(self, derivative: Union[str, tuple]) -> OperatorBase:
        """Determine the derivative of the feature map circuit.

        Args:
            derivative (str or tuple): String or tuple of parameters for specifying the derivation.

        Return:
            Derivative circuit in Qiskit Opflow format.

        """
        if isinstance(derivative, str):
            if derivative == "I":
                opflow = self.opflow
            elif derivative == "dx":
                opflow = self._differentiation_from_tuple((self._x,)).copy()
            elif derivative == "dxdx":
                opflow = self._differentiation_from_tuple((self._x, self._x)).copy()
            elif derivative == "dpdxdx":
                opflow = self._differentiation_from_tuple((self._p, self._x, self._x)).copy()
            elif derivative == "laplace":
                list_sum = []
                for xi in self._x:
                    list_sum.append(self._differentiation_from_tuple((xi, xi)).copy())
                opflow = SummedOp(list_sum)
            elif derivative == "laplace_dp":
                list_sum = []
                for xi in self._x:
                    list_sum.append(self._differentiation_from_tuple((self._p, xi, xi)).copy())
                opflow = SummedOp(list_sum)
            elif derivative == "dp":
                opflow = self._differentiation_from_tuple((self._p,)).copy()
            elif derivative == "dpdp":
                opflow = self._differentiation_from_tuple((self._p, self._p)).copy()
            elif derivative == "dpdx":
                opflow = self._differentiation_from_tuple((self._p, self._x)).copy()
            elif derivative == "dxdp":
                opflow = self._differentiation_from_tuple((self._x, self._p)).copy()
            else:
                raise ValueError("Unknown string command:", derivative)
        elif isinstance(derivative, tuple):
            opflow = self._differentiation_from_tuple(derivative)
        else:
            raise TypeError("Input is neither string nor tuple, but:", type(derivative))

        # remove mesurement operator

        return _remove_measure(opflow)

    def get_differentiation_from_tuple(self, diff_tuple: tuple) -> OperatorBase:
        """Returns the derivative of the feature map circuit for a tuple of parameters.

        The tuple describes the differentiation with respect to the parameters in the tuple.

        Args:
            diff_tuple (tuple): Tuple of parameters for specifying the derivation.

        Return:
            Derivaitve circuit in Qiskit Opflow format.
        """
        return self.get_derivative(diff_tuple)

    def get_derivation_from_string(self, input_string: str) -> OperatorBase:
        """Returns the derivative of the feature map circuit for a string abbreviation.

        The table for the abbreviations can be found in the documentation of the class.

        Args:
            input_string (str): String for specifying the derivation.

        Return:
            Derivaitve circuit in Qiskit Opflow format.
        """
        return self.get_derivative(input_string)

    def _differentiation_from_tuple(self, diff_tuple: tuple) -> OperatorBase:
        """Recursive routine for automatic differentiating the feature map

        Variables for the differentiation are supplied by a tuple
        (x,param,param_op) from left to right -> dx dparam dparam_op PQC(x,param,param_op)


        Args:
            diff_tuple (tuple): tuple containing ParameterVectors or ParameterExpressions or Strings
                                determining the derivation

        Return:
            Derivaitve circuit in Qiskit Opflow format.
        """
        # TODO: support tuple of lists of parameter elements

        if diff_tuple == ():
            # Cancel the recursion by returning the opflow of the simply measured feature map
            return self.opflow.copy()
        else:
            # Check if differentiating tuple is already stored in opflow_cache
            if self.opflow_caching == True and (diff_tuple,) in self.opflow_cache:
                # If stored -> return
                return self.opflow_cache[(diff_tuple,)].copy()
            else:
                # Recursive differentiation with the most left object
                circ = _opflow_differentiation(
                    self._differentiation_from_tuple(diff_tuple[1:]), diff_tuple[0]
                )
                # Store result in the opflow_cache
                if self.opflow_caching == True:
                    self.opflow_cache[(diff_tuple,)] = circ
                return circ

    @property
    def parameter_vector(self) -> ParameterVector:
        """Parameter ParameterVector ``p`` utilized in the feature map circuit."""
        return self._p

    @property
    def feature_vector(self) -> ParameterVector:
        """Feature ParameterVector ``x`` utilized in the feature map circuit."""
        return self._x

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the feature map circuit."""
        return len(self._p)

    @property
    def num_features(self) -> int:
        """Number of features in the feature map circuit."""
        return len(self._x)

    def assign_parameters(
        self, opflow: OperatorBase, features: np.ndarray, parameters: np.ndarray
    ) -> OperatorBase:
        """
        Assigns numerical values to the ParameterVector elements of the feature map circuit.

        Args:
            opflow (OperatorBase): Opflow object to be assigned.
            features (np.ndarray): Numerical values of the feature vector.
            parameters (np.ndarray): Numerical values of the parameter vector.

        Return:
            Opflow object with assigned numerical values.
        """

        if opflow is None:
            return None

        todo_list = []  # list for the variables
        multi_list = []  # list of return ListOp or no list
        param_list = []  # list of parameters that are substituted

        # check shape of the x and adjust to [[]] form if necessary
        if features is not None:
            xx, multi_x = _adjust_input(features, len(self._x))
            todo_list.append(xx)
            param_list.append(self._x)
            multi_list.append(multi_x)
        if parameters is not None:
            pp, multi_p = _adjust_input(parameters, len(self._p))
            todo_list.append(pp)
            param_list.append(self._p)
            multi_list.append(multi_p)

        # Recursive construction of the assignment dictionary and list structure
        def rec_assign(dic, todo_list, param_list, multi_list):
            if len(todo_list) <= 0:
                return None
            return_list = []
            for x_ in todo_list[0]:
                for A, B in zip(param_list[0], x_):
                    dic[A] = B
                if len(multi_list[1:]) > 0:
                    return_list.append(
                        rec_assign(dic.copy(), todo_list[1:], param_list[1:], multi_list[1:])
                    )
                else:
                    return_list.append(opflow.assign_parameters(dic))

            if multi_list[0]:
                return ListOp(return_list)
            else:
                return return_list[0]

        return rec_assign({}, todo_list, param_list, multi_list)


def _adjust_input(x, x_length) -> tuple:
    """Adjust the input to the form [[]] if necessary.

    Args:
        x (np.ndarray): Input array.
        x_length (int): Dimension of the input array, e.g. feature dimension.

    Return:
        Adjusted input array and a boolean flag for multiple inputs.
    """

    # check shape of the x and adjust to [[]] form if necessary
    multiple_inputs = False
    error = False
    shape = np.shape(x)
    if shape == () and x_length == 1:
        # Single floating point number
        xx = np.array([[x]])
    elif len(shape) == 1:
        if x_length == 1:
            xx = np.array([np.array([xx]) for xx in x])
            multiple_inputs = True
        else:
            # We have a single multi dimensional x (e.g. parameter vector)
            if len(x) == x_length:
                xx = np.array([x])
            else:
                error = True
    elif len(shape) == 2:
        if shape[1] == x_length:
            xx = x
            multiple_inputs = True
        else:
            error = True
    else:
        error = True

    if error:
        raise ValueError("Wrong format of an input variable.")

    return xx, multiple_inputs


def _remove_measure(operator: OperatorBase) -> OperatorBase:
    """Replace all measurements from the inputted Opflow expression

    Args:
        operator (OperatorBase): Opflow expression from which the measurements are removed

    Return:
        New opflow expression with replaced measurement operator
    """
    # We reached a ComposedOp term -> replace the measurement
    if isinstance(operator, ComposedOp):
        # Remove the measurement from the children
        for i in range(len(operator.oplist)):
            if isinstance(operator.oplist[i], CircuitStateFn):
                return_op = operator.oplist[i]
                return_op._coeff = operator.coeff
                return return_op

        return ~Zero @ One

    # We reached a CircuitStateFn term -> apply the measurement to the wavefunction
    elif isinstance(operator, CircuitStateFn):
        return operator

    # We reached a list of opflow terms, recursive call of
    # the function to all children in the list
    elif isinstance(operator, ListOp):
        op_list = [_remove_measure(op) for op in operator.oplist]

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
            raise ValueError("Unknown ListOp type in _remove_measure:", type(operator))
    else:
        raise ValueError("Unknown type in _remove_measure:", type(operator))


def measure_feature_map_derivative(
    operator: OperatorBase, measurement: OperatorBase
) -> OperatorBase:
    """
    Applied the measurement operator to each circuit in the given opflow structure.

    Args:
        operator (OperatorBase): Opflow expression that is measured
        measurement (OperatorBase): Measurement operator

    Return:
        New opflow expression with replaced measurement operator
    """
    # We reached a ComposedOp term -> replace the measurement
    if isinstance(operator, ComposedOp):
        for i in range(len(operator.oplist)):
            if isinstance(operator.oplist[i], OperatorStateFn):
                if hasattr(operator.oplist[i], "replace"):
                    if operator.oplist[i].replace:
                        operator.oplist[i] = measurement
                else:
                    operator.oplist[i] = measurement
        return operator

    # We reached a CircuitStateFn term -> apply the measurement to the wavefunction
    elif isinstance(operator, CircuitStateFn):
        coeff = operator._coeff
        operator._coeff = 1.0
        measured_op = measurement @ operator
        measured_op._coeff = coeff
        return measured_op

    # We reached a list of opflow terms, recursive call of
    # the function to all children in the list
    elif isinstance(operator, ListOp):
        op_list = [measure_feature_map_derivative(op, measurement) for op in operator.oplist]

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
            raise ValueError("Unknown ListOp type in measure_derivative:", type(operator))
    else:
        raise ValueError("Unknown type in measure_derivative:", type(operator))


def _opflow_differentiation(
    opflow: OperatorBase, parameters: Union[list, tuple, ParameterVectorElement, ParameterVector]
):
    """
    Routine for the automatic differentiation based on qiskit routines

    Args:
        opflow : Input opflow expression
        params (list | ParameterVector): variables which are used in the
                                            differentiation (have to be the same type )
    Returns:
        opflow structure of the differentiated input opflow
    """

    # Make list if input is not a list
    if isinstance(parameters, ParameterVectorElement):
        parameters = [parameters]
    if isinstance(parameters, tuple):
        parameters = list(parameters)

    # Call the automatic differentiation routine
    # Separate routine for for one dim. or multi dim. variables
    if len(parameters) == 1:
        # Automatic qiskit differentiation
        # ParameterExpression(0) is removed by clean_opflow_circ
        # Gates are transpiled in Gradient(), this can yield different set of gates
        # than supported by the QC hardware, this is also fixed by clean_opflow_circ
        return _clean_opflow_circ(Gradient().convert(operator=opflow, params=parameters)[0])
    else:
        # If multiple variables are differentiated -> results are returned in array

        # Check if the same variables are the same type
        params_name = parameters[0].name.split("[", 1)[0]
        for p in parameters:
            if p.name.split("[", 1)[0] != params_name:
                raise TypeError("Differentiable variables are not the same type.")

        # Automatic qiskit differentiation
        # ParameterExpression(0) is removed by clean_opflow_circ
        # Gates are transpiled in Gradient(), this can yield different set of gates
        # than supported by the QC hardware, this is also fixed by clean_opflow_circ
        return _clean_opflow_circ(Gradient().get_gradient(operator=opflow, params=parameters))


def _clean_opflow_circ(operator: OperatorBase, instruction_set: list[str] = None) -> OperatorBase:
    """
    Function for cleaning the opflow structure.

    - Removes all ParameterExpression(0) contributions
    - Transpiling back to the given instruction set

    Args:
        operator (OperatorBase): input opflow expression
        instruction_set (list): list of gates which are supported by the QC hardware

    Return:
        The opflow expression without the zero contributions and transpiled circuits
    """

    # We reached the Composed object or the wavefunction
    if isinstance(operator, CircuitStateFn):
        # If coefficient is zero replace by Zero @ One which is sorted out later
        ZeroParameterExpr = False
        if isinstance(operator.coeff, ParameterExpression):
            # Check for ParameterExpression(0)
            ZeroParameterExpr = operator.coeff.sympify() == 0
        if operator.coeff != 0.0 and operator.coeff != 0 and ZeroParameterExpr == False:
            if instruction_set is None:
                return operator
            else:
                # Transpile back to the original basis set
                transpiled_circ = transpile(
                    operator.primitive,
                    basis_gates=instruction_set,
                    optimization_level=1,  # 1 for reducing number of gates
                    layout_method="trivial",
                )

                transpiled_circ = operator.primitive
                return CircuitStateFn(
                    primitive=transpiled_circ,
                    coeff=operator.coeff,
                    is_measurement=operator.is_measurement,
                )
        else:
            return ~Zero @ One
    elif isinstance(operator, OperatorStateFn):
        # Measurement operator -> do nothing
        return operator
    elif isinstance(operator, ComposedOp):
        # If coefficient is zero replace by Zero @ One which is sorted out later
        ZeroParameterExpr = False
        if isinstance(operator.coeff, ParameterExpression):
            # Check for ParameterExpression(0)
            ZeroParameterExpr = operator.coeff.sympify() == 0
        if operator.coeff != 0.0 and operator.coeff != 0 and ZeroParameterExpr == False:
            # None-zero coefficient found, iterate through composed operators
            op_list = [_clean_opflow_circ(op, instruction_set) for op in operator.oplist]
            if ~Zero @ One in op_list:
                return ~Zero @ One
            else:
                return ComposedOp(op_list, coeff=operator.coeff)
        else:
            return ~Zero @ One

    elif isinstance(operator, ListOp):
        # List object reached, recursive call of the function
        op_list = [_clean_opflow_circ(op, instruction_set) for op in operator.oplist]

        # Sort out the Zero @ One terms
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
            raise ValueError("Unknown Type in ListOp")
    else:
        raise ValueError("Unknown Type in OperatorBase:", type(operator))
