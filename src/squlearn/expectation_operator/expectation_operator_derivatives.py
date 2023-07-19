from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.opflow import StateFn, OperatorStateFn
from qiskit.opflow import SummedOp, ListOp, PauliOp, TensoredOp, PauliSumOp
from qiskit.opflow import Zero, One
from qiskit.opflow import OperatorBase
from qiskit.opflow.list_ops.list_op import ListOp as real_ListOp
from qiskit.opflow.gradients.derivative_base import _coeff_derivative
from qiskit.quantum_info import Pauli, SparsePauliOp
from typing import Union
import numpy as np

from .expectation_operator_base import ExpectationOperatorBase
from ..util.data_preprocessing import adjust_input


class ExpectationOperatorDerivatives:
    r"""Class for calculating derivatives of expectation operators.

    The derivatives are calculated by automatic differentiation of parameter in the expectation
    operator. Also, squaring of the operator is implemented.
    The class can either applied on a single operator, or on a list of operators.
    results are cached for faster evaluation.

    Args:
        expectation_operator (Union[ExpectationOperatorBase, list]): Expectation operator or list
                                                                     of expectation operators from
                                                                     which the derivatives are
                                                                     obtained.
        opflow_caching (bool): If True, the opflow structure of the expectation operator is cached

    .. list-table:: Strings that are recognized by the :meth:`get_derivative` method
       :widths: 25 75
       :header-rows: 1

       * - String
         - Derivative
       * - ``"O"``
         - Expectation operator :math:`\hat{O}`
       * - ``"OO"``
         - Squared expectation operator :math:`\hat{O}^2`
       * - ``"dop"`` or ``"Odop"``
         - First-order derivative of the expectation operator:
           :math:`\frac{d}{dp}\hat{O}(p)`
       * - ``"dopdop"`` or ``"Odopdop"``
         - Second-order derivative of the expectation operator:
           :math:`\frac{d^2}{dp^2}\hat{O}(p)`
       * - ``"OOdop"``
         - First-order derivative of the squared expectation operator:
           :math:`\frac{d}{dp}\hat{O}^2(p)`
       * - ``"OOdopdop"``
         - Second-order derivative of the squared expectation operator:
           :math:`\frac{d^2}{dp^2}\hat{O}^2(p)`
       * - ``"I"``
         - Returns a identity operator with the same number of qubits as the provided
           expectation operator

    **Example: first-order derivative of the Ising Hamiltonian**

    .. code-block:: python

       from squlearn.expectation_operator import IsingHamiltonian,ExpectationOperatorDerivatives
       op = IsingHamiltonian(num_qubits=3)
       print(ExpectationOperatorDerivatives(op).get_derivative("dop"))

    **Example: Squared summed Pauli Operator**

    .. code-block:: python

       from squlearn.expectation_operator import SummedPaulis,ExpectationOperatorDerivatives
       op = SummedPaulis(num_qubits=3)
       print(ExpectationOperatorDerivatives(op).get_operator_squared())

    Attributes:
        parameter_vector (ParameterVector): Parameter vector used in the expectation operator
        num_parameters (int): Total number of trainable parameters in the expectation operator
        num_operators (int): Number operators in case of multiple expectation operators


    """

    def __init__(
        self,
        expectation_operator: Union[ExpectationOperatorBase, list],
        opflow_caching=True,
    ):
        self.expectation_operator = expectation_operator

        # Contains the OperatorMeasurement() of the expectation-operator for later replacement
        if isinstance(self.expectation_operator, ExpectationOperatorBase):
            # 1d output by a single expectation-operator
            self.multiple_output = False
            self._num_operators = 1
            self._parameter_vector = ParameterVector("p_op", expectation_operator.num_parameters)

            opflow = self.expectation_operator.get_operator(self._parameter_vector)
            opflow.replace = False
        else:
            # multi dimensional output by multiple Expectation-operators
            expectation_op_list = []
            self.multiple_output = True
            self._num_operators = len(expectation_operator)
            try:
                n_oper = 0
                for op in self.expectation_operator:
                    n_oper = n_oper + op.num_parameters

                self._parameter_vector = ParameterVector("p_op", n_oper)
                ioff = 0
                for op in self.expectation_operator:
                    expectation_op_list.append(op.get_operator(self._parameter_vector[ioff:]))
                    ioff = ioff + op.num_parameters
                opflow = ListOp(expectation_op_list)
                opflow.replace = False
            except:
                raise ValueError("Unknown structure of the Expectation operator!")

        self.opflow = opflow
        self.opflow_cache = {}
        self.opflow_caching = opflow_caching

        if self.opflow_caching:
            self.opflow_cache["O"] = opflow

    def get_derivative(self, derivative: Union[str, tuple]):
        """Determine the derivative of the expectation operator.

        Args:
            derivative (str or tuple): String or tuple of parameters for specifying the derivation.
                                       See :class:`ExpectationOperatorDerivatives` for more information.

        Return:
            Differentiated expectation operator in opflow StateFn format
        """
        if isinstance(derivative, str):
            # todo change with replaced operator
            if derivative == "I":
                measure_op = StateFn(
                    PauliOp(Pauli("I" * self.opflow.num_qubits)), is_measurement=True
                )
            elif derivative == "O":
                measure_op = self.opflow
            elif derivative == "OO":
                measure_op = self.get_operator_squared()
            elif derivative == "dop" or derivative == "Odop":
                measure_op = self._differentiation_from_tuple(
                    self.opflow.copy(), (self._parameter_vector,), "O"
                )
            elif derivative == "dopdop" or derivative == "Odopdop":
                measure_op = self._differentiation_from_tuple(
                    self.opflow.copy(), (self._parameter_vector, self._parameter_vector), "O"
                )
            elif derivative == "OOdop":
                measure_op = self._differentiation_from_tuple(
                    self.get_operator_squared(), (self._parameter_vector,), "OO"
                )
            elif derivative == "OOdopdop":
                measure_op = self._differentiation_from_tuple(
                    self.get_operator_squared(),
                    (self._parameter_vector, self._parameter_vector),
                    "OO",
                )
            else:
                raise ValueError("Unknown string command:", derivative)

        elif isinstance(derivative, tuple):
            measure_op = self._differentiation_from_tuple(self.opflow, derivative, "O")
        else:
            raise TypeError("Input is neither string nor tuple, but:", type(derivative))

        measure_op.replace = False
        return measure_op

    def get_differentiation_from_tuple(self, diff_tuple: tuple):
        """Computes the derivative of the expectation operator from a tuple of parameters

        Args:
            diff_tuple (tuple): Tuple containing ParameterVectors or ParameterExpressions

        Return:
            Differentiated expectation operator
        """

        return self.get_derivative(diff_tuple)

    def get_derivation_from_string(self, input_string: str):
        """Returns the derivative of the expectation operator for a string abbreviation.

        The table for the abbreviations can be found at :class:`ExpectationOperatorDerivatives`.

        Args:
            input_string (str): String for specifying the derivation.

        Return:
            Differentiated expectation operator
        """
        return self.get_derivative(input_string)

    def _differentiation_from_tuple(
        self, expectation_op: OperatorBase, diff_tuple: tuple, expectation_op_label: str
    ):
        """Recursive routine for automatic differentiating the expectation_operator

        Args:
            expectation_op (OperatorBase): opflow structure of the expectation operator
            diff_tuple (tuple): Tuple containing ParameterVectors or ParameterExpressions
            expectation_op_label (str): string for labeling the expectation operator

        Return:
            The differentiated opflow expression
        """

        if diff_tuple == ():
            # Cancel the recursion by returning the opflow operator
            return expectation_op
        else:
            # Check if differentiating tuple is already stored in opflow_cache
            if (
                self.opflow_caching == True
                and (diff_tuple, expectation_op_label) in self.opflow_cache
            ):
                # If stored -> return
                return self.opflow_cache[(diff_tuple, expectation_op_label)].copy()
            else:
                # Recursive differentiation with the most left object
                measure = operator_differentiation(
                    self._differentiation_from_tuple(
                        expectation_op, diff_tuple[1:], expectation_op_label
                    ),
                    diff_tuple[0],
                )
                # Store result in the opflow_cache
                if self.opflow_caching == True:
                    self.opflow_cache[(diff_tuple, expectation_op_label)] = measure
                return measure

    def get_operator_squared(self):
        "Builds and caches the squared form of the expectation operator OO=O^2"
        if self.opflow_caching == True and "OO" in self.opflow_cache:
            return self.opflow_cache["OO"].copy()
        else:
            # Get the operator and store it in ListOp structure
            if not isinstance(self.opflow, ListOp):
                op_list = ListOp([self.opflow])
            else:
                op_list = self.opflow

            # Loop through the ListOp and generate the squared mesaurment operator
            o2_list = []
            for op in op_list:
                o2_list.append(
                    StateFn(
                        (op.primitive @ op.primitive).reduce().reduce(),
                        is_measurement=True,
                    )
                )

            # Suitable export format
            if not isinstance(self.opflow, ListOp):
                O2 = o2_list[0]
            else:
                O2 = ListOp(o2_list)

            # If caching is enabled, store in the dictionary
            if self.opflow_caching == True:
                self.opflow_cache["OO"] = O2
            return O2

    @property
    def parameter_vector(self):
        """Parameter vector of the expectation operator"""
        return self._parameter_vector

    @property
    def num_parameters(self):
        """Total number of trainable parameters in the expectation operator"""
        return len(self._parameter_vector)

    @property
    def num_operators(self):
        """Number operators in case of multiple expectation operators"""
        return self._num_operators

    def assign_parameters(self, operator: OperatorBase, parameters: np.ndarray):
        """Assign parameters to a derivative that is obtained from this class.

        Args:
            operator (OperatorBase): Operator to which the parameters are assigned
            parameters (np.ndarray): Parameters values that replace the parameters in the operator

        Return:
            Operator with assigned parameters
        """
        param_op_inp, multi_param_op = adjust_input(parameters, len(self._parameter_vector))

        return_list = []
        for p in param_op_inp:
            dic = dict(zip(self._parameter_vector, p))
            return_list.append(_convert_to_sparse(operator.assign_parameters(dic)))

        if multi_param_op:
            return ListOp(return_list)
        else:
            return return_list[0]


def operator_differentiation(
    opflow: OperatorBase, parameters: Union[ParameterVector, list, ParameterExpression]
):
    """Core routine for differentiating a given expectation operator w.r.t. to its parameters

    Args:
        expectation_op (OperatorBase): opflow structure of the expectation operator, can also be a
                                       list of expectation operators
        parameters: Union[ParameterVector, list, ParameterExpression]: Parameters that are used for
                                                                       the differentiation.
    Returns:
        Differentiated expectation operator
    """
    # Make a list if input is not a list
    if parameters == None or parameters == []:
        return None

    if isinstance(parameters, ParameterVectorElement):
        parameters = [parameters]

    if not isinstance(opflow, ListOp):
        expectation_op_ = ListOp([opflow])
    else:
        expectation_op_ = opflow

    list_op = []
    for op in expectation_op_:
        if len(parameters) == 1:
            # In case of a single parameter no array has to be returned
            expectation_op_grad = _operator_differentiation(op, parameters[0])
            list_op.append(expectation_op_grad)
        else:
            # Build list and derive the differentiated circuit for each parameter separately
            expectation_op_grad_list = []
            for p in parameters:
                expectation_op_grad = _operator_differentiation(op, p)
                expectation_op_grad_list.append(expectation_op_grad)
            list_op.append(ListOp(expectation_op_grad_list))

    if not isinstance(opflow, ListOp):
        return list_op[0]
    else:
        return ListOp(list_op)


def _operator_differentiation(
    operator: ListOp, parameters: Union[ParameterVector, list, ParameterExpression]
):
    """
    Function for differentiating a list of operators w.r.t. a list of parameters

    Args:
        operator (ListOp): List of operators to be differentiated
        parameters (Union[ParameterVector, list, ParameterExpression]): Parameters to differentiate

    Return:
        List of differentiated operators
    """

    # Helper function for getting the coefficient
    def is_coeff_c(coeff, c):
        if isinstance(coeff, ParameterExpression):
            expr = coeff._symbol_expr
            return expr == c
        return coeff == c

    # Check for parameter lists, if so iterate through the list can call the function again
    if isinstance(parameters, (ParameterVector, list)):
        grad_list = [_operator_differentiation(operator, p) for p in parameters]
        return ListOp(grad_list)

    # In case input operator is 0 return 0
    if operator == 0:
        return 0.0 * PauliOp(Pauli("I" * operator.num_qubits))

    if isinstance(operator, OperatorStateFn):
        op = _operator_differentiation(operator.primitive, parameters)
        if op == 0:
            return StateFn(
                PauliOp(Pauli("I" * operator.num_qubits)),
                is_measurement=True,
                coeff=0.0,
            )
        return StateFn(op, is_measurement=True)

    # Handle SummedOp but also ListOps!!
    if isinstance(operator, ListOp):
        # Iterate the gradient derivation through the list
        grad_ops = [_operator_differentiation(op, parameters) for op in operator.oplist]

        # pylint: disable=comparison-with-callable
        if isinstance(operator, SummedOp):
            without_zeros = [grad for grad in grad_ops if grad != 0]
            if len(without_zeros) == 0:
                return 0
            return SummedOp(oplist=without_zeros, coeff=operator.coeff).reduce()
        elif isinstance(operator, real_ListOp):
            return ListOp(grad_ops, coeff=operator.coeff, combo_fn=operator.combo_fn)
        else:
            raise TypeError("Only SummedOp or ListOp are allowed! Type found:", type(operator))

    # No we can finally do the differentiation
    if not is_coeff_c(operator._coeff, 1.0) and not is_coeff_c(operator._coeff, 1.0j):
        coeff = operator._coeff
        if is_coeff_c(coeff, 0.0):
            return ~Zero @ One
        op = operator / coeff

        # Call qiskit function for parameter differentiation
        d_coeff = _coeff_derivative(coeff, parameters)
        grad_op = 0
        if op != ~Zero @ One and not is_coeff_c(d_coeff, 0.0):
            grad_op += d_coeff * op

        return grad_op

    # Only operator without coefficients is left, return 0 since a constant value is differentiated
    # to zero
    return 0.0 * PauliOp(Pauli("I" * operator.num_qubits))


def _convert_to_sparse(operator: OperatorBase):
    """
    Function for converging a operator into PauliSUmOp form (speed up in evaluation)

    Args:
        operator (OperatorBase): Operator to be converted

    Return:
        Operator in PauliSumOp form
    """
    # convert operator into sparse form
    if isinstance(operator, OperatorStateFn):
        if not isinstance(operator.primitive, PauliOp) and not isinstance(
            operator.primitive, PauliSumOp
        ):
            # Recreate OperatorStateFn with PauliSumOp instead of SummedOp -> large speed up!
            return OperatorStateFn(
                sum(operator.primitive.oplist),
                coeff=operator.coeff,
                is_measurement=operator.is_measurement,
            )
        elif isinstance(operator.primitive, PauliOp):
            return OperatorStateFn(
                PauliSumOp(
                    SparsePauliOp(operator.primitive.primitive, coeffs=[operator.primitive.coeff])
                ),
                coeff=operator.coeff,
                is_measurement=operator.is_measurement,
            )
        else:
            return operator

    elif isinstance(operator, ListOp):
        op_list = [_convert_to_sparse(op) for op in operator.oplist]

        if isinstance(operator, SummedOp):
            return SummedOp(oplist=op_list, coeff=operator.coeff)
        elif isinstance(operator, TensoredOp):
            return TensoredOp(oplist=op_list, coeff=operator.coeff)
        elif isinstance(operator, real_ListOp):
            return ListOp(oplist=op_list, coeff=operator.coeff)
        else:
            raise ValueError("Unknown ListOp type in _convert_to_sparse:", type(operator))
    else:
        raise ValueError("Unsupported type in _convert_to_sparse:", type(operator))
