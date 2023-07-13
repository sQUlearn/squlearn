from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.opflow import StateFn, OperatorStateFn
from qiskit.opflow import SummedOp, ListOp, PauliOp, TensoredOp, PauliSumOp
from qiskit.opflow import Zero, One
from qiskit.opflow.list_ops.list_op import ListOp as real_ListOp
from qiskit.opflow.gradients.derivative_base import _coeff_derivative
from qiskit.quantum_info import Pauli, SparsePauliOp
from typing import Union
import numpy as np

from .expectation_operator_base import ExpectationOperatorBase
from ..util.data_preprocessing import adjust_input


class ExpectationOperatorDerivatives:
    def __init__(
        self,
        expectation_operator: Union[ExpectationOperatorBase, list],
        opflow_caching=True,
    ):
        # handling multiple Expectation_operators also done here

        self.expectation_operator = expectation_operator

        # Contains the OperatorMeasurement() of the expectation-operator for later replacement
        if isinstance(self.expectation_operator, ExpectationOperatorBase):
            # 1d output by a single expectation-operator
            self.multiple_output = False
            self._num_operators = 1
            self.parameters = ParameterVector("p_op", expectation_operator.num_parameters)

            opflow = self.expectation_operator.get_operator(self.parameters)
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

                self.parameters = ParameterVector("p_op", n_oper)
                ioff = 0
                for op in self.expectation_operator:
                    expectation_op_list.append(op.get_operator(self.parameters[ioff:]))
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

    def get_derivative(self, input: Union[str, tuple]):
        """return the opflow structure of the wavefunction"""
        if isinstance(input, str):
            # todo change with replaced operator
            if input == "I":
                measure_op = StateFn(
                    PauliOp(Pauli("I" * self.opflow.num_qubits)), is_measurement=True
                )
            elif input == "O":
                measure_op = self.opflow
            elif input == "OO":
                measure_op = self.get_operator_squared()
            elif input == "dop" or input == "Odop":
                measure_op = self._differentiation_from_tuple(
                    self.opflow.copy(), (self.parameters,), "O"
                )
            elif input == "dopdop" or input == "Odopdop":
                measure_op = self._differentiation_from_tuple(
                    self.opflow.copy(), (self.parameters, self.parameters), "O"
                )
            elif input == "OOdop":
                measure_op = self._differentiation_from_tuple(
                    self.get_operator_squared(), (self.parameters,), "OO"
                )
            elif input == "OOdopdop":
                measure_op = self._differentiation_from_tuple(
                    self.get_operator_squared(),
                    (self.parameters, self.parameters),
                    "OO",
                )
            else:
                raise ValueError("Unknown string command:", input)

        elif isinstance(input, tuple):
            measure_op = self._differentiation_from_tuple(self.opflow, input, "O")
        else:
            raise TypeError("Input is neither string nor tuple, but:", type(input))

        measure_op.replace = False
        return measure_op

    def get_differentiation_from_tuple(self, diff_tuple: tuple):
        return self.get_derivative(diff_tuple)

    def get_derivation_from_string(self, input_string: str):
        return self.get_derivative(input_string)

    def _differentiation_from_tuple(self, expectation_op, diff_tuple: tuple, expectation_op_label):
        """Recursive routine for automatic differentiating the expectation_operator
        Args:
            diff_tuple : tuple containing ParameterVectors or ParameterExpressions
        Returns:
            The differentiated opflow expression
        """
        # TODO: outer function for input checking

        if diff_tuple == ():
            # Cancel the recursion by returning the opflow of the simply measured feature map
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
                measure = _opflow_differentiation(
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

    def get_parameter_vector(self):
        return self.parameters

    @property
    def num_parameters(self):
        return len(self.parameters)

    @property
    def num_operators(self):
        return self._num_operators

    def assign_parameters(self, operator, parameters: np.ndarray):
        param_op_inp, multi_param_op = adjust_input(parameters, len(self.parameters))

        return_list = []
        for p in param_op_inp:
            dic = dict(zip(self.parameters, p))
            return_list.append(_convert_to_sparse(operator.assign_parameters(dic)))

        if multi_param_op:
            return ListOp(return_list)
        else:
            return return_list[0]


def _opflow_differentiation(opflow, parameters):
    """Special routines for differentiating a given expectation operator
    w.r.t. the Expectation operator parameters

    Args:
        expectation_op : opflow structure of the expectation operator
        param : parameters of the differentiation
    Returns:
        opflow structure of the differentiated input opflow
        return ListOp for multiple parameters
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


def _operator_differentiation(operator, parameters):
    """
    Computes the differentiation of a single operator
    """

    # Helper function for getting the coefficient
    def is_coeff_c(coeff, c):
        if isinstance(coeff, ParameterExpression):
            expr = coeff._symbol_expr
            return expr == c
        return coeff == c

    # Check for parameter lists, this is not supported here!

    # TODO also iterationable
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

        d_coeff = _coeff_derivative(coeff, parameters)
        grad_op = 0
        if op != ~Zero @ One and not is_coeff_c(d_coeff, 0.0):
            grad_op += d_coeff * op

        return grad_op

    # Only operator without coefficients is left, return 0 since a constant value is differentiated
    return 0.0 * PauliOp(Pauli("I" * operator.num_qubits))


def _convert_to_sparse(operator):
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
