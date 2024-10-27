import numpy as np
from typing import List, Union, Set
import copy

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.compiler import transpile

from .optree import (
    OpTreeNodeBase,
    OpTreeList,
    OpTreeSum,
    OpTreeCircuit,
    OpTreeOperator,
    OpTreeValue,
    OpTreeExpectationValue,
    OpTreeMeasuredOperator,
)


def _circuit_parameter_shift(
    element: Union[OpTreeCircuit, QuantumCircuit, OpTreeValue],
    parameter: ParameterExpression,
) -> Union[None, OpTreeSum, OpTreeValue]:
    """
    Build the parameter shift derivative of a circuit w.r.t. a single parameter.

    Args:
        element (Union[OpTreeLeafCircuit, QuantumCircuit]): The circuit to be differentiated.
        parameter (ParameterExpression): The parameter w.r.t. which the circuit is differentiated.

    Returns:
        The parameter shift derivative of the circuit (always a OpTreeNodeSum)
    """

    def _param_in_instruction(instruction, parameter):
        if len(instruction.params) == 0:
            return False
        if not isinstance(instruction.params[0], ParameterExpression):
            return parameter == instruction.params[0]
        else:
            return parameter in instruction.params[0].parameters

    if isinstance(element, OpTreeValue):
        return OpTreeValue(0.0)

    if isinstance(element, OpTreeCircuit):
        circuit = element.circuit
        input_type = "leaf"
    elif isinstance(element, QuantumCircuit):
        circuit = element
        input_type = "circuit"
    else:
        raise ValueError("element must be a CircuitTreeLeaf or a QuantumCircuit")

    # Transpile to gates that are supported in the parameter shift rule
    circuit = OpTreeDerivative.transpile_to_supported_instructions(circuit)

    # Return None when the parameter is not in the circuit
    if parameter not in circuit.parameters:
        return OpTreeValue(0.0)

    shift_sum = OpTreeSum()

    primitives_v2 = False
    iref_to_data_index = None
    param_table = []
    if hasattr(circuit, "_parameter_table"):
        param_table = circuit._parameter_table[parameter]  # pylint: disable=protected-access
        iref_to_data_index = {id(inst.operation): idx for idx, inst in enumerate(circuit.data)}
    else:
        primitives_v2 = True
        param_table = []
        operator_index = 0
        for inst in circuit.data:
            if _param_in_instruction(inst, parameter):
                param_table.append((inst.operation, operator_index))
            operator_index += 1

    # Loop through all parameter occurences in the circuit
    for param_reference in param_table:

        # Get the gate in which the parameter is located
        if primitives_v2:
            original_gate, m = param_reference
        else:
            original_gate, _ = param_reference
            m = iref_to_data_index[id(original_gate)]

        fac = original_gate.params[0].gradient(parameter)

        # Copy the circuit for the shifted ones
        pshift_circ = copy.deepcopy(circuit)
        mshift_circ = copy.deepcopy(circuit)

        # Get the gates instance in which the parameter is located
        pshift_gate = pshift_circ.data[m].operation
        mshift_gate = mshift_circ.data[m].operation

        # Get the parameter instances in the shited circuits
        p_param = pshift_gate.params[0]
        m_param = mshift_gate.params[0]

        # Shift the parameter in the gates
        # For analytic gradients the circuit parameters are shifted once by +pi/2 and
        # once by -pi/2.
        shift_constant = 0.5
        pshift_gate.params[0] = p_param + (np.pi / (4 * shift_constant))
        mshift_gate.params[0] = m_param - (np.pi / (4 * shift_constant))

        # Save replaced gates in the circuit
        pshift_circ.data[m] = pshift_circ.data[m].replace(operation=pshift_gate)
        mshift_circ.data[m] = mshift_circ.data[m].replace(operation=mshift_gate)

        # Append the shifted circuits to the sum
        if input_type == "leaf":
            shift_sum.append(OpTreeCircuit(pshift_circ), shift_constant * fac)
            shift_sum.append(OpTreeCircuit(mshift_circ), -shift_constant * fac)
        else:
            shift_sum.append(pshift_circ, shift_constant * fac)
            shift_sum.append(mshift_circ, -shift_constant * fac)

    return shift_sum


def _operator_differentiation(
    element: Union[OpTreeOperator, SparsePauliOp, OpTreeValue],
    parameter: ParameterExpression,
) -> Union[OpTreeOperator, SparsePauliOp, OpTreeValue]:
    """
    Obtain the derivative of an operator w.r.t. a single parameter.

    Args:
        element (Union[OpTreeLeafOperator, SparsePauliOp]): The operator to be differentiated.
        parameter (ParameterExpression): The parameter w.r.t. which the operator is differentiated.

    Returns:
        Operator derivative as OpTreeLeafOperator or SparsePauliOp

    """
    if isinstance(element, OpTreeValue):
        return OpTreeValue(0.0)

    if isinstance(element, OpTreeOperator):
        operator = element.operator
        input_type = "leaf"
    elif isinstance(element, SparsePauliOp):
        operator = element
        input_type = "sparse_pauli_op"

    # Return None when the parameter is part of the operator
    if parameter not in operator.parameters:
        return OpTreeValue(0.0)

    # Rebuild the operator with the differentiated coefficients
    op_list = []
    param_list = []
    for i, coeff in enumerate(operator.coeffs):
        if isinstance(coeff, ParameterExpression):
            # the 1j fixes a bug in qiskit
            d_coeff = -1j * ((1j * coeff).gradient(parameter))
            if isinstance(d_coeff, complex):
                if d_coeff.imag == 0:
                    d_coeff = d_coeff.real
        else:
            d_coeff = 0.0

        if d_coeff != 0.0:
            op_list.append(operator.paulis[i])
            param_list.append(d_coeff)

    if len(op_list) > 0:
        operator_grad = SparsePauliOp(op_list, param_list)
        if input_type == "leaf":
            return OpTreeOperator(operator_grad)
        return operator_grad
    return None


def _differentiate_inplace(
    tree_node: OpTreeNodeBase,
    parameter: ParameterExpression,
) -> None:
    """
    Create the derivative of a OpTreeNode w.r.t. a single parameter, modifies the tree inplace.

    Functions returns nothing, since the OpTree is modified inplace.

    Args:
        tree_node (OpTreeNodeBase): The OpTree Node to be differentiated.
        parameter (ParameterExpression): The parameter w.r.t. which the circuit is differentiated.

    """
    if isinstance(tree_node, OpTreeNodeBase):
        remove_list = []
        for i, child in enumerate(tree_node.children):
            if isinstance(tree_node.factor[i], ParameterExpression):
                grad_fac = tree_node.factor[i].gradient(parameter)
            else:
                grad_fac = 0.0

            if isinstance(child, (QuantumCircuit, OpTreeCircuit)):
                # reached a circuit leaf -> grad by parameter shift function
                grad = _circuit_parameter_shift(child, parameter)
            elif isinstance(child, (SparsePauliOp, OpTreeOperator)):
                grad = _operator_differentiation(child, parameter)
            elif isinstance(child, OpTreeMeasuredOperator):
                grad_op = _operator_differentiation(child.operator, parameter)
                if isinstance(grad_op, OpTreeValue):
                    grad = grad_op
                else:
                    grad = OpTreeMeasuredOperator(child.circuit, grad_op)
            elif isinstance(child, OpTreeValue):
                grad = OpTreeValue(0.0)
            elif isinstance(child, OpTreeExpectationValue):
                raise NotImplementedError("Expectation value differentiation not implemented yet")
            else:
                # Node -> recursive call
                _differentiate_inplace(child, parameter)
                grad = child

            # Product rule for differentiation
            if isinstance(grad_fac, float):
                # if grad_fac is a numeric value
                if grad_fac == 0.0:
                    tree_node.children[i] = grad
                else:
                    tree_node.children[i] = OpTreeSum(
                        [child, grad], [grad_fac, tree_node.factor[i]]
                    )
                    tree_node.factor[i] = 1.0
            else:
                # if grad_fac is still a parameter
                tree_node.children[i] = OpTreeSum([child, grad], [grad_fac, tree_node.factor[i]])
                tree_node.factor[i] = 1.0

        if len(remove_list) > 0:
            tree_node.remove(remove_list)
    else:
        raise ValueError("tree_node must be a OpTreeNodeSum or a OpTreeNodeList")


def _differentiate_copy(
    element: Union[OpTreeNodeBase, OpTreeCircuit, QuantumCircuit, OpTreeOperator, SparsePauliOp],
    parameter: ParameterExpression,
) -> OpTreeNodeBase:
    """
    Create the derivative of a OpTree or circuit w.r.t. a single parameter, the input is untouched.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): The OpTree (or circuit) to be differentiated.
        parameter (ParameterExpression): The parameter w.r.t. which the circuit is differentiated.

    Returns:
        The derivative of the circuit as an OpTree
    """

    if isinstance(element, OpTreeNodeBase):
        children_list = []
        factor_list = []
        for i, child in enumerate(element.children):
            if isinstance(element.factor[i], ParameterExpression):
                # get derivative of factor
                grad_fac = element.factor[i].gradient(parameter)
                fac = element.factor[i]
                # recursive call to get the gradient for the child
                grad = _differentiate_copy(child, parameter)

                # Product rule for differentiation
                if isinstance(grad_fac, float):
                    if grad_fac == 0.0:
                        children_list.append(grad)
                        factor_list.append(fac)
                    else:
                        children_list.append(OpTreeSum([child, grad], [grad_fac, fac]))
                        factor_list.append(1.0)
                else:
                    children_list.append(OpTreeSum([child, grad], [grad_fac, fac]))
                    factor_list.append(1.0)
            else:
                # No parameter in factor -> just call recursive call for the children
                children_list.append(_differentiate_copy(child, parameter))
                factor_list.append(element.factor[i])

        # Rebuild the tree with the new children and factors (copy part)
        if isinstance(element, OpTreeSum):
            return OpTreeSum(children_list, factor_list)
        elif isinstance(element, OpTreeList):
            return OpTreeList(children_list, factor_list)
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

    elif isinstance(element, (QuantumCircuit, OpTreeCircuit)):
        # Reached a circuit leaf -> grad by parameter shift function
        return _circuit_parameter_shift(element, parameter)
    elif isinstance(element, (SparsePauliOp, OpTreeOperator)):
        # Reached a operator leaf -> grad by parameter shift function
        return _operator_differentiation(element, parameter)
    elif isinstance(element, OpTreeMeasuredOperator):
        grad_op = _operator_differentiation(element.operator, parameter)
        if isinstance(grad_op, OpTreeValue):
            return grad_op
        return OpTreeMeasuredOperator(element.circuit, grad_op)
    elif isinstance(element, OpTreeExpectationValue):
        raise NotImplementedError("Expectation value differentiation not implemented yet")
    elif isinstance(element, OpTreeValue):
        return OpTreeValue(0.0)
    else:
        raise ValueError("Unsupported element type: " + str(type(element)))


class OpTreeDerivative:
    """Static class for differentiation of a OpTrees, circuits, or operators."""

    SUPPORTED_GATES = {
        "s",
        "sdg",
        "t",
        "tdg",
        "ecr",
        "sx",
        "x",
        "y",
        "z",
        "h",
        "rx",
        "ry",
        "rz",
        "p",
        "cx",
        "cy",
        "cz",
    }

    @staticmethod
    def transpile_to_supported_instructions(
        circuit: QuantumCircuit, supported_gates: Set[str] = SUPPORTED_GATES
    ) -> QuantumCircuit:
        """Function for transpiling a circuit to a supported instruction set for gradient calculation.

        Args:
            circuit (QuantumCircuit): Circuit to transpile.
            supported_gates (Set[str]): Set of supported gates (Default set given).

        Returns:
            Circuit which is transpiled to the supported instruction set.
        """

        unique_ops = set(circuit.count_ops())
        if not unique_ops.issubset(supported_gates):
            circuit = transpile(
                circuit,
                basis_gates=list(supported_gates),
                optimization_level=0,
                layout_method="trivial",
            )
        return circuit

    @staticmethod
    def differentiate(
        element: Union[
            OpTreeNodeBase, OpTreeCircuit, QuantumCircuit, OpTreeOperator, SparsePauliOp
        ],
        parameters: Union[ParameterExpression, List[ParameterExpression], ParameterVector],
    ) -> OpTreeNodeBase:
        """
        Calculates the derivative of a OpTree (or circuit) w.r.t. to a parameter or a list of parameters.

        Args:
            element (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): OpTree (or circuit)
                                                                                to be differentiated.
            parameters (Union[ParameterExpression, List[ParameterExpression], ParameterVector]): Parameter(s) w.r.t.
                                                                                                    the OpTree is
                                                                                                    differentiated

        Returns:
            The derivative of the OpTree (or circuit) in OpTree form.
        """

        # Preprocessing
        # ParameterExpression as list
        is_list = True
        if isinstance(parameters, ParameterExpression):
            parameters = [parameters]
            is_list = False

        if isinstance(element, (QuantumCircuit, OpTreeCircuit)):
            if isinstance(element, OpTreeCircuit):
                element = OpTreeCircuit(
                    OpTreeDerivative.transpile_to_supported_instructions(element.circuit)
                )
            else:
                element = OpTreeCircuit(
                    OpTreeDerivative.transpile_to_supported_instructions(element)
                )

        # For inplace operation, the input must be a OpTreeNodeList
        is_node = True
        if not isinstance(element, OpTreeNodeBase):
            is_node = False
            start = OpTreeList([element], [1.0])
        else:
            start = element

        derivative_list = []
        fac_list = []
        for dp in parameters:
            # copy the circuit tree for inplace operation during derivative calculation
            res = copy.deepcopy(start)
            _differentiate_inplace(res, dp)
            if is_node:
                derivative_list.append(res)
                fac_list.append(1.0)
            else:
                # if the input was a circuit, get rid ouf the outer OpTreeNodeList container
                # from the preprocessing
                if len(res.children) > 0:
                    derivative_list.append(res.children[0])
                    fac_list.append(res.factor[0])

        # Return either in list form or as single OpTreeNode
        if is_list or len(derivative_list) == 0:
            return OpTreeList(derivative_list, fac_list)
        else:
            return derivative_list[0]

    @staticmethod
    def differentiate_v2(
        element: Union[
            OpTreeNodeBase, OpTreeCircuit, QuantumCircuit, OpTreeOperator, SparsePauliOp
        ],
        parameters: Union[ParameterExpression, List[ParameterExpression], ParameterVector],
    ) -> OpTreeNodeBase:
        """
        Calculates the derivative of a OpTree (or circuit) w.r.t. to a parameter or a list of parameters.

        Second implementation, in which the derivative is calculated during the recursive derivative
        computation.

        Args:
            element (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): OpTree (or circuit)
                                                                                to be differentiated.
            parameters (Union[ParameterExpression, List[ParameterExpression], ParameterVector]): Parameter(s) w.r.t.
                                                                                                    the OpTree is
                                                                                                    differentiated

        Returns:
            The derivative of the OpTree (or circuit) in OpTree form.
        """

        # Preprocessing -> ParameterExpression as list
        is_list = True
        if isinstance(parameters, ParameterExpression):
            parameters = [parameters]
            is_list = False

        if isinstance(element, (QuantumCircuit, OpTreeCircuit)):
            if isinstance(element, OpTreeCircuit):
                element = OpTreeCircuit(
                    OpTreeDerivative.transpile_to_supported_instructions(element.circuit)
                )
            else:
                element = OpTreeCircuit(
                    OpTreeDerivative.transpile_to_supported_instructions(element)
                )

        # Loop through all parameters and calculate the derivative
        derivative_list = []
        fac_list = []
        for dp in parameters:
            derivative_list.append(_differentiate_copy(element, dp))
            fac_list.append(1.0)

        # Adjust the output for single parameter input
        if is_list or len(derivative_list) == 0:
            return OpTreeList(derivative_list, fac_list)
        else:
            return derivative_list[0]
