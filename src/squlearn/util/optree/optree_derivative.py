import numpy as np
from typing import List, Union
import copy

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.quantum_info import SparsePauliOp
#from qiskit.opflow.gradients.derivative_base import _coeff_derivative

from .optree import (
    OpTreeNodeBase,
    OpTreeLeafBase,
    OpTreeNodeList,
    OpTreeNodeSum,
    OpTreeLeafCircuit,
    OpTreeLeafContainer,
    OpTreeLeafOperator,
)


def circuit_parameter_shift(
    element: Union[OpTreeLeafCircuit, QuantumCircuit], parameter: ParameterExpression
) -> OpTreeNodeSum:
    """
    Build the parameter shift derivative of a circuit wrt a single parameter.

    Args:
        element (Union[OpTreeLeafCircuit, QuantumCircuit]): The circuit to be differentiated.
        parameter (ParameterExpression): The parameter wrt which the circuit is differentiated.

    Returns:
        The parameter shift derivative of the circuit (always a OpTreeNodeSum)
    """

    # TODO: fix what happens if parameter is not present!

    if isinstance(element, OpTreeLeafCircuit):
        circuit = element.circuit
        input_type = "leaf"
    elif isinstance(element, QuantumCircuit):
        circuit = element
        input_type = "circuit"
    else:
        raise ValueError("element must be a CircuitTreeLeaf or a QuantumCircuit")

    iref_to_data_index = {id(inst.operation): idx for idx, inst in enumerate(circuit.data)}
    shift_sum = OpTreeNodeSum()
    # Loop through all parameter occurences in the circuit
    for param_reference in circuit._parameter_table[parameter]:  # pylint: disable=protected-access
        # Get the gate in which the parameter is located
        original_gate, param_index = param_reference
        m = iref_to_data_index[id(original_gate)]

        # Get derivative of the factor of the gate
        fac = original_gate.params[0].gradient(parameter)

        # Copy the circuit for the shifted ones
        pshift_circ = copy.deepcopy(circuit)
        mshift_circ = copy.deepcopy(circuit)

        # Get the gates instance in which the parameter is located
        pshift_gate = pshift_circ.data[m].operation
        mshift_gate = mshift_circ.data[m].operation

        # Get the parameter instances in the shited circuits
        p_param = pshift_gate.params[param_index]
        m_param = mshift_gate.params[param_index]

        # Shift the parameter in the gates
        # For analytic gradients the circuit parameters are shifted once by +pi/2 and
        # once by -pi/2.
        shift_constant = 0.5
        pshift_gate.params[param_index] = p_param + (np.pi / (4 * shift_constant))
        mshift_gate.params[param_index] = m_param - (np.pi / (4 * shift_constant))

        # Append the shifted circuits to the sum
        if input_type == "leaf":
            shift_sum.append(OpTreeLeafCircuit(pshift_circ), shift_constant * fac)
            shift_sum.append(OpTreeLeafCircuit(mshift_circ), -shift_constant * fac)
        else:
            shift_sum.append(pshift_circ, shift_constant * fac)
            shift_sum.append(mshift_circ, -shift_constant * fac)

    return shift_sum

def operator_derivative(element:Union[OpTreeLeafOperator, SparsePauliOp], parameter: ParameterExpression):
    """
    Obtain the derivative of an operator wrt a single parameter.
    """

    if isinstance(element, OpTreeLeafOperator):
        operator = element.operator
        input_type = "leaf"
    elif isinstance(element, SparsePauliOp):
        operator = element
        input_type = "operator"

    # Rebuild the operator with the differentiated coefficients
    op_list=[]
    param_list=[]
    for i, coeff in enumerate(operator.coeffs):
        if isinstance(coeff, ParameterExpression):
            # the 1j fixes a bug in qiskit
            d_coeff = -1j*((1j*coeff).gradient(parameter))
            if isinstance(d_coeff,complex):
                if d_coeff.imag == 0:
                    d_coeff = d_coeff.real
        else:
            d_coeff = 0.0

        if d_coeff != 0.0:
            op_list.append(operator.paulis[i])
            param_list.append(d_coeff)

    if len(op_list) > 0:
        operator_grad = SparsePauliOp(op_list,param_list)
        if input_type=="leaf":
            return OpTreeLeafOperator(operator_grad)
        return operator_grad
    return None


def circuit_derivative_inplace(
    tree_node: OpTreeNodeBase,
    parameter: ParameterExpression,
) -> None:
    """ "
    Create the derivative of a OpTreeNode wrt a single parameter, modifies the tree inplace.

    Functions returns nothing, since the OpTree is modified inplace.

    Args:
        tree_node (OpTreeNodeBase): The OpTree Node to be differentiated.
        parameter (ParameterExpression): The parameter wrt which the circuit is differentiated.

    """
    if isinstance(tree_node, OpTreeNodeBase):
        for i, child in enumerate(tree_node.children):
            if isinstance(tree_node.factor[i], ParameterExpression):
                grad_fac = tree_node.factor[i].gradient(parameter)
            else:
                grad_fac = 0.0

            if isinstance(child, QuantumCircuit) or isinstance(child, OpTreeLeafCircuit):
                # reached a leaf -> grad by parameter shift function
                grad = circuit_parameter_shift(child, parameter)
            else:
                # Node -> recursive call
                circuit_derivative_inplace(child, parameter)
                grad = child

            # Product rule for differentiation
            if isinstance(grad_fac, float):
                # if grad_fac is a numeric value
                if grad_fac == 0.0:
                    tree_node.children[i] = grad
                else:
                    tree_node.children[i] = OpTreeNodeSum(
                        [child, grad], [grad_fac, tree_node.factor[i]]
                    )
                    tree_node.factor[i] = 1.0
            else:
                # if grad_fac is still a parameter
                tree_node.children[i] = OpTreeNodeSum(
                    [child, grad], [grad_fac, tree_node.factor[i]]
                )
                tree_node.factor[i] = 1.0

    else:
        raise ValueError("tree_node must be a CircuitTreeSum or a CircuitTreeList")


def circuit_derivative(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    parameters: Union[ParameterExpression, List[ParameterExpression], ParameterVector],
) -> OpTreeNodeBase:
    """
    Calculates the derivative of a OpTree (or circuit) wrt to a parameter or a list of parameters.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): OpTree (or circuit)
                                                                            to be differentiated.
        parameters (Union[ParameterExpression, List[ParameterExpression], ParameterVector]): Parameter(s) wrt
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
    is_not_circuit = True
    # QuantumCircuit or OpTreeLeafCircuit as single OpTreeNodeList,
    #  otherwise inplace operation does not work
    if isinstance(element, QuantumCircuit) or isinstance(element, OpTreeLeafCircuit):
        is_not_circuit = False
        start = OpTreeNodeList([element], [1.0])
    else:
        start = element

    derivative_list = []
    fac_list = []
    for dp in parameters:
        # copy the circuit tree for inplace operation during derivative calculation
        res = copy.deepcopy(start)
        circuit_derivative_inplace(res, dp)
        if is_not_circuit:
            derivative_list.append(res)
        else:
            # if the input was a circuit, get rid ouf the outer OpTreeNodeList container
            # from the preprocessing
            derivative_list.append(res.children[0])
        fac_list.append(1.0)

    # Return either in list form or as single OpTreeNode
    if is_list:
        return OpTreeNodeList(derivative_list, fac_list)
    else:
        return derivative_list[0]


def circuit_derivative_copy(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    parameter: ParameterExpression,
) -> OpTreeNodeBase:
    """
    Create the derivative of a OpTree or circuit wrt a single parameter, the input is untouched.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): The OpTree (or circuit) to be differentiated.
        parameter (ParameterExpression): The parameter wrt which the circuit is differentiated.

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
                grad = circuit_derivative_copy(child, parameter)

                # Product rule for differentiation
                if isinstance(grad_fac, float):
                    if grad_fac == 0.0:
                        children_list.append(grad)
                        factor_list.append(fac)
                    else:
                        children_list.append(OpTreeNodeSum([child, grad], [grad_fac, fac]))
                        factor_list.append(1.0)
                else:
                    children_list.append(OpTreeNodeSum([child, grad], [grad_fac, fac]))
                    factor_list.append(1.0)
            else:
                # No parameter in factor -> just call recursive call for the children
                children_list.append(circuit_derivative_copy(child, parameter))
                factor_list.append(element.factor[i])

        # Rebuild the tree with the new children and factors (copy part)
        if isinstance(element, OpTreeNodeSum):
            return OpTreeNodeSum(children_list, factor_list)
        elif isinstance(element, OpTreeNodeList):
            return OpTreeNodeList(children_list, factor_list)
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

    else:
        # Reached a leaf -> grad by parameter shift function
        return circuit_parameter_shift(element, parameter)


def circuit_derivative_v2(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    parameters: Union[ParameterExpression, List[ParameterExpression], ParameterVector],
) -> OpTreeNodeBase:
    """
    Calculates the derivative of a OpTree (or circuit) wrt to a parameter or a list of parameters.

    Second implementation, in which the derivative is calculated during the recursive derivative
    computation.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): OpTree (or circuit)
                                                                            to be differentiated.
        parameters (Union[ParameterExpression, List[ParameterExpression], ParameterVector]): Parameter(s) wrt
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

    # Loop through all parameters and calculate the derivative
    derivative_list = []
    fac_list = []
    for dp in parameters:
        derivative_list.append(circuit_derivative_copy(element, dp))
        fac_list.append(1.0)

    # Adjust the output for single parameter input
    if is_list:
        return OpTreeNodeList(derivative_list, fac_list)
    else:
        return derivative_list[0]


def simplify_copy(
    element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp]
) -> Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp]:
    """
    Function for simplifying an OpTree structure, the input is kept untouched.

    Merges double sums and identifies identical branches or leafs in sums.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp]): The OpTree to be simplified.

    Returns:
        A simplified copy of the OpTree.
    """

    def combine_two_ops(op1, op2):
        """Helper function for combining two operations into one.

        TODO: not used/tested yet
        """
        if op1 is None and op2 is None:
            return None
        elif op1 is None and op2 is not None:
            return op2
        elif op1 is not None and op2 is None:
            return op1
        else:
            return lambda x: op1(op2(x))

    if isinstance(element, OpTreeNodeBase):
        if len(element.children) > 0:
            # Recursive call for all children
            children_list = []
            factor_list = []
            operation_list = []
            for i, child in enumerate(element.children):
                children_list.append(simplify_copy(child))  # Recursive call
                factor_list.append(element.factor[i])
                operation_list.append(element.operation[i])

            if isinstance(element, OpTreeNodeSum):
                new_element = OpTreeNodeSum(children_list, factor_list, operation_list)
            elif isinstance(element, OpTreeNodeList):
                new_element = OpTreeNodeList(children_list, factor_list, operation_list)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

            # Check for double sum if the element is a sum and one of the children is a sums
            if isinstance(new_element, OpTreeNodeSum) and any(
                [isinstance(child, OpTreeNodeSum) for child in new_element.children]
            ):
                # Merge the sum of a sum into the parent sum
                children_list = []
                factor_list = []
                operation_list = []
                for i, child in enumerate(new_element.children):
                    if isinstance(child, OpTreeNodeSum):
                        for j, childs_child in enumerate(child.children):
                            children_list.append(childs_child)
                            factor_list.append(new_element.factor[i] * child.factor[j])
                            operation_list.append(
                                combine_two_ops(new_element.operation[i], child.operation[j])
                            )
                    else:
                        children_list.append(child)
                        factor_list.append(new_element.factor[i])
                        operation_list.append(new_element.operation[i])
                # Create OpTreeSum with the new (potentially merged) children
                new_element = OpTreeNodeSum(children_list, factor_list, operation_list)

            # Check for similar branches in the Sum and merge them into a single branch
            if isinstance(new_element, OpTreeNodeSum):
                children_list = []
                factor_list = []
                operation_list = []

                # TODO check for operator
                for i, child in enumerate(new_element.children):
                    # Chick if child already exists in the list
                    # (branch is already present -> merging)
                    if child in children_list:
                        index = children_list.index(child)
                        factor_list[index] += new_element.factor[i]
                    else:
                        children_list.append(child)
                        factor_list.append(new_element.factor[i])
                        operation_list.append(new_element.operation[i])

                # Create new OpTreeSum with the merged branches
                new_element = OpTreeNodeSum(children_list, factor_list, operation_list)

            return new_element

        else:
            # Reached an empty Node -> cancel the recursion
            return copy.deepcopy(element)
    else:
        # Reached a leaf -> cancel the recursion
        return copy.deepcopy(element)


def simplify_inplace(
    element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp]
) -> None:
    """
    Function for simplifying an OpTree structure, the input is modified inplace.

    Merges double sums and identifies identical branches or leafs in sums.
    The function returns nothing, since the OpTree is modified inplace.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp]): The OpTree to be simplified.

    """
    raise NotImplementedError("Not implemented yet")