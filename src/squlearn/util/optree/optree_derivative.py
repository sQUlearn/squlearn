import numpy as np
from typing import List, Union
import copy

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.quantum_info import SparsePauliOp

from .optree import (
    OpTreeNodeBase,
    OpTreeLeafBase,
    OpTreeNodeList,
    OpTreeNodeSum,
    OpTreeLeafCircuit,
    OpTreeLeafOperator,
)


def circuit_parameter_shift(
    element: Union[OpTreeLeafCircuit, QuantumCircuit], parameter: ParameterExpression
) -> Union[None,OpTreeNodeSum]:
    """
    Build the parameter shift derivative of a circuit wrt a single parameter.

    Args:
        element (Union[OpTreeLeafCircuit, QuantumCircuit]): The circuit to be differentiated.
        parameter (ParameterExpression): The parameter wrt which the circuit is differentiated.

    Returns:
        The parameter shift derivative of the circuit (always a OpTreeNodeSum)
    """

    if isinstance(element, OpTreeLeafCircuit):
        circuit = element.circuit
        input_type = "leaf"
    elif isinstance(element, QuantumCircuit):
        circuit = element
        input_type = "circuit"
    else:
        raise ValueError("element must be a CircuitTreeLeaf or a QuantumCircuit")

    # Return None when the parameter is not in the circuit
    if parameter not in circuit._parameter_table:
        return None

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


def operator_derivative(
    element: Union[OpTreeLeafOperator, SparsePauliOp], parameter: ParameterExpression
) -> Union[OpTreeLeafOperator, SparsePauliOp]:
    """
    Obtain the derivative of an operator wrt a single parameter.

    Args:
        element (Union[OpTreeLeafOperator, SparsePauliOp]): The operator to be differentiated.
        parameter (ParameterExpression): The parameter wrt which the operator is differentiated.

    Returns:
        Operator derivative as OpTreeLeafOperator or SparsePauliOp

    """

    if isinstance(element, OpTreeLeafOperator):
        operator = element.operator
        input_type = "leaf"
    elif isinstance(element, SparsePauliOp):
        operator = element
        input_type = "operator"

    # Return None when the parameter is part of the operator
    if parameter not in operator._parameter_table:
        return None

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
            return OpTreeLeafOperator(operator_grad)
        return operator_grad
    return None


def derivative_inplace(
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

            if isinstance(child, (QuantumCircuit, OpTreeLeafCircuit)):
                # reached a circuit leaf -> grad by parameter shift function
                grad = circuit_parameter_shift(child, parameter)
            elif isinstance(child, (SparsePauliOp, OpTreeLeafOperator)):
                grad = operator_derivative(child, parameter)
            else:
                # Node -> recursive call
                derivative_inplace(child, parameter)
                grad = child

            print("grad",grad)
            print("grad_fac",grad_fac)
            
            # Product rule for differentiation
            if isinstance(grad_fac, float):
                # if grad_fac is a numeric value
                if grad_fac == 0.0 and grad is not None:
                    print("a")
                    tree_node.children[i] = grad
                elif grad is not None:
                    print("b")
                    tree_node.children[i] = OpTreeNodeSum(
                        [child, grad], [grad_fac, tree_node.factor[i]]
                    )
                    tree_node.factor[i] = 1.0
                else:
                    print("c")
                    tree_node.factor[i] = grad_fac
            else:
                # if grad_fac is still a parameter
                if grad is not None:
                    tree_node.children[i] = OpTreeNodeSum(
                        [child, grad], [grad_fac, tree_node.factor[i]]
                    )
                    tree_node.factor[i] = 1.0
                else:
                    tree_node.factor[i] = grad_fac

    else:
        raise ValueError("tree_node must be a CircuitTreeSum or a CircuitTreeList")


def derivative(
    element: Union[
        OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit, OpTreeLeafOperator, SparsePauliOp
    ],
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
    # For inplace operation, the input must be a OpTreeNodeList
    is_node = True
    if not isinstance(element, OpTreeNodeBase):
        is_node = False
        start = OpTreeNodeList([element], [1.0])
    else:
        start = element

    derivative_list = []
    fac_list = []
    for dp in parameters:
        # copy the circuit tree for inplace operation during derivative calculation
        res = copy.deepcopy(start)
        derivative_inplace(res, dp)
        if is_node:
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


def derivative_copy(
    element: Union[
        OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit, OpTreeLeafOperator, SparsePauliOp
    ],
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
                grad = derivative_copy(child, parameter)

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
                children_list.append(derivative_copy(child, parameter))
                factor_list.append(element.factor[i])

        # Rebuild the tree with the new children and factors (copy part)
        if isinstance(element, OpTreeNodeSum):
            return OpTreeNodeSum(children_list, factor_list)
        elif isinstance(element, OpTreeNodeList):
            return OpTreeNodeList(children_list, factor_list)
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

    elif isinstance(element, (QuantumCircuit, OpTreeLeafCircuit)):
        # Reached a circuit leaf -> grad by parameter shift function
        return circuit_parameter_shift(element, parameter)
    elif isinstance(element, (SparsePauliOp, OpTreeLeafOperator)):
        # Reached a operator leaf -> grad by parameter shift function
        return operator_derivative(element, parameter)
    else:
        raise ValueError("Unsupported element type: " + str(type(element)))


def derivative_v2(
    element: Union[
        OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit, OpTreeLeafOperator, SparsePauliOp
    ],
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
        derivative_list.append(derivative_copy(element, dp))
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
            children_list = [simplify_copy(child) for child in element.children]
            factor_list = element.factor
            operation_list = element.operation

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
    elif isinstance(element, (SparsePauliOp, OpTreeLeafOperator)):
        return simplify_operator(element)
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


def simplify_operator(
    element: Union[SparsePauliOp, OpTreeLeafOperator]
) -> Union[SparsePauliOp, OpTreeLeafOperator]:
    if isinstance(element, OpTreeLeafOperator):
        operator = element.operator
        input_type = "leaf"
    else:
        operator = element
        input_type = "operator"

        pauli_list = []
        coeff_list = []

        # check for identical paulis and merge them
        for i, pauli in enumerate(operator.paulis):
            # Check if pauli already exists in the list
            if pauli in pauli_list:
                index = pauli_list.index(pauli)
                coeff_list[index] += operator.coeffs[i]
            else:
                pauli_list.append(pauli)
                coeff_list.append(operator.coeffs[i])

    if len(pauli_list) > 0:
        operator_simp = SparsePauliOp(pauli_list, coeff_list)
        if input_type == "leaf":
            return OpTreeLeafOperator(operator_simp)
        return operator_simp
    else:
        return None
