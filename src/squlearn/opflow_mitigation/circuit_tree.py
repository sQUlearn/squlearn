import numpy as np
import copy
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression
from typing import List, Union, Callable, Any
import time
from qiskit.primitives import Estimator
from qiskit.converters import circuit_to_dag
from hashlib import blake2b

import dill as pickle


def hash_circuit(circuit: QuantumCircuit) -> tuple:
    """Hashes a circuit using the qiskit _circuit_key function.

    Args:
        circuit (QuantumCircuit): The circuit to be hashed.

    Returns:
        a tuple containing the circuit information that can be used for comparison.

    """
    # TODO: can be replaced by whatever hash function is used in qiskit in the future.
    from qiskit.primitives.utils import _circuit_key

    return _circuit_key(circuit)
    # return blake2b(str(_circuit_key(circuit)).encode("utf-8"), digest_size=20).hexdigest() # faster for comparison slower for generation


class OpTreeElementBase:
    """Base class for elements of the OpTree."""

    pass


class OpTreeLeafBase(OpTreeElementBase):
    """Base class for Leafs of the OpTree."""

    pass


class OpTreeNodeBase(OpTreeElementBase):
    """Base class for nodes in the OpTree.

    Args:
        children_list (list): A list of children of the node.
        factor_list (list): A list of factors for each child.
        operation_list (list): A list of operations that are applied to each child.
    """

    def __init__(
        self,
        children_list: Union[None, List[OpTreeElementBase]] = None,
        factor_list: Union[None, List[float]] = None,
        operation_list: Union[None, List[Callable], List[None]] = None,
    ) -> None:
        # Initialize with a given list of children
        # Factors default to 1.0
        # Operations default to None
        if children_list is not None:
            self._children_list = children_list
            if factor_list is not None:
                if len(children_list) != len(factor_list):
                    raise ValueError("circuit_list and factor_list must have the same length")
                self._factor_list = factor_list
            else:
                self._factor_list = [1.0 for i in range(len(children_list))]

            if operation_list is not None:
                if len(children_list) != len(operation_list):
                    raise ValueError("circuit_list and operation_list must have the same length")
                self._operation_list = operation_list
            else:
                self._operation_list = [None for i in range(len(children_list))]

        else:
            # Initialize empty
            self._children_list = []
            self._factor_list = []
            self._operation_list = []

    @property
    def children(self) -> List[OpTreeElementBase]:
        """Returns the list of children of the node."""
        return self._children_list

    @property
    def factor(self) -> List[float]:
        """Returns the list of factors of the node."""
        return self._factor_list

    @property
    def operation(self) -> List[Union[Callable, None]]:
        """Returns the list of operations of the node."""
        return self._operation_list

    def append(
        self,
        children: OpTreeElementBase,
        factor: float = 1.0,
        operation: Union[None, Callable] = None,
    ):
        """Appends a child to the node.

        Args:
            children (OpTreeElementBase): The child to be appended.
            factor (float, optional): The factor that is applied to the child. Defaults to 1.0.
            operation ([type], optional): The operation that is applied to the child. Defaults to None.
        """

        self._children_list.append(children)
        self._factor_list.append(factor)
        self._operation_list.append(operation)

    def __eq__(self, other) -> bool:
        """Function for comparing two OpTreeNodes.

        Checks the following in this order:
        - Type of the nodes is the same
        - The number of children is the same
        - The factors are the same
        - The children are the same
        - The operations and factors of the children are the same
        """

        if isinstance(other, type(self)):
            # Fast checks: number of children is equal
            if len(self._children_list) != len(other._children_list):
                return False
            # Medium fast check: len and set of factors are equal
            fac_set_self = set(self._factor_list)
            fac_set_other = set(other._factor_list)
            if len(fac_set_self) != len(fac_set_other):
                return False
            if fac_set_self != fac_set_other:
                return False
            # Slow check: compare are all children and check factors and operations
            for child in self._children_list:
                if child not in other._children_list:
                    return False
                else:
                    index = other._children_list.index(child)
                    if (
                        self._factor_list[self._children_list.index(child)]
                        != other._factor_list[index]
                        and self._operation_list[self._children_list.index(child)]
                        != other._operation_list[index]
                    ):
                        return False
            return True
        else:
            return False


class OpTreeNodeList(OpTreeNodeBase):
    """A OpTree node that represents its children as a list/array/vector."""

    def __str__(self) -> str:
        """Returns a string representation of the node as a list of its children."""
        text = "["
        for i, child in enumerate(self._children_list):
            if isinstance(child, QuantumCircuit):
                text += str(self._factor_list[i]) + "*" + "\n" + str(child) + "\n"
            else:
                text += str(self._factor_list[i]) + "*" + str(child)
            if i < len(self._children_list) - 1:
                text += ", "
        text += "]"
        return text


class OpTreeNodeSum(OpTreeNodeBase):
    """A OpTree node that sums over its children."""

    def __str__(self) -> str:
        """Returns a string representation of the node as a sum of its children."""
        text = "("
        for i, child in enumerate(self._children_list):
            if isinstance(child, QuantumCircuit):
                text += str(self._factor_list[i]) + "*" + "\n" + str(child) + "\n"
            else:
                text += str(self._factor_list[i]) + "*" + str(child)
            if i < len(self._factor_list) - 1:
                text += " + "
        text += ")"
        return text


class OpTreeLeafCircuit(OpTreeLeafBase):
    """A leaf of the OpTree that represents a circuit.

    Args:
        circuit (QuantumCircuit): The circuit that is represented by the leaf.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._hashvalue = hash_circuit(circuit)  # Hash tuple for fast comparison

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the circuit that is represented by the leaf."""
        return self._circuit

    @property
    def hashvalue(self) -> tuple:
        """Returns the hashvalue of the circuit."""
        return self._hashvalue

    def __str__(self) -> str:
        """Returns the string representation of the circuit."""
        return "\n" + str(self._circuit) + "\n"

    def __eq__(self, other) -> bool:
        """Function for comparing two OpTreeLeafCircuits."""
        if isinstance(other, OpTreeLeafCircuit):
            return self._hashvalue == other._hashvalue
        return False


class OperatorType:
    """Dummy class for representing operators."""

    pass


class OpTreeLeafOperator(OpTreeLeafBase):
    """A leaf of the OpTree that represents an operator.

    Args:
        operator (TODO): The operator that is represented by the leaf.
    """

    def __init__(self, operator: OperatorType) -> None:
        self._operator = operator

    def __str__(self) -> str:
        """Returns the string representation of the operator."""
        return str(self._operator)

    def __eq__(self, other) -> bool:
        """Function for comparing two OpTreeLeafOperators."""
        if isinstance(other, OpTreeLeafOperator):
            # TODO: check for same operators
            return False
        return False

class OpTreeLeafContainer(OpTreeLeafBase):
    """
    A container for arbitrary objects that can be used as leafs in the OpTree.
    """
    def __init__(self, item: Any) -> None:
        self.item = item

    def __str__(self) -> str:
        """Returns the string representation of the object."""
        return str(self.item)

    def __eq__(self, other) -> bool:
        """Function for comparing two OpTreeLeafContainers."""
        if isinstance(other, OpTreeLeafContainer):
            return self.item == other.item


class OpTree:
    """Dummy in case we want a full tree structure."""

    pass


def get_first_leaf(
    element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, OperatorType]
) -> QuantumCircuit:
    if isinstance(element, OpTreeNodeBase):
        return get_first_leaf(element.children[0])
    elif isinstance(element, OpTreeLeafCircuit):
        return element.circuit
    elif isinstance(element, OpTreeLeafOperator):
        return element.operator
    elif isinstance(element, OperatorType) or isinstance(element, QuantumCircuit):
        return element
    else:
        raise ValueError("Unknown element type: " + str(type(element)))


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
    if isinstance(element, OpTreeLeafCircuit):
        circuit = element.circuit
        type = "leaf"
    elif isinstance(element, QuantumCircuit):
        circuit = element
        type = "circuit"
    else:
        raise ValueError("element must be a CircuitTreeLeaf or a QuantumCircuit")

    iref_to_data_index = {id(inst.operation): idx for idx, inst in enumerate(circuit.data)}
    shift_sum = OpTreeNodeSum()
    # Loop through all parameter occurences in the circuit
    for param_reference in circuit._parameter_table[parameter]: # pylint: disable=protected-access

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
        if type == "leaf":
            shift_sum.append(OpTreeLeafCircuit(pshift_circ), shift_constant * fac)
            shift_sum.append(OpTreeLeafCircuit(mshift_circ), -shift_constant * fac)
        else:
            shift_sum.append(pshift_circ, shift_constant * fac)
            shift_sum.append(mshift_circ, -shift_constant * fac)

    return shift_sum


def circuit_derivative_inplace(
    tree_node: OpTreeNodeBase,
    parameter: ParameterExpression,
) -> None:
    """"
    Create the derivative of a OpTreeNode wrt a single parameter, modifies the tree inplace.

    Functions returns nothing, since the OpTree is modified inplace.

    Args:
        tree_node (OpTreeNodeBase): The OpTree Node to be differentiated.
        parameter (ParameterExpression): The parameter wrt which the circuit is differentiated.

    """
    if isinstance(tree_node, OpTreeNodeBase):
        for i,child in enumerate(tree_node.children):

            if isinstance(tree_node.factor[i], ParameterExpression):
                grad_fac = tree_node.factor[i].gradient(parameter)
            else:
                grad_fac = 0.0

            if isinstance(child, QuantumCircuit) or isinstance(
                child, OpTreeLeafCircuit
            ):
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
                    tree_node.children[i] = OpTreeNodeSum([child, grad], [grad_fac, tree_node.factor[i]])
                    tree_node.factor[i] = 1.0
            else:
                # if grad_fac is still a parameter
                tree_node.children[i] = OpTreeNodeSum([child, grad], [grad_fac, tree_node.factor[i]])
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
        for i,child in enumerate(element.children):
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


def simplify_copy(element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, OperatorType]) -> Union[OpTreeNodeBase,OpTreeLeafBase,QuantumCircuit,OperatorType]:
    """
    Function for simplifying an OpTree structure, the input is kept untouched.

    Merges double sums and identifies identical branches or leafs in sums.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, OperatorType]): The OpTree to be simplified.

    Returns:
        A simplified copy of the OpTree.
    """

    def combine_two_ops(op1, op2):
        """ Helper function for combining two operations into one.

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
            for i,child in enumerate(element.children):
                children_list.append(simplify_copy(child)) # Recursive call
                factor_list.append(element.factor[i])
                operation_list.append(element.operation[i])

            if isinstance(element, OpTreeNodeSum):
                new_element = OpTreeNodeSum(children_list, factor_list, operation_list)
            elif isinstance(element, OpTreeNodeList):
                new_element = OpTreeNodeList(children_list, factor_list, operation_list)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

            # Check for double sum if the element is a sum and one of the children is a sums
            if (isinstance(new_element, OpTreeNodeSum) and
                any([isinstance(child, OpTreeNodeSum) for child in new_element.children])
            ):
                # Merge the sum of a sum into the parent sum
                children_list = []
                factor_list = []
                operation_list = []
                for i,child in enumerate(new_element.children):
                    if isinstance(child, OpTreeNodeSum):
                        for j,childs_child in enumerate(child.children):
                            children_list.append(childs_child)
                            factor_list.append(new_element.factor[i] * child.factor[j])
                            operation_list.append(combine_two_ops(new_element.operation[i],child.operation[j])
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

                #TODO check for operator
                for i,child in enumerate(new_element.children):
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

def simplify_inplace(element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, OperatorType]) -> None:
    """
    Function for simplifying an OpTree structure, the input is modified inplace.

    Merges double sums and identifies identical branches or leafs in sums.
    The function returns nothing, since the OpTree is modified inplace.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, OperatorType]): The OpTree to be simplified.

    """
    raise NotImplementedError("Not implemented yet")


def _evaluate_index_tree(element: Union[OpTreeNodeBase, OpTreeLeafContainer], result_array) -> Union[np.ndarray,float]:
    """
    Function for evaluating an OpTree structure that has been indexed with a given result array.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafContainer]): The OpTree to be evaluated.
                                                              Has to be index first, such that the
                                                              leafs point to the
                                                              correct result array entries
                                                              and all factors have to be numeric.
        result_array (np.ndarray): The result array that contains the results to be placed in
                                   the leafs of the OpTree.

    Returns:
        The evaluated OpTree structure as a numpy array or a float.

    """
    if isinstance(element, OpTreeNodeBase):

        if any(not isinstance(fac, float) for fac in element.factor):
            raise ValueError("All factors must be numeric for evaluation")

        # Recursive construction of the data array
        temp = np.array(
            [
                element.factor[i] * _evaluate_index_tree(child, result_array)
                for i, child in enumerate(element.children)
            ]
        )
        if isinstance(element, OpTreeNodeSum):
            # OpTreeNodeSum -> sum over the array
            return np.sum(temp, axis=0)
        elif isinstance(element, OpTreeNodeList):
            # OpTreeNodeList -> return just the array
            return temp
        else:
            raise ValueError("element must be a OpTreeNodeSum or a OpTreeNodeList")
    elif isinstance(element, OpTreeLeafContainer):
        # Return value from the result array
        return result_array[element.item]
    else:
        raise ValueError("element must be a OpTreeNode or a OpTreeLeafContainer")


def evaluate(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    estimator,
    operator,
    dictionary,
    detect_circuit_duplicates: bool = False,
):
    """
    TODO: docstring
    """
    
    
    
    
    # TODO: dictionary might be slow!

    # create a list of circuit and a copy of the circuit tree with indices pointing to the circuit
    circuit_list = []
    if detect_circuit_duplicates:
        circuit_hash_list = []
    parameter_list = []

    circuit_counter = 0

    def build_lists_and_index_tree(element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit]):
        """
        Helper function for building the circuit list and the parameter list, and
        creates a indexed copy of the OpTree structure that references the circuits in the list.
        """

        # Global counter for indexing the circuits, circuit list and hash list, and parameter list
        nonlocal circuit_counter
        nonlocal circuit_list
        nonlocal circuit_hash_list
        nonlocal parameter_list
        if isinstance(element, OpTreeNodeBase):
            # Index circuits and bind parameters in the OpTreeNode structure
            child_list_indexed = [build_lists_and_index_tree(c) for c in element.children]
            factor_list_bound = []
            for fac in element.factor:
                if isinstance(fac, ParameterExpression):
                    factor_list_bound.append(
                        float(fac.bind(dictionary, allow_unknown_parameters=True))
                    )
                else:
                    factor_list_bound.append(fac)
            op = element.operation # TODO: check if this is correct

            # Recursive rebuild of the OpTree structure
            if isinstance(element, OpTreeNodeSum):
                return OpTreeNodeSum(child_list_indexed, factor_list_bound, op)
            elif isinstance(element, OpTreeNodeList):
                return OpTreeNodeList(child_list_indexed, factor_list_bound, op)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

        else:
            # Reached a CircuitTreeLeaf
            # Get the circuit, and if needed also the hash of the circuit
            if isinstance(element, QuantumCircuit):
                circuit = element
                if detect_circuit_duplicates:
                    circuit_hash = hash_circuit(circuit)
            elif isinstance(element, OpTreeLeafCircuit):
                circuit = element.circuit
                if detect_circuit_duplicates:
                    circuit_hash = element.hashvalue
            else:
                raise ValueError("element must be a CircuitTreeLeaf or a QuantumCircuit")

            # In case of duplicate detection, check if the circuit is already in the
            # circuit list and return the index of the circuit if it is already present
            if detect_circuit_duplicates:
                if circuit_hash in circuit_hash_list:
                    return OpTreeLeafContainer(circuit_list.index(circuit))
                circuit_hash_list.append(circuit_hash)

            # Otherwise append the circuit to the circuit list, copy the paramerters into vector form
            # and append them to the parameter list, increase the counter and return the index
            # in the OpTreeLeafContainer
            circuit_list.append(circuit)
            parameter_list.append(np.array([dictionary[p] for p in circuit.parameters]))
            circuit_counter += 1
            return OpTreeLeafContainer(circuit_counter - 1)

    # Build the circuit list and the parameter list, and create a indexed copy
    # of the OpTree structure
    start = time.time()
    index_tree = build_lists_and_index_tree(element)
    print("build_lists_and_index_tree", time.time() - start)

    # Build operator list
    operator_list = [operator] * len(circuit_list)

    # Evaluation via the estimator
    start = time.time()
    estimator_result = estimator.run(circuit_list, operator_list, parameter_list).result().values
    print("run time", time.time() - start)

    return _evaluate_index_tree(index_tree, estimator_result)