from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import List, Union, Callable, Any


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


def hash_operator(operator: SparsePauliOp) -> tuple:
    """Hashes an operator using the qiskit _observable_key function.

    Args:
        operator (SparsePauliOp): The operator to be hashed.

    Returns:
        A tuple containing the operator information that can be used for comparison.
    """
    # TODO: can be replaced by whatever hash function is used in qiskit in the future.
    from qiskit.primitives.utils import _observable_key

    return _observable_key(operator)


class OpTreeElementBase:
    """Base class for elements of the OpTree."""

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


class OpTreeLeafBase(OpTreeElementBase):
    """Base class for Leafs of the OpTree."""

    pass


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


class OpTreeLeafOperator(OpTreeLeafBase):
    """A leaf of the OpTree that represents an operator.

    Args:
        operator (TODO): The operator that is represented by the leaf.
    """

    def __init__(self, operator: SparsePauliOp) -> None:
        self._operator = operator
        self._hashvalue = hash_operator(operator)  # Hash tuple for fast comparison

    @property
    def operator(self) -> SparsePauliOp:
        """Returns the operator that is represented by the leaf."""
        return self._operator

    def hashvalue(self) -> tuple:
        """Returns the hashvalue of the circuit."""
        return self._hashvalue

    def __str__(self) -> str:
        """Returns the string representation of the operator."""
        return str(self._operator)

    def __eq__(self, other) -> bool:
        """Function for comparing two OpTreeLeafOperators."""
        if isinstance(other, OpTreeLeafOperator):
            return self._hashvalue == other._hashvalue
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


def get_number_of_leafs(tree: OpTreeElementBase) -> int:
    """Returns the number of leafs of the OpTree.

    Args:
        tree (OpTreeElementBase): The OpTree.

    Returns:
        int: The number of leafs of the OpTree.
    """
    if isinstance(tree, OpTreeLeafBase):
        return 1
    else:
        num = 0
        for child in tree.children:
            num += get_number_of_leafs(child)
        return num


def get_tree_depth(tree: OpTreeElementBase) -> int:
    """Returns the depth of the OpTree.

    Args:
        tree (OpTreeElementBase): The OpTree.

    Returns:
        int: The depth of the OpTree.
    """
    if isinstance(tree, OpTreeLeafBase):
        return 0
    else:
        depth = 0
        for child in tree.children:
            depth = max(depth, get_tree_depth(child))
        return depth + 1


def get_first_leaf(
    element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp]
) -> QuantumCircuit:
    if isinstance(element, OpTreeNodeBase):
        return get_first_leaf(element.children[0])
    elif isinstance(element, OpTreeLeafCircuit):
        return element.circuit
    elif isinstance(element, OpTreeLeafOperator):
        return element.operator
    elif isinstance(element, SparsePauliOp) or isinstance(element, QuantumCircuit):
        return element
    else:
        raise ValueError("Unknown element type: " + str(type(element)))
