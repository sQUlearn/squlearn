from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import List, Union, Callable, Any
import copy


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

    def remove(self, index: Union[List[int], int]):
        """Removes children from the node.

        Args:
            index (int): The list of indices of the children to be removed.
                         Can also be a single index.
        """

        if isinstance(index, int):
            index = [index]

        if len(index) > len(self._children_list):
            raise ValueError("index must not be larger than the number of children")

        if len(index) == 0:
            return None

        self._children_list = [
            child for i, child in enumerate(self._children_list) if i not in index
        ]
        self._factor_list = [
            factor for i, factor in enumerate(self._factor_list) if i not in index
        ]
        self._operation_list = [
            operation for i, operation in enumerate(self._operation_list) if i not in index
        ]

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

    def copy(self):
        """ Function for copying a OpTreeNodeBase object. """
        return type(self)(copy.deepcopy(self._children_list), copy.deepcopy(self._factor_list), copy.deepcopy(self._operation_list))

class OpTreeNodeList(OpTreeNodeBase):
    """A OpTree node that represents its children as a list/array/vector.

    Args:
        children_list (list): A list of children of the list.
        factor_list (list): A list of factors for each child.
        operation_list (list): A list of operations that are applied to each child.
    """

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
    """A OpTree node that sums over its children.

    Args:
        children_list (list): A list of children of the summation.
        factor_list (list): A list of factors for each child.
        operation_list (list): A list of operations that are applied to each child.
    """

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

    def copy(self):
        """ Function for copying a OpTreeLeafCircuit object. """
        return OpTreeLeafCircuit(self._circuit.copy())


class OpTreeLeafOperator(OpTreeLeafBase):
    """A leaf of the OpTree that represents an operator.

    Args:
        operator (SparsePauliOp): The operator that is represented by the leaf.
    """

    def __init__(self, operator: SparsePauliOp) -> None:
        self._operator = operator
        self._hashvalue = hash_operator(operator)  # Hash tuple for fast comparison

    @property
    def operator(self) -> SparsePauliOp:
        """Returns the operator that is represented by the leaf."""
        return self._operator

    @property
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

    def copy(self):
        """ Function for copying a OpTreeLeafOperator object. """
        return OpTreeLeafOperator(self._operator.copy())


class OpTreeLeafExpectationValue(OpTreeLeafBase):
    """
    Leaf of the OpTree that represents an expectation value of a circuit and an operator.

    Args:
        circuit (Union[OpTreeLeafCircuit, QuantumCircuit]): The circuit in the expectation value.
        operator (Union[OpTreeLeafOperator, SparsePauliOp]): The operator in the expectation value.
    """

    def __init__(
        self,
        circuit: Union[OpTreeLeafCircuit, QuantumCircuit],
        operator: Union[OpTreeLeafOperator, SparsePauliOp],
    ) -> None:
        if isinstance(circuit, QuantumCircuit):
            circuit = OpTreeLeafCircuit(circuit)
        self._circuit = circuit

        if isinstance(operator, SparsePauliOp):
            operator = OpTreeLeafOperator(operator)
        self._operator = operator

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the circuit that is represented by the leaf."""
        return self._circuit.circuit

    @property
    def operator(self) -> SparsePauliOp:
        """Returns the operator that is represented by the leaf."""
        return self._operator.operator

    @property
    def hashvalue(self) -> tuple:
        """Returns the hashvalue of the circuit."""
        return self._circuit.hashvalue + self._operator.hashvalue

    def __str__(self) -> str:
        """Returns the string representation of the expectation value."""
        return str(self._circuit) + "\n with observable \n" + str(self._operator) + "\n"

    def __eq__(self, other) -> bool:
        """Function for comparing two OpTreeLeafOperators."""
        if isinstance(other, OpTreeLeafExpectationValue):
            return self._circuit == other._circuit and self._operator == other._operator
        return False

    def copy(self):
        """ Function for copying a OpTreeLeafExpectationValue object. """
        return OpTreeLeafExpectationValue(self._circuit.copy(), self._operator.copy())

class OpTreeLeafMeasuredOperator(OpTreeLeafExpectationValue):
    """
    Leaf of the OpTree that represents an measurement.

    The circuit in the class represents the circuit that is measured for the given operator.
    """

    def measure_circuit(
        self, circuit: Union[QuantumCircuit, OpTreeLeafCircuit]
    ) -> OpTreeLeafExpectationValue:
        """
        Applies the measurement of the leaf to the circuit and returns an expectation value.

        Args:
            circuit (Union[QuantumCircuit, OpTreeLeafCircuit]): The circuit that is measured.

        Returns:
            OpTreeLeafExpectationValue: The expectation value leaf with the measured circuit.
        """
        circuit_ = circuit
        if isinstance(circuit, OpTreeLeafCircuit):
            circuit_ = circuit
        return OpTreeLeafExpectationValue(circuit_.compose(self.circuit), self.operator)

    def copy(self):
        """ Function for copying a OpTreeLeafMeasuredOperator object."""
        return OpTreeLeafMeasuredOperator(self._circuit.copy(), self._operator.copy())

class OpTreeLeafContainer(OpTreeLeafBase):
    """
    A container for arbitrary objects that can be used as leafs in the OpTree.

    Args:
        item (Any): Any kind of item that is represented by the leaf.
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

    def copy(self):
        """ Function for copying a OpTreeLeafContainer object. """
        return OpTreeLeafContainer(copy.deepcopy(self.item))

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
) -> Union[OpTreeLeafBase, QuantumCircuit, SparsePauliOp]:
    """Returns the first leaf of the supplied OpTree.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp]): The OpTree.

    Returns:
        The first found leaf of the OpTree.
    """
    if isinstance(element, OpTreeNodeBase):
        return get_first_leaf(element.children[0])
    else:
        return element


def gen_expectation_tree(
    circuit_tree: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    operator_tree: Union[
        OpTreeNodeBase, OpTreeLeafMeasuredOperator, OpTreeLeafOperator, SparsePauliOp
    ],
):
    """
    Function that generates an expectation tree from a circuit tree and an operator tree.

    The operator tree is applied to each leaf of the circuit tree and the
    resulting expectation values are returned as ``OpTreeExpectationValueLeafs``.

    Args:
        circuit_tree (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): The circuit tree.
        operator_tree (Union[OpTreeNodeBase, OpTreeLeafMeasuredOperator, OpTreeLeafOperator, SparsePauliOp]): The operator tree.

    Returns:
        The combined tree with ``OpTreeExpectationValueLeafs`` at the leafs.
    """
    if isinstance(circuit_tree, OpTreeNodeBase):
        children_list = [
            gen_expectation_tree(child, operator_tree) for child in circuit_tree.children
        ]
        factor_list = circuit_tree.factor
        operation_list = circuit_tree.operation

        if isinstance(circuit_tree, OpTreeNodeSum):
            return OpTreeNodeSum(children_list, factor_list, operation_list)
        elif isinstance(circuit_tree, OpTreeNodeList):
            return OpTreeNodeList(children_list, factor_list, operation_list)
        else:
            raise ValueError("wrong type of circuit_tree")

    elif isinstance(circuit_tree, (OpTreeLeafCircuit, QuantumCircuit)):
        # Reached a circuit node -> append operation tree

        if isinstance(operator_tree, OpTreeNodeBase):
            children_list = [
                gen_expectation_tree(circuit_tree, child) for child in operator_tree.children
            ]
            factor_list = operator_tree.factor
            operation_list = operator_tree.operation

            if isinstance(operator_tree, OpTreeNodeSum):
                return OpTreeNodeSum(children_list, factor_list, operation_list)
            elif isinstance(operator_tree, OpTreeNodeList):
                return OpTreeNodeList(children_list, factor_list, operation_list)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")
        elif isinstance(operator_tree, (OpTreeLeafOperator, SparsePauliOp)):
            return OpTreeLeafExpectationValue(circuit_tree, operator_tree)
        elif isinstance(operator_tree, OpTreeLeafMeasuredOperator):
            return operator_tree.measure_circuit(circuit_tree)
        else:
            raise ValueError("wrong type of operator_tree")
    else:
        raise ValueError(
            "circuit_tree must be a CircuitTreeSum or a CircuitTreeList", type(circuit_tree)
        )
