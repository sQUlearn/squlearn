from typing import List, Union, Callable, Any
import copy

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterExpression


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
        """Function for copying a OpTreeNodeBase object."""
        return type(self)(
            copy.deepcopy(self._children_list),
            copy.deepcopy(self._factor_list),
            copy.deepcopy(self._operation_list),
        )


class OpTreeList(OpTreeNodeBase):
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


class OpTreeSum(OpTreeNodeBase):
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


class OpTreeCircuit(OpTreeLeafBase):
    """A leaf of the OpTree that represents a circuit.

    Args:
        circuit (QuantumCircuit): The circuit that is represented by the leaf.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._hashvalue = OpTree.hash_circuit(circuit)  # Hash tuple for fast comparison

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
        if isinstance(other, OpTreeCircuit):
            return self._hashvalue == other._hashvalue
        return False

    def copy(self):
        """Function for copying a OpTreeLeafCircuit object."""
        return OpTreeCircuit(self._circuit.copy())


class OpTreeOperator(OpTreeLeafBase):
    """A leaf of the OpTree that represents an operator.

    Args:
        operator (SparsePauliOp): The operator that is represented by the leaf.
    """

    def __init__(self, operator: SparsePauliOp) -> None:
        self._operator = operator
        self._hashvalue = OpTree.hash_operator(operator)  # Hash tuple for fast comparison

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
        if isinstance(other, OpTreeOperator):
            return self._hashvalue == other._hashvalue
        return False

    def copy(self):
        """Function for copying a OpTreeLeafOperator object."""
        return OpTreeOperator(self._operator.copy())


class OpTreeExpectationValue(OpTreeLeafBase):
    """
    Leaf of the OpTree that represents an expectation value of a circuit and an operator.

    Args:
        circuit (Union[OpTreeLeafCircuit, QuantumCircuit]): The circuit in the expectation value.
        operator (Union[OpTreeLeafOperator, SparsePauliOp]): The operator in the expectation value.
    """

    def __init__(
        self,
        circuit: Union[OpTreeCircuit, QuantumCircuit],
        operator: Union[OpTreeOperator, SparsePauliOp],
    ) -> None:
        if isinstance(circuit, QuantumCircuit):
            circuit = OpTreeCircuit(circuit)
        if not isinstance(circuit, OpTreeCircuit):
            raise ValueError("Wrong format of the given circuit!")
        self._circuit = circuit

        if isinstance(operator, SparsePauliOp):
            operator = OpTreeOperator(operator)
        if not isinstance(operator, OpTreeOperator):
            raise ValueError("Wrong format of the given operator!")
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
        if isinstance(other, OpTreeExpectationValue):
            return self._circuit == other._circuit and self._operator == other._operator
        return False

    def copy(self):
        """Function for copying a OpTreeLeafExpectationValue object."""
        return OpTreeExpectationValue(self._circuit.copy(), self._operator.copy())


class OpTreeMeasuredOperator(OpTreeExpectationValue):
    """
    Leaf of the OpTree that represents an measurement.

    The circuit in the class represents the circuit that is measured for the given operator.
    """

    def measure_circuit(
        self, circuit: Union[QuantumCircuit, OpTreeCircuit]
    ) -> OpTreeExpectationValue:
        """
        Applies the measurement of the leaf to the circuit and returns an expectation value.

        Args:
            circuit (Union[QuantumCircuit, OpTreeLeafCircuit]): The circuit that is measured.

        Returns:
            OpTreeLeafExpectationValue: The expectation value leaf with the measured circuit.
        """
        circuit_ = circuit
        if isinstance(circuit, OpTreeCircuit):
            circuit_ = circuit
        return OpTreeExpectationValue(circuit_.compose(self.circuit), self.operator)

    def copy(self):
        """Function for copying a OpTreeLeafMeasuredOperator object."""
        return OpTreeMeasuredOperator(self._circuit.copy(), self._operator.copy())


class OpTreeContainer(OpTreeLeafBase):
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
        if isinstance(other, OpTreeContainer):
            return self.item == other.item

    def copy(self):
        """Function for copying a OpTreeLeafContainer object."""
        return OpTreeContainer(copy.deepcopy(self.item))


class OpTreeValue(OpTreeLeafBase):
    """
    A leaf that contains an evaluated value.

    Args:
        value (float): A float value that is represented by the leaf.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def __str__(self) -> str:
        """Returns the string representation of the value."""
        return str(self.value)

    def __eq__(self, other) -> bool:
        """Function for comparing two OpTreeLeafValues."""
        if isinstance(other, OpTreeValue):
            return self.value == other.value

    def copy(self):
        """Function for copying a OpTreeLeafValue object."""
        return OpTreeValue(self.value)


def _simplify_operator(
    element: Union[SparsePauliOp, OpTreeOperator],
) -> Union[SparsePauliOp, OpTreeOperator]:
    if isinstance(element, OpTreeOperator):
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
            return OpTreeOperator(operator_simp)
        return operator_simp
    else:
        return None


class OpTree:
    """Static class containing functions for working with OpTrees objects."""

    from .optree_derivative import OpTreeDerivative

    derivative = OpTreeDerivative

    from .optree_evaluate import OpTreeEvaluate

    evaluate = OpTreeEvaluate

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
                num += OpTree.get_number_of_leafs(child)
            return num

    @staticmethod
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
                depth = max(depth, OpTree.get_tree_depth(child))
            return depth + 1

    @staticmethod
    def get_num_nested_lists(tree: OpTreeElementBase) -> int:
        """Returns the number of nested lists in the OpTree.

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
                depth = max(depth, OpTree.get_num_nested_lists(child))
            if isinstance(tree, OpTreeList):
                return depth + 1
            return depth

    @staticmethod
    def get_first_leaf(
        element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp],
    ) -> Union[OpTreeLeafBase, QuantumCircuit, SparsePauliOp]:
        """Returns the first leaf of the supplied OpTree.

        Args:
            element (Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp]): The OpTree.

        Returns:
            The first found leaf of the OpTree.
        """
        if isinstance(element, OpTreeNodeBase):
            return OpTree.get_first_leaf(element.children[0])
        else:
            return element

    @staticmethod
    def gen_expectation_tree(
        circuit_tree: Union[OpTreeNodeBase, OpTreeCircuit, QuantumCircuit],
        operator_tree: Union[
            OpTreeNodeBase, OpTreeMeasuredOperator, OpTreeOperator, SparsePauliOp
        ],
    ):
        """
        Function that generates an expectation tree from a circuit tree and an operator tree.

        .. currentmodule:: squlearn.util.optree

        The operator tree is applied to each leaf of the circuit tree and the
        resulting expectation values are returned as :class:`OpTreeExpectationValue`.

        Args:
            circuit_tree (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): The circuit tree.
            operator_tree (Union[OpTreeNodeBase, OpTreeLeafMeasuredOperator, OpTreeLeafOperator, SparsePauliOp]): The operator tree.

        Returns:
            The combined tree with :class:`OpTreeExpectationValue` at the leafs.
        """
        if isinstance(circuit_tree, OpTreeNodeBase):
            children_list = [
                OpTree.gen_expectation_tree(child, operator_tree)
                for child in circuit_tree.children
            ]
            factor_list = circuit_tree.factor
            operation_list = circuit_tree.operation

            if isinstance(circuit_tree, OpTreeSum):
                return OpTreeSum(children_list, factor_list, operation_list)
            elif isinstance(circuit_tree, OpTreeList):
                return OpTreeList(children_list, factor_list, operation_list)
            else:
                raise ValueError("wrong type of circuit_tree")

        elif isinstance(circuit_tree, (OpTreeCircuit, QuantumCircuit)):
            # Reached a circuit node -> append operation tree

            if isinstance(operator_tree, OpTreeNodeBase):
                children_list = [
                    OpTree.gen_expectation_tree(circuit_tree, child)
                    for child in operator_tree.children
                ]
                factor_list = operator_tree.factor
                operation_list = operator_tree.operation

                if isinstance(operator_tree, OpTreeSum):
                    return OpTreeSum(children_list, factor_list, operation_list)
                elif isinstance(operator_tree, OpTreeList):
                    return OpTreeList(children_list, factor_list, operation_list)
                else:
                    raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")
            elif isinstance(operator_tree, (OpTreeOperator, SparsePauliOp)):
                return OpTreeExpectationValue(circuit_tree, operator_tree)
            elif isinstance(operator_tree, OpTreeMeasuredOperator):
                return operator_tree.measure_circuit(circuit_tree)
            else:
                raise ValueError("wrong type of operator_tree")
        else:
            raise ValueError(
                "circuit_tree must be a CircuitTreeSum or a CircuitTreeList", type(circuit_tree)
            )

    @staticmethod
    def simplify(
        element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, SparsePauliOp],
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
                children_list = [OpTree.simplify(child) for child in element.children]
                factor_list = element.factor
                operation_list = element.operation

                if isinstance(element, OpTreeSum):
                    new_element = OpTreeSum(children_list, factor_list, operation_list)
                elif isinstance(element, OpTreeList):
                    new_element = OpTreeList(children_list, factor_list, operation_list)
                else:
                    raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

                # Check for double sum if the element is a sum and one of the children is a sums
                if isinstance(new_element, OpTreeSum) and any(
                    [isinstance(child, OpTreeSum) for child in new_element.children]
                ):
                    # Merge the sum of a sum into the parent sum
                    children_list = []
                    factor_list = []
                    operation_list = []
                    for i, child in enumerate(new_element.children):
                        if isinstance(child, OpTreeSum):
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
                    new_element = OpTreeSum(children_list, factor_list, operation_list)

                # Check for similar branches in the Sum and merge them into a single branch
                if isinstance(new_element, OpTreeSum):
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
                    new_element = OpTreeSum(children_list, factor_list, operation_list)

                return new_element

            else:
                # Reached an empty Node -> cancel the recursion
                return copy.deepcopy(element)
        elif isinstance(element, (SparsePauliOp, OpTreeOperator)):
            return _simplify_operator(element)
        else:
            # Reached a leaf -> cancel the recursion
            return copy.deepcopy(element)

    @staticmethod
    def assign_parameters(
        element: Union[OpTreeNodeBase, OpTreeCircuit, QuantumCircuit],
        dictionary,
        inplace: bool = False,
    ):
        """
        Assigns the parameters of the OpTree structure to the values in the dictionary.

        Args:
            element (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): The OpTree for which
                                                                                all parameters are
                                                                                assigned.
            dictionary (dict): The dictionary that contains the parameter names as keys
                            and the parameter values as values.

        Returns:
            The OpTree structure with all parameters assigned, (copied if inplace=False)
        """

        if isinstance(element, OpTreeNodeBase):
            if inplace:
                for c in element.children:
                    OpTree.assign_parameters(c, dictionary, inplace=True)
                for i, fac in enumerate(element.factor):
                    if isinstance(fac, ParameterExpression):
                        element.factor[i] = float(
                            fac.bind(dictionary, allow_unknown_parameters=True)
                        )

            else:
                # Index circuits and bind parameters in the OpTreeNode structure
                child_list_assigned = [
                    OpTree.assign_parameters(c, dictionary) for c in element.children
                ]
                factor_list_bound = []
                for fac in element.factor:
                    if isinstance(fac, ParameterExpression):
                        factor_list_bound.append(
                            float(fac.bind(dictionary, allow_unknown_parameters=True))
                        )
                    else:
                        factor_list_bound.append(fac)

                # Recursive rebuild of the OpTree structure
                if isinstance(element, OpTreeSum):
                    return OpTreeSum(child_list_assigned, factor_list_bound, element.operation)
                elif isinstance(element, OpTreeList):
                    return OpTreeList(child_list_assigned, factor_list_bound, element.operation)
                else:
                    raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")
        elif isinstance(element, OpTreeCircuit):
            # Assign the parameters to the circuit
            if inplace:
                element.circuit.assign_parameters(
                    [dictionary[p] for p in element.circuit.parameters], inplace=True
                )
            else:
                return OpTreeCircuit(
                    element.circuit.assign_parameters(
                        [dictionary[p] for p in element.circuit.parameters], inplace=False
                    )
                )
        elif isinstance(element, QuantumCircuit):
            # Assign the parameters to the circuit
            if inplace:
                element.assign_parameters(
                    [dictionary[p] for p in element.parameters], inplace=True
                )
            else:
                return element.assign_parameters(
                    [dictionary[p] for p in element.parameters], inplace=False
                )
        elif isinstance(element, (OpTreeExpectationValue, OpTreeMeasuredOperator)):
            # Assign the parameters to the circuit and operator
            if inplace:
                element.circuit.assign_parameters(
                    [dictionary[p] for p in element.circuit.parameters], inplace=True
                )
                element.operator.assign_parameters(
                    [dictionary[p] for p in element.operator.parameters], inplace=True
                )
            else:
                return OpTreeExpectationValue(
                    element.circuit.assign_parameters(
                        [dictionary[p] for p in element.circuit.parameters], inplace=False
                    ),
                    element.operator.assign_parameters(
                        [dictionary[p] for p in element.operator.parameters], inplace=False
                    ),
                )
        elif isinstance(element, OpTreeOperator):
            # Assign the parameters to the operator
            if inplace:
                element.operator.assign_parameters(
                    [dictionary[p] for p in element.operator.parameters], inplace=True
                )
            else:
                return OpTreeOperator(
                    element.operator.assign_parameters(
                        [dictionary[p] for p in element.operator.parameters], inplace=False
                    )
                )
        elif isinstance(element, SparsePauliOp):
            # Assign the parameters to the operator
            if inplace:
                element.assign_parameters(
                    [dictionary[p] for p in element.parameters], inplace=True
                )
            else:
                return element.assign_parameters(
                    [dictionary[p] for p in element.parameters], inplace=False
                )
        else:
            raise ValueError(
                "element must be a OpTreeNodeBase, OpTreeLeafCircuit or a QuantumCircuit"
            )
