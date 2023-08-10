import numpy as np
import copy
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression
from typing import List, Union, Callable
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
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    parameter: ParameterExpression,
) -> OpTreeNodeBase:
    """"
    Create the derivative of a Tree (or circuit) wrt a single parameter, modifies the tree inplace.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): The circuit to be differentiated.
        parameter (ParameterExpression): The parameter wrt which the circuit is differentiated.

    Returns:
        The derivative of the circuit
    """
    if isinstance(element, OpTreeNodeBase):
        for i,child in enumerate(element.children):

            if isinstance(element.factor[i], ParameterExpression):
                grad_fac = element.factor[i].gradient(parameter)
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

            if isinstance(grad_fac, float):
                # if grad_fac is a numeric value
                if grad_fac == 0.0:
                    element.children[i] = grad
                else:
                    element.children[i] = OpTreeNodeSum([child, grad], [grad_fac, element.factor[i]])
                    element.factor[i] = 1.0
            else:
                # if grad_fac is still a parameter
                element.children[i] = OpTreeNodeSum([child, grad], [grad_fac, element.factor[i]])
                element.factor[i] = 1.0

    else:
        # reached a leaf -> replace with parameter shift derivative
        element = circuit_parameter_shift(element, parameter)


def circuit_derivative(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    parameters: Union[ParameterExpression, List[ParameterExpression], ParameterVector],
):
    # preprocessing

    is_list = True
    if isinstance(parameters, ParameterExpression):
        parameters = [parameters]
        is_list = False

    is_not_circuit = True
    if isinstance(element, QuantumCircuit) or isinstance(element, OpTreeLeafCircuit):
        is_not_circuit = False
        start = OpTreeNodeList([element], [1.0])
    else:
        start = element
    # start = element

    derivative_list = []
    fac_list = []
    for dp in parameters:
        res = copy.deepcopy(start)
        circuit_derivative_inplace(res, dp)
        if is_not_circuit:
            derivative_list.append(res)
        else:
            derivative_list.append(res.children[0])
        fac_list.append(1.0)

    if is_list:
        return OpTreeNodeList(derivative_list, fac_list)
    else:
        return derivative_list[0]


def circuit_derivative_copy(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    parameter: ParameterExpression,
):
    if isinstance(element, OpTreeNodeBase):
        children_list = []
        factor_list = []
        for i in range(len(element.children)):
            if isinstance(element.factor[i], ParameterExpression):
                # get derivative of factor
                df = element.factor[i].gradient(parameter)
                f = element.factor[i]
                l = element.children[i]
                grad = circuit_derivative_copy(element.children[i], parameter)

                if isinstance(df, float):
                    if df == 0.0:
                        children_list.append(grad)
                        factor_list.append(f)
                    else:
                        children_list.append(OpTreeNodeSum([l, grad], [df, f]))
                        factor_list.append(1.0)
                else:
                    children_list.append(OpTreeNodeSum([l, grad], [df, f]))
                    factor_list.append(1.0)

            else:
                children_list.append(circuit_derivative_copy(element.children[i], parameter))
                factor_list.append(element.factor[i])

        if isinstance(element, OpTreeNodeSum):
            return OpTreeNodeSum(children_list, factor_list)
        elif isinstance(element, OpTreeNodeList):
            return OpTreeNodeList(children_list, factor_list)
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

    else:
        return circuit_parameter_shift(element, parameter)


def circuit_derivative_v2(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    parameters: Union[ParameterExpression, List[ParameterExpression], ParameterVector],
):
    # preprocessing

    is_list = True
    if isinstance(parameters, ParameterExpression):
        parameters = [parameters]
        is_list = False

    start = element

    derivative_list = []
    fac_list = []
    for dp in parameters:
        derivative_list.append(circuit_derivative_copy(start, dp))
        fac_list.append(1.0)

    if is_list:
        return OpTreeNodeList(derivative_list, fac_list)
    else:
        return derivative_list[0]


def simplify_copy(element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]):
    def _combine_two_ops(op1, op2):
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
            l = []
            f = []
            op = []
            for i in range(len(element.children)):
                l.append(simplify_copy(element.children[i]))
                f.append(element.factor[i])
                op.append(element._operation_list[i])

            if isinstance(element, OpTreeNodeSum):
                new_element = OpTreeNodeSum(l, f, op)
            elif isinstance(element, OpTreeNodeList):
                new_element = OpTreeNodeList(l, f, op)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

            # Check for double sum
            if isinstance(new_element, OpTreeNodeSum) and isinstance(
                new_element.children[0], OpTreeNodeSum
            ):
                # detected double sum -> combine double some

                l = []
                f = []
                op = []
                for i in range(len(new_element.children)):
                    for j in range(len(new_element.children[i].children)):
                        l.append(new_element.children[i].children[j])
                        f.append(new_element.factor[i] * new_element.children[i].factor[j])
                        op.append(
                            _combine_two_ops(
                                new_element.operation[i],
                                new_element.children[i].operation[j],
                            )
                        )
                new_element = OpTreeNodeSum(l, f, op)

            # Check for double circuits in sum
            if isinstance(new_element, OpTreeNodeSum):
                dic = {}
                l = []
                f = []
                op = []

                # Better but slower
                for i in range(len(new_element.children)):
                    if new_element.children[i] in l:
                        index = l.index(new_element.children[i])
                        f[index] += new_element.factor[i]
                    else:
                        l.append(new_element.children[i])
                        f.append(new_element.factor[i])
                        op.append(new_element.operation[i])

                # for i in range(len(new_element.children)):
                #     element_str = str(new_element.children[i])
                #     if element_str in dic:
                #         f[dic[element_str]] += new_element.factor[i]
                #         print("double circuit in sum")
                #     else:
                #         dic[element_str] = len(f)
                #         l.append(new_element.children[i])
                #         f.append(new_element.factor[i])
                #         op.append(new_element._operation_list[i])
                #
                new_element = OpTreeNodeSum(l, f, op)

            return new_element

        else:
            return copy.deepcopy(element)
    else:
        return copy.deepcopy(element)


def evaluate_index_tree(element: Union[OpTreeNodeBase, OpTreeLeafCircuit], result_array):
    if isinstance(element, OpTreeNodeBase):
        temp = np.array(
            [
                element.factor[i] * evaluate_index_tree(element.children[i], result_array)
                for i in range(len(element.children))
            ]
        )
        if isinstance(element, OpTreeNodeSum):
            return np.sum(temp, axis=0)
        elif isinstance(element, OpTreeNodeList):
            return temp
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")
    elif isinstance(element, int):
        return result_array[element]
    else:
        raise ValueError("element must be a Tree element or a integer pointer")


def evaluate(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    estimator,
    operator,
    dictionary,
    detect_circuit_duplicates: bool = False,
):
    # dictionary might be slow!

    # create a list of circuit and a copy of the circuit tree with indices pointing to the circuit
    circuit_list = []
    if detect_circuit_duplicates:
        circuit_hash_list = []
    parameter_list = []
    global circuit_counter
    circuit_counter = 0

    def build_lists_and_index_tree(element):
        global circuit_counter
        if isinstance(element, OpTreeNodeBase):
            l = [build_lists_and_index_tree(c) for c in element.children]
            f = []
            for i in range(len(element.factor)):
                if isinstance(element.factor[i], ParameterExpression):
                    f.append(
                        float(element.factor[i].bind(dictionary, allow_unknown_parameters=True))
                    )
                else:
                    f.append(element.factor[i])
            op = element.operation
            if isinstance(element, OpTreeNodeSum):
                return OpTreeNodeSum(l, f, op)
            elif isinstance(element, OpTreeNodeList):
                return OpTreeNodeList(l, f, op)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

        else:
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

            if detect_circuit_duplicates:
                if circuit_hash in circuit_hash_list:
                    return circuit_list.index(circuit)
                circuit_hash_list.append(circuit_hash)

            circuit_list.append(circuit)

            parameter_list.append(np.array([dictionary[p] for p in circuit.parameters]))
            circuit_counter += 1
            return circuit_counter - 1

    start = time.time()
    index_tree = build_lists_and_index_tree(element)
    print("build_lists_and_index_tree", time.time() - start)

    print("len(circuit_list)", len(circuit_list))

    # print(circuit_tree_index)

    parameter_list = [parameter_list[0]] * len(circuit_list)

    op_list = [operator] * len(circuit_list)

    # print("circuit_list",circuit_list)
    # print("op_list",op_list)
    # print("parameter_list",parameter_list)

    # print("inital_circuit",circuit_list[0])

    start = time.time()
    res1 = Estimator().run(circuit_list, op_list, parameter_list)
    print("run", time.time() - start)
    start = time.time()
    res2 = res1.result()
    print("res2", time.time() - start)
    start = time.time()
    result = res2.values
    print("result time", time.time() - start)

    print("result", result)

    return evaluate_index_tree(index_tree, result)
