import numpy as np
from typing import Union, List
import time

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.primitives import BaseEstimator,BaseSampler
from qiskit.quantum_info import SparsePauliOp

from .optree import (
    hash_circuit,
    OpTreeNodeBase,
    OpTreeLeafBase,
    OpTreeNodeList,
    OpTreeNodeSum,
    OpTreeLeafCircuit,
    OpTreeLeafOperator,
    OpTreeLeafContainer,
)


def _evaluate_index_tree(
    element: Union[OpTreeNodeBase, OpTreeLeafContainer], result_array
) -> Union[np.ndarray, float]:
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
        print(type(element))
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
    start = time.time()
    circuit_list, parameter_list, index_tree = build_circuit_list(
        element, dictionary, detect_circuit_duplicates
    )

    print("build_lists_and_index_tree", time.time() - start)

    # Build operator list
    operator_list = [operator] * len(circuit_list)

    # Evaluation via the estimator
    start = time.time()
    estimator_result = estimator.run(circuit_list, operator_list, parameter_list).result().values
    print("run time", time.time() - start)

    return _evaluate_index_tree(index_tree, estimator_result)


def build_circuit_list(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    dictionary: dict,
    detect_circuit_duplicates: bool = False,
):
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
            op = element.operation  # TODO: check if this is correct

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

    index_tree = build_lists_and_index_tree(element)

    return circuit_list, parameter_list, index_tree


def build_operator_list(
    element: Union[OpTreeNodeBase, OpTreeLeafOperator, SparsePauliOp],
    dictionary: dict,
    detect_operator_duplicates: bool = False,
):
    # create a list of circuit and a copy of the circuit tree with indices pointing to the circuit
    operator_list = []
    if detect_operator_duplicates:
        operator_hash_list = []  # TODO

    operator_counter = 0

    def build_lists_and_index_tree(
        element: Union[OpTreeNodeBase, OpTreeLeafOperator, SparsePauliOp]
    ):
        """
        Helper function for building the circuit list and the parameter list, and
        creates a indexed copy of the OpTree structure that references the circuits in the list.
        """

        # Global counter for indexing the circuits, circuit list and hash list, and parameter list
        nonlocal operator_counter
        nonlocal operator_list
        nonlocal operator_hash_list

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
            op = element.operation  # TODO: check if this is correct

            # Recursive rebuild of the OpTree structure
            if isinstance(element, OpTreeNodeSum):
                return OpTreeNodeSum(child_list_indexed, factor_list_bound, op)
            elif isinstance(element, OpTreeNodeList):
                return OpTreeNodeList(child_list_indexed, factor_list_bound, op)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

        else:
            # Reached a Operator

            if isinstance(element, SparsePauliOp):
                operator = element
            elif isinstance(element, OpTreeLeafOperator):
                operator = element.operator
            else:
                raise ValueError("element must be a OpTreeLeafOperator or a SparsePauliOp")

            # Assign parameters
            operator = operator.assign_parameters(
                [dictionary[p] for p in operator.parameters], inplace=False
            )

            # Todo check if it makes a difference in speed if not complex numbers are used

            if len(operator.parameters) != 0:
                raise ValueError("Not all parameters are assigned in the operator!")

            if detect_operator_duplicates:
                if operator in operator_list:
                    return OpTreeLeafContainer(operator_list.index(operator))

            operator_list.append(operator)
            operator_counter += 1
            return OpTreeLeafContainer(operator_counter - 1)

    index_tree = build_lists_and_index_tree(element)

    return operator_list, index_tree


def _add_offset_to_tree(element: Union[OpTreeNodeBase, OpTreeLeafContainer], offset: int):
    """
    Adds a constant offset to all leafs of the OpTree structure.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafContainer]): The OpTree to be adjusted.
        offset (int): The offset to be added to all leafs.

    Returns:
        Returns a copy with the offset added to all leafs.

    """
    if offset == 0:
        return element

    if isinstance(element, OpTreeNodeBase):
        # Recursive call and reconstruction of the data array
        children_list = [_add_offset_to_tree(child, offset) for child in element.children]

        # Rebuild the tree with the new children and factors (copy part)
        if isinstance(element, OpTreeNodeSum):
            return OpTreeNodeSum(children_list, element.factor)  # TODO: operation
        elif isinstance(element, OpTreeNodeList):
            return OpTreeNodeList(children_list, element.factor)  # TODO: operation
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

    elif isinstance(element, OpTreeLeafContainer):
        # Return value from the result array
        return OpTreeLeafContainer(element.item + offset)
    else:
        raise ValueError("element must be a OpTreeNode or a OpTreeLeafContainer")


def evaluate_sampler(
    circuit: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    operator: Union[OpTreeNodeBase, OpTreeLeafOperator, SparsePauliOp],
    dictionary_circuit: Union[dict, List[dict]],
    dictionary_operator: Union[dict, List[dict]],
    sampler: BaseSampler,
    detect_circuit_duplicates: bool = False,
    detect_operator_duplicates: bool = False,
    dictionaries_combined: bool = False,
):

    multiple_circuit_dict = True
    if not isinstance(dictionary_circuit, list):
        dictionary_circuit = [dictionary_circuit]
        multiple_circuit_dict = False

    multiple_operator_dict = True
    if not isinstance(dictionary_operator, list):
        dictionary_operator = [dictionary_operator]
        multiple_operator_dict = False

    if dictionaries_combined:
        if len(dictionary_circuit) != len(dictionary_operator):
            raise ValueError("The length of the circuit and operator dictionary must be the same")

    total_circle_list = []
    total_parameter_list = []

    tree_circuit=[]
    for i,dictionary_circuit__ in enumerate(dictionary_circuit):

        circuit_list, parameter_list, circuit_tree = build_circuit_list(
            circuit, dictionary_circuit__, detect_circuit_duplicates
        )

        circuit_list = [circuit.measure_all(inplace=False) for circuit in circuit_list]

        tree_circuit.append(_add_offset_to_tree(circuit_tree,len(total_circle_list)))

        total_circle_list += circuit_list
        total_parameter_list += parameter_list

    if multiple_circuit_dict:
        evaluation_tree = OpTreeNodeList(tree_circuit)
    else:
        evaluation_tree = tree_circuit[0]

    sampler_result = (
        sampler.run(total_circle_list, total_parameter_list).result().quasi_dists
    )

    print("sampler_result", sampler_result)
    print("len(ampler_result)", len(sampler_result))

    # TODO: find out how expectation values are calculated from the distribution

    # if dictionaries_combined:
    #     dictionary_operator_ = [dictionary_operator[i]]
    # else:
    #     dictionary_operator_ = dictionary_operator

    # tree_operator=[]
    # for dictionary_operator__ in dictionary_operator_:

    #     # Build operator list and circuit list from the corresponding Optrees
    #     operator_list, operator_tree = build_operator_list(
    #         operator, dictionary_operator__, detect_operator_duplicates
    #     )



def evaluate_estimator(
    circuit: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
    operator: Union[OpTreeNodeBase, OpTreeLeafOperator, SparsePauliOp],
    dictionary_circuit: Union[dict, List[dict]],
    dictionary_operator: Union[dict, List[dict]],
    estimator: BaseEstimator,
    detect_circuit_duplicates: bool = False,
    detect_operator_duplicates: bool = False,
    dictionaries_combined: bool = False,
):
    """ """

    # Combine the operator tree and the circuit tree into a single tree
    def adjust_tree_operators(element, operator_tree, operator_list_length):
        if isinstance(element, OpTreeNodeBase):
            children_list = [
                adjust_tree_operators(child, operator_tree, operator_list_length)
                for child in element.children
            ]

            # Rebuild the tree with the new children and factors (copy part)
            if isinstance(element, OpTreeNodeSum):
                return OpTreeNodeSum(children_list, element.factor)  # TODO: operation
            elif isinstance(element, OpTreeNodeList):
                return OpTreeNodeList(children_list, element.factor)  # TODO: operation
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

        elif isinstance(element, OpTreeLeafContainer):
            k = element.item
            return _add_offset_to_tree(operator_tree, k * operator_list_length)

    total_circle_list = []
    total_operator_list = []
    total_parameter_list = []


    multiple_circuit_dict = True
    if not isinstance(dictionary_circuit, list):
        dictionary_circuit = [dictionary_circuit]
        multiple_circuit_dict = False

    multiple_operator_dict = True
    if not isinstance(dictionary_operator, list):
        dictionary_operator = [dictionary_operator]
        multiple_operator_dict = False

    if dictionaries_combined:
        if len(dictionary_circuit) != len(dictionary_operator):
            raise ValueError("The length of the circuit and operator dictionary must be the same")


    tree_circuit=[]
    for i,dictionary_circuit__ in enumerate(dictionary_circuit):

        if dictionaries_combined:
            dictionary_operator_ = [dictionary_operator[i]]
        else:
            dictionary_operator_ = dictionary_operator

        tree_operator=[]
        for dictionary_operator__ in dictionary_operator_:

            # Build operator list and circuit list from the corresponding Optrees
            operator_list, operator_tree = build_operator_list(
                operator, dictionary_operator__, detect_operator_duplicates
            )
            circuit_list, parameter_list, circuit_tree = build_circuit_list(
                circuit, dictionary_circuit__, detect_circuit_duplicates
            )

            # Combine the operator tree and the circuit tree into a single tree
            # Adjust the offset to match the operator list and the parameter list
            tree_operator.append(
                _add_offset_to_tree(
                    adjust_tree_operators(circuit_tree, operator_tree, len(operator_list)),
                    len(total_circle_list),
                )
            )

            # Add everything to the total lists that are evaluated by the estimator
            for i, circ in enumerate(circuit_list):
                for op in operator_list:
                    total_circle_list.append(circ)
                    total_operator_list.append(op)
                    total_parameter_list.append(parameter_list[i])

        if multiple_operator_dict and not dictionaries_combined:
            tree_circuit.append(OpTreeNodeList(tree_operator))
        else:
            tree_circuit.append(tree_operator[0])

    if multiple_circuit_dict:
        evaluation_tree = OpTreeNodeList(tree_circuit)
    else:
        evaluation_tree = tree_circuit[0]

    # Evaluation via the estimator
    start = time.time()
    estimator_result = (
        estimator.run(total_circle_list, total_operator_list, total_parameter_list).result().values
    )
    print("run time", time.time() - start)

    print("estimator_result", estimator_result)

    print("evaluation_tree", evaluation_tree)

    return _evaluate_index_tree(evaluation_tree, estimator_result)


def assign_circuit_parameters(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit],
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
        The OpTree structure with all parameters assigned, (copied)
    """

    if isinstance(element, OpTreeNodeBase):
        if inplace:
            for c in element.children:
                assign_circuit_parameters(c, dictionary, inplace=True)
            for i, fac in enumerate(element.factor):
                if isinstance(fac, ParameterExpression):
                    element.factor[i] = float(fac.bind(dictionary, allow_unknown_parameters=True))

        else:
            # Index circuits and bind parameters in the OpTreeNode structure
            child_list_assigned = [
                assign_circuit_parameters(c, dictionary) for c in element.children
            ]
            factor_list_bound = []
            for fac in element.factor:
                if isinstance(fac, ParameterExpression):
                    factor_list_bound.append(
                        float(fac.bind(dictionary, allow_unknown_parameters=True))
                    )
                else:
                    factor_list_bound.append(fac)
            op = element.operation  # TODO: check if this is correct

            # Recursive rebuild of the OpTree structure
            if isinstance(element, OpTreeNodeSum):
                return OpTreeNodeSum(child_list_assigned, factor_list_bound, op)
            elif isinstance(element, OpTreeNodeList):
                return OpTreeNodeList(child_list_assigned, factor_list_bound, op)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")
    elif isinstance(element, OpTreeLeafCircuit):
        # Assign the parameters to the circuit
        if inplace:
            element.circuit.assign_parameters(dictionary, inplace=True)
        else:
            return OpTreeLeafCircuit(element.circuit.assign_parameters(dictionary, inplace=False))
    elif isinstance(element, QuantumCircuit):
        # Assign the parameters to the circuit
        if inplace:
            element.assign_parameters(dictionary, inplace=True)
        else:
            return element.assign_parameters(dictionary, inplace=False)
    else:
        raise ValueError("element must be a OpTreeNodeBase, OpTreeLeafCircuit or a QuantumCircuit")
