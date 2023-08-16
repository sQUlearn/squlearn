import numpy as np
from typing import Union, List

import time

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit.primitives.backend_estimator import _pauli_expval_with_variance
from qiskit.primitives.base import SamplerResult

from .optree import (
    hash_circuit,
    hash_operator,
    OpTreeNodeBase,
    OpTreeLeafBase,
    OpTreeNodeList,
    OpTreeNodeSum,
    OpTreeLeafCircuit,
    OpTreeLeafOperator,
    OpTreeLeafContainer,
    OpTreeLeafExpectationValue,
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
                    return OpTreeLeafContainer(circuit_hash_list.index(circuit_hash))
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
        operator_hash_list = []

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
                if detect_operator_duplicates:
                    operator_hash = hash_operator(operator)
            elif isinstance(element, OpTreeLeafOperator):
                operator = element.operator
                if detect_operator_duplicates:
                    operator_hash = element.hashvalue
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
                if operator_hash in operator_hash_list:
                    return OpTreeLeafContainer(operator_hash_list.index(operator_hash))
                operator_hash_list.append(operator_hash)

            operator_list.append(operator)
            operator_counter += 1
            return OpTreeLeafContainer(operator_counter - 1)

    index_tree = build_lists_and_index_tree(element)

    return operator_list, index_tree

def build_expectation_list(
    element: Union[OpTreeNodeBase, OpTreeLeafExpectationValue],
    dictionary: dict,
    detect_expectation_duplicates: bool = False,
):
    # create a list of circuit and a copy of the circuit tree with indices pointing to the circuit
    circuit_list = []
    operator_list = []
    if detect_expectation_duplicates:
        expectation_hash_list = []
    parameter_list = []

    expectation_counter = 0

    def build_lists_and_index_tree(
        element: Union[OpTreeNodeBase, OpTreeLeafExpectationValue]
    ):
        """
        Helper function for building the circuit list and the parameter list, and
        creates a indexed copy of the OpTree structure that references the circuits in the list.
        """

        # Global counter for indexing the circuits, circuit list and hash list, and parameter list
        nonlocal expectation_counter
        nonlocal circuit_list
        nonlocal operator_list
        nonlocal parameter_list
        nonlocal expectation_hash_list

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
            # Reached a Expecation Value Leaf

            operator = element.operator
            circuit = element.circuit
            hashvalue = element.hashvalue

            if detect_expectation_duplicates:
                if hashvalue in expectation_hash_list:
                    return OpTreeLeafContainer(expectation_hash_list.index(hashvalue))
                expectation_hash_list.append(hashvalue)

            # Assign parameters to operator
            operator = operator.assign_parameters(
                [dictionary[p] for p in operator.parameters], inplace=False
            )
            if len(operator.parameters) != 0:
                raise ValueError("Not all parameters are assigned in the operator!")

            operator_list.append(operator)
            circuit_list.append(circuit)
            parameter_list.append(np.array([dictionary[p] for p in circuit.parameters]))
            expectation_counter += 1
            return OpTreeLeafContainer(expectation_counter - 1)

    index_tree = build_lists_and_index_tree(element)

    return circuit_list, operator_list, parameter_list, index_tree


def build_expectation_list_v2(
    element: Union[OpTreeNodeBase, OpTreeLeafExpectationValue],
    dictionary: dict,
    detect_expectation_duplicates: bool = False,
):
    # create a list of circuit and a copy of the circuit tree with indices pointing to the circuit
    circuit_list = []
    operator_list = []
    circuit_hash_list = []
    if detect_expectation_duplicates:
        expectation_hash_list = []
    parameter_list = []
    circuit_eval_list = []
    expectation_counter = 0
    circuit_eval_counter = 0

    def build_lists_and_index_tree(
        element: Union[OpTreeNodeBase, OpTreeLeafExpectationValue]
    ):
        """
        Helper function for building the circuit list and the parameter list, and
        creates a indexed copy of the OpTree structure that references the circuits in the list.
        """

        # Global counter for indexing the circuits, circuit list and hash list, and parameter list
        nonlocal expectation_counter
        nonlocal circuit_list
        nonlocal operator_list
        nonlocal parameter_list
        nonlocal expectation_hash_list
        nonlocal circuit_hash_list
        nonlocal circuit_eval_list
        nonlocal circuit_eval_counter

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
            # Reached a Expecation Value Leaf

            operator = element.operator
            circuit = element.circuit

            hashvalue_circuit = element._circuit.hashvalue

            circuit_already_in_list = False
            if hashvalue_circuit in circuit_hash_list:
                index = circuit_hash_list.index(hashvalue_circuit)
                if np.array_equal(parameter_list[index], np.array([dictionary[p] for p in circuit.parameters])):
                    circuit_already_in_list = True

            if circuit_already_in_list:
                circuit_eval_list.append(index)
            else:
                circuit_eval_list.append(circuit_eval_counter)
                circuit_hash_list.append(hashvalue_circuit)
                parameter_list.append(np.array([dictionary[p] for p in circuit.parameters]))
                circuit_list.append(circuit)
                circuit_eval_counter += 1

            # Assign parameters to operator
            operator = operator.assign_parameters(
                [dictionary[p] for p in operator.parameters], inplace=False
            )
            if len(operator.parameters) != 0:
                raise ValueError("Not all parameters are assigned in the operator!")

            operator_list.append(operator)
            expectation_counter += 1
            return OpTreeLeafContainer(expectation_counter - 1)

    index_tree = build_lists_and_index_tree(element)

    return circuit_list, operator_list, parameter_list,circuit_eval_list, index_tree







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

def evaluate_expectation_from_sampler(
    observable: Union[List[SparsePauliOp], SparsePauliOp],
    results: SamplerResult,
    index_slice: Union[None, slice] = None,
    index_list: Union[None, List[int]] = None,
):
    """
    Function for evaluating the expectation value of an observable from the results of a sampler.

    Args:
        observable (SparsePauliOp): The observable to be evaluated.
        results (BaseSamplerResult): The results of the sampler primitive.
        index_slice (Union[None, slice]): The index slice that is used to select a subset of the
                                          results.

    Returns:
        The expectation value of the observable as a numpy array.
    """

    if index_slice is not None and index_list is not None:
        raise ValueError("Only one of index_slice and index_list can be specified")

    # Check if the observable is a list or not
    no_list = False
    if not isinstance(observable, list):
        no_list = True
        observable = [observable]

    # Create a list of PauliList objects from the observable
    op_pauli_list = [PauliList(obs.paulis) for obs in observable]

    # Index slice that is used to select a subset of the results
    if index_slice is None:
        index_slice_ = slice(0, len(results.quasi_dists))
    else:
        index_slice_ = index_slice

    if index_list is None:

        # Calulate the expectation value with internal Qiskit function
        exp_val = np.array(
            [
                [
                    np.real_if_close(
                        np.dot(
                            _pauli_expval_with_variance(quasi.binary_probabilities(), pauli)[0],
                            observable[i].coeffs,
                        )
                    )
                    for i, pauli in enumerate(op_pauli_list)
                ]
                for quasi in results.quasi_dists[index_slice_]
            ]
        )
    else:
        # Calulate the expectation value with internal Qiskit function
        exp_val = np.array(
            [
                    np.real_if_close(
                        np.dot(
                            _pauli_expval_with_variance(results.quasi_dists[j].binary_probabilities(), op_pauli_list[i])[0],
                            observable[i].coeffs,
                        )
                    )
                for i,j in enumerate(index_list)
            ]
        )

    # Format results
    if no_list:
        exp_val.resize(exp_val.shape[0])
        return exp_val

    return exp_val











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
    start = time.time()
    # Preprocess the circuit dictionary
    multiple_circuit_dict = True
    if not isinstance(dictionary_circuit, list):
        dictionary_circuit = [dictionary_circuit]
        multiple_circuit_dict = False

    # Preprocess the operator dictionary
    multiple_operator_dict = True
    if not isinstance(dictionary_operator, list):
        dictionary_operator = [dictionary_operator]
        multiple_operator_dict = False

    if dictionaries_combined:
        if len(dictionary_circuit) != len(dictionary_operator):
            raise ValueError("The length of the circuit and operator dictionary must be the same")

    total_circle_list = []
    total_parameter_list = []
    index_offsets = [0]
    tree_circuit = []
    # Build list of circuits and parameters that are evaluated by the sampler
    # Also creates the evluation tree for the assembling the final results
    for i, dictionary_circuit_ in enumerate(dictionary_circuit):
        circuit_list, parameter_list, circuit_tree = build_circuit_list(
            circuit, dictionary_circuit_, detect_circuit_duplicates
        )

        circuit_list = [circuit.measure_all(inplace=False) for circuit in circuit_list]

        if multiple_circuit_dict and dictionaries_combined:
            tree_circuit.append(circuit_tree)
        else:
            tree_circuit.append(_add_offset_to_tree(circuit_tree, len(total_circle_list)))

        total_circle_list += circuit_list
        total_parameter_list += parameter_list
        index_offsets.append(len(total_circle_list))

    if multiple_circuit_dict:
        evaluation_tree = OpTreeNodeList(tree_circuit)
    else:
        evaluation_tree = tree_circuit[0]
    print("post processing", time.time() - start)

    # Run the sampler primtive
    start = time.time()
    sampler_result = sampler.run(total_circle_list, total_parameter_list).result()
    print("sampler run time", time.time() - start)

    start = time.time()
    final_result = []
    # Assemble the final results from the sampler measurements
    for i, dictionary_operator_ in enumerate(dictionary_operator):
        operator_list, operator_tree = build_operator_list(
            operator, dictionary_operator_, detect_operator_duplicates
        )

        if multiple_circuit_dict and dictionaries_combined:
            # Pick subset of the circuits that are linked to the current operator dictionary
            circ_tree = evaluation_tree.children[i]
            index_slice = slice(index_offsets[i], index_offsets[i + 1] + 1)
        else:
            # Pick all circuits
            circ_tree = evaluation_tree
            index_slice = None

        # Evaluate the expectation value of the current operator and operator dict and
        # (a subset of) the circuit measurements
        expec = evaluate_expectation_from_sampler(operator_list, sampler_result, index_slice)
        # Evaluate the operator tree for assembling the final operator values
        expec2 = [_evaluate_index_tree(operator_tree, ee) for ee in expec]
        # Evluate the circuit tree for assembling the final circuit values
        final_result.append(_evaluate_index_tree(circ_tree, expec2))
    print("post processing", time.time() - start)


    if multiple_operator_dict and multiple_circuit_dict and not dictionaries_combined:
        # Swap axes to match the order of the dictionaries (circuit dict first, operator dict second)
        return np.swapaxes(np.array(final_result), 0, 1)
    if multiple_operator_dict and multiple_circuit_dict and dictionaries_combined:
        return np.array(final_result)
    elif multiple_operator_dict and not multiple_circuit_dict:
        return np.array(final_result)
    else:
        return np.array(final_result[0])


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

    start = time.time()
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

    tree_circuit = []
    for i, dictionary_circuit__ in enumerate(dictionary_circuit):
        if dictionaries_combined:
            dictionary_operator_ = [dictionary_operator[i]]
        else:
            dictionary_operator_ = dictionary_operator

        tree_operator = []
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
    print("pre-processing", time.time() - start)


    # Evaluation via the estimator
    start = time.time()
    estimator_result = (
        estimator.run(total_circle_list, total_operator_list, total_parameter_list).result().values
    )
    print("run time", time.time() - start)

    start = time.time()
    final_results = _evaluate_index_tree(evaluation_tree, estimator_result)
    print("post processing", time.time() - start)

    return final_results

def evaluate_expectation_tree_from_estimator(expectation_tree: Union[OpTreeNodeBase,OpTreeLeafExpectationValue], dictionary:Union[List[dict],dict], estimator:BaseEstimator,detect_expectation_duplicates:bool=False):
    """
    Evaluate a expectation tree with an estimator.

    Args:
        expectation_tree (Union[OpTreeNodeBase,OpTreeLeafExpectationValue]): The expectation tree to be evaluated.
        dictionary (Union[List(dict),dict]): The dictionary that contains the parameter and their values. Can be list for the evaluation of multiple dictionaries.
        estimator (BaseEstimator): The estimator that is used for the evaluation.
        detect_expectation_duplicates (bool, optional): If True, duplicate expectation values are detected and only evaluated once. Defaults to False.
    """

    total_circle_list, total_operator_list, total_parameter_list, total_tree_list = [], [], [],[]

    multiple_dict = True
    if not isinstance(dictionary, list):
        dictionary = [dictionary]
        multiple_dict = False

    for dict_ in dictionary:
        circuit_list, operator_list, parameter_list, tree_list = build_expectation_list(expectation_tree, dict_,detect_expectation_duplicates)
        total_tree_list.append(_add_offset_to_tree(tree_list,len(total_circle_list)))
        total_circle_list += circuit_list
        total_operator_list += operator_list
        total_parameter_list += parameter_list

    if multiple_dict:
        evaluation_tree = OpTreeNodeList(total_tree_list)
    else:
        evaluation_tree = total_tree_list[0]

    # Evaluation via the estimator
    start = time.time()
    estimator_result = (
        estimator.run(total_circle_list, total_operator_list, total_parameter_list).result().values
    )
    print("post processing", time.time() - start)
    print("estimator_result", estimator_result)
    return _evaluate_index_tree(evaluation_tree, estimator_result)




def evaluate_expectation_tree_from_sampler(expectation_tree, dictionary, sampler,detect_expectation_duplicates:bool=False):


    # TODO: multiple dictionaries

    start = time.time()
    circuit_list, operator_list, parameter_list, circuit_eval_list, index_tree = build_expectation_list_v2(expectation_tree, dictionary)
    circuit_list = [circuit.measure_all(inplace=False) for circuit in circuit_list]
    print("build_lists_and_index_tree", time.time() - start)

    # Evaluation via the estimator
    start = time.time()
    sampler_result = (
        sampler.run(circuit_list, parameter_list).result()
    )
    print("run time", time.time() - start)

    start = time.time()
    expec = evaluate_expectation_from_sampler(operator_list, sampler_result, index_list=circuit_eval_list)
    result = _evaluate_index_tree(index_tree, expec)
    print("post processing", time.time() - start)
    print("expec", expec)

    return result








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
