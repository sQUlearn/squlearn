import numpy as np
from typing import Union
import time

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression

from .optree import (
    hash_circuit,
    OpTreeNodeBase,
    OpTreeLeafBase,
    OpTreeNodeList,
    OpTreeNodeSum,
    OpTreeLeafCircuit,
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


def assign_circuit_parameters(
    element: Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit], dictionary, inplace:bool=False
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
                assign_circuit_parameters(c,dictionary,inplace=True)
            for i,fac in enumerate(element.factor):
                if isinstance(fac, ParameterExpression):
                    element.factor[i] = float(fac.bind(dictionary, allow_unknown_parameters=True))

        else:
            # Index circuits and bind parameters in the OpTreeNode structure
            child_list_assigned = [assign_circuit_parameters(c,dictionary) for c in element.children]
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



