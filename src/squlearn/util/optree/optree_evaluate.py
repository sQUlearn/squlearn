import numpy as np
from typing import Union, List, Tuple

import time
from packaging import version

from qiskit.circuit import QuantumCircuit
from qiskit import __version__ as qiskit_version
from qiskit.circuit import ParameterExpression, Clbit
from qiskit.primitives import BackendEstimator
from qiskit.quantum_info import SparsePauliOp, PauliList, Pauli
from qiskit.primitives.backend_estimator import _pauli_expval_with_variance
from qiskit.primitives.base import SamplerResult
from ...util.decompose_to_std import decompose_to_std

from .optree import (
    OpTreeNodeBase,
    OpTreeLeafBase,
    OpTreeList,
    OpTreeSum,
    OpTreeCircuit,
    OpTreeOperator,
    OpTreeContainer,
    OpTreeValue,
    OpTreeExpectationValue,
    OpTreeMeasuredOperator,
)

QISKIT_SMALLER_1_2 = version.parse(qiskit_version) < version.parse("1.2.0")

if QISKIT_SMALLER_1_2:

    class BitArray:
        """Dummy BitArray."""

else:
    from qiskit.primitives import BitArray

from ..executor import BaseSamplerV1, BaseEstimatorV1, BaseSamplerV2, BaseEstimatorV2


def _check_tree_for_matrix_compatibility(element: Union[OpTreeNodeBase, OpTreeLeafBase]):
    """
    Function for checking if an OpTree structure requires nested lists with different dimensions

    Necessary to check if data can be stored in a numpy float array or in a numpy object array.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafContainer]): The OpTree to be checked.

    Returns:
        True if the OpTree structure is compatible with a numpy float array, False otherwise.
    """

    def _get_dimensions(element: Union[OpTreeNodeBase, OpTreeLeafBase]):
        """
        Helper function for checking the dimensions of the OpTree structure.

        Args:
            element (Union[OpTreeNodeBase, OpTreeLeafContainer]): The OpTree to be checked.

        Returns:
            Outer dimension of the root of the OpTree structure.
        """
        if isinstance(element, OpTreeList):
            dim_list = [_get_dimensions(child) for child in element.children]
            if not all(dim == dim_list[0] for dim in dim_list):
                raise ValueError("All leafs must have the same dimension")
            return len(dim_list)
        elif isinstance(element, OpTreeSum):
            dim_list = [_get_dimensions(child) for child in element.children]
            if not all(dim == dim_list[0] for dim in dim_list):
                raise ValueError("All leafs must have the same dimension")
            return dim_list[0]
        elif isinstance(element, OpTreeLeafBase):
            return 1
        else:
            raise ValueError("element must be a OpTreeNode or a OpTreeLeafContainer")

    try:
        _get_dimensions(element)
        return True
    except ValueError:
        return False


def _evaluate_index_tree(
    element: Union[OpTreeNodeBase, OpTreeContainer, OpTreeValue],
    result_array: np.ndarray,
    datatype: str = "auto",
) -> Union[np.ndarray, float]:
    """
    Function for evaluating an OpTree structure that has been indexed with a given result array.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafContainer]): The OpTree to be evaluated.
                                                              Has to be indexed first, such that
                                                              the leafs contain the address of the
                                                              result array entries
                                                              and all factors have to be numeric.
        result_array (np.ndarray): The result array that contains the results to be placed in
                                   the leafs of the OpTree.
        datatype (str): The datatype of the result array. Can be ``'float'``,``'object'``,
                        or ``'auto'``. If 'auto', the function will check if the OpTree structure
                        is compatible with a numpy float array, and if not, it will use a numpy
                        object array. Defaults to ``'auto'``.

    Returns:
        The evaluated OpTree structure as a numpy array or a float.
    """

    def _evaluate_index_tree_recursive(
        element: Union[OpTreeNodeBase, OpTreeContainer, OpTreeValue],
        result_array: np.ndarray,
        datatype: str = "float",
    ) -> Union[np.ndarray, float]:
        if isinstance(element, OpTreeNodeBase):
            if any(not isinstance(fac, float) for fac in element.factor):
                raise ValueError("All factors must be numeric for evaluation")

            # Recursive construction of the data array
            if datatype == "float":
                try:
                    temp = np.array(
                        [
                            element.factor[i]
                            * _evaluate_index_tree_recursive(
                                child, result_array, datatype=datatype
                            )
                            for i, child in enumerate(element.children)
                        ],
                        dtype=float,
                    )
                except ValueError:
                    temp = np.array(
                        [
                            element.factor[i]
                            * _evaluate_index_tree_recursive(
                                child, result_array, datatype=datatype
                            )
                            for i, child in enumerate(element.children)
                        ],
                        dtype=object,
                    )
            elif datatype == "object":
                temp = np.array(
                    [
                        element.factor[i]
                        * _evaluate_index_tree_recursive(child, result_array, datatype=datatype)
                        for i, child in enumerate(element.children)
                    ],
                    dtype=object,
                )
            else:
                raise ValueError("datatype must be 'float' or 'object'")

            if isinstance(element, OpTreeSum):
                # OpTreeNodeSum -> sum over the array
                return np.sum(temp, axis=0)
            elif isinstance(element, OpTreeList):
                # OpTreeNodeList -> return just the array
                return temp
            else:
                raise ValueError("element must be a OpTreeNodeSum or a OpTreeNodeList")
        elif isinstance(element, OpTreeContainer):
            # Return value from the result array
            return result_array[element.item]
        elif isinstance(element, OpTreeValue):
            # Return the value
            return element.value
        else:
            raise ValueError("element must be a OpTreeNode or a OpTreeLeafContainer")

    if datatype == "auto":
        if _check_tree_for_matrix_compatibility(element):
            datatype = "float"
        else:
            datatype = "object"

    return _evaluate_index_tree_recursive(element, result_array, datatype=datatype)


def _build_circuit_list(
    optree_element: Union[OpTreeNodeBase, OpTreeCircuit, QuantumCircuit, OpTreeValue],
    dictionary: dict,
    detect_circuit_duplicates: bool = True,
) -> Tuple[
    List[QuantumCircuit],
    List[np.ndarray],
    Union[OpTreeNodeBase, OpTreeContainer, OpTreeValue],
]:
    """
    Helper function for creating a list of circuits from an OpTree structure.

    The function also creates a copy of the OpTree structure that references
    the circuits in the list. Furthermore, the function also creates the list
    of parameters that can be supplied to the Qiskit primitive for evaluation.

    The function also checks if the same circuit occurs multiple times
    in the OpTree structure and only adds it once to the circuit list.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): The OpTree to be
                                                                            converted.
        dictionary (dict): The dictionary that contains the values for the parameters in the
                           circuit and the OpTree structure.
        detect_circuit_duplicates (bool): If True, the removes duplicate circuits from the
                                          circuit list.

    Returns:
        A tuple containing the circuit list, the parameter list and the indexed copy of the
        OpTree structure.

    """
    from .optree import OpTree

    circuit_list = []
    if detect_circuit_duplicates:
        circuit_hash_dict = {}
    parameter_list = []
    circuit_counter = 0

    def _build_lists_and_index_tree(
        optree_element: Union[OpTreeNodeBase, OpTreeLeafBase, QuantumCircuit, OpTreeValue],
    ):
        """
        Helper function for building the circuit list and the parameter list, and
        creates a indexed copy of the OpTree structure that references the circuits in the list.
        """

        # Global counter for indexing the circuits, circuit list, hash list, and parameter list
        nonlocal circuit_list
        nonlocal circuit_hash_dict
        nonlocal parameter_list
        nonlocal circuit_counter

        if isinstance(optree_element, OpTreeNodeBase):
            # Recursive copy of the OpTreeNode structure and binding of the parameters
            # in the OpTree structure.
            child_list_indexed = [_build_lists_and_index_tree(c) for c in optree_element.children]
            factor_list_bound = []
            for fac in optree_element.factor:
                if isinstance(fac, ParameterExpression):
                    factor_list_bound.append(
                        float(fac.bind(dictionary, allow_unknown_parameters=True))
                    )
                else:
                    factor_list_bound.append(fac)

            # Recursive rebuild of the OpTree structure
            if isinstance(optree_element, OpTreeSum):
                return OpTreeSum(child_list_indexed, factor_list_bound, optree_element.operation)
            elif isinstance(optree_element, OpTreeList):
                return OpTreeList(child_list_indexed, factor_list_bound, optree_element.operation)
            else:
                raise ValueError("element must be a OpTreeNodeSum or a OpTreeNodeList")

        else:
            # Reached a CircuitTreeLeaf
            # Get the circuit, and check for duplicates if necessary.
            if isinstance(optree_element, QuantumCircuit):
                circuit = optree_element
                if detect_circuit_duplicates:
                    circuit_hash = OpTree.hash_circuit(circuit)
            elif isinstance(optree_element, OpTreeCircuit):
                circuit = optree_element.circuit
                if detect_circuit_duplicates:
                    circuit_hash = optree_element.hashvalue
            elif isinstance(optree_element, OpTreeValue):
                return optree_element  # Add nothing to the lists
            else:
                raise ValueError("element must be a CircuitTreeLeaf or a QuantumCircuit")

            # In case of duplicate detection, check if the circuit is already in the
            # circuit list and return the index of the circuit if it is already present
            if detect_circuit_duplicates:
                if circuit_hash in circuit_hash_dict:
                    return OpTreeContainer(circuit_hash_dict[circuit_hash])
                circuit_hash_dict[circuit_hash] = circuit_counter

            # Otherwise append the circuit to the circuit list, copy the paramerters into vector
            # form and append them to the parameter list, increase the counter and return the index
            # in the OpTreeLeafContainer
            circuit_list.append(circuit)
            parameter_list.append(np.array([dictionary[p] for p in circuit.parameters]))
            circuit_counter += 1
            return OpTreeContainer(circuit_counter - 1)

    index_tree = _build_lists_and_index_tree(optree_element)

    return circuit_list, parameter_list, index_tree


def _build_operator_list(
    optree_element: Union[
        OpTreeNodeBase,
        OpTreeOperator,
        OpTreeExpectationValue,
        OpTreeMeasuredOperator,
        OpTreeValue,
        SparsePauliOp,
    ],
    dictionary: dict,
    detect_operator_duplicates: bool = True,
) -> Tuple[List[SparsePauliOp], Union[OpTreeNodeBase, OpTreeContainer, OpTreeValue]]:
    """
    Helper function for creating a list of operators from an OpTree structure.

    The function also creates a copy of the OpTree structure that references
    the operators in the list. Furthermore, it assigns the parameters from the
    dictionary to the operators.

    Measurement operators are not added to the list of operators, and only the operator are
    returned for an OpTreeLeafExpectationValue or OpTreeLeafMeasuredOperator.

    The function also checks if the same operator occurs multiple times, and only adds it once
    to the operator list.

    Args:
        optree_element (Union[OpTreeNodeBase, OpTreeLeafOperator, OpTreeLeafExpectationValue, OpTreeLeafMeasuredOperator, SparsePauliOp]): The Operator in OpTree format to be converted.
        dictionary (dict): The dictionary that contains the values for the parameters in the
                           operator and the OpTree structure.
        detect_operator_duplicates (bool): If True, the removes duplicate operators from the
                                           operator list.

    Returns:
        A tuple containing the operator list and the indexed copy of the OpTree structure.
    """

    from .optree import OpTree

    operator_list = []
    if detect_operator_duplicates:
        operator_dict = {}
    operator_counter = 0

    def _build_lists_and_index_tree(
        optree_element: Union[OpTreeNodeBase, OpTreeOperator, SparsePauliOp, OpTreeValue],
    ):
        """
        Helper function for building the circuit list and the parameter list, and
        creates a indexed copy of the OpTree structure that references the circuits in the list.
        """

        # Global counter for indexing the operator, the operator list and the hash dictionary
        nonlocal operator_counter
        nonlocal operator_list
        nonlocal operator_dict

        if isinstance(optree_element, OpTreeNodeBase):
            # Recursive copy of the OpTreeNode structure and binding of the parameters
            # in the OpTree structure.
            child_list_indexed = [_build_lists_and_index_tree(c) for c in optree_element.children]
            factor_list_bound = []
            for fac in optree_element.factor:
                if isinstance(fac, ParameterExpression):
                    factor_list_bound.append(
                        float(fac.bind(dictionary, allow_unknown_parameters=True))
                    )
                else:
                    factor_list_bound.append(fac)
            # Recursive rebuild of the OpTree structure
            if isinstance(optree_element, OpTreeSum):
                return OpTreeSum(child_list_indexed, factor_list_bound, optree_element.operation)
            elif isinstance(optree_element, OpTreeList):
                return OpTreeList(child_list_indexed, factor_list_bound, optree_element.operation)
            else:
                raise ValueError("element must be a OpTreeNodeSum or a OpTreeNodeList")
        elif isinstance(optree_element, OpTreeValue):
            return optree_element  # Add nothing to the lists
        else:
            # Reached a Operator

            if isinstance(optree_element, SparsePauliOp):
                operator = optree_element
                if detect_operator_duplicates:
                    operator_hash = OpTree.hash_operator(operator)
            elif isinstance(
                optree_element,
                (OpTreeOperator, OpTreeExpectationValue, OpTreeMeasuredOperator),
            ):
                operator = optree_element.operator
                if detect_operator_duplicates:
                    operator_hash = optree_element.hashvalue
            else:
                raise ValueError("element must be a OpTreeLeafOperator or a SparsePauliOp")

            # Assign parameters
            operator = operator.assign_parameters(
                [dictionary[p] for p in operator.parameters], inplace=False
            )

            # TODO check if it makes a difference in speed if not complex numbers are used

            if len(operator.parameters) != 0:
                raise ValueError("Not all parameters are assigned in the operator!")

            # Check if the operator is already part of the evaluation list
            # If that is the case, return the index of the operator
            if detect_operator_duplicates:
                if operator_hash in operator_dict:
                    return OpTreeContainer(operator_dict[operator_hash])
                operator_dict[operator_hash] = operator_counter

            operator_list.append(operator)
            operator_counter += 1
            return OpTreeContainer(operator_counter - 1)

    index_tree = _build_lists_and_index_tree(optree_element)

    return operator_list, index_tree


def _build_measurement_list(
    optree_element: Union[
        OpTreeNodeBase,
        OpTreeMeasuredOperator,
        OpTreeOperator,
        SparsePauliOp,
        OpTreeValue,
    ],
    detect_measurement_duplicates: bool = True,
    detect_operator_duplicates: bool = True,
) -> Tuple[List[QuantumCircuit], List[List[int]]]:
    """
    Helper function for creating a list of measurement circuits from an OpTree structure.

    The function creates also a list of indices that connect a operator with a measurement.
    If no measurement is present, the measurement circuit is set to None.
    In this list, the outer list contains the measurement index, and the inner list contains
    the operator indices that can be calculated from the measurement.

    The function also checks if the same measurement occurs multiple times, and only adds it once
    to the measurement list.

    Args:
        optree_element (Union[OpTreeNodeBase, OpTreeLeafMeasuredOperator, OpTreeLeafOperator, SparsePauliOp]): The Operator in OpTree format to be converted.
        detect_measurement_duplicates (bool): If True, the removes duplicate measurements from the
                                                measurement list.
        detect_operator_duplicates (bool): If True, the removes duplicate operators from the
                                             operator list.

    Returns:
        A tuple containing the measurement circuit list and the operator measurement list.
    """

    measurement_circuits = []
    if detect_measurement_duplicates:
        measurement_hash_dict = {}
    if detect_operator_duplicates:
        operator_hash_dict = {}
    measurement_counter = 0
    operator_counter = 0
    operator_measurement_list = []

    from .optree import OpTree

    def build_list(
        optree_element: Union[OpTreeNodeBase, OpTreeOperator, SparsePauliOp, OpTreeValue],
    ):
        """
        Helper function for building the circuit list and the parameter list, and
        creates a indexed copy of the OpTree structure that references the circuits in the list.
        """

        # Global counter for indexing the circuits, circuit list and hash list, and parameter list
        nonlocal measurement_circuits
        nonlocal measurement_hash_dict
        nonlocal operator_hash_dict
        nonlocal measurement_counter
        nonlocal operator_counter
        nonlocal operator_measurement_list

        if isinstance(optree_element, OpTreeNodeBase):
            # Recursive Call of the function for traversing the OpTree structure
            for c in optree_element.children:
                build_list(c)

        elif isinstance(optree_element, OpTreeMeasuredOperator):
            # Measurement operator detected, check for duplicates of the operator if necessary
            if detect_operator_duplicates:
                if optree_element.hashvalue in operator_hash_dict:
                    return None
                operator_hash_dict[optree_element.hashvalue] = operator_counter

            circuit = optree_element.circuit
            if circuit.num_clbits == 0:
                raise ValueError("Circuit missing a measurement")

            # check for duplicates of the measurement circuit if necessary
            if detect_measurement_duplicates:
                measurement_hash = OpTree.hash_circuit(circuit)
                if measurement_hash in measurement_hash_dict:
                    operator_measurement_list[measurement_hash_dict[measurement_hash]].append(
                        operator_counter
                    )
                    operator_counter += 1
                    return None
                measurement_hash_dict[measurement_hash] = measurement_counter

            # Add everything to the list
            measurement_circuits.append(circuit)
            operator_measurement_list.append([operator_counter])
            measurement_counter += 1
            operator_counter += 1
            return None

        elif isinstance(optree_element, OpTreeValue):
            return None

        elif isinstance(optree_element, (SparsePauliOp, OpTreeOperator)):
            # Measure free Operator detected, check for duplicates of the operator if necessary
            # (Conversion of X and Y Paulis is done elsewhere!)
            if detect_operator_duplicates:
                if isinstance(optree_element, SparsePauliOp):
                    hashvalue = OpTree.hash_operator(optree_element)
                else:
                    hashvalue = optree_element.hashvalue
                if hashvalue in operator_hash_dict:
                    return None
                operator_hash_dict[hashvalue] = operator_counter

            # check for duplicates of the measurement circuit if necessary
            if detect_measurement_duplicates:
                measurement_hash = "None"
                if measurement_hash in measurement_hash_dict:
                    operator_measurement_list[measurement_hash_dict[measurement_hash]].append(
                        operator_counter
                    )
                    operator_counter += 1
                    return None
                measurement_hash_dict[measurement_hash] = measurement_counter

            # Add everything to the list
            measurement_circuits.append(None)
            operator_measurement_list.append([operator_counter])
            measurement_counter += 1
            operator_counter += 1
            return None

        else:
            raise ValueError("Wrong OpTree type detected!")

    build_list(optree_element)

    return measurement_circuits, operator_measurement_list


def _build_expectation_list(
    optree_element: Union[OpTreeNodeBase, OpTreeExpectationValue, OpTreeValue],
    dictionary: dict,
    detect_expectation_duplicates: bool = True,
    group_circuits: bool = True,
) -> Tuple[
    List[QuantumCircuit],
    List[SparsePauliOp],
    List[np.ndarray],
    List[List[int]],
    Union[OpTreeNodeBase, OpTreeContainer, OpTreeValue],
]:
    """
    Helper function for creating a lists of circuits and operator from an expectation OpTree.

    The function also creates a copy of the OpTree structure that references the expectation
    values in the list. Furthermore, it assigns the parameters from the dictionary to the
    operators and builds a list of parameters that can be supplied to the Qiskit primitive.

    The function detects and removes duplicate expectation values from the evaluation.
    It also groups the same circuits together, and only adds them once to the circuit list.
    This can be turned off with the option ``detect_expectation_duplicates`` and
    ``group_circuits``.

    Args:
        optree_element (Union[OpTreeNodeBase, OpTreeLeafExpectationValue]): The expectation values
                                                                            in OpTree format to
                                                                            be converted.
        dictionary (dict): The dictionary that contains the values for the parameters in the
                           circuits, operators and the OpTree structure.
        detect_expectation_duplicates (bool): If True, the removes duplicate expectation values
                                                  from the evaluation list. Defaults to True.
        group_circuits (bool): If True, the removes duplicate circuits from the evaluation list.
                                 Defaults to True.

    Returns:
        A tuple containing the circuit list, the operator list, the parameter list, the list
        that connects the circuits with the operators, and the indexed copy of the OpTree
        structure for the final assembly.

    """

    # create a list of circuit and a copy of the circuit tree with indices pointing to the circuit
    circuit_list = []
    operator_list = []
    circuit_dict = {}
    expectation_dict = {}
    parameter_list = []
    expectation_counter = 0
    circuit_eval_counter = 0
    circuit_operator_list = []

    def build_lists_and_index_tree(
        optree_element: Union[OpTreeNodeBase, OpTreeExpectationValue, OpTreeValue],
    ):
        """
        Helper function for building the circuit list and the parameter list, and
        creates a indexed copy of the OpTree structure that references the circuits in the list.
        """

        # Global counter for indexing the circuits, circuit list and hash list, and parameter list

        nonlocal circuit_list
        nonlocal operator_list
        nonlocal circuit_dict
        nonlocal expectation_dict
        nonlocal parameter_list
        nonlocal expectation_counter
        nonlocal circuit_operator_list
        nonlocal circuit_eval_counter

        if isinstance(optree_element, OpTreeNodeBase):
            # Index circuits and bind parameters in the OpTreeNode structure
            child_list_indexed = [build_lists_and_index_tree(c) for c in optree_element.children]
            factor_list_bound = []
            for fac in optree_element.factor:
                if isinstance(fac, ParameterExpression):
                    factor_list_bound.append(
                        float(fac.bind(dictionary, allow_unknown_parameters=True))
                    )
                else:
                    factor_list_bound.append(fac)

            # Recursive rebuild of the OpTree structure
            if isinstance(optree_element, OpTreeSum):
                return OpTreeSum(child_list_indexed, factor_list_bound, optree_element.operation)
            elif isinstance(optree_element, OpTreeList):
                return OpTreeList(child_list_indexed, factor_list_bound, optree_element.operation)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

        elif isinstance(optree_element, OpTreeValue):
            return optree_element  # Add nothing to the lists

        elif isinstance(optree_element, OpTreeExpectationValue):
            # Reached a Expecation Value Leaf

            operator = optree_element.operator
            circuit = optree_element.circuit

            # Check if the same expectation value is already in the list
            if detect_expectation_duplicates:
                hashvalue = optree_element.hashvalue
                if hashvalue in expectation_dict:
                    return OpTreeContainer(expectation_dict[hashvalue])
                expectation_dict[hashvalue] = expectation_counter

            # If circuits are grouped, check if the same circuit is already in the list
            circuit_already_in_list = False
            if group_circuits:
                hashvalue_circuit = optree_element._circuit.hashvalue
                if hashvalue_circuit in circuit_dict:
                    index = circuit_dict[hashvalue_circuit]
                    circuit_already_in_list = True
                    circuit_operator_list[index].append(expectation_counter)
            if not circuit_already_in_list:
                circuit_operator_list.append([expectation_counter])
                if group_circuits:
                    circuit_dict[hashvalue_circuit] = circuit_eval_counter
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
            return OpTreeContainer(expectation_counter - 1)

        else:
            raise ValueError("element must be a OpTreeNode or a OpTreeLeafContainer")

    index_tree = build_lists_and_index_tree(optree_element)

    return circuit_list, operator_list, parameter_list, circuit_operator_list, index_tree


def _add_offset_to_tree(
    optree_element: Union[OpTreeNodeBase, OpTreeContainer, OpTreeValue], offset: int
):
    """
    Helper function for adding a constant offset to all leafs of the OpTree structure.

    Args:
        element (Union[OpTreeNodeBase, OpTreeLeafContainer]): The OpTree to be adjusted.
        offset (int): The offset to be added to all leafs.

    Returns:
        Returns a copy with the offset added to all leafs.

    """
    if offset == 0:
        return optree_element

    if isinstance(optree_element, OpTreeNodeBase):
        # Recursive call and reconstruction of the data array
        children_list = [_add_offset_to_tree(child, offset) for child in optree_element.children]

        # Rebuild the tree with the new children and factors (copy part)
        if isinstance(optree_element, OpTreeSum):
            return OpTreeSum(children_list, optree_element.factor, optree_element.operation)
        elif isinstance(optree_element, OpTreeList):
            return OpTreeList(children_list, optree_element.factor, optree_element.operation)
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

    elif isinstance(optree_element, OpTreeContainer):
        # Change the item value
        if isinstance(optree_element.item, int):
            return OpTreeContainer(optree_element.item + offset)
        else:
            raise ValueError("Offset can only be added to integer leafs")
    elif isinstance(optree_element, OpTreeValue):
        # Return the value
        return optree_element
    else:
        raise ValueError("element must be a OpTreeNode or a OpTreeLeafContainer")


def _evaluate_expectation_from_sampler(
    operator: List[SparsePauliOp],
    results: Union[SamplerResult, List[BitArray]],
    operator_measurement_list: Union[None, List[List[int]]] = None,
    offset: int = 0,
):
    """
    Function for evaluating the expectation value of an operator from the results of a sampler.

    operator_measurement_list connects the measured circuits with the operators.
    If operator_measurement_list is two level nested (e.g. [[0,1],[2]]), this means that
    the outer list represents the circuits, and the inner list represents the operators.

    As an alternative the depth of operator_measurement_list can be 3, where the outer list
    represents the circuits, the middle list represents different measurements of the same circuit,
    and the inner list represents the operators that are computed. In this case, the results have
    to be sorted in the same way as the operators occur in the observable list.

    Args:
        observable (SparsePauliOp): The observable to be evaluated.
        results (BaseSamplerResult): The results of the sampler primitive.
        operator_measurement_list (Union[None, List[List[int]]]): The index list that is used to
                                                                  connect the measured circuit
                                                                  with the inputted operator list.
        offset (int): The offset that is added to the index of the circuits in the sampler results.

    Returns:
        The expectation value of the operator as a numpy array.
    """

    # Get depth of a nested list
    depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1

    # Create a list of PauliList objects from the observable
    op_pauli_list = [PauliList(obs.paulis) for obs in operator]

    # Check if only the Z and I Paulis are used in the observable
    # Too late for a basis change
    for p in op_pauli_list:
        if p.x.any():
            raise ValueError(
                "Observable only with Z and I Paulis are supported, "
                + "run transform_to_zbasis first"
            )

    primitives_v2 = False
    if isinstance(results, SamplerResult):
        num_results = len(results.quasi_dists)
    else:
        num_results = len(results)
        primitives_v2 = True

    # If no measurement is present, create one where every circuit is connected to all
    # operators
    if operator_measurement_list is None:
        operator_measurement_list_ = list(range(0, len(operator))) * num_results
    else:
        operator_measurement_list_ = operator_measurement_list

    depth_om_list = depth(operator_measurement_list_)
    if depth_om_list == 3:
        # Flatten the list to avoid nested loops
        # three nested lists -> circuit and measurements are considered separated
        flatted_resort_list = [item2 for item in operator_measurement_list_ for item2 in item]
    elif depth_om_list == 2:
        # two nested lists -> circuit and measurements are not separated
        flatted_resort_list = operator_measurement_list_
    else:
        raise ValueError("Wrong depth of operator_measurement_list")

    # Calulate the expectation value with internal Qiskit function
    if primitives_v2:
        exp_val = np.array(
            [
                np.real_if_close(results[icirc + offset].expectation_values(operator[iop]))
                for icirc, oplist in enumerate(flatted_resort_list)
                for iop in oplist
            ]
        )
    else:
        exp_val = np.array(
            [
                np.real_if_close(
                    np.dot(
                        _pauli_expval_with_variance(
                            results.quasi_dists[icirc + offset].binary_probabilities(),
                            op_pauli_list[iop],
                        )[0],
                        operator[iop].coeffs,
                    )
                )
                for icirc, oplist in enumerate(flatted_resort_list)
                for iop in oplist
            ]
        )

    if depth_om_list == 3:
        # Resort results into the operator order (so far in measurement order)
        ioff = 0
        index_list = []
        for outer_circ in operator_measurement_list_:
            flatted_resort_list_circ = [item2 for item in outer_circ for item2 in item]
            index_list.append(list(np.argsort(flatted_resort_list_circ) + ioff))
            ioff += len(flatted_resort_list_circ)
        return np.take(exp_val, index_list)
    else:
        # two nested lists -> No need to resort
        return exp_val


def _transform_operator_to_zbasis(
    operator: Union[OpTreeOperator, SparsePauliOp], abelian_grouping: bool = True
) -> Union[OpTreeOperator, OpTreeNodeBase, OpTreeSum, OpTreeMeasuredOperator]:
    """
    Takes an operator and transforms it to the Z basis by adding measurement circuits.

    Basis changes to the Z basis are achieved by adding the associated gates into the
    measurement circuits.

    Args:
        operator (Union[OpTreeLeafOperator, SparsePauliOp]): The operator to be transformed.
        abelian_grouping (bool, optional): If True, the operator is grouped into commuting terms.
                                        Defaults to True.

    Return:
        Returns the transformed operator together with the measurement circuits either as
        an OpTreeLeafMeasuredOperator or an OpTreeNodeSum if the operator is grouped into different
        terms. If no transformation is needed, the input operator is returned without any changes.
    """

    if QISKIT_SMALLER_1_2:
        measurement_circuit = BackendEstimator._measurement_circuit
    else:
        from qiskit.primitives.backend_estimator_v2 import (
            _measurement_circuit as measurement_circuit,
        )

    # Adjust measurements to be possible in Z basis
    if isinstance(operator, OpTreeOperator):
        operator = operator.operator

    children_list = []

    # fast check, if there is anything to do (checks for X and Y gates)
    any_changes = False
    for op in operator:
        any_changes = any_changes or op.paulis.x.any()
    if not any_changes:
        return operator

    if abelian_grouping:
        for op in operator.group_commuting(qubit_wise=True):
            # Build the measurement circuit and the adjusted measurements
            basis = Pauli((np.logical_or.reduce(op.paulis.z), np.logical_or.reduce(op.paulis.x)))
            meas_circuit, indices = measurement_circuit(op.num_qubits, basis)
            z_list = [
                [
                    op.paulis.z[j, indices][i] or op.paulis.x[j, indices][i]
                    for i in range(len(op.paulis.z[0, indices]))
                ]
                for j in range(len(op.paulis.z[:, indices]))
            ]
            x_list = [
                [False for i in range(len(op.paulis.z[0, indices]))]
                for j in range(len(op.paulis.z[:, indices]))
            ]
            paulis = PauliList.from_symplectic(
                z_list, x_list, op.paulis.phase
            )  # TODO: Check Phase

            # Build the expectation value leaf with the adjusted measurements
            children_list.append(
                OpTreeMeasuredOperator(meas_circuit, SparsePauliOp(paulis, op.coeffs))
            )

    else:
        for basis, op in zip(operator.paulis, operator):  # type: ignore
            # Build the measurement circuit and the adjusted measurements
            meas_circuit, indices = measurement_circuit(op.num_qubits, basis)
            z_list = [
                [
                    op.paulis.z[j, indices][i] or op.paulis.x[j, indices][i]
                    for i in range(len(op.paulis.z[0, indices]))
                ]
                for j in range(len(op.paulis.z[:, indices]))
            ]
            x_list = [
                [False for i in range(len(op.paulis.z[0, indices]))]
                for j in range(len(op.paulis.z[:, indices]))
            ]
            paulis = PauliList.from_symplectic(z_list, x_list, op.paulis.phase)

            # Build the expectation value leaf with the adjusted measurements
            children_list.append(
                OpTreeMeasuredOperator(meas_circuit, SparsePauliOp(paulis, op.coeffs))
            )

    if len(children_list) == 1:
        return children_list[0]
    # If there are multiple measurements necessary convert to OpTreeSum
    return OpTreeSum(children_list)


def _measure_all_unmeasured(circ_in, final_measurements: bool = False):
    """Helper function for circuits with in-circuit measurements."""

    def change_order(n, reorder_list):
        """Helper function for to map by a given mapping."""
        for i in reorder_list:
            if i[1] == n:
                return i[0]

    if circ_in.num_clbits == 0:
        return circ_in.measure_all(inplace=False)
    else:
        qubits = [i for i in range(circ_in.num_qubits)]
        circ_in = decompose_to_std(circ_in)
        for instruction, qargs, cargs in circ_in.data:
            if instruction.name == "measure":
                for qubit in qargs:
                    if circ_in.find_bit(qubit)[0] in qubits:
                        qubits.remove(circ_in.find_bit(qubit)[0])
                    else:
                        raise ValueError(
                            "There are multiple measurements on the same qubit."
                            "Please remove measurements accordingly. Note that this can happen,"
                            " if one defines an observable with X,Y Pauli measurements on a qubit,"
                            " which is already measured in an in-circuit measurement."
                        )

        circ = circ_in.copy()
        if not final_measurements:
            # Add measurements to all non measured qubits if not measured
            new_creg = circ._create_creg(len(qubits), "meas")
            circ.add_register(new_creg)
            if not final_measurements:
                circ.measure(qubits, new_creg)
                if len(qubits) == circ_in.num_qubits:
                    return circ

        new_ordering = []
        for instruction, qargs, cargs in circ.data:
            if instruction.name == "measure":
                for n in range(len(qargs)):
                    new_ordering.append([circ.find_bit(qargs[n])[0], circ.find_bit(cargs[n])[0]])

        circ_new = QuantumCircuit(circ.num_qubits)
        if not final_measurements:
            new_creg = circ_new._create_creg(circ.num_qubits, "meas")
        else:
            new_creg = circ_new._create_creg(circ_in.num_clbits, "meas")
        circ_new.add_register(new_creg)
        for instruction, qargs, cargs in circ.data:
            if (
                instruction.name == "measure" and not final_measurements
            ):  # to adjust the clbits of measurements
                clbits = [circ.find_bit(i)[0] for i in qargs]
            else:
                clbits = [circ.find_bit(i)[0] for i in cargs]
            operation = instruction.copy()
            if instruction.condition:  # to adjust the clbits of c_if
                operation.condition = (
                    Clbit(
                        circ_new.cregs[0],
                        change_order(circ.find_bit(instruction.condition[0])[0], new_ordering),
                    ),
                    instruction.condition[1],
                )
            circ_new.append(operation, [circ.find_bit(i)[0] for i in qargs], clbits)
        return circ_new


class OpTreeEvaluate:
    """Static class for evaluating OpTree structures with Qiskit's primitives."""

    @staticmethod
    def evaluate_with_sampler(
        circuit: Union[OpTreeNodeBase, OpTreeCircuit, QuantumCircuit],
        operator: Union[OpTreeNodeBase, OpTreeOperator, SparsePauliOp, OpTreeMeasuredOperator],
        dictionary_circuit: Union[dict, List[dict]],
        dictionary_operator: Union[dict, List[dict]],
        sampler: Union[BaseSamplerV1, BaseSamplerV2],
        dictionaries_combined: bool = False,
        detect_duplicates: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        Function for evaluating the expectation value with the sampler primitive.

        Inputted are a circuit and operator in OpTree format, and a dictionaries that contain the
        values for the parameters in the circuit and operator.
        Dictionary can be a list of dictionaries, in which case the function evaluates the expectation
        value for all combinations of the dictionaries, or if ``dictionaries_combined==True``, the
        function evaluates the expectation value for the same index of the dictionaries.

        The function also checks if the same circuit or operator occurs multiple times, and only adds
        it once to the evaluation list. This can be turned off with the ``detect_duplicates`` flag.

        Args:
            circuit (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): The circuit or
                OpTree with circuits to be evaluated.
            operator (Union[OpTreeNodeBase, OpTreeLeafOperator, SparsePauliOp, OpTreeLeafMeasuredOperator]):
                The operator or OpTree in the expectation values.
            dictionary_circuit (Union[dict, List[dict]]): The dictionary or list of dictionaries that
                contain the values for the parameters in the circuit (or the circuit OpTree).
            dictionary_operator (Union[dict, List[dict]]): The dictionary or list of dictionaries
                that contain the values for the parameters in the operator
                (or the operator OpTree).
            sampler (Union[BaseSamplerV1, BaseSamplerV2]): The sampler primitive that is used
                for the evaluation.
            dictionaries_combined (bool): If True, the function evaluates the expectation value for
                the same index of the dictionaries (both have to be Lists). Defaults to False.
            detect_duplicates (bool): If True, the removes duplicate circuits and operators from the
                evaluation list. Defaults to True.

        Returns:
            The expectation value of the expectation values as a numpy array.
        """

        def _max_from_nested_list(l: Union[list, int]):
            """Helper function for finding the maximum value of a nested list"""
            if isinstance(l, int):
                return l
            else:
                return np.max([_max_from_nested_list(ll) for ll in l])

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
                raise ValueError(
                    "The length of the circuit and operator dictionary must be the same"
                )

        # Build list of measurement circuits that are used for the evaluation
        # Also creates a list of indices that connect a operator with a measurement
        measurement_circuits, operator_measurement_list = _build_measurement_list(
            operator, detect_duplicates, detect_duplicates
        )

        # Build list of measured circuits and parameters to be evaluated by the sampler
        offset = 0
        offset_for_dict_combo = []
        total_circuit_list = []
        total_parameter_list = []
        index_offsets = [0]
        tree_circuit = []
        circuit_operator_list = []
        for dictionary_circuit_ in dictionary_circuit:
            # Create the circuit list, the parameter list and the indexed copy of the circuit tree
            # for the final assembly of the results
            circuit_list, parameter_list, circuit_tree = _build_circuit_list(
                circuit, dictionary_circuit_, detect_duplicates
            )

            # List for distinguishing between the different dictionaries later
            offset_for_dict_combo.append(len(total_circuit_list))

            # Loop for creating the measured circuits
            for i, circ_unmeasured in enumerate(circuit_list):
                for measure in measurement_circuits:
                    if measure is None:
                        total_circuit_list.append(_measure_all_unmeasured(circ_unmeasured))
                    else:
                        total_circuit_list.append(
                            _measure_all_unmeasured(
                                circ_unmeasured.compose(measure, inplace=False),
                                final_measurements=True,
                            )
                        )
                total_parameter_list += [parameter_list[i]] * len(operator_measurement_list)
                circuit_operator_list.append(operator_measurement_list)

            if multiple_circuit_dict and dictionaries_combined:
                tree_circuit.append(circuit_tree)
            else:
                tree_circuit.append(_add_offset_to_tree(circuit_tree, offset))

            offset += len(circuit_list)
            index_offsets.append(offset)

        if multiple_circuit_dict:
            evaluation_tree = OpTreeList(tree_circuit)
        else:
            evaluation_tree = tree_circuit[0]
        # print("Pre-processing: ", time.time() - start)

        # Run the sampler primtive
        start = time.time()
        # print("Number of circuits for sampler: ", len(total_circuit_list))

        if len(total_circuit_list) > 0:
            if isinstance(sampler, BaseSamplerV1):
                sampler_result = sampler.run(total_circuit_list, total_parameter_list).result()
            elif isinstance(sampler, BaseSamplerV2):
                pubs = list(zip(total_circuit_list, total_parameter_list))
                sampler_result = [result.data.meas for result in sampler.run(pubs).result()]
            else:
                raise ValueError("Unknown sampler type!")
        else:
            sampler_result = []
        # print("Sampler run time: ", time.time() - start)

        # Compute the expectation value from the sampler results
        start = time.time()
        final_result = []

        for i, dictionary_operator_ in enumerate(dictionary_operator):
            # Create the operator list and the indexed copy of the operator tree
            # for assembling the expectation values from the sampler results
            operator_list, operator_tree = _build_operator_list(
                operator, dictionary_operator_, detect_duplicates
            )

            if multiple_circuit_dict and dictionaries_combined:
                # Pick subset of the circuits that are linked to the current operator dictionary
                circ_tree = evaluation_tree.children[i]
                index_slice = slice(index_offsets[i], index_offsets[i + 1])
                offset = offset_for_dict_combo[i]
            else:
                # Pick all circuits
                circ_tree = evaluation_tree
                index_slice = slice(0, len(total_circuit_list))
                offset = 0

            if len(total_circuit_list) == 0:
                # If no circuits are present, return evaluated operator with no circuits
                if len(operator_list) == 0:
                    return np.array([])
                else:
                    expec2 = _evaluate_index_tree(operator_tree, [])
                    final_result.append(_evaluate_index_tree(circ_tree, [expec2]))
            else:
                if _max_from_nested_list(operator_measurement_list) != len(operator_list) - 1:
                    raise ValueError("Operator measurement list does not match operator list!")

                # Evaluate the expectation values from the sampler results
                expec = _evaluate_expectation_from_sampler(
                    operator_list,
                    sampler_result,
                    operator_measurement_list=circuit_operator_list[index_slice],
                    offset=offset,
                )

                # Evaluate the operator tree
                expec2 = [_evaluate_index_tree(operator_tree, ee) for ee in expec]

                # Evaluate the circuit tree for assembling the final results
                final_result.append(_evaluate_index_tree(circ_tree, expec2))
        # print("Post processing: ", time.time() - start)

        if multiple_operator_dict and multiple_circuit_dict and not dictionaries_combined:
            # Swap axes to match the order of the dictionaries
            # (circuit dict first, operator dict second)
            return np.swapaxes(np.array(final_result), 0, 1)
        if multiple_operator_dict and multiple_circuit_dict and dictionaries_combined:
            return np.array(final_result)
        elif multiple_operator_dict and not multiple_circuit_dict:
            return np.array(final_result)
        else:
            return_val = np.array(final_result[0])
            if len(return_val.shape) == 0:
                return float(return_val)
            return return_val

    @staticmethod
    def evaluate_with_estimator(
        circuit: Union[OpTreeNodeBase, OpTreeCircuit, QuantumCircuit],
        operator: Union[OpTreeNodeBase, OpTreeOperator, SparsePauliOp],
        dictionary_circuit: Union[dict, List[dict]],
        dictionary_operator: Union[dict, List[dict]],
        estimator: Union[BaseEstimatorV1, BaseEstimatorV2],
        dictionaries_combined: bool = False,
        detect_duplicates: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        Function for evaluating the expectation value with the estimator primitive.

        Inputted are a circuit and operator in OpTree format, and a dictionaries that contain the
        values for the parameters in the circuit and operator.
        Dictionary can be a list of dictionaries, in which case the function evaluates the
        expectation
        value for all combinations of the dictionaries, or if ``dictionaries_combined==True``, the
        function evaluates the expectation value for the same index of the dictionaries.

        The function also checks if the same circuit or operator occurs multiple times, and only
        adds
        it once to the evaluation list. This can be turned off with the ``detect_duplicates`` flag.

        Args:
            circuit (Union[OpTreeNodeBase, OpTreeLeafCircuit, QuantumCircuit]): The circuit or
                OpTree with circuits to be evaluated.
            operator (Union[OpTreeNodeBase, OpTreeLeafOperator, SparsePauliOp]): The operator or
                OpTree in the expectation values.
            dictionary_circuit (Union[dict, List[dict]]): The dictionary or list of dictionaries
                that contain the values for the parameters in the circuit (or the circuit OpTree).
            dictionary_operator (Union[dict, List[dict]]): The dictionary or list of dictionaries
                that contain the values for the parameters in the operator
                (or the operator OpTree).
            estimator (Union[BaseEstimatorV1,BaseEstimatorV2]): The estimator primitive that is
                used for the evaluation.
            dictionaries_combined (bool): If True, the function evaluates the expectation value for
                the same index of the dictionaries (both have to be Lists). Defaults to False.
            detect_duplicates (bool): If True, the removes duplicate circuits and operators from
                the evaluation list. Defaults to True.

        Returns:
            The expectation value of the expectation values as a numpy array.
        """

        def adjust_tree_operators(
            circuit_tree: Union[OpTreeNodeBase, OpTreeContainer],
            operator_tree: Union[OpTreeNodeBase, OpTreeContainer],
            offset: int,
        ):
            """Helper function for merging the operator tree and the circuit tree into a single tree.

            Args:
                circuit_tree (Union[OpTreeNodeBase,OpTreeLeafContainer]): The indexed circuit tree.
                operator_tree (Union[OpTreeNodeBase,OpTreeLeafContainer]): The indexed operator tree.
                operator_list_length (int): The length of the operator list.

            Returns:
                The merged tree.
            """
            if isinstance(circuit_tree, OpTreeNodeBase):
                children_list = [
                    adjust_tree_operators(child, operator_tree, offset)
                    for child in circuit_tree.children
                ]

                # Rebuild the tree with the new children and factors (copy part)
                if isinstance(circuit_tree, OpTreeSum):
                    return OpTreeSum(children_list, circuit_tree.factor, circuit_tree.operation)
                elif isinstance(circuit_tree, OpTreeList):
                    return OpTreeList(children_list, circuit_tree.factor, circuit_tree.operation)
                else:
                    raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

            elif isinstance(circuit_tree, OpTreeContainer):
                k = circuit_tree.item
                return _add_offset_to_tree(operator_tree, k * offset)

        start = time.time()

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
                raise ValueError(
                    "The length of the circuit and operator dictionary must be the same"
                )

        tree_circuit = []
        total_circuit_list = []
        total_operator_list = []
        total_parameter_list = []
        for i, dictionary_circuit__ in enumerate(dictionary_circuit):
            # Build circuit list and circuit tree from the circuit OpTree
            circuit_list, parameter_list, circuit_tree = _build_circuit_list(
                circuit, dictionary_circuit__, detect_duplicates
            )

            # Handle operator dictionary
            if dictionaries_combined:
                dictionary_operator_ = [dictionary_operator[i]]
            else:
                dictionary_operator_ = dictionary_operator

            tree_operator = []
            for dictionary_operator__ in dictionary_operator_:
                # Build operator list and and operator tree from the operator OpTree
                operator_list, operator_tree = _build_operator_list(
                    operator, dictionary_operator__, detect_duplicates
                )

                # Combine the operator tree and the circuit tree into a single tree
                # Adjust the offset to match the operator list and the parameter list
                tree_operator.append(
                    _add_offset_to_tree(
                        adjust_tree_operators(circuit_tree, operator_tree, len(operator_list)),
                        len(total_circuit_list),
                    )
                )

                # Add everything to the total lists that are evaluated by the estimator
                for i, circ in enumerate(circuit_list):
                    total_circuit_list += [circ] * len(operator_list)
                    total_parameter_list += [parameter_list[i]] * len(operator_list)
                    for op in operator_list:
                        total_operator_list.append(op)

            if multiple_operator_dict and not dictionaries_combined:
                tree_circuit.append(OpTreeList(tree_operator))
            else:
                tree_circuit.append(tree_operator[0])

        # Create evaluation tree for final assembly of values
        if multiple_circuit_dict:
            evaluation_tree = OpTreeList(tree_circuit)
        else:
            evaluation_tree = tree_circuit[0]
        # print("Pre-processing: ", time.time() - start)

        # Evaluation via the estimator
        start = time.time()
        # print("Number of circuits for estimator: ", len(total_circuit_list))
        if len(total_circuit_list) == 0:
            return _evaluate_index_tree(evaluation_tree, [])

        if isinstance(estimator, BaseEstimatorV1):
            estimator_result = (
                estimator.run(total_circuit_list, total_operator_list, total_parameter_list)
                .result()
                .values
            )
        elif isinstance(estimator, BaseEstimatorV2):
            pubs = list(zip(total_circuit_list, total_operator_list, total_parameter_list))
            estimator_result = np.array([r.data.evs for r in estimator.run(pubs).result()])
        else:
            raise ValueError("Unknown estimator type!")

        # print("Estimator run time: ", time.time() - start)

        # Assembly the final values from the evaluation tree
        start = time.time()
        final_results = _evaluate_index_tree(evaluation_tree, estimator_result)
        # print("Post-processing: ", time.time() - start)

        return final_results

    @staticmethod
    def evaluate_tree_with_estimator(
        expectation_tree: Union[OpTreeNodeBase, OpTreeExpectationValue],
        dictionary: Union[List[dict], dict],
        estimator: Union[BaseEstimatorV1, BaseEstimatorV2],
        detect_duplicates: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Evaluate a expectation tree with an estimator.

        The OpTree can only contain expectation values, and the estimator is used to evaluate the
        expectation values.
        Dictionary can be a list of dictionaries, in which case the function evaluates the
        expectation value for all combinations of the dictionaries.
        The function also checks if the same expectation value occurs multiple times, and only adds
        it once to the evaluation list. This can be turned off with the ``detect_duplicates`` flag.

        Args:
            expectation_tree (Union[OpTreeNodeBase,OpTreeExpectationValue]): The expectation OpTree
                to be evaluated.
            dictionary (Union[List(dict),dict]): The dictionary that contains the parameter and
                their values. Can be list for the evaluation of multiple dictionaries.
            estimator (Union[BaseEstimatorV1,BaseEstimatorV2]): The estimator primitive that is
                used for the evaluation.
            detect_expectation_duplicates (bool, optional): If True, duplicate expectation values
                are detected and only evaluated once. Defaults to True.

        Returns:
            The expectation value of the expectation OpTree as a numpy array.
        """

        start = time.time()

        # Preprocess the dictionary
        multiple_dict = True
        if not isinstance(dictionary, list):
            dictionary = [dictionary]
            multiple_dict = False

        total_circuit_list = []
        total_operator_list = []
        total_parameter_list = []
        total_tree_list = []
        for dict_ in dictionary:
            # Loop over all dictionaries and sum up the lists from the expecation tree
            (
                circuit_list,
                operator_list,
                parameter_list,
                _,
                index_tree,
            ) = _build_expectation_list(expectation_tree, dict_, detect_duplicates, False)

            total_tree_list.append(_add_offset_to_tree(index_tree, len(total_circuit_list)))
            total_circuit_list += [
                (
                    circuit
                    if circuit.num_clbits == 0
                    else circuit.remove_final_measurements(inplace=False)
                )
                for circuit in circuit_list
            ]
            total_operator_list += operator_list
            total_parameter_list += parameter_list

        if multiple_dict:
            evaluation_tree = OpTreeList(total_tree_list)
        else:
            evaluation_tree = total_tree_list[0]
        # print("Pre-processing: ", time.time() - start)

        # Evaluation via the estimator
        start = time.time()
        # print("Number of circuits for estimator: ", len(total_circuit_list))
        if len(total_circuit_list) == 0:
            return _evaluate_index_tree(evaluation_tree, [])

        if isinstance(estimator, BaseEstimatorV1):
            estimator_result = (
                estimator.run(total_circuit_list, total_operator_list, total_parameter_list)
                .result()
                .values
            )
        elif isinstance(estimator, BaseEstimatorV2):
            pubs = list(zip(total_circuit_list, total_operator_list, total_parameter_list))
            estimator_result = np.array([r.data.evs for r in estimator.run(pubs).result()])
        else:
            raise ValueError("Unknown estimator type!")
        # print("Run time of estimator: ", time.time() - start)

        # Final assembly of the results
        start = time.time()
        final_result = _evaluate_index_tree(evaluation_tree, estimator_result)
        # print("Post-processing: ", time.time() - start)
        return final_result

    @staticmethod
    def evaluate_tree_with_sampler(
        expectation_tree: Union[OpTreeNodeBase, OpTreeExpectationValue],
        dictionary: Union[List[dict], dict],
        sampler: Union[BaseSamplerV1, BaseSamplerV2],
        detect_duplicates: bool = True,
    ):
        """
        Evaluate a expectation tree with a sampler.

        The OpTree can only contain expectation values, and the sampler is used to evaluate the
        expectation values.
        Dictionary can be a list of dictionaries, in which case the function evaluates the
        expectation value for all combinations of the dictionaries.
        The function also checks if the same circuit or expectation value occurs multiple times,
        and only adds it once to the evaluation list.
        This can be turned off with the ``detect_duplicates`` flag.

        Args:
            expectation_tree (Union[OpTreeNodeBase, OpTreeExpectationValue]): The expectation
                OpTree to be evaluated.
            dictionary (Union[List(dict),dict]): The dictionary that contains the parameter and
                their values. Can be list for the evaluation of multiple dictionaries.
            sampler (Union[BaseSamplerV1, BaseSamplerV2]): The sampler primitive that is used
                for the evaluation.
            detect_expectation_duplicates (bool, optional): If True, duplicate expectation values
                and circuits are detected and only evaluated once.

        Returns:
            The expectation value of the expectation OpTree as a numpy array.
        """

        # Preprocess the dictionary
        multiple_dict = True
        if not isinstance(dictionary, list):
            dictionary = [dictionary]
            multiple_dict = False

        # Build circuit and parameter list
        total_circuit_list = []
        total_operator_list = []
        total_parameter_list = []
        total_circuit_operator_list = []
        total_tree_list = []
        for dict_ in dictionary:
            start = time.time()
            # convert tree to lists
            (
                circuit_list,
                operator_list,
                parameter_list,
                circuit_operator_list,
                index_tree,
            ) = _build_expectation_list(
                expectation_tree,
                dict_,
                detect_expectation_duplicates=detect_duplicates,
                group_circuits=detect_duplicates,
            )

            # Add offset to the index tree for multiple dictionaries
            offset = len(total_operator_list)
            total_tree_list.append(_add_offset_to_tree(index_tree, offset))
            # index management for expectation value evaluation of redundant circuits
            total_circuit_operator_list += [
                [iop + offset for iop in icirc] for icirc in circuit_operator_list
            ]
            total_circuit_list += [_measure_all_unmeasured(circuit) for circuit in circuit_list]
            total_operator_list += operator_list
            total_parameter_list += parameter_list

        # Final assembly of the index tree for the evaluation
        if multiple_dict:
            evaluation_tree = OpTreeList(total_tree_list)
        else:
            evaluation_tree = total_tree_list[0]

        # Evaluation via the sampler
        start = time.time()
        # print("Number of circuits for sampler: ", len(total_circuit_list))
        if len(total_circuit_list) == 0:
            _evaluate_index_tree(evaluation_tree, [])

        if isinstance(sampler, BaseSamplerV1):
            sampler_result = sampler.run(total_circuit_list, total_parameter_list).result()
        elif isinstance(sampler, BaseSamplerV2):
            pubs = list(zip(total_circuit_list, total_parameter_list))
            sampler_result = [result.data.meas for result in sampler.run(pubs).result()]
        else:
            raise ValueError("Unknown sampler type!")
        # print("Sampler run time: ", time.time() - start)

        # Computation of the expectation values from the sampler results
        start = time.time()
        expec = _evaluate_expectation_from_sampler(
            total_operator_list,
            sampler_result,
            operator_measurement_list=total_circuit_operator_list,
        )

        # Final assembly of the results
        result = _evaluate_index_tree(evaluation_tree, expec)
        # print("Post-processing: ", time.time() - start)

        return result

    @staticmethod
    def transform_to_zbasis(
        optree_element: Union[
            OpTreeNodeBase, OpTreeOperator, OpTreeExpectationValue, SparsePauliOp
        ],
        abelian_grouping: bool = True,
    ):
        """
        Function for transforming an OpTree structure to the Z basis.

        This can be applied to OpTree structures that contain operators or expectation values.
        The function transforms the operators to the Z basis by adding measurement circuits.

        Args:
            optree_element (Union[OpTreeNodeBase, OpTreeLeafOperator, OpTreeLeafExpectationValue, SparsePauliOp]):
                The OpTree structure to be transformed.
            abelian_grouping (bool, optional): If True, the operator is grouped into commuting terms.
                Defaults to True.

        Returns:
            The transformed OpTree structure.
        """
        if isinstance(optree_element, OpTreeNodeBase):
            # Recursive call for all children
            children_list = [
                OpTreeEvaluate.transform_to_zbasis(child, abelian_grouping)
                for child in optree_element.children
            ]
            if isinstance(optree_element, OpTreeSum):
                return OpTreeSum(children_list, optree_element.factor, optree_element.operation)
            elif isinstance(optree_element, OpTreeList):
                return OpTreeList(children_list, optree_element.factor, optree_element.operation)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")
        elif isinstance(optree_element, OpTreeOperator):
            return _transform_operator_to_zbasis(optree_element.operator, abelian_grouping)
        elif isinstance(optree_element, SparsePauliOp):
            return _transform_operator_to_zbasis(optree_element, abelian_grouping)
        elif isinstance(optree_element, OpTreeExpectationValue):
            operator_in_zbasis = _transform_operator_to_zbasis(optree_element.operator)
            from .optree import OpTree

            return OpTree.gen_expectation_tree(optree_element.circuit, operator_in_zbasis)
        else:
            raise ValueError("Wrong type of Optree Element:", type(optree_element))
