import numpy as np
from qiskit.circuit import ParameterVector
from qiskit_algorithms.gradients import LinCombQGT, QFI

import pennylane as qml
import pennylane.numpy as pnp

from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from .executor import Executor, BaseEstimatorV2
from .data_preprocessing import adjust_features, adjust_parameters
from .pennylane import PennyLaneCircuit


def get_quantum_fisher(
    encoding_circuit: EncodingCircuitBase,
    x: np.ndarray,
    p: np.ndarray,
    executor: Executor,
    mode: str = "p",
):
    """
    Function for evaluating the Quantum Fisher Information Matrix of a encoding circuit.

    The Quantum Fisher Information Matrix (QFIM) is evaluated the supplied numerical
    features and parameter value.

    Mode enables the user to choose between different modes of evaluation:
    * ``"p"`` : QFIM for parameters only
    * ``"x"`` : QFIM for features only
    * ``"px"`` : QFIM for parameters and features (order parameters first)

    In case of multiple inputs for ``x`` and ``p``, the QFIM is evaluated for each input separately
    and returned as a numpy matrix.

    Args:
        encoding_circuit (EncodingCircuitBase): Encoding circuit for which the QFIM is evaluated
        x (np.ndarray): Input data values for replacing the features in the encoding circuit
        p (np.ndarray): Parameter values for replacing the parameters in the encoding circuit
        executor (Executor): Executor for evaluating the QFIM (utilizes estimator)
        mode (str): Mode for evaluating the QFIM, possibilities: ``"p"``, ``"x"``,
                    ``"px"`` (default: ``"p"``)

    Return:
        Numpy matrix with the QFIM, in case of multiple inputs, the array is nested.
    """

    if executor.quantum_framework == "qiskit":
        return _get_quantum_fisher_qiskit(encoding_circuit, x, p, executor, mode)
    else:
        return _get_quantum_fisher_pennylane(encoding_circuit, x, p, executor, mode)


def _get_quantum_fisher_qiskit(
    encoding_circuit: EncodingCircuitBase,
    x: np.ndarray,
    p: np.ndarray,
    executor: Executor,
    mode: str = "p",
):
    """
    Qiskit implementation of the Quantum Fisher Information Matrix of a encoding circuit.

    The Quantum Fisher Information Matrix (QFIM) is evaluated the supplied numerical
    features and parameter value.

    Mode enables the user to choose between different modes of evaluation:
    * ``"p"`` : QFIM for parameters only
    * ``"x"`` : QFIM for features only
    * ``"px"`` : QFIM for parameters and features (order parameters first)

    In case of multiple inputs for ``x`` and ``p``, the QFIM is evaluated for each input separately
    and returned as a numpy matrix.

    Args:
        encoding_circuit (EncodingCircuitBase): Encoding circuit for which the QFIM is evaluated
        x (np.ndarray): Input data values for replacing the features in the encoding circuit
        p (np.ndarray): Parameter values for replacing the parameters in the encoding circuit
        executor (Executor): Executor for evaluating the QFIM (utilizes estimator)
        mode (str): Mode for evaluating the QFIM, possibilities: ``"p"``, ``"x"``,
                    ``"px"``, (default: ``"p"``)

    Return:
        Numpy matrix with the QFIM, in case of multiple inputs, the array is nested.
    """
    estimator = executor.get_estimator()
    if isinstance(estimator, BaseEstimatorV2):
        raise ValueError(
            "Incompatible Qiskit version for QFI calculation with Qiskit Algorithms. "
            "Please downgrade to Qiskit 1.0 or consider using PennyLane."
        )

    # Get Qiskit QFI primitive
    qfi = QFI(LinCombQGT(estimator))

    p_ = ParameterVector("p", encoding_circuit.num_parameters)
    x_ = ParameterVector("x", encoding_circuit.num_features)
    circuit = encoding_circuit.get_circuit(x_, p_)

    # Adjust input
    x_list, multi_x = adjust_features(x, encoding_circuit.num_features)
    p_list, multi_p = adjust_parameters(p, encoding_circuit.num_parameters)

    circ_list = []
    param_values_list = []
    param_list = []
    if mode == "p":
        for xval in x_list:
            circ_temp = circuit.assign_parameters(dict(zip(x_, xval)))
            for pval in p_list:
                circ_list.append(circ_temp)
                param_values_list.append(pval)
                param_list.append(p_)
    elif mode == "x":
        for xval in x_list:
            for pval in p_list:
                circ_list.append(circuit.assign_parameters(dict(zip(p_, pval))))
                param_values_list.append(xval)
                param_list.append(x_)
    elif mode == "px":
        for xval in x_list:
            for pval in p_list:
                circ_list.append(circuit)
                param_values_list.append(np.concatenate((pval, xval)))
                param_list.append(list(p_) + list(x_))
    else:
        raise ValueError("Invalid mode for QFI evaluation.")

    # Evaluate QFIM with Qiskit Primitive
    qfis = np.array(qfi.run(circ_list, param_values_list, param_list).result().qfis)

    # Reformating in case of multiple inputs
    reshape_list = []
    if multi_x:
        reshape_list.append(len(x_list))
    if multi_p:
        reshape_list.append(len(p_list))

    if len(reshape_list) > 0:
        qfis = qfis.reshape(reshape_list + list(qfis[0].shape))
    else:
        qfis = qfis[0]

    executor.clear_estimator_cache()
    return qfis


def _get_quantum_fisher_pennylane(
    encoding_circuit: EncodingCircuitBase,
    x: np.ndarray,
    p: np.ndarray,
    executor: Executor,
    mode: str = "p",
):
    """
    PennyLane implementation of the Quantum Fisher Information Matrix of a encoding circuit.

    The Quantum Fisher Information Matrix (QFIM) is evaluated the supplied numerical
    features and parameter value.

    Mode enables the user to choose between different modes of evaluation:
    * ``"p"`` : QFIM for parameters only
    * ``"x"`` : QFIM for features only
    * ``"px"`` : QFIM for parameters and features (order parameters first)

    In case of multiple inputs for ``x`` and ``p``, the QFIM is evaluated for each input
    separately and returned as a numpy matrix.

    Args:
        encoding_circuit (EncodingCircuitBase): Encoding circuit for which the QFIM is evaluated
        x (np.ndarray): Input data values for replacing the features in the encoding circuit
        p (np.ndarray): Parameter values for replacing the parameters in the encoding circuit
        executor (Executor): Executor for evaluating the QFIM (utilizes estimator)
        mode (str): Mode for evaluating the QFIM, possibilities: ``"p"``, ``"x"``,
                    ``"px"`` (default: ``"p"``)

    Return:
        Numpy matrix with the QFIM, in case of multiple inputs, the array is nested.
    """

    parameter_vector = ParameterVector("p", encoding_circuit.num_parameters)
    feature_vector = ParameterVector("x", encoding_circuit.num_features)
    circuit = encoding_circuit.get_circuit(feature_vector, parameter_vector)

    # Adjust input
    x_adjusted, multi_x = adjust_features(x, encoding_circuit.num_features)
    p_adjusted, multi_p = adjust_parameters(p, encoding_circuit.num_parameters)

    fisher_list = []
    if mode == "p":
        pennylane_circuit = PennyLaneCircuit(circuit, "probs", executor)
        fisher_func = qml.metric_tensor(pennylane_circuit.pennylane_circuit)
        for x_values in x_adjusted:
            x_values = pnp.array(x_values, requires_grad=False)
            for p_values in p_adjusted:
                p_values = pnp.array(p_values, requires_grad=True)
                # pylint: disable=not-callable
                fisher_list.append(4.0 * np.array(fisher_func(p_values, x_values)))

    elif mode == "x":
        pennylane_circuit = PennyLaneCircuit(circuit, "probs", executor)
        fisher_func = qml.metric_tensor(pennylane_circuit.pennylane_circuit)
        for x_values in x_adjusted:
            x_values = pnp.array(x_values, requires_grad=True)
            for p_values in p_adjusted:
                p_values = pnp.array(p_values, requires_grad=False)
                # pylint: disable=not-callable
                fisher_list.append(4.0 * np.array(fisher_func(p_values, x_values)))

    elif mode == "px":
        px_ = ParameterVector(
            "px", encoding_circuit.num_parameters + encoding_circuit.num_features
        )
        dictionary = dict(zip(list(parameter_vector) + list(feature_vector), list(px_)))
        circuit.assign_parameters(dictionary, inplace=True)
        pennylane_circuit = PennyLaneCircuit(circuit, "probs", executor)
        fisher_func = qml.metric_tensor(pennylane_circuit.pennylane_circuit)
        for x_values in x_adjusted:
            for p_values in p_adjusted:
                x_values = pnp.array(np.concatenate((p_values, x_values)), requires_grad=True)
                # pylint: disable=not-callable
                fisher_list.append(4.0 * np.array(fisher_func(x_values)))
    else:
        raise ValueError("Invalid mode for QFI evaluation.")

    fisher_list = np.array(fisher_list)

    # Reformating in case of multiple inputs
    reshape_list = []
    if multi_x:
        reshape_list.append(len(x_adjusted))
    if multi_p:
        reshape_list.append(len(p_adjusted))

    if len(reshape_list) > 0:
        fisher_list = fisher_list.reshape(reshape_list + list(fisher_list[0].shape))
    else:
        fisher_list = fisher_list[0]

    return fisher_list
