import numpy as np
from typing import Union, List

from qiskit.circuit.parametervector import ParameterVectorElement

from qulacs import QuantumState, CausalConeSimulator

from .qulacs_circuit import QulacsCircuit


def qulacs_evaluate(circuit: QulacsCircuit, **kwargs) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Computes the expectation values of the observables defined in the circuit data structure.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        kwargs: Parameters to evaluate the circuit given as keyword arguments

    Returns:
        np.ndarray: Result of the evaluation
    """

    obs_param_list = sum([list(kwargs[param]) for param in circuit.observable_parameter_names], [])

    circ = circuit.get_circuit_func()(
        *[kwargs[param] for param in circuit.circuit_parameter_names]
    )
    state = QuantumState(circuit.num_qubits)
    circ.update_quantum_state(state)

    operators = circuit.get_observable_func()(*obs_param_list)

    param_values = np.array([o.get_expectation_value(state) for o in operators])

    values = np.real_if_close(param_values)

    if not circuit.multiple_observables:
        return values[0]

    return values


def qulacs_evaluate_statevector(circuit: QulacsCircuit, **kwargs) -> np.ndarray:
    """
    Function to evaluate the statevector of the Qulacs circuit with the given parameters.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        kwargs: Parameters to evaluate the circuit given as keyword arguments

    Returns:
        np.ndarray: Statevector solution of the circuit
    """

    circ = circuit.get_circuit_func()(
        *[kwargs[param] for param in circuit.circuit_parameter_names]
    )
    state = QuantumState(circuit.num_qubits)
    circ.update_quantum_state(state)

    return state.get_vector()


def qulacs_evaluate_probabilities(circuit: QulacsCircuit, **kwargs) -> np.ndarray:
    """
    Function to evaluate the probabilities of the Qulacs circuit with the given parameters.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        kwargs: Parameters to evaluate the circuit given as keyword arguments

    Returns:
        np.ndarray: Probabilites of the circuit
    """

    # Collects the args values connected to the observable parameters
    circ = circuit.get_circuit_func()(
        *[kwargs[param] for param in circuit.circuit_parameter_names]
    )
    state = QuantumState(circuit.num_qubits)
    circ.update_quantum_state(state)

    return np.square(np.abs(state.get_vector()))


def qulacs_gradient(
    circuit: QulacsCircuit,
    parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Computes the gradient of the expectation values of the observables defined in the
    circuit data structure.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        parameters (Union[None, ParameterVectorElement, List[ParameterVectorElement]]): Parameters
            that are differentiated in the circuit. If None, no differentiation is performed.
        kwargs: Parameters to evaluate the circuit given as keyword arguments

    Returns:
        np.ndarray: Result of the evaluation
    """

    obs_param_list = sum([list(kwargs[param]) for param in circuit.observable_parameter_names], [])

    qulacs_circuit = circuit.get_circuit_func(parameters)(
        *[kwargs[param] for param in circuit.circuit_parameter_names]
    )

    outer_jacobian = circuit.get_outer_jacobian_circuit(parameters)(
        *[kwargs[param] for param in circuit.circuit_parameter_names]
    )
    operators = circuit.get_observable_func()(*obs_param_list)

    if isinstance(parameters, ParameterVectorElement):
        parameters = [parameters]
    parameters = list(parameters) if parameters is not None else []

    is_parameterized = len(parameters)

    if is_parameterized:
        param_values = np.array(
            [outer_jacobian.T @ np.array(qulacs_circuit.backprop(o)) for o in operators]
        )
    else:
        param_values = np.array([[]])

    values = np.real_if_close(param_values)

    if not circuit.multiple_observables:
        return values[0]

    return values


def qulacs_operator_gradient(
    circuit: QulacsCircuit,
    parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Computes the gradient of the expectation values of the observables defined in the
    circuit data structure.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        parameters (Union[None, ParameterVectorElement, List[ParameterVectorElement]]): Parameters
            that are differentiated in the observable. If None, no differentiation is performed.
        kwargs: Parameters to evaluate the circuit given as keyword arguments

    Returns:
        np.ndarray: Result of the evaluation
    """

    obs_param_list = [kwargs[param] for param in circuit.observable_parameter_names]
    outer_jacobian = circuit.get_outer_jacobian_observables(parameters)(*obs_param_list)

    circ = circuit.get_circuit_func()(
        *[kwargs[param] for param in circuit.circuit_parameter_names]
    )
    state = QuantumState(circuit.num_qubits)
    circ.update_quantum_state(state)
    operators = circuit.get_observables_for_gradient(parameters)()

    param_obs_values = [
        outer_jacobian[i].T
        @ np.array(
            [o if isinstance(o, float) else o.get_expectation_value(state) for o in operator]
        )
        for i, operator in enumerate(operators)
    ]

    values = np.real_if_close(param_obs_values)

    if not circuit.multiple_observables:
        return values[0]

    return values


def qulacs_evaluate_causalcone(circuit: QulacsCircuit, **kwargs) -> np.ndarray:
    """
    Function to evaluate the Qulacs circuit with the given parameters.

    Function to evaluate the expectation values of the observables defined in the
    circuit data structure using the Causal Cone Simulator.

    Args:
        circuit (QulacsCircuit): Qulacs circuit to evaluate
        kwargs: Parameters to evaluate the circuit given as keyword arguments

    Returns:
        np.ndarray: Result of the evaluation
    """

    obs_param_list = sum([list(kwargs[param]) for param in circuit.observable_parameter_names], [])

    circ = circuit.get_circuit_func()(
        *[kwargs[param] for param in circuit.circuit_parameter_names]
    )

    operators = circuit.get_observable_func()(*obs_param_list)

    param_values = np.array(
        [CausalConeSimulator(circ, o).get_expectation_value() for o in operators]
    )

    values = np.real_if_close(param_values)

    if not circuit.multiple_observables:
        return values[0]

    return values
