from qiskit.circuit.library import standard_gates
from typing import Union

from qc_executor import QuantumCircuit


def decompose_to_std(
    circuit: QuantumCircuit, gate_list: Union[None, list] = None
) -> QuantumCircuit:
    """
    Function to decompose the circuit to standard gates.

    Args:
        circuit (QuantumCircuit): The qc_executor circuit, which is supposed to be decomposed.
        gate_list (Union[None,list]): List of gates, which are considered as standard gates.
            If the gate is not in the list, it will be decomposed. If the list is empty,
            all gates from `qiskit.circuit.library.standard_gates` and
            `["cx","cy","cz","measure"]` are considered as standard gates.

    Returns:
        QuantumCircuit: The decomposed circuit, as a qc_executor circuit.
    """
    if not gate_list:
        gate_list = [*dir(standard_gates), "cx", "cy", "cz", "measure"]

    qiskit_circuit = circuit.qiskit_circuit
    decompose_names = [
        instruction.operation.name
        for instruction in qiskit_circuit.data
        if instruction.operation.name not in gate_list
    ]
    qiskit_circuit_new = qiskit_circuit.decompose(decompose_names)

    while decompose_names and qiskit_circuit != qiskit_circuit_new:
        qiskit_circuit = qiskit_circuit_new
        decompose_names = [
            instruction.operation.name
            for instruction in qiskit_circuit.data
            if instruction.operation.name not in gate_list
        ]
        qiskit_circuit_new = qiskit_circuit.decompose(decompose_names)

    return QuantumCircuit.from_qiskit(qiskit_circuit_new)
