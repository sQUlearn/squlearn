from qiskit.circuit.library import standard_gates
from qiskit.circuit import QuantumCircuit
from typing import Union


def decompose_to_std(
    circuit: QuantumCircuit, gate_list: Union[None, list] = None
) -> QuantumCircuit:
    """
    Function to decompose the circuit to standard gates.

    Args:
        circuit (QuantumCircuit): The Circuit, which is supposed to be decomposed.
        gate_list (Union[None,list]): List of gates, which are considered as standard gates.
            If the gate is not in the list, it will be decomposed. If the list is empty,
            all gates from `qiskit.circuit.library.standard_gates` and
            `["cx","cy","cz","measure"]` are considered as standard gates.

    Returns:
        QuantumCircuit: The decomposed circuit.
    """
    if not gate_list:
        gate_list = [*dir(standard_gates), "cx", "cy", "cz", "measure"]
    decompose_names = [
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name not in gate_list
    ]
    circuit_new = circuit.decompose(decompose_names)

    while decompose_names and circuit != circuit_new:
        circuit = circuit_new
        decompose_names = [
            instruction.operation.name
            for instruction in circuit.data
            if instruction.operation.name not in gate_list
        ]
        circuit_new = circuit.decompose(decompose_names)

    return circuit_new
