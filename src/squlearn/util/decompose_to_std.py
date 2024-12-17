from qiskit.circuit.library import standard_gates
from qiskit.circuit import QuantumCircuit


def decompose_to_std(circuit: QuantumCircuit, gate_list: list = []) -> QuantumCircuit:
    """
    Function to decompose the circuit to standard gates.

    Args:
        circuit (QuantumCiruit): The Circuit, which is supposed to be decomposed.
        gate_list (list): List of gates, which are considered as standard gates. If the gate is not in the list, it will be decomposed. If the list is empty, all gates from qiskit.circuit.library.standard_gates and ["cx","cy","cz","measure"] are considered as standard gates.

    Returns:
        QuantumCiruit: The decomposed circuit.
    """
    if len(gate_list) == 0:
        gate_list = [*dir(standard_gates), "cx", "cy", "cz", "measure"]
    decompose_names = [
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name not in gate_list
    ]
    circuit_new = circuit.decompose(decompose_names)

    while len(decompose_names) > 0 and circuit != circuit_new:
        circuit = circuit_new
        decompose_names = [
            instruction.operation.name
            for instruction in circuit.data
            if instruction.operation.name not in gate_list
        ]
        circuit_new = circuit.decompose(decompose_names)

    return circuit_new
