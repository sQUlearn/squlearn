from qiskit.transpiler import Target
from qulacs import ParametricQuantumCircuit
from qulacs.gate import Identity, H, X, Y, Z, S, Sdag, T, Tdag, SWAP, CZ, RX, RY, RZ, U1


def qulacs_gate_i(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Identity gate."""
    circuit.add_gate(Identity(qubit))


def qulacs_gate_h(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Hadamard gate."""
    circuit.add_gate(H(qubit))


def qulacs_gate_x(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Pauli-X gate."""
    circuit.add_gate(X(qubit))


def qulacs_gate_y(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Pauli-Y gate."""
    circuit.add_gate(Y(qubit))


def qulacs_gate_z(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Pauli-Z gate."""
    circuit.add_gate(Z(qubit))


def qulacs_gate_s(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """S gate."""
    circuit.add_gate(S(qubit))


def qulacs_gate_sdg(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """S-dagger gate."""
    circuit.add_gate(Sdag(qubit))


def qulacs_gate_t(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """T gate."""
    circuit.add_gate(T(qubit))


def qulacs_gate_tdg(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """T-dagger gate."""
    circuit.add_gate(Tdag(qubit))


def qulacs_gate_swap(circuit: ParametricQuantumCircuit, qubit1: int, qubit2: int) -> None:
    """SWAP gate."""
    circuit.add_gate(SWAP(qubit1, qubit2))


def qulacs_gate_cnot(circuit: ParametricQuantumCircuit, control: int, target: int) -> None:
    """CNOT or CX gate."""
    circuit.add_CNOT_gate(control, target)


# CY Gate is not available in Qulacs, so we do not define it.


def qulacs_gate_cz(circuit: ParametricQuantumCircuit, control: int, target: int) -> None:
    """CZ gate."""
    circuit.add_gate(CZ(control, target))


def qulacs_gate_rx(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """RX gate."""
    circuit.add_gate(RX(index, angle))


def qulacs_param_gate_rx(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """Parameterized RX gate."""
    circuit.add_parametric_RX_gate(index, angle)


def qulacs_gate_ry(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """RY gate."""
    circuit.add_gate(RY(index, angle))


def qulacs_param_gate_ry(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """Parameterized RY gate."""
    circuit.add_parametric_RY_gate(index, angle)


def qulacs_gate_rz(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """RZ gate."""
    circuit.add_gate(RZ(index, angle))


def qulacs_param_gate_rz(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """Parameterized RZ gate."""
    circuit.add_parametric_RZ_gate(index, angle)


def qulacs_gate_U1(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """U1 gate."""
    circuit.add_gate(U1(index, angle))


# Dictionary of conversion Qiskit gates (from string) to Quilacs gate functions
qiskit_qulacs_gate_dict = {
    "h": qulacs_gate_h,
    "x": qulacs_gate_x,
    "y": qulacs_gate_y,
    "z": qulacs_gate_z,
    "s": qulacs_gate_s,
    "t": qulacs_gate_t,
    "swap": qulacs_gate_swap,
    "cx": qulacs_gate_cnot,
    # "cy" does not exist in Qulacs
    "cz": qulacs_gate_cz,
    "sdg": qulacs_gate_sdg,
    "tdg": qulacs_gate_tdg,
    "rx": qulacs_gate_rx,
    "ry": qulacs_gate_ry,
    "rz": qulacs_gate_rz,
}

qiskit_qulacs_param_gate_dict = {
    "rx": qulacs_param_gate_rx,
    "ry": qulacs_param_gate_ry,
    "rz": qulacs_param_gate_rz,
}

custom_name_mapping = {
    "i": qulacs_gate_i,
    "cnot": qulacs_gate_cnot,
}

qiskit_qulacs_target = Target.from_configuration(
    basis_gates=qiskit_qulacs_gate_dict.keys(), custom_name_mapping=custom_name_mapping
)
