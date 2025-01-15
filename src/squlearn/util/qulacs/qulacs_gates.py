
from qulacs import ParametricQuantumCircuit

def qulacs_gate_i(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Identity gate."""
    pass # TODO: Check if this is correct

def qulacs_gate_h(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Hadamard gate."""
    circuit.add_H_gate(qubit)

def qulacs_gate_x(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Pauli-X gate."""
    circuit.add_X_gate(qubit)

def qulacs_gate_y(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Pauli-Y gate."""
    circuit.add_Y_gate(qubit)

def qulacs_gate_z(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Pauli-Z gate."""
    circuit.add_Z_gate(qubit)

def qulacs_gate_s(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """S gate."""
    circuit.add_S_gate(qubit)

def qulacs_gate_t(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """T gate."""
    circuit.add_T_gate(qubit)

# def RXX(theta, wires):
#     """RXX gate."""
#     return qml.PauliRot(theta, "XX", wires=wires)


# def RYY(theta, wires):
#     """RYY gate."""
#     return qml.PauliRot(theta, "YY", wires=wires)


# def RZZ(theta, wires):
#     """RZZ gate."""
#     return qml.PauliRot(theta, "ZZ", wires=wires)


# def RXZ(theta, wires):
#     """RXZ gate."""
#     return qml.PauliRot(theta, "XZ", wires=wires)


# def reset(wires):
#     """Reset gate, implemented by measure and reset."""
#     return qml.measure(wires=wires, reset=True)


# def tdg(wires):
#     """T-dagger gate."""
#     return qml.adjoint(qml.T(wires=wires))


# def sdg(wires):
#     """S-dagger gate."""
#     return qml.adjoint(qml.S(wires=wires))


# def cs(wires):
#     """CS gate."""
#     if len(wires) != 2:
#         raise ValueError("CS gate requires two wires.")
#     return qml.ctrl(qml.S(wires[1]), wires[0])


# def csx(wires):
#     """CSX gate."""
#     if len(wires) != 2:
#         raise ValueError("CSX gate requires two wires.")
#     return qml.ctrl(qml.SX(wires[1]), wires[0])

# Dictionary of conversion Qiskit gates (from string) to Quilacs gate functions
qiskit_qulacs_gate_dict = {
    "i": qulacs_gate_i,
    "h": qulacs_gate_h,
    "x": qulacs_gate_x,
    "y": qulacs_gate_y,
    "z": qulacs_gate_z,
    "s": qulacs_gate_s,
    "t": qulacs_gate_t,
}


# qiskit_qulacs_gate_dict = {
#     "i": qml.Identity,
#     "h": qml.Hadamard,
#     "x": qml.PauliX,
#     "y": qml.PauliY,
#     "z": qml.PauliZ,
#     "s": qml.S,
#     "t": qml.T,
#     "toffoli": qml.Toffoli,
#     "sx": qml.SX,
#     "swap": qml.SWAP,
#     "iswap": qml.ISWAP,
#     "cswap": qml.CSWAP,
#     "ecr": qml.ECR,
#     "ch": qml.CH,
#     "rx": qml.RX,
#     "ry": qml.RY,
#     "rz": qml.RZ,
#     "p": qml.PhaseShift,
#     "cp": qml.ControlledPhaseShift,
#     "cx": qml.CNOT,
#     "cnot": qml.CNOT,
#     "cy": qml.CY,
#     "cz": qml.CZ,
#     "crx": qml.CRX,
#     "cry": qml.CRY,
#     "crz": qml.CRZ,
#     "rxx": RXX,
#     "ryy": RYY,
#     "rzz": RZZ,
#     "rxz": RXZ,
#     "barrier": qml.Barrier,
#     "u": qml.U3,
#     "measure": qml.measure,
#     "reset": reset,
#     "tdg": tdg,
#     "sdg": sdg,
#     "cs": cs,
#     "csx": csx,
# }
