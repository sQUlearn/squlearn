
from qulacs import ParametricQuantumCircuit

from qulacs.gate import *

def qulacs_gate_i(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    # print("Adding I gate on qubit", qubit)
    circuit.add_gate(Identity(qubit))

def qulacs_gate_h(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Hadamard gate."""
    # print("Adding H gate on qubit", qubit)
    circuit.add_gate(H(qubit))

def qulacs_gate_x(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Pauli-X gate."""
    # print("Adding X gate on qubit", qubit)
    circuit.add_gate(X(qubit))

def qulacs_gate_y(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Pauli-Y gate."""
    # print("Adding Y gate on qubit", qubit)
    circuit.add_gate(Y(qubit))

def qulacs_gate_z(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """Pauli-Z gate."""
    # print("Adding Z gate on qubit", qubit)
    circuit.add_gate(Z(qubit))

def qulacs_gate_s(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """S gate."""
    # print("Adding S gate on qubit", qubit)
    circuit.add_gate(S(qubit))

def qulacs_gate_sdg(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """S-dagger gate."""
    # print("Adding S-dagger gate on qubit", qubit)
    circuit.add_gate(Sdag(qubit))

def qulacs_gate_t(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """T gate."""
    # print("Adding T gate on qubit", qubit)
    circuit.add_gate(T(qubit))

def qulacs_gate_tdg(circuit: ParametricQuantumCircuit, qubit: int) -> None:
    """T-dagger gate."""
    # print("Adding T-dagger gate on qubit", qubit)
    circuit.add_gate(Tdag(qubit))

def qulacs_gate_swap(circuit: ParametricQuantumCircuit, qubit1: int, qubit2: int) -> None:
    """SWAP gate."""
    # print("Adding SWAP gate between qubits", qubit1, "and", qubit2)
    circuit.add_gate(SWAP(qubit1, qubit2))

def qulacs_gate_cnot(circuit: ParametricQuantumCircuit, control: int, target: int) -> None:
    """CNOT gate."""
    # print("Adding CNOT gate with control qubit", control, "and target qubit", target)
    #circuit.add_gate(CNOT(control, target))
    circuit.add_CNOT_gate(control, target)

def qulacs_gate_cz(circuit: ParametricQuantumCircuit, control: int, target: int) -> None:
    """CZ gate."""
    # print("Adding CZ gate with control qubit", control, "and target qubit", target)
    circuit.add_gate(CZ(control, target))

def qulacs_gate_rx(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """RX gate."""
    # print("Adding RX gate with angle", angle, "on qubit", index)
    circuit.add_gate(RX(index, angle))

def qulacs_param_gate_rx(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """Parameterized RX gate."""
    # print("Adding Parameterized RX gate with angle", angle, "on qubit", index)
    #circuit.add_gate(ParametricRX(index, angle))
    circuit.add_parametric_RX_gate(index, angle)

def qulacs_gate_ry(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """RY gate."""
    # print("Adding RY gate with angle", angle, "on qubit", index)
    circuit.add_gate(RY(index, angle))

def qulacs_param_gate_ry(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """Parameterized RY gate."""
    # print("Adding Parameterized RY gate with angle", angle, "on qubit", index)
    #circuit.add_gate(ParametricRY(index, angle))
    circuit.add_parametric_RY_gate(index, angle)

def qulacs_gate_rz(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """RZ gate."""
    # print("Adding RZ gate with angle", angle, "on qubit", index)
    circuit.add_gate(RZ(index, angle))

def qulacs_param_gate_rz(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """Parameterized RZ gate."""
    # print("Adding Parameterized RZ gate with angle", angle, "on qubit", index)
    #circuit.add_gate(ParametricRZ(index, angle))
    circuit.add_parametric_RZ_gate(index, angle)

# def qulacs_gate_cy(circuit: ParametricQuantumCircuit, control: int, target: int) -> None:
#     """CY gate."""
#     circuit.add_gate(CY(control, target)) # NOT IMPLEMENTED

def qulacs_gate_cp(circuit: ParametricQuantumCircuit, control: int, target: int) -> None:
    """CP gate."""
    # print("Adding CP gate with control qubit", control, "and target qubit", target)
    circuit.add_gate(CP(control, target))

def qulacs_gate_U1(circuit: ParametricQuantumCircuit, angle: float, index: int) -> None:
    """U1 gate."""
    # print("Adding U1 gate with angle", angle, "on qubit", index)
    circuit.add_gate(U1(index, angle))

# TODO
# Toffoli gate



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
    "swap": qulacs_gate_swap,
    "cx": qulacs_gate_cnot,
    "cnot": qulacs_gate_cnot,
    #"cy" does not exist in Qulacs
    "cz": qulacs_gate_cz,
    "sdg": qulacs_gate_sdg,
    "tdg": qulacs_gate_tdg,
    "cp": qulacs_gate_cp,
    "rx": qulacs_gate_rx,
    "ry": qulacs_gate_ry,
    "rz": qulacs_gate_rz,
}

qiskit_qulacs_param_gate_dict = {
    "rx": qulacs_param_gate_rx,
    "ry": qulacs_param_gate_ry,
    "rz": qulacs_param_gate_rz,
}


# qiskit_qulacs_gate_dict = {
#     "toffoli": qml.Toffoli,
#     "sx": qml.SX,
#     "iswap": qml.ISWAP,
#     "cswap": qml.CSWAP,
#     "ecr": qml.ECR,
#     "ch": qml.CH,
#     "rx": qml.RX,
#     "ry": qml.RY,
#     "rz": qml.RZ,
#     "p": qml.PhaseShift,
#     "cp": qml.ControlledPhaseShift,
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

#     "cs": cs,
#     "csx": csx,
# }
