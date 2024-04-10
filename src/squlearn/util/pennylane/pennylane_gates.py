import pennylane as qml


def RXX(theta, wires):
    return qml.PauliRot(theta, "XX", wires=wires)


def RYY(theta, wires):
    return qml.PauliRot(theta, "YY", wires=wires)


def RZZ(theta, wires):
    return qml.PauliRot(theta, "ZZ", wires=wires)


def RXZ(theta, wires):
    return qml.PauliRot(theta, "XZ", wires=wires)

def reset(wires):
    return qml.measure(wires=wires,reset=True)

qiskit_pennyland_gate_dict = {
    "i": qml.Identity,
    "h": qml.Hadamard,
    "x": qml.PauliX,
    "y": qml.PauliY,
    "z": qml.PauliZ,
    "s": qml.S,
    "t": qml.T,
    "toffoli": qml.Toffoli,
    "sx": qml.SX,
    "swap": qml.SWAP,
    "iswap": qml.ISWAP,
    "cswap": qml.CSWAP,
    "ecr": qml.ECR,
    "ch": qml.CH,
    "rx": qml.RX,
    "ry": qml.RY,
    "rz": qml.RZ,
    "p": qml.PhaseShift,
    "cp": qml.ControlledPhaseShift,
    "cx": qml.CNOT,
    "cnot": qml.CNOT,
    "cy": qml.CY,
    "cz": qml.CZ,
    "crx": qml.CRX,
    "cry": qml.CRY,
    "crz": qml.CRZ,
    "rxx": RXX,
    "ryy": RYY,
    "rzz": RZZ,
    "rxz": RXZ,
    "barrier": qml.Barrier,
    "u": qml.U3,
    "measure": qml.measure,
    "reset": reset,
}
