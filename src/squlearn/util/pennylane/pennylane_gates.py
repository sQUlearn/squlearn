import pennylane as qml

def RXX(theta, wires):
    return qml.PauliRot(theta, 'XX', wires=wires)

def RYY(theta, wires):
    return qml.PauliRot(theta, 'YY', wires=wires)

def RZZ(theta, wires):
    return qml.PauliRot(theta, 'ZZ', wires=wires)

def RXZ(theta, wires):
    return qml.PauliRot(theta, 'XZ', wires=wires)

qiskit_pennyland_gate_dict = {
    "h": qml.Hadamard,
    "x": qml.PauliX,
    "y": qml.PauliY,
    "z": qml.PauliZ,
    "rz": qml.RZ,
    "ry": qml.RY,
    "rx": qml.RX,
    "cx": qml.CNOT,
    "cnot": qml.CNOT,
    "crx": qml.CRX,
    "cry": qml.CRY,
    "crz": qml.CRZ,
    "rxx": RXX,
    "ryy": RYY,
    "rzz": RZZ,
    "rxz": RXZ,
    "u": qml.U3,
    }
