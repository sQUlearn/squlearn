import pennylane as qml
from qiskit.transpiler import Target
from qiskit.circuit import IfElseOp


def RXX(theta, wires):
    """RXX gate."""
    return qml.PauliRot(theta, "XX", wires=wires)


def RYY(theta, wires):
    """RYY gate."""
    return qml.PauliRot(theta, "YY", wires=wires)


def RZZ(theta, wires):
    """RZZ gate."""
    return qml.PauliRot(theta, "ZZ", wires=wires)


def RZX(theta, wires):
    """RZX gate."""
    return qml.PauliRot(theta, "ZX", wires=wires)


def reset(wires):
    """Reset gate, implemented by measure and reset."""
    return qml.measure(wires=wires, reset=True)


def tdg(wires):
    """T-dagger gate."""
    return qml.adjoint(qml.T(wires=wires))


def sdg(wires):
    """S-dagger gate."""
    return qml.adjoint(qml.S(wires=wires))


def cs(wires):
    """CS gate."""
    if len(wires) != 2:
        raise ValueError("CS gate requires two wires.")
    return qml.ctrl(qml.S(wires[1]), wires[0])


def csx(wires):
    """CSX gate."""
    if len(wires) != 2:
        raise ValueError("CSX gate requires two wires.")
    return qml.ctrl(qml.SX(wires[1]), wires[0])


# Dictionary of conversion Qiskit gates (from string) to PennyLane gates
qiskit_pennylane_gate_dict = {
    "h": qml.Hadamard,
    "x": qml.PauliX,
    "y": qml.PauliY,
    "z": qml.PauliZ,
    "s": qml.S,
    "t": qml.T,
    "ccx": qml.Toffoli,
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
    "cy": qml.CY,
    "cz": qml.CZ,
    "crx": qml.CRX,
    "cry": qml.CRY,
    "crz": qml.CRZ,
    "rxx": RXX,
    "ryy": RYY,
    "rzz": RZZ,
    "rzx": RZX,
    "u": qml.U3,
    "measure": qml.measure,
    "reset": reset,
    "tdg": tdg,
    "sdg": sdg,
    "cs": cs,
    "csx": csx,
}

custom_name_mapping = {
    "i": qml.Identity,
    "toffoli": qml.Toffoli,
    "cnot": qml.CNOT,
    "barrier": qml.Barrier,
}

qiskit_pennylane_target = Target.from_configuration(
    basis_gates=qiskit_pennylane_gate_dict.keys(), custom_name_mapping=custom_name_mapping
)

qiskit_pennylane_target.add_instruction(IfElseOp, name="if_else")
