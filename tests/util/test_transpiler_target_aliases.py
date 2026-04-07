"""Tests for transpiler target alias normalization on PennyLane/Qulacs targets."""

from qiskit import QuantumCircuit, transpile

from squlearn.util.pennylane.pennylane_gates import qiskit_pennylane_target
from squlearn.util.qulacs.qulacs_gates import qiskit_qulacs_target


class TestTranspilerTargetAliases:
    """Ensure alias names are normalized to canonical basis-gate names."""

    def test_pennylane_target_normalizes_aliases(self):
        qc = QuantumCircuit(3)
        qc.id(0)
        qc.cx(0, 1)
        qc.ccx(0, 1, 2)

        # Alias names in Qiskit are normalized by instruction names during transpilation.
        transpiled = transpile(qc, target=qiskit_pennylane_target, optimization_level=0)
        op_names = [inst.operation.name for inst in transpiled.data]

        assert all(name in {"id", "cx", "ccx"} for name in op_names)

    def test_qulacs_target_normalizes_aliases(self):
        qc = QuantumCircuit(2)
        qc.id(0)
        qc.cx(0, 1)

        transpiled = transpile(qc, target=qiskit_qulacs_target, optimization_level=0)
        op_names = [inst.operation.name for inst in transpiled.data]

        assert all(name in {"id", "cx"} for name in op_names)
