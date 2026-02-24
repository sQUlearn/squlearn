"""Tests for PennyLaneCircuit, focusing on if_else gate handling."""

import pytest
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector

from squlearn.util.pennylane.pennylane_circuit import PennyLaneCircuit


class TestPennyLaneCircuitIfElse:
    """Tests for if_else gate handling in PennyLaneCircuit."""

    def test_if_else_gate_appended_to_main_lists(self):
        """Test that gates inside an if_else block are appended to the main gate lists."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.measure(0, cr[0])
        with qc.if_test((cr[0], 1)):
            qc.x(1)

        pl_circuit = PennyLaneCircuit(qc, "probs")

        # 3 gates: H on q0, measure on q0, X on q1 (conditional)
        assert len(pl_circuit._pennylane_gates) == 3
        assert len(pl_circuit._pennylane_gates_wires) == 3
        assert len(pl_circuit._pennylane_conditions) == 3

    def test_if_else_wire_mapping(self):
        """Test that qubit wires in an if_else body are correctly mapped to outer circuit wires."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Measure qubit 1 and conditionally apply X to qubit 0.
        qc.h(1)
        qc.measure(1, cr[0])
        with qc.if_test((cr[0], 1)):
            qc.x(0)  # X on qubit 0 (not qubit 1)

        pl_circuit = PennyLaneCircuit(qc, "probs")

        # The conditional X gate should be on wire 0, not wire 1
        assert pl_circuit._pennylane_gates_wires[2] == [0]
        assert pl_circuit._pennylane_conditions[2] == (0, True)

    def test_if_else_condition_stored_correctly(self):
        """Test that if_else conditions are stored with correct bit index and value."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.x(0)
        qc.measure(0, cr[0])
        with qc.if_test((cr[0], 1)):
            qc.x(1)

        pl_circuit = PennyLaneCircuit(qc, "probs")

        # H and measure have None condition; X inside if_else has condition (0, True)
        assert pl_circuit._pennylane_conditions[0] is None
        assert pl_circuit._pennylane_conditions[1] is None
        assert pl_circuit._pennylane_conditions[2] == (0, True)

    def test_if_else_with_parametrized_gate(self):
        """Test that a parametrized gate inside an if_else block is handled correctly."""
        p = ParameterVector("p", 1)
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.measure(0, cr[0])
        with qc.if_test((cr[0], 1)):
            qc.rx(p[0], 1)

        pl_circuit = PennyLaneCircuit(qc, "probs")

        # 3 gates: H, measure, RX (conditional)
        assert len(pl_circuit._pennylane_gates) == 3
        # The conditional RX is on wire 1
        assert pl_circuit._pennylane_gates_wires[2] == [1]
        # Parameter vector 'p' should be registered
        assert "p" in pl_circuit.circuit_parameter_names

    def test_if_else_multiple_gates_in_body(self):
        """Test that multiple gates in an if_else body are all appended correctly."""
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.measure(0, cr[0])
        with qc.if_test((cr[0], 1)):
            qc.x(1)
            qc.x(2)

        pl_circuit = PennyLaneCircuit(qc, "probs")

        # 4 gates: H, measure, X on q1 (conditional), X on q2 (conditional)
        assert len(pl_circuit._pennylane_gates) == 4
        assert pl_circuit._pennylane_gates_wires[2] == [1]
        assert pl_circuit._pennylane_gates_wires[3] == [2]
        assert pl_circuit._pennylane_conditions[2] == (0, True)
        assert pl_circuit._pennylane_conditions[3] == (0, True)
