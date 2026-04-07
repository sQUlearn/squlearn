"""Tests for PennyLaneCircuit, focusing on if_else gate handling."""

import pytest

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

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

    def test_nested_if_else_combines_conditions(self):
        """Nested if_else keeps both outer and inner conditions for the inner gate."""
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr, cr)

        qc.x(0)
        qc.x(1)
        qc.measure(0, cr[0])
        qc.measure(1, cr[1])

        with qc.if_test((cr[0], 1)):
            with qc.if_test((cr[1], 1)):
                qc.x(2)

        pl_circuit = PennyLaneCircuit(qc, "probs")

        # The last gate (x on q2) must depend on both conditions.
        nested_condition = pl_circuit._pennylane_conditions[-1]
        assert isinstance(nested_condition, list)
        assert nested_condition == [(0, True), (1, True)]

    def test_barrier_is_ignored_in_conversion(self):
        """Barrier does not create a PennyLane gate and does not raise during conversion."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.barrier()
        qc.x(0)

        pl_circuit = PennyLaneCircuit(qc, "probs")

        # Only H and X remain as executable gates.
        assert len(pl_circuit._pennylane_gates) == 2
        assert len(pl_circuit._pennylane_conditions) == 2

    def test_parameter_tuple_keeps_all_gate_params(self):
        """Gate parameter tuples keep all entries, including constant expressions."""
        p = ParameterVector("p", 1)
        qc = QuantumCircuit(1)
        qc.u(0 * p[0], p[0], 0.0, 0)

        pl_circuit = PennyLaneCircuit(qc, "probs")
        gate_params = pl_circuit._pennylane_gates_param_function[0]

        assert gate_params is not None
        assert len(gate_params) == 3

    def test_observable_constant_parameter_expression_not_dropped(self):
        """ParameterExpression coefficients that simplify to constants are retained."""
        p = ParameterVector("p", 1)
        qc = QuantumCircuit(1)
        qc.x(0)
        observable = 2.0 * SparsePauliOp(["Z"], coeffs=[0 * p[0]]) + SparsePauliOp(
            ["I"], coeffs=[1.0]
        )

        pl_circuit = PennyLaneCircuit(qc, observable)

        assert len(pl_circuit._pennylane_obs_param_function) == 2
