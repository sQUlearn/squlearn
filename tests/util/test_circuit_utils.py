"""Tests for compose_keep_parameters."""

import numpy as np
import pytest
from qc_executor import Parameters, QuantumCircuit

from squlearn.util.circuit_utils import compose_keep_parameters


def _names(circuit):
    return sorted(str(parameter) for parameter in circuit.parameters)


class TestComposeKeepParameters:
    def test_shared_parameter_keeps_its_name(self):
        """The case QuantumCircuitBase.compose gets wrong: x[0] is shared, p differs."""
        x, p = Parameters("x", 1), Parameters("p", 2)
        first = QuantumCircuit(1)
        first.rx(0, x[0])
        first.ry(0, p[0])
        second = QuantumCircuit(1)
        second.rz(0, x[0])
        second.ry(0, p[1])

        compose_keep_parameters(first, second)

        assert _names(first) == ["p[0]", "p[1]", "x[0]"]

    def test_the_library_compose_would_rename(self):
        """Documents why the helper exists."""
        x, p = Parameters("x", 1), Parameters("p", 2)
        first = QuantumCircuit(1)
        first.rx(0, x[0])
        first.ry(0, p[0])
        second = QuantumCircuit(1)
        second.rz(0, x[0])
        second.ry(0, p[1])

        first.compose(second)

        assert "x[0]" not in _names(first)

    def test_appends_in_place_and_returns_the_circuit(self):
        first, second = QuantumCircuit(2), QuantumCircuit(2)
        first.h(0)
        second.cx(0, 1)

        assert compose_keep_parameters(first, second) is first
        assert first.circuit_metrics() == {"h": 1, "cx": 1}

    def test_qubit_mapping(self):
        first, second = QuantumCircuit(3), QuantumCircuit(2)
        second.cx(0, 1)

        compose_keep_parameters(first, second, qubits=[2, 0])

        assert first.ir[0].qubits == (2, 0)

    def test_result_runs(self):
        x = Parameters("x", 1)
        first, second = QuantumCircuit(1), QuantumCircuit(1)
        first.rx(0, x[0])
        second.rx(0, x[0])

        compose_keep_parameters(first, second)
        bound = first.assign_parameters({x[0]: np.pi / 2})

        assert [instruction.params for instruction in bound.ir] == [(np.pi / 2,), (np.pi / 2,)]

    def test_wrong_number_of_qubits_is_rejected(self):
        with pytest.raises(ValueError):
            compose_keep_parameters(QuantumCircuit(3), QuantumCircuit(2), qubits=[0])
