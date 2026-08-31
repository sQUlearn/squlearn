"""Regression tests for the big-endian Pauli-label convention.

qc_executor's ``QuantumOperator`` treats the leftmost label character as
qubit 0. ``SinglePauli``, ``SingleProbability`` and ``IsingHamiltonian``
build their labels internally from a ``qubit: int`` argument and must
therefore address the physically correct qubit.
"""

import numpy as np
import pytest

from qc_executor import QuantumCircuit

from squlearn import Executor
from squlearn.observables import IsingHamiltonian, SingleProbability, SinglePauli

NUM_QUBITS = 4
FRAMEWORKS = ["qiskit", "pennylane", "qulacs"]


def _excite(qubits) -> QuantumCircuit:
    """A circuit with an X gate on each of the given qubits, identity elsewhere."""
    circuit = QuantumCircuit(NUM_QUBITS)
    for q in qubits:
        circuit.x(q)
    return circuit


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("excited_qubit", range(NUM_QUBITS))
def test_single_pauli_z_addresses_physical_qubit(framework, excited_qubit):
    """SinglePauli(qubit=i) must measure Z on the physically excited qubit i, not its mirror."""
    executor = Executor(framework).as_qc_executor
    circuit = _excite([excited_qubit])
    operator = SinglePauli(NUM_QUBITS, excited_qubit, op_str="Z").get_pauli(np.array([]))

    value = executor.expectation_value(circuit, operator)

    assert np.isclose(value, -1.0), (
        f"SinglePauli(qubit={excited_qubit}) on framework={framework} did not measure "
        f"the qubit excited by X: expected -1.0, got {value}"
    )


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("excited_qubit", range(NUM_QUBITS))
@pytest.mark.parametrize("measured_qubit", range(NUM_QUBITS))
def test_single_probability_addresses_physical_qubit(framework, excited_qubit, measured_qubit):
    """SingleProbability(qubit=i) must report P(|1>) for physical qubit i only."""
    executor = Executor(framework).as_qc_executor
    circuit = _excite([excited_qubit])
    operator = SingleProbability(NUM_QUBITS, measured_qubit, one_state=True).get_pauli(
        np.array([])
    )

    value = executor.expectation_value(circuit, operator)

    expected = 1.0 if measured_qubit == excited_qubit else 0.0
    assert np.isclose(value, expected), (
        f"SingleProbability(qubit={measured_qubit}) on framework={framework} with X on "
        f"qubit={excited_qubit}: expected {expected}, got {value}"
    )


ZZ_PAIRS = [(i, j) for i in range(NUM_QUBITS) for j in range(i)]


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("i, j", ZZ_PAIRS)
def test_ising_hamiltonian_zz_term_addresses_physical_qubits(framework, i, j):
    """The two-character ZZ term of IsingHamiltonian must address qubits i and j, i != j.

    Z(x)Z gives +1 both when both addressed qubits are excited *and* when neither is,
    so exciting {i, j} together and expecting +1 cannot tell "addresses the right pair"
    apart from "addresses some other, unexcited pair". Exciting exactly one of {i, j}
    and requiring -1 is airtight: that value occurs iff the excited qubit is exactly one
    of the two qubits the term addresses, regardless of what the other one is. Doing this
    for both i and j together pins down the addressed pair to exactly {i, j}.
    """
    executor = Executor(framework).as_qc_executor

    ob = IsingHamiltonian(NUM_QUBITS, I="N", Z="N", X="N", ZZ="F")
    pairs = [(a, b) for a in range(NUM_QUBITS) for b in range(a)]
    weight = np.zeros(len(pairs))
    weight[pairs.index((i, j))] = 1.0
    operator = ob.get_pauli(weight)

    for excited_qubit in (i, j):
        value = executor.expectation_value(_excite([excited_qubit]), operator)
        assert np.isclose(value, -1.0), (
            f"IsingHamiltonian ZZ({i},{j}) on framework={framework} with only qubit "
            f"{excited_qubit} excited: expected -1.0, got {value}"
        )
