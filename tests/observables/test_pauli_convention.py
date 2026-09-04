"""Regression tests for the big-endian Pauli-label convention.

qc_executor's ``QuantumOperator`` treats the leftmost label character as
qubit 0. ``SinglePauli``, ``SingleProbability``, ``IsingHamiltonian``,
``SummedPaulis`` and ``SummedProbabilities`` build their labels internally
from a qubit index and must therefore address the physically correct qubit;
``CustomObservable`` passes the user's string through unchanged, so the
convention is part of its public contract.
"""

import numpy as np
import pytest

from qc_executor import QuantumCircuit

from squlearn import Executor
from squlearn.observables import (
    CustomObservable,
    IsingHamiltonian,
    SinglePauli,
    SingleProbability,
    SummedPaulis,
    SummedProbabilities,
)

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
    executor = Executor(framework)
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
    executor = Executor(framework)
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
    executor = Executor(framework)

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


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("excited_qubit", range(NUM_QUBITS))
def test_custom_observable_label_addresses_physical_qubit(framework, excited_qubit):
    """A user's Pauli string is passed through unchanged, so its leftmost character
    must land on qubit 0. This is the only observable whose convention is visible in
    the public API, which is why it needs pinning down explicitly."""
    executor = Executor(framework)
    label = "I" * excited_qubit + "Z" + "I" * (NUM_QUBITS - excited_qubit - 1)
    operator = CustomObservable(NUM_QUBITS, label).get_pauli(np.array([]))

    value = executor.expectation_value(_excite([excited_qubit]), operator)

    assert np.isclose(value, -1.0), (
        f"CustomObservable('{label}') on framework={framework} did not measure qubit "
        f"{excited_qubit}, the one excited by X: expected -1.0, got {value}"
    )


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("excited_qubit", range(NUM_QUBITS))
def test_summed_paulis_weights_address_physical_qubits(framework, excited_qubit):
    """With full_sum=True every qubit owns one weight, so weighting a single qubit
    turns SummedPaulis into a SinglePauli on it. A mirrored label assignment would
    move the weight to qubit NUM_QUBITS-1-i and flip the sign of the result.

    A uniform weighting is invariant under mirroring and would catch nothing, which
    is exactly why the pre-existing SummedPaulis tests never saw the bug.
    """
    executor = Executor(framework)

    ob = SummedPaulis(NUM_QUBITS, op_str="Z", full_sum=True, include_identity=False)
    weight = np.zeros(ob.num_parameters)
    weight[excited_qubit] = 1.0
    operator = ob.get_pauli(weight)

    value = executor.expectation_value(_excite([excited_qubit]), operator)

    assert np.isclose(value, -1.0), (
        f"SummedPaulis weight {excited_qubit} on framework={framework} did not land on "
        f"the qubit excited by X: expected -1.0, got {value}"
    )


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("excited_qubit", range(NUM_QUBITS))
@pytest.mark.parametrize("weighted_qubit", range(NUM_QUBITS))
def test_summed_probabilities_weights_address_physical_qubits(
    framework, excited_qubit, weighted_qubit
):
    """SummedProbabilities with a single unit weight must report P(|1>) of exactly
    that physical qubit."""
    executor = Executor(framework)

    ob = SummedProbabilities(
        NUM_QUBITS, one_state=True, full_sum=True, include_identity=False
    )
    weight = np.zeros(ob.num_parameters)
    weight[weighted_qubit] = 1.0
    operator = ob.get_pauli(weight)

    value = executor.expectation_value(_excite([excited_qubit]), operator)

    expected = 1.0 if weighted_qubit == excited_qubit else 0.0
    assert np.isclose(value, expected), (
        f"SummedProbabilities weight {weighted_qubit} on framework={framework} with X on "
        f"qubit={excited_qubit}: expected {expected}, got {value}"
    )
