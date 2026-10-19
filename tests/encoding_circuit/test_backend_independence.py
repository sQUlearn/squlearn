"""Encoding circuits that start from a Qiskit circuit must still run on every backend.

``PrunedEncodingCircuit`` and ``QiskitEncodingCircuit`` used to hand out a Qiskit-bound
circuit. Converting one of those to another backend yields an *empty* circuit rather than
an error, so these tests compare the actual states instead of only checking types.
"""

import numpy as np
import pytest
from qc_executor import Executor, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

from squlearn.encoding_circuit import (
    HubregtsenEncodingCircuit,
    PrunedEncodingCircuit,
    QiskitEncodingCircuit,
)

BACKENDS = ["pennylane", "qulacs"]


def _statevector(backend: str, circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(Executor.create(backend).statevector(circuit))


def _pruned():
    return PrunedEncodingCircuit(HubregtsenEncodingCircuit(3, 2), [0, 2])


def _circuits():
    qiskit_enc = QiskitEncodingCircuit(ZZFeatureMap(2), mode="x", feature_label="x")
    pruned = _pruned()
    features = np.array([0.2, 0.5])
    parameters = pruned.generate_initial_parameters(num_features=2, seed=1)
    return {
        "qiskit_encoding_circuit": qiskit_enc.get_circuit(np.array([0.3, -0.4]), np.array([])),
        "pruned_encoding_circuit": pruned.get_circuit(features, parameters),
    }


@pytest.mark.parametrize("name", ["qiskit_encoding_circuit", "pruned_encoding_circuit"])
def test_get_circuit_is_framework_independent(name):
    assert isinstance(_circuits()[name], QuantumCircuit)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["qiskit_encoding_circuit", "pruned_encoding_circuit"])
def test_state_matches_qiskit(name, backend):
    circuit = _circuits()[name]
    reference = _statevector("qiskit", circuit)

    overlap = np.abs(np.vdot(reference, _statevector(backend, circuit)))

    assert np.isclose(overlap, 1.0)


@pytest.mark.parametrize("backend", ["qiskit", *BACKENDS])
def test_pruned_circuit_composes_with_its_inverse(backend):
    """``inverse()`` and ``compose()`` must see the pruned gates, so U^dagger U = 1."""
    pruned = _pruned()
    features = np.array([0.2, 0.5])
    parameters = pruned.generate_initial_parameters(num_features=2, seed=1)
    # PrunedEncodingCircuit only sets num_qubits inside get_circuit (see backlog), so
    # compose() needs one call first.
    pruned.get_circuit(features, parameters)
    fidelity = pruned.compose(
        pruned.inverse(), concatenate_features=False, concatenate_parameters=False
    )

    state = _statevector(backend, fidelity.get_circuit(features, parameters))

    assert np.isclose(abs(state[0]) ** 2, 1.0)
