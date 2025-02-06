import pytest
import copy
import numpy as np
from qiskit import QuantumCircuit
from sklearn.preprocessing import MinMaxScaler
from squlearn import Executor
from squlearn.encoding_circuit import ChebyshevTower
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.kernel.matrix.fidelity_kernel import FidelityKernel
from squlearn.kernel.ml.qgpr import QGPR


class TestChebyshevTower:
    def test_init(self):
        circuit = ChebyshevTower(num_qubits=2, num_chebyshev=2, num_features=2)
        assert circuit.num_features == 2
        assert circuit.num_qubits == 2
        assert circuit.num_chebyshev == 2
        assert circuit.alpha == 1.0
        assert circuit.num_layers == 1
        assert circuit.rotation_gate == "ry"
        assert circuit.hadamard_start is True
        assert circuit.arrangement == "block"
        assert circuit.nonlinearity == "arccos"

        with pytest.raises(ValueError):
            ChebyshevTower(num_qubits=2, num_chebyshev=2, num_features=2, rotation_gate="invalid")

        with pytest.raises(ValueError):
            ChebyshevTower(num_qubits=2, num_chebyshev=2, num_features=2, arrangement="invalid")

        with pytest.raises(ValueError):
            ChebyshevTower(num_qubits=2, num_chebyshev=2, num_features=2, nonlinearity="invalid")

    def test_feature_bounds(self):
        circuit = ChebyshevTower(
            num_features=2, num_qubits=2, num_chebyshev=2, num_layers=1, nonlinearity="arccos"
        )
        bounds = circuit.feature_bounds
        assert bounds.shape == (circuit.num_features, 2)
        assert bounds[0, 0] == -1.0
        assert bounds[0, 1] == 1.0

        circuit = ChebyshevTower(
            num_features=2, num_qubits=2, num_chebyshev=2, num_layers=1, nonlinearity="arctan"
        )
        bounds = circuit.feature_bounds
        assert bounds.shape == (circuit.num_features, 2)
        assert bounds[0, 0] == -np.inf
        assert bounds[0, 1] == np.inf

    def test_get_params(self):
        circuit = ChebyshevTower(num_features=2, num_qubits=2, num_layers=1, num_chebyshev=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": 2,
            "num_qubits": 2,
            "num_layers": 1,
            "num_chebyshev": 2,
            "alpha": 1.0,
            "rotation_gate": "ry",
            "hadamard_start": True,
            "arrangement": "block",
            "nonlinearity": "arccos",
        }

    def test_set_params(self):
        circuit = ChebyshevTower(num_features=2, num_qubits=2, num_layers=1, num_chebyshev=1)
        circuit.set_params(
            num_features=3,
            num_qubits=3,
            num_layers=2,
            num_chebyshev=2,
            alpha=3.0,
            rotation_gate="rx",
            hadamard_start=False,
            arrangement="alternating",
            nonlinearity="arctan",
        )
        assert circuit.num_features == 3
        assert circuit.num_qubits == 3
        assert circuit.num_layers == 2
        assert circuit.num_chebyshev == 2
        assert circuit.alpha == 3.0
        assert circuit.rotation_gate == "rx"
        assert circuit.hadamard_start is False
        assert circuit.arrangement == "alternating"
        assert circuit.nonlinearity == "arctan"

        with pytest.raises(ValueError):
            circuit.set_params(
                num_features=3, num_qubits=3, num_layers=2, num_chebyshev=1, nonlinearity="invalid"
            )

    def test_get_circuit(self):
        circuit = ChebyshevTower(num_features=2, num_qubits=2, num_layers=1, num_chebyshev=1)
        features = np.array([0.5, -0.5])

        qc = circuit.get_circuit(features=features)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        with pytest.raises(EncodingSlotsMismatchError):
            ChebyshevTower(
                num_features=1, num_qubits=1, num_chebyshev=2, num_layers=1
            ).get_circuit(features=features)

    def test_minimal_fit(self):
        circuit = ChebyshevTower(num_features=2, num_qubits=2, num_chebyshev=2)

        X_train = np.array([[-1.0, -0.8], [-0.6, -0.4], [-0.2, 0.0], [0.4, 0.6], [0.8, 1.0]])
        y_train = np.array([-0.6, -0.2, 0.2, 0.6, 1.0])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)
