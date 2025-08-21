import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from squlearn import Executor
from squlearn.encoding_circuit import HighDimEncodingCircuit
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR


class TestHighDimEncodingCircuit:

    def test_init(self):
        circuit = HighDimEncodingCircuit(num_qubits=2)
        assert circuit.num_qubits == 2
        assert circuit.cycling is True
        assert circuit.cycling_type == "saw"
        assert circuit.num_layers is None
        assert circuit.layer_type == "rows"
        assert circuit.entangling_gate == "iswap"

        with pytest.raises(ValueError):
            HighDimEncodingCircuit(num_qubits=2, cycling_type="invalid")

        with pytest.raises(ValueError):
            HighDimEncodingCircuit(num_qubits=2, layer_type="invalid")

        with pytest.raises(ValueError):
            HighDimEncodingCircuit(num_qubits=2, entangling_gate="invalid")

    def test_get_params(self):
        circuit = HighDimEncodingCircuit(num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "num_layers": None,
            "cycling": True,
            "cycling_type": "saw",
            "layer_type": "rows",
            "entangling_gate": "iswap",
        }

    def test_get_circuit(self):
        circuit = HighDimEncodingCircuit(num_qubits=2)
        features = np.array([0.5, -0.5])

        qc = circuit.get_circuit(features=features)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        circuit = HighDimEncodingCircuit(
            num_qubits=4,
        )
        features = ParameterVector("x", 12)
        qc = circuit.get_circuit(features)
        assert circuit.num_layers == 2

    def test_minimal_fit(self):
        circuit = HighDimEncodingCircuit(num_qubits=2)

        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([5, 7, 9, 11, 13])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = HighDimEncodingCircuit(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)
