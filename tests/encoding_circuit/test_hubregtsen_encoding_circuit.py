import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR


class TestHubregtsenEncodingCircuit:
    def test_init(self):
        circuit = HubregtsenEncodingCircuit(num_qubits=2)
        assert circuit.num_qubits == 2
        assert circuit.num_layers == 1
        assert circuit.closed is True
        assert circuit.final_encoding is False

    def test_num_parameters_closed(self):
        qubits = 3
        layers = 1
        circuit = HubregtsenEncodingCircuit(num_qubits=qubits, num_layers=layers, closed=True)
        assert circuit.num_parameters == qubits * layers + qubits * layers

    def test_num_parameters_none_closed(self):
        qubits = 3
        layers = 1
        circuit = HubregtsenEncodingCircuit(num_qubits=qubits, num_layers=layers, closed=False)
        assert circuit.num_parameters == qubits * layers + (qubits - 1) * layers

    def test_parameter_bounds(self):
        qubits = 3
        circuit = HubregtsenEncodingCircuit(num_qubits=qubits, num_layers=1, closed=False)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)
        assert np.all(bounds[:qubits] == [-np.pi, np.pi])
        assert np.all(bounds[qubits:] == [-2.0 * np.pi, 2.0 * np.pi])

    def test_get_params(self):
        circuit = HubregtsenEncodingCircuit(num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "num_layers": 1,
            "closed": True,
            "final_encoding": False,
        }

    def test_get_circuit(self):
        circuit = HubregtsenEncodingCircuit(num_qubits=2)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        with pytest.raises(EncodingSlotsMismatchError):
            HubregtsenEncodingCircuit(num_qubits=1).get_circuit(
                features=features, parameters=params
            )

    def test_minimal_fit(self):
        circuit = HubregtsenEncodingCircuit(num_qubits=2)

        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([5, 7, 9, 11, 13])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = HubregtsenEncodingCircuit(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)
