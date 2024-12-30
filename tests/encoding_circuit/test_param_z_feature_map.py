import numpy as np
from qiskit import QuantumCircuit
from squlearn.encoding_circuit import ParamZFeatureMap


class TestParamZFeatureMap:
    def test_init(self):
        circuit = ParamZFeatureMap(num_qubits=2, num_features=2)
        assert circuit.num_features == 2
        assert circuit.num_qubits == 2
        assert circuit.num_layers == 2
        assert circuit._entangling is False

    def test_num_parameters(self):
        features = 2
        qubits = 3
        layers = 2
        circuit = ParamZFeatureMap(num_features=features, num_qubits=qubits, num_layers=layers)
        assert circuit.num_parameters == max(qubits, features) * layers

    def test_get_params(self):
        circuit = ParamZFeatureMap(num_features=2, num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": 2,
            "num_qubits": 2,
            "num_layers": 2,
            "entangling": False,
        }

    def test_get_ciruit(self):
        circuit = ParamZFeatureMap(num_features=2, num_qubits=2)
        features = np.array([0.5, -0.5])
        params = np.array([0.1, 0.2, 0.3, 0.4])

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2
