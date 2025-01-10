import numpy as np
import pytest
import copy
from qiskit import QuantumCircuit
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError


class TestHubregtsenEncodingCircuit:
    def test_init(self):
        circuit = HubregtsenEncodingCircuit(num_qubits=2, num_features=2)
        assert circuit.num_qubits == 2
        assert circuit.num_features == 2
        assert circuit.num_layers == 1
        assert circuit.closed is True
        assert circuit.final_encoding is False

    def test_num_parameters_closed(self):
        features = 2
        qubits = 3
        layers = 1
        circuit = HubregtsenEncodingCircuit(
            num_features=features, num_qubits=qubits, num_layers=layers, closed=True
        )
        assert circuit.num_parameters == qubits * layers + qubits * layers

    def test_num_parameters_none_closed(self):
        features = 2
        qubits = 3
        layers = 1
        circuit = HubregtsenEncodingCircuit(
            num_features=features, num_qubits=qubits, num_layers=layers, closed=False
        )
        assert circuit.num_parameters == qubits * layers + (qubits - 1) * layers

    def test_parameter_bounds(self):
        qubits = 3
        circuit = HubregtsenEncodingCircuit(num_qubits=qubits, num_layers=1, closed=False)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)
        assert np.all(bounds[:qubits] == [-np.pi, np.pi])
        assert np.all(bounds[qubits:] == [-2.0 * np.pi, 2.0 * np.pi])

    def test_feature_bounds(self):
        circuit = HubregtsenEncodingCircuit(num_features=2, num_qubits=2)
        bounds = circuit.feature_bounds
        assert bounds.shape == (circuit.num_features, 2)
        assert np.all(bounds == [-np.pi, np.pi])

    def test_get_params(self):
        circuit = HubregtsenEncodingCircuit(num_features=2, num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": 2,
            "num_qubits": 2,
            "num_layers": 1,
            "closed": True,
            "final_encoding": False,
        }

    def test_get_circuit(self):
        circuit = HubregtsenEncodingCircuit(num_features=2, num_qubits=2)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        with pytest.raises(EncodingSlotsMismatchError):
            HubregtsenEncodingCircuit(num_features=1, num_qubits=1).get_circuit(
                features=features, parameters=params
            )

    def test_drawing_does_not_violate_circuit_parameters(self):
        circuit = HubregtsenEncodingCircuit(num_features=2, num_qubits=2, num_layers=1)

        params_with_features_before = copy.deepcopy(circuit.get_params())
        circuit.draw(output="mpl")
        params_with_features_after = copy.deepcopy(circuit.get_params())

        assert params_with_features_before == params_with_features_after

        # same but with num_features=None
        circuit = HubregtsenEncodingCircuit(num_features=2, num_qubits=2, num_layers=1)

        params_without_features_before = copy.deepcopy(circuit.get_params())
        circuit.draw(output="mpl")
        params_without_features_after = copy.deepcopy(circuit.get_params())

        assert params_without_features_before == params_without_features_after
