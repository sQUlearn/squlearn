from black import Encoding
import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn.encoding_circuit import KyriienkoEncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError


class TestKyriienkoEncodingCircuit:
    def test_init(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=2, num_features=2)
        assert circuit.num_features == 2
        assert circuit.num_qubits == 2
        assert circuit.encoding_style == "chebyshev_tower"
        assert circuit.variational_arrangement == "HEA"
        assert circuit.num_encoding_layers == 1
        assert circuit.num_variational_layers == 1
        assert circuit.rotation_gate == "ry"
        assert circuit.block_width == 2
        assert circuit.block_depth == 1

        with pytest.raises(ValueError):
            KyriienkoEncodingCircuit(
                num_qubits=2, num_features=2, variational_arrangement="invalid"
            )

        with pytest.raises(ValueError):
            KyriienkoEncodingCircuit(num_qubits=2, num_features=2, rotation_gate="invalid")

        with pytest.raises(ValueError):
            KyriienkoEncodingCircuit(num_qubits=1, num_features=2)

    def test_parameter_bounds(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=2, num_features=2)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)
        assert np.all(bounds == [-2.0 * np.pi, 2.0 * np.pi])

    def test_num_parameters(self):
        qubits = 2
        features = 2
        variational_layers = 1
        block_depth = 1
        circuit = KyriienkoEncodingCircuit(
            num_qubits=qubits,
            num_features=features,
            variational_arrangement="HEA",
            num_variational_layers=variational_layers,
            block_depth=block_depth,
        )
        assert circuit.num_parameters == 3 * qubits * variational_layers

        circuit = KyriienkoEncodingCircuit(
            num_qubits=qubits,
            num_features=features,
            variational_arrangement="ABA",
            num_variational_layers=variational_layers,
            block_depth=block_depth,
        )
        assert circuit.num_parameters == 3 * qubits * block_depth * variational_layers

    def test_get_params(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=2, num_features=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": 2,
            "num_qubits": 2,
            "encoding_style": "chebyshev_tower",
            "variational_arrangement": "HEA",
            "num_encoding_layers": 1,
            "num_variational_layers": 1,
            "rotation_gate": "ry",
            "block_width": 2,
            "block_depth": 1,
        }

    def test_get_circuit(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=2, num_features=2)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert qc.num_qubits == 2
        assert isinstance(qc, QuantumCircuit)

        with pytest.raises(EncodingSlotsMismatchError):
            KyriienkoEncodingCircuit(num_qubits=2, num_features=1).get_circuit(
                features=[0.3, 0.5, -0.5], parameters=params
            )
