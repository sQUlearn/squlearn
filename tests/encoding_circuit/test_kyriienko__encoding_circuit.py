import numpy as np
import pytest
import copy
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import KyriienkoEncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.kernel.matrix.fidelity_kernel import FidelityKernel
from squlearn.kernel.ml.qgpr import QGPR


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

    def test_drawing_does_not_violate_circuit_parameters(self):
        circuit = KyriienkoEncodingCircuit(num_features=2, num_qubits=2)

        params_with_features_before = copy.deepcopy(circuit.get_params())
        circuit.draw(output="mpl")
        params_with_features_after = copy.deepcopy(circuit.get_params())

        assert params_with_features_before == params_with_features_after

        # same but with num_features=None
        circuit = KyriienkoEncodingCircuit(num_features=2, num_qubits=2)

        params_without_features_before = copy.deepcopy(circuit.get_params())
        circuit.draw(output="mpl")
        params_without_features_after = copy.deepcopy(circuit.get_params())

        assert params_without_features_before == params_without_features_after

    def test_minimal_fit(self):
        circuit = KyriienkoEncodingCircuit(num_features=2, num_qubits=2)

        X_train = np.array([[-1.0, -0.8], [-0.6, -0.4], [-0.2, 0.0], [0.4, 0.6], [0.8, 1.0]])
        y_train = np.array([-0.6, -0.2, 0.2, 0.6, 1.0])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)
