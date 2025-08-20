import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn.encoding_circuit import YZ_CX_EncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR
from squlearn.util.executor import Executor


class TestYZ_CX_EncodingCircuit:
    def test_init(self):
        circuit = YZ_CX_EncodingCircuit(num_qubits=2)
        assert circuit.num_qubits == 2
        assert circuit.num_layers == 1
        assert circuit.closed is True
        assert circuit.c == 1.0

    def test_num_parameters(self):
        qubits = 3
        layers = 1
        circuit = YZ_CX_EncodingCircuit(
            num_qubits=qubits,
            num_layers=layers,
            closed=True,
        )
        assert circuit.num_parameters == 2 * qubits * layers

    def test_parameter_bounds(self):
        circuit = YZ_CX_EncodingCircuit(num_qubits=2, num_layers=1)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)
        assert np.all(bounds == [-np.pi, np.pi])

    def test_get_params(self):
        circuit = YZ_CX_EncodingCircuit(num_qubits=2)
        named_params = circuit.get_params()
        print(named_params)
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "num_layers": 1,
            "c": 1.0,
        }

    def test_get_circuit(self):
        circuit = YZ_CX_EncodingCircuit(num_qubits=2)
        features = np.array([0.5, -0.5])
        params = np.array([0.1, 0.2, 0.3, 0.4])

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        with pytest.raises(EncodingSlotsMismatchError):
            YZ_CX_EncodingCircuit(num_qubits=1).get_circuit(features=features, parameters=params)

    def test_minimal_fit(self):
        circuit = YZ_CX_EncodingCircuit(num_qubits=2)

        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([5, 7, 9, 11, 13])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = YZ_CX_EncodingCircuit(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)
