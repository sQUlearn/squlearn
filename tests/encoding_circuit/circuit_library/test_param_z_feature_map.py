import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import ParamZFeatureMap
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR
from tests.qiskit_circuit_equivalence import assert_circuits_equal


def _build_expected_paramz_circuit(
    num_qubits: int,
    num_layers: int,
    entangling: bool,
    features: np.ndarray,
    parameters: np.ndarray,
):
    num_features = len(features)
    num_params = len(parameters)
    circuit = QuantumCircuit(num_qubits)
    index_offset = 0

    for layer in range(num_layers):
        for i in range(num_qubits):
            circuit.h(i)
            circuit.p(parameters[index_offset % num_params] * features[i % num_features], i)
            index_offset += 1

        if entangling:
            if num_layers % 2 == 0:
                for j in range(num_qubits - 1):
                    circuit.cx(j, j + 1)
            else:
                for j in range(1, num_qubits - 1, 2):
                    circuit.cx(j, j + 1)

    return circuit


class TestParamZFeatureMap:
    def test_init(self):
        circuit = ParamZFeatureMap(num_qubits=2)
        assert circuit.num_qubits == 2
        assert circuit.num_layers == 2
        assert circuit._entangling is False

    def test_num_parameters(self):
        qubits = 3
        layers = 2
        circuit = ParamZFeatureMap(num_qubits=qubits, num_layers=layers)
        assert circuit.num_parameters == qubits * layers

    def test_get_params(self):
        circuit = ParamZFeatureMap(num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "num_layers": 2,
            "entangling": False,
        }

    def test_get_ciruit(self):
        circuit = ParamZFeatureMap(num_qubits=2)
        features = np.array([0.5, -0.5])
        params = np.array([0.1, 0.2, 0.3, 0.4])

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

    def test_minimal_fit(self):
        circuit = ParamZFeatureMap(num_qubits=2, num_layers=2)

        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([5, 7, 9, 11, 13])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = ParamZFeatureMap(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)

    @pytest.mark.parametrize(
        "num_qubits,num_layers,entangling",
        [
            (2, 1, False),
            (2, 1, True),
            (3, 2, False),
            (3, 2, True),
            (4, 2, True),
        ],
    )
    def test_paramzfeaturemap_circuit_matches_ground_truth(
        self, num_qubits, num_layers, entangling
    ):

        circuit_obj = ParamZFeatureMap(
            num_qubits=num_qubits, num_layers=num_layers, entangling=entangling
        )

        num_features = min(2, circuit_obj.num_encoding_slots)
        features = np.linspace(-0.9, 0.9, num_features)
        rng = np.random.RandomState(42)
        parameters = rng.uniform(-np.pi, np.pi, size=circuit_obj.num_parameters)

        qc_actual = circuit_obj.get_circuit(features=features, parameters=parameters)

        qc_expected = _build_expected_paramz_circuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            entangling=entangling,
            features=features,
            parameters=parameters,
        )

        assert_circuits_equal(qc_actual, qc_expected)
