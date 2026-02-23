import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR
from tests.qiskit_circuit_equivalence import assert_circuits_equal


def _build_expected_hubregtsen_circuit(
    num_qubits: int,
    num_layers: int,
    closed: bool,
    final_encoding: bool,
    features: np.ndarray,
    parameters: np.ndarray,
):
    QC = QuantumCircuit(num_qubits)
    index_offset = 0
    num_features = len(features)
    num_params = len(parameters)

    # initial Hadamard on all qubits
    QC.h(range(num_qubits))

    # Layer loops
    for layer in range(num_layers):
        n_feature_loop = int(np.ceil(num_features / num_qubits))
        for i in range(n_feature_loop * num_qubits):
            if (i // num_qubits) % 2 == 0:
                QC.rz(features[i % num_features], i % num_qubits)
            else:
                QC.rx(features[i % num_features], i % num_qubits)

        # single theta Ry
        for i in range(num_qubits):
            QC.ry(parameters[index_offset % num_params], i)
            index_offset += 1

        # Entangled theta CRZ gates
        if num_qubits > 2:
            istop = num_qubits if closed else num_qubits - 1
            for i in range(istop):
                QC.crz(parameters[index_offset % num_params], i, (i + 1) % num_qubits)
                index_offset += 1

    # final encoding
    if final_encoding:
        n_feature_loop = int(np.ceil(num_features / num_qubits))
        for i in range(n_feature_loop * num_qubits):
            if int(np.ceil(i / num_qubits)) % 2 == 0:
                QC.rz(features[i % num_features], i % num_qubits)
            else:
                QC.rx(features[i % num_features], i % num_qubits)

    return QC


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

    @pytest.mark.parametrize(
        "num_qubits,num_layers,closed,final_encoding",
        [
            (2, 1, True, False),
            (3, 1, False, False),
            (3, 2, True, True),
            (4, 2, True, False),
        ],
    )
    def test_hubregtsen_get_circuit_matches_ground_truth(
        self, num_qubits, num_layers, closed, final_encoding
    ):
        circuit = HubregtsenEncodingCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            closed=closed,
            final_encoding=final_encoding,
        )

        num_encoding_slots = circuit.num_encoding_slots
        num_features = min(2, num_encoding_slots) if num_encoding_slots >= 2 else 1
        features = np.linspace(-0.9, 0.9, num_features)

        rng = np.random.RandomState(42)
        parameters = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)

        qc_actual = circuit.get_circuit(features=features, parameters=parameters)

        qc_expected = _build_expected_hubregtsen_circuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            closed=closed,
            final_encoding=final_encoding,
            features=features,
            parameters=parameters,
        )

        assert_circuits_equal(qc_actual, qc_expected)
