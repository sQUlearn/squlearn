import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import MultiControlEncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR
from tests.qiskit_circuit_equivalence import assert_circuits_equal


def _build_expected_multi_control_circuit(
    num_qubits: int,
    num_layers: int,
    closed: bool,
    final_encoding: bool,
    features: np.ndarray,
    parameters: np.ndarray,
):
    QC = QuantumCircuit(num_qubits)
    index_offset = 0
    feature_offset = 0
    num_params = len(parameters)
    num_features = len(features)

    for _ in range(num_layers):
        # ZZ encoding: H + Rz(features)
        QC.h(range(num_qubits))
        for i in range(num_qubits):
            QC.rz(features[feature_offset % num_features], i)
            feature_offset += 1

        istop = num_qubits if closed else num_qubits - 1

        # even pairs: CRx, CRy, CRz
        for i in range(0, istop, 2):
            QC.crx(parameters[index_offset % num_params], i, (i + 1) % num_qubits)
            index_offset += 1
            QC.cry(parameters[index_offset % num_params], i, (i + 1) % num_qubits)
            index_offset += 1
            QC.crz(parameters[index_offset % num_params], i, (i + 1) % num_qubits)
            index_offset += 1

        # odd pairs: CRx, CRy, CRz
        if num_qubits >= 2:
            for i in range(1, istop, 2):
                QC.crx(parameters[index_offset % num_params], i, (i + 1) % num_qubits)
                index_offset += 1
                QC.cry(parameters[index_offset % num_params], i, (i + 1) % num_qubits)
                index_offset += 1
                QC.crz(parameters[index_offset % num_params], i, (i + 1) % num_qubits)
                index_offset += 1

    if final_encoding:
        for i in range(num_qubits):
            QC.rz(features[feature_offset % num_features], i)
            feature_offset += 1
    return QC


class TestMultiControlEncodingCircuit:

    def test_init(self):
        circuit = MultiControlEncodingCircuit(num_qubits=2)
        assert circuit.num_qubits == 2
        assert circuit.num_layers == 1
        assert circuit.closed is True
        assert circuit.final_encoding is False

        with pytest.raises(ValueError):
            MultiControlEncodingCircuit(num_qubits=1)

    def test_num_parameters_closed(self):
        qubits = 3
        layers = 1
        circuit = MultiControlEncodingCircuit(
            num_qubits=qubits,
            num_layers=layers,
            closed=True,
        )
        assert circuit.num_parameters == 3 * (qubits - 1) * layers + 3 * layers

    def test_num_parameters_none_closed(self):
        qubits = 3
        layers = 1
        circuit = MultiControlEncodingCircuit(
            num_qubits=qubits,
            num_layers=layers,
            closed=False,
        )
        assert circuit.num_parameters == 3 * (qubits - 1) * layers

    def test_parameter_bounds(self):
        qubits = 3
        circuit = MultiControlEncodingCircuit(num_qubits=qubits, num_layers=1)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)
        assert np.all(bounds == [-2.0 * np.pi, 2.0 * np.pi])

    def test_get_params(self):
        circuit = MultiControlEncodingCircuit(num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "num_layers": 1,
            "closed": True,
            "final_encoding": False,
        }

    def test_get_circuit(self):
        circuit = MultiControlEncodingCircuit(num_qubits=2)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, circuit.num_parameters)
        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit(features=np.array([0.3, 0.5, -0.5]), parameters=params)

    def test_minimal_fit(self):
        circuit = MultiControlEncodingCircuit(num_qubits=2)

        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([5, 7, 9, 11, 13])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = MultiControlEncodingCircuit(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)

    @pytest.mark.parametrize(
        "num_qubits,num_layers,closed,final_encoding",
        [
            (2, 1, True, False),
            (3, 1, False, True),
            (3, 2, True, False),
            (4, 2, False, True),
        ],
    )
    def test_multi_control_get_circuit_ground_truth(
        self, num_qubits, num_layers, closed, final_encoding
    ):
        circuit = MultiControlEncodingCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            closed=closed,
            final_encoding=final_encoding,
        )

        num_features = min(2, circuit.num_encoding_slots)
        features = np.linspace(-0.9, 0.9, num_features)

        rng = np.random.RandomState(42)
        parameters = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)

        qc_actual = circuit.get_circuit(features=features, parameters=parameters)

        qc_expected = _build_expected_multi_control_circuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            closed=closed,
            final_encoding=final_encoding,
            features=features,
            parameters=parameters,
        )

        assert_circuits_equal(qc_actual, qc_expected)
