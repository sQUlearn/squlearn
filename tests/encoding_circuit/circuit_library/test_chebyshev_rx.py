from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import ChebyshevRx
import pytest
import numpy as np

from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR
from tests.qiskit_circuit_equivalence import assert_circuits_equal


def _build_expected_chebyshev_rx_circuit(
    num_qubits: int,
    num_layers: int,
    closed: bool,
    nonlinearity: str,
    features: np.ndarray,
    parameters: np.ndarray,
):
    if nonlinearity == "arccos":

        def mapping(a, x):
            return a * np.arccos(x)

    else:

        def mapping(a, x):
            return a * np.arctan(x)

    QC = QuantumCircuit(num_qubits)
    index_offset = 0
    feature_offset = 0

    def entangle_layer_local(QC_local: QuantumCircuit) -> QuantumCircuit:
        for i in range(0, num_qubits + (1 if closed else 0) - 1, 2):
            QC_local.cx(i, (i + 1) % num_qubits)
        if num_qubits > 2:
            for i in range(1, num_qubits + (1 if closed else 0) - 1, 2):
                QC_local.cx(i, (i + 1) % num_qubits)
        return QC_local

    for _ in range(num_layers):
        for i in range(num_qubits):
            QC.rx(
                mapping(
                    parameters[index_offset % len(parameters)],
                    features[feature_offset % len(features)],
                ),
                i,
            )
            index_offset += 1
            feature_offset += 1

        for i in range(num_qubits):
            QC.rx(parameters[index_offset % len(parameters)], i)
            index_offset += 1

        QC = entangle_layer_local(QC)

    return QC


class TestChebyshevRx:
    def test_init(self):
        circuit = ChebyshevRx(num_qubits=2)
        assert circuit.num_qubits == 2
        assert circuit.num_layers == 1
        assert circuit.closed is False
        assert circuit.alpha == 4.0
        assert circuit.nonlinearity == "arccos"

        with pytest.raises(ValueError):
            ChebyshevRx(num_qubits=2, nonlinearity="invalid")

    def test_num_parameters(self):
        qubits = 2
        layers = 1
        circuit = ChebyshevRx(num_qubits=qubits, num_layers=layers, closed=True)
        assert circuit.num_parameters == 2 * qubits * layers

    def test_parameter_bounds(self):
        qubits = 2
        layers = 1
        circuit = ChebyshevRx(num_qubits=qubits, num_layers=layers)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)
        assert np.all(bounds[: qubits * layers, 0] == 0.0)
        assert np.all(bounds[: qubits * layers, 1] == circuit.alpha)
        assert np.all(bounds[qubits * layers :, 0] == -np.pi)
        assert np.all(bounds[qubits * layers :, 1] == np.pi)

    def test_feature_bounds(self):
        circuit = ChebyshevRx(num_qubits=2, num_layers=1, nonlinearity="arccos")
        bounds = circuit.get_feature_bounds(num_features=2)
        assert bounds.shape == (2, 2)
        assert bounds[0, 0] == -1.0
        assert bounds[0, 1] == 1.0

        circuit = ChebyshevRx(num_qubits=2, num_layers=1, nonlinearity="arctan")
        bounds = circuit.get_feature_bounds(num_features=2)
        assert bounds.shape == (2, 2)
        assert bounds[0, 0] == -np.inf
        assert bounds[0, 1] == np.inf

    def test_generate_initial_parameters(self):
        circuit = ChebyshevRx(num_qubits=2)
        params = circuit.generate_initial_parameters(seed=42, num_features=2)
        assert len(params) == circuit.num_parameters

    def test_get_params(self):
        circuit = ChebyshevRx(num_qubits=2, num_layers=1)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "num_layers": 1,
            "closed": False,
            "alpha": 4.0,
            "nonlinearity": "arccos",
        }

    def test_set_params(self):
        circuit = ChebyshevRx(num_qubits=2, num_layers=1)
        circuit.set_params(
            num_qubits=3,
            num_layers=2,
            closed=True,
            alpha=3.0,
            nonlinearity="arctan",
        )
        assert circuit.num_qubits == 3
        assert circuit.num_layers == 2
        assert circuit.closed is True
        assert circuit.alpha == 3.0
        assert circuit.nonlinearity == "arctan"

        with pytest.raises(ValueError):
            circuit.set_params(num_qubits=3, num_layers=2, closed=False, nonlinearity="invalid")

    def test_get_circuit(self):
        circuit = ChebyshevRx(num_qubits=2, num_layers=1)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        used_params = {param for instruction in qc.data for param in instruction.operation.params}
        assert len(used_params) == len(params)

        with pytest.raises(EncodingSlotsMismatchError):
            ChebyshevRx(num_qubits=1, num_layers=1).get_circuit(
                features=features, parameters=params
            )

    def test_minimal_fit(self):
        circuit = ChebyshevRx(num_qubits=2, num_layers=2, closed=True)

        X_train = np.array([[-1.0, -0.8], [-0.6, -0.4], [-0.2, 0.0], [0.4, 0.6], [0.8, 1.0]])
        y_train = np.array([-0.6, -0.2, 0.2, 0.6, 1.0])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-1)

    def test_feature_consistency(self):
        circuit = ChebyshevRx(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)

    @pytest.mark.parametrize(
        "num_qubits,num_layers,closed,nonlinearity",
        [
            (2, 1, False, "arccos"),
            (3, 1, True, "arccos"),
            (3, 2, False, "arctan"),
            (4, 2, True, "arctan"),
        ],
    )
    def test_chebyshev_rx_get_circuit_matches_ground_truth(
        self, num_qubits, num_layers, closed, nonlinearity
    ):

        circuit = ChebyshevRx(
            num_qubits=num_qubits,
            num_layers=num_layers,
            closed=closed,
            nonlinearity=nonlinearity,
        )

        num_encoding_slots = circuit.num_encoding_slots
        num_features = min(2, num_encoding_slots) if num_encoding_slots >= 2 else 1
        features = np.linspace(-0.9, 0.9, num_features)

        rng = np.random.RandomState(42)
        parameters = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)

        qc_actual = circuit.get_circuit(features=features, parameters=parameters)

        qc_expected = _build_expected_chebyshev_rx_circuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            closed=closed,
            nonlinearity=nonlinearity,
            features=features,
            parameters=parameters,
        )

        assert_circuits_equal(qc_actual, qc_expected)
