import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import ChebyshevPQC
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR
from tests.qiskit_circuit_equivalence import assert_circuits_equal


def _build_expected_chebyshev_circuit(
    num_qubits: int,
    num_layers: int,
    closed: bool,
    entangling_gate: str,
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

    if entangling_gate not in ["crz", "rzz"]:
        raise ValueError("Unknown entangling gate")

    # basis change at beginning
    for i in range(num_qubits):
        QC.ry(parameters[index_offset % len(parameters)], i)
        index_offset += 1

    for _ in range(num_layers):
        # chebyshev rx encodings
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

        # even pairs (0,1), (2,3), ...
        for i in range(0, num_qubits + (1 if closed else 0) - 1, 2):
            if entangling_gate == "crz":
                QC.crz(parameters[index_offset % len(parameters)], i, (i + 1) % num_qubits)
            else:
                QC.rzz(parameters[index_offset % len(parameters)], i, (i + 1) % num_qubits)
            index_offset += 1

        # odd pairs (1,2), (3,4), ...
        if num_qubits > 2:
            for i in range(1, num_qubits + (1 if closed else 0) - 1, 2):
                if entangling_gate == "crz":
                    QC.crz(parameters[index_offset % len(parameters)], i, (i + 1) % num_qubits)
                else:
                    QC.rzz(parameters[index_offset % len(parameters)], i, (i + 1) % num_qubits)
                index_offset += 1

    # final basis change
    for i in range(num_qubits):
        QC.ry(parameters[index_offset % len(parameters)], i)
        index_offset += 1

    return QC


class TestChebyshevPQC:
    def test_init(self):
        circuit = ChebyshevPQC(num_qubits=2)
        assert circuit.num_qubits == 2
        assert circuit.num_layers == 1
        assert circuit.closed is True
        assert circuit.entangling_gate == "crz"
        assert circuit.alpha == 4.0
        assert circuit.nonlinearity == "arccos"

        with pytest.raises(ValueError):
            ChebyshevPQC(
                num_features=2, num_qubits=2, entangling_gate="crz", nonlinearity="invalid"
            )

        with pytest.raises(ValueError):
            ChebyshevPQC(
                num_features=2, num_qubits=2, entangling_gate="invalid", nonlinearity="arccos"
            )

    @pytest.mark.parametrize(
        "closed, expected",
        [
            (True, 2 * 3 + 3 * 1 + 3 * 1),  # 12
            (False, 2 * 3 + 3 * 1 + (3 - 1) * 1),  # 11
        ],
    )
    def test_num_parameters(self, closed, expected):
        qubits = 3
        layers = 1
        circuit = ChebyshevPQC(num_qubits=qubits, num_layers=layers, closed=closed)
        assert circuit.num_parameters == expected

    def test_parameter_bounds(self):
        circuit = ChebyshevPQC(num_qubits=2, num_layers=1)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)

    def test_generate_initial_parameters(self):
        circuit = ChebyshevPQC(num_qubits=2)
        params = circuit.generate_initial_parameters(seed=42, num_features=2)
        assert len(params) == circuit.num_parameters

    def test_feature_bounds(self):
        circuit = ChebyshevPQC(num_qubits=2, num_layers=1, nonlinearity="arccos")
        bounds = circuit.get_feature_bounds(num_features=2)
        assert bounds.shape == (2, 2)
        assert bounds[0, 0] == -1.0
        assert bounds[0, 1] == 1.0

        circuit = ChebyshevPQC(num_qubits=2, num_layers=1, nonlinearity="arctan")
        bounds = circuit.get_feature_bounds(num_features=2)
        assert bounds.shape == (2, 2)
        assert bounds[0, 0] == -np.inf
        assert bounds[0, 1] == np.inf

    def test_get_params(self):
        circuit = ChebyshevPQC(num_qubits=2, num_layers=1)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "num_layers": 1,
            "closed": True,
            "entangling_gate": "crz",
            "alpha": 4.0,
            "nonlinearity": "arccos",
        }

    def test_set_params(self):
        circuit = ChebyshevPQC(num_qubits=2, num_layers=1)
        circuit.set_params(
            num_qubits=3,
            num_layers=2,
            closed=False,
            alpha=3.0,
            nonlinearity="arctan",
        )
        assert circuit.num_qubits == 3
        assert circuit.num_layers == 2
        assert circuit.closed is False
        assert circuit.alpha == 3.0
        assert circuit.nonlinearity == "arctan"

        with pytest.raises(ValueError):
            circuit.set_params(num_qubits=3, num_layers=2, closed=False, nonlinearity="invalid")

    def test_get_circuit(self):
        circuit = ChebyshevPQC(num_qubits=2, num_layers=1)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        used_params = {param for instruction in qc.data for param in instruction.operation.params}
        assert len(used_params) == len(params)

        with pytest.raises(EncodingSlotsMismatchError):
            ChebyshevPQC(num_qubits=1, num_layers=1).get_circuit(
                features=features, parameters=params
            )

    def test_minimal_fit(self):
        circuit = ChebyshevPQC(num_qubits=2, num_layers=2, closed=True)

        X_train = np.array([[-1.0, -0.8], [-0.6, -0.4], [-0.2, 0.0], [0.4, 0.6], [0.8, 1.0]])
        y_train = np.array([-0.6, -0.2, 0.2, 0.6, 1.0])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-1)

    def test_feature_consistency(self):
        circuit = ChebyshevPQC(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)

    @pytest.mark.parametrize(
        "num_qubits,num_layers,closed,entangling_gate,nonlinearity",
        [
            (2, 1, True, "crz", "arccos"),
            (3, 1, False, "crz", "arccos"),
            (3, 2, True, "rzz", "arctan"),
            (4, 2, False, "rzz", "arccos"),
        ],
    )
    def test_get_circuit_matches_ground_truth(
        self, num_qubits, num_layers, closed, entangling_gate, nonlinearity
    ):

        circuit = ChebyshevPQC(
            num_qubits=num_qubits,
            num_layers=num_layers,
            closed=closed,
            entangling_gate=entangling_gate,
            nonlinearity=nonlinearity,
        )

        num_encoding_slots = circuit.num_encoding_slots
        num_features = min(2, num_encoding_slots) if num_encoding_slots >= 2 else 1
        features = np.linspace(-0.9, 0.9, num_features)

        rng = np.random.RandomState(42)
        parameters = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)

        qc_actual = circuit.get_circuit(features=features, parameters=parameters)

        qc_expected = _build_expected_chebyshev_circuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            closed=closed,
            entangling_gate=entangling_gate,
            nonlinearity=nonlinearity,
            features=features,
            parameters=parameters,
        )

        assert_circuits_equal(qc_actual, qc_expected)
