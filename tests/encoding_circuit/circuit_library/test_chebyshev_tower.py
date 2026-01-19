import pytest
import numpy as np
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import ChebyshevTower
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR
from tests.qiskit_circuit_equivalence import assert_circuits_equal


def _build_expected_chebyshev_tower(
    num_qubits: int,
    num_chebyshev: int,
    num_layers: int,
    alpha: float,
    rotation_gate: str,
    hadamard_start: bool,
    arrangement: str,
    nonlinearity: str,
    features: np.ndarray,
):
    # mapping same as in implementation: mapping(x,i) = alpha * i * arccos/atan(x)
    if nonlinearity == "arccos":

        def mapping(x, i):
            return alpha * i * np.arccos(x)

    else:

        def mapping(x, i):
            return alpha * i * np.arctan(x)

    def entangle_layer(QC: QuantumCircuit):
        for i in range(0, num_qubits - 1, 2):
            QC.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            QC.cx(i, i + 1)
        return QC

    QC = QuantumCircuit(num_qubits)

    if hadamard_start:
        QC.h(range(num_qubits))

    for layer in range(num_layers):
        index_offset = 0
        iqubit = 0
        icheb = 1

        if arrangement == "block":
            outer = len(features)
            inner = num_chebyshev
        elif arrangement == "alternating":
            inner = len(features)
            outer = num_chebyshev
        else:
            raise ValueError("Arrangement must be either 'block' or 'alternating'")

        for outer_ in range(outer):
            for inner_ in range(inner):
                angle = mapping(features[index_offset % len(features)], icheb)
                target = iqubit % num_qubits
                if rotation_gate.lower() == "rx":
                    QC.rx(angle, target)
                elif rotation_gate.lower() == "ry":
                    QC.ry(angle, target)
                elif rotation_gate.lower() == "rz":
                    QC.rz(angle, target)
                else:
                    raise ValueError("Rotation gate {} not supported".format(rotation_gate))
                iqubit += 1
                if arrangement == "block":
                    icheb += 1
                elif arrangement == "alternating":
                    index_offset += 1

            if arrangement == "block":
                index_offset += 1
                icheb = 1
            elif arrangement == "alternating":
                icheb += 1

        # entangling layer only if more layers follow
        if layer + 1 < num_layers:
            QC = entangle_layer(QC)

    return QC


class TestChebyshevTower:
    def test_init(self):
        circuit = ChebyshevTower(num_qubits=2, num_chebyshev=2)
        assert circuit.num_qubits == 2
        assert circuit.num_chebyshev == 2
        assert circuit.alpha == 1.0
        assert circuit.num_layers == 1
        assert circuit.rotation_gate == "ry"
        assert circuit.hadamard_start is True
        assert circuit.arrangement == "block"
        assert circuit.nonlinearity == "arccos"

        with pytest.raises(ValueError):
            ChebyshevTower(num_qubits=2, num_chebyshev=2, rotation_gate="invalid")

        with pytest.raises(ValueError):
            ChebyshevTower(num_qubits=2, num_chebyshev=2, arrangement="invalid")

        with pytest.raises(ValueError):
            ChebyshevTower(num_qubits=2, num_chebyshev=2, nonlinearity="invalid")

    def test_feature_bounds(self):
        circuit = ChebyshevTower(
            num_qubits=2, num_chebyshev=2, num_layers=1, nonlinearity="arccos"
        )
        bounds = circuit.get_feature_bounds(num_features=2)
        assert bounds.shape == (2, 2)
        assert bounds[0, 0] == -1.0
        assert bounds[0, 1] == 1.0

        circuit = ChebyshevTower(
            num_qubits=2, num_chebyshev=2, num_layers=1, nonlinearity="arctan"
        )
        bounds = circuit.get_feature_bounds(num_features=2)
        assert bounds.shape == (2, 2)
        assert bounds[0, 0] == -np.inf
        assert bounds[0, 1] == np.inf

    def test_get_params(self):
        circuit = ChebyshevTower(num_qubits=2, num_layers=1, num_chebyshev=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "num_layers": 1,
            "num_chebyshev": 2,
            "alpha": 1.0,
            "rotation_gate": "ry",
            "hadamard_start": True,
            "arrangement": "block",
            "nonlinearity": "arccos",
        }

    def test_set_params(self):
        circuit = ChebyshevTower(num_qubits=2, num_layers=1, num_chebyshev=1)
        circuit.set_params(
            num_qubits=3,
            num_layers=2,
            num_chebyshev=2,
            alpha=3.0,
            rotation_gate="rx",
            hadamard_start=False,
            arrangement="alternating",
            nonlinearity="arctan",
        )
        assert circuit.num_qubits == 3
        assert circuit.num_layers == 2
        assert circuit.num_chebyshev == 2
        assert circuit.alpha == 3.0
        assert circuit.rotation_gate == "rx"
        assert circuit.hadamard_start is False
        assert circuit.arrangement == "alternating"
        assert circuit.nonlinearity == "arctan"

        with pytest.raises(ValueError):
            circuit.set_params(num_qubits=3, num_layers=2, num_chebyshev=1, nonlinearity="invalid")

    def test_get_circuit(self):
        circuit = ChebyshevTower(num_qubits=2, num_layers=1, num_chebyshev=1)
        features = np.array([0.5, -0.5])

        qc = circuit.get_circuit(features=features)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        with pytest.raises(EncodingSlotsMismatchError):
            ChebyshevTower(num_qubits=1, num_chebyshev=2, num_layers=1).get_circuit(
                features=features
            )

    def test_minimal_fit(self):
        circuit = ChebyshevTower(num_qubits=2, num_chebyshev=2)

        X_train = np.array([[-1.0, -0.8], [-0.6, -0.4], [-0.2, 0.0], [0.4, 0.6], [0.8, 1.0]])
        y_train = np.array([-0.6, -0.2, 0.2, 0.6, 1.0])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = ChebyshevTower(num_qubits=4, num_features=3, num_chebyshev=1)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)

    @pytest.mark.parametrize(
        "num_qubits,num_chebyshev,num_layers,rotation_gate,hadamard_start,arrangement,nonlinearity",
        [
            (4, 2, 2, "ry", True, "block", "arccos"),
            (3, 1, 1, "rx", False, "alternating", "arctan"),
            (2, 3, 1, "rz", True, "block", "arccos"),
            (4, 2, 1, "ry", True, "alternating", "arccos"),
        ],
    )
    def test_chebyshevtower_get_circuit_matches_ground_truth(
        self,
        num_qubits,
        num_chebyshev,
        num_layers,
        rotation_gate,
        hadamard_start,
        arrangement,
        nonlinearity,
    ):

        circuit = ChebyshevTower(
            num_qubits=num_qubits,
            num_chebyshev=num_chebyshev,
            num_layers=num_layers,
            rotation_gate=rotation_gate,
            hadamard_start=hadamard_start,
            arrangement=arrangement,
            nonlinearity=nonlinearity,
            alpha=1.5,
        )

        num_encoding_slots = circuit.num_encoding_slots
        num_features = min(2, num_encoding_slots) if num_encoding_slots >= 2 else 1
        features = np.linspace(-0.8, 0.8, num_features)

        qc_actual = circuit.get_circuit(features=features, parameters=None)

        qc_expected = _build_expected_chebyshev_tower(
            num_qubits=num_qubits,
            num_chebyshev=num_chebyshev,
            num_layers=num_layers,
            alpha=circuit.alpha,
            rotation_gate=rotation_gate,
            hadamard_start=hadamard_start,
            arrangement=arrangement,
            nonlinearity=nonlinearity,
            features=features,
        )

        assert_circuits_equal(qc_actual, qc_expected)
