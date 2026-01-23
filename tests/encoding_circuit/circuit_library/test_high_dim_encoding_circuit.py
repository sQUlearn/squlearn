import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from squlearn import Executor
from squlearn.encoding_circuit import HighDimEncodingCircuit
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR
from tests.qiskit_circuit_equivalence import assert_circuits_equal


def _build_expected_highdim_circuit(
    num_qubits: int,
    num_features: int,
    layer_type: str,
    cycling: bool,
    cycling_type: str,
    features: np.ndarray,
    num_layers: int,
    entangling_gate: str,
):
    QC = QuantumCircuit(num_qubits)
    QC.h(range(num_qubits))

    def build_layer(QC, feature_vec, index_offset):
        if layer_type == "rows":
            rows = True
        else:
            rows = False

        for i in range(3 * num_qubits):
            if rows:
                iqubit = i // 3
            else:
                iqubit = i % num_qubits

            ii = index_offset + i
            if cycling:
                if cycling_type == "saw":
                    ii = ii % num_features
                elif cycling_type == "hat":
                    itest = ii % max(num_features + num_features - 2, 1)
                    ii = itest if itest < num_features else num_features + num_features - 2 - itest
            else:
                if ii >= num_features:
                    break

            # Rz,Ry,Rz
            if rows:
                if i % 3 == 0:
                    QC.rz(feature_vec[ii], iqubit)
                elif i % 3 == 1:
                    QC.ry(feature_vec[ii], iqubit)
                else:
                    QC.rz(feature_vec[ii], iqubit)
            else:
                if i // num_qubits == 0:
                    QC.rz(feature_vec[ii], iqubit)
                elif i // num_qubits == 1:
                    QC.ry(feature_vec[ii], iqubit)
                else:
                    QC.rz(feature_vec[ii], iqubit)
        return QC

    def entangle_layer_cx(QC):
        for i in range(0, num_qubits - 1, 2):
            QC.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            QC.cx(i, i + 1)
        return QC

    def entangle_layer_iswap(QC):
        siswap = QuantumCircuit(2)
        siswap.cx(0, 1)
        siswap.cs(1, 0)
        siswap.ch(1, 0)
        siswap.cs(1, 0)
        siswap.cx(0, 1)
        siswap_gate = siswap.to_gate(label="siswap")

        for i in range(0, num_qubits - 1, 2):
            QC.append(siswap_gate, [i, i + 1])
        for i in range(1, num_qubits - 1, 2):
            QC.append(siswap_gate, [i, i + 1])
        return QC

    index_offset = 0
    for i in range(num_layers):
        if i != 0:
            if entangling_gate == "cx":
                QC = entangle_layer_cx(QC)
            else:
                QC = entangle_layer_iswap(QC)
        QC = build_layer(QC, features, index_offset)
        index_offset += num_qubits * 3
        if not cycling and index_offset >= num_features:
            index_offset = 0
    return QC


class TestHighDimEncodingCircuit:

    def test_init(self):
        circuit = HighDimEncodingCircuit(num_qubits=2)
        assert circuit.num_qubits == 2
        assert circuit.cycling is True
        assert circuit.cycling_type == "saw"
        assert circuit.num_layers is None
        assert circuit.layer_type == "rows"
        assert circuit.entangling_gate == "iswap"

        with pytest.raises(ValueError):
            HighDimEncodingCircuit(num_qubits=2, cycling_type="invalid")

        with pytest.raises(ValueError):
            HighDimEncodingCircuit(num_qubits=2, layer_type="invalid")

        with pytest.raises(ValueError):
            HighDimEncodingCircuit(num_qubits=2, entangling_gate="invalid")

    def test_get_params(self):
        circuit = HighDimEncodingCircuit(num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "num_layers": None,
            "cycling": True,
            "cycling_type": "saw",
            "layer_type": "rows",
            "entangling_gate": "iswap",
        }

    def test_get_circuit(self):
        circuit = HighDimEncodingCircuit(num_qubits=2)
        features = np.array([0.5, -0.5])

        qc = circuit.get_circuit(features=features)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        circuit = HighDimEncodingCircuit(
            num_qubits=4,
        )
        features = ParameterVector("x", 12)
        qc = circuit.get_circuit(features)
        assert circuit.num_layers == 2

    def test_minimal_fit(self):
        circuit = HighDimEncodingCircuit(num_qubits=2)

        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([5, 7, 9, 11, 13])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = HighDimEncodingCircuit(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)

    @pytest.mark.parametrize(
        "num_qubits,num_layers,cycling,cycling_type,layer_type,entangling_gate",
        [
            (2, 1, True, "saw", "rows", "cx"),
            (3, 2, True, "hat", "columns", "iswap"),
            (4, 1, False, "saw", "rows", "cx"),
            (3, 2, True, "saw", "columns", "cx"),
        ],
    )
    def test_highdim_get_circuit_matches_ground_truth(
        self, num_qubits, num_layers, cycling, cycling_type, layer_type, entangling_gate
    ):
        circuit = HighDimEncodingCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            cycling=cycling,
            cycling_type=cycling_type,
            layer_type=layer_type,
            entangling_gate=entangling_gate,
        )

        num_features = 5
        features = np.linspace(-0.8, 0.8, num_features)

        parameters = None

        qc_actual = circuit.get_circuit(features=features, parameters=parameters)

        qc_expected = _build_expected_highdim_circuit(
            num_qubits,
            num_features,
            layer_type,
            cycling,
            cycling_type,
            features,
            num_layers,
            entangling_gate,
        )
        assert_circuits_equal(qc_actual, qc_expected)
