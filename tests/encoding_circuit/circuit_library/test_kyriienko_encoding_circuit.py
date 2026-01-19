import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import KyriienkoEncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError
from squlearn.encoding_circuit.layered_encoding_circuit import Layer, LayeredEncodingCircuit
from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR
from tests.qiskit_circuit_equivalence import assert_circuits_equal


def _variational_gate_block_expected(
    QC: QuantumCircuit,
    parameters: np.ndarray,
    index_offset: int,
    num_layers: int,
    num_qubits: int,
    shift_parameter: int = 0,
    variational_arrangement: str = "HEA",
    global_num_qubits: int = None,
):
    """
    Helper-Function for building expected variational gate blocks. Matches exactly the logic in
    KyriienkoEncodingCircuit.get_circuit.
    """
    if global_num_qubits is None:
        global_num_qubits = num_qubits

    qubit_starting_index = shift_parameter
    num_qubits_local = num_qubits + shift_parameter

    for layer in range(num_layers):
        for i_raw in range(qubit_starting_index, num_qubits_local):
            i = i_raw
            if i >= global_num_qubits:
                i = i - global_num_qubits
            QC.rz(parameters[index_offset], i)
            QC.rx(parameters[index_offset + 1], i)
            QC.rz(parameters[index_offset + 2], i)
            index_offset += 3

        if variational_arrangement == "HEA":
            for start in (0, 1):
                for i in range(start, num_qubits_local - 1, 2):
                    QC.cx(i, i + 1)
        elif variational_arrangement == "ABA":
            for start in (qubit_starting_index, qubit_starting_index + 1):
                for i_raw in range(start, num_qubits_local - 1, 2):
                    i = i_raw
                    if i + 1 < global_num_qubits:
                        QC.cx(i, i + 1)
                    else:
                        QC.cx(i, 0)
    return QC, index_offset


def _build_expected_kyriienko_circuit(
    num_qubits: int,
    num_features: int,
    encoding_style: str,
    variational_arrangement: str,
    num_encoding_layers: int,
    num_variational_layers: int,
    rotation_gate: str,
    parameters: np.ndarray,
    features: np.ndarray,
    block_width: int = 2,
    block_depth: int = 1,
    alpha: float = 2.0,
):
    """
    Build expected Kyriienko encoding circuit according to the specified parameters.
    """
    # product branch: delegated to LayeredEncodingCircuit
    if encoding_style == "chebyshev_product":
        QC_layered = LayeredEncodingCircuit(num_qubits=num_qubits, num_features=num_features)
        layer = Layer(QC_layered)
        # mapping in product branch: mapping layer.Rx/Ry/Rz ("x" name) with encoding=np.arcsin
        {"rx": layer.Rx, "ry": layer.Ry, "rz": layer.Rz}[rotation_gate]("x", encoding=np.arcsin)
        QC_layered.add_layer(layer, num_layers=num_encoding_layers)
        QC = QC_layered.get_circuit(features, [])
        # Now add variational layers same as below
    else:
        QC = QuantumCircuit(num_qubits)
        # chebyshev_tower or chebyshev_sparse
        iqubit = 0
        icheb = 1
        inner = num_features
        outer = num_qubits  # num_chebyshev in code equals num_qubits

        for _layer in range(num_encoding_layers):
            index_offset_encoding = 0
            iqubit = 0
            icheb = 1
            for outer_ in range(outer):
                for inner_ in range(inner):
                    angle = (
                        alpha * icheb * np.arccos(features[index_offset_encoding % num_features])
                    )
                    # pick rotation gate
                    if rotation_gate == "rx":
                        QC.rx(angle, iqubit % num_qubits)
                    elif rotation_gate == "ry":
                        QC.ry(angle, iqubit % num_qubits)
                    elif rotation_gate == "rz":
                        QC.rz(angle, iqubit % num_qubits)
                    iqubit += 1
                    index_offset_encoding += 1
                # tower: ncheb = 1 + icheb else stays 1 (sparse)
                if encoding_style == "chebyshev_tower":
                    icheb = 1 + icheb
                else:
                    icheb = 1

    # now add variational layers
    index_offset_variational = 0
    if variational_arrangement == "HEA":
        QC, index_offset_variational = _variational_gate_block_expected(
            QC,
            parameters,
            index_offset_variational,
            num_layers=num_variational_layers,
            num_qubits=num_qubits,
            variational_arrangement="HEA",
            global_num_qubits=num_qubits,
        )
    elif variational_arrangement == "ABA":
        if num_qubits % block_width != 0:
            raise ValueError(
                "Test precondition: block_width must divide num_qubits for ABA tests."
            )
        number_of_blocks = int(np.ceil(num_qubits / block_width))
        shifting_factor = int(np.floor(block_width / 2))

        for layer in range(num_variational_layers):
            if layer % 2 == 0:  # even layer
                for block in range(number_of_blocks):
                    start = int(block * block_width)
                    QC, index_offset_variational = _variational_gate_block_expected(
                        QC,
                        parameters,
                        index_offset_variational,
                        num_layers=block_depth,
                        num_qubits=block_width,
                        shift_parameter=start,
                        variational_arrangement="ABA",
                        global_num_qubits=num_qubits,
                    )
            else:
                for block in range(number_of_blocks):
                    start = int(shifting_factor + block * block_width)
                    QC, index_offset_variational = _variational_gate_block_expected(
                        QC,
                        parameters,
                        index_offset_variational,
                        num_layers=block_depth,
                        num_qubits=block_width,
                        shift_parameter=start,
                        variational_arrangement="ABA",
                        global_num_qubits=num_qubits,
                    )
            QC.barrier()
    return QC


class TestKyriienkoEncodingCircuit:
    def test_init(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=2)
        assert circuit.num_qubits == 2
        assert circuit.encoding_style == "chebyshev_tower"
        assert circuit.variational_arrangement == "HEA"
        assert circuit.num_encoding_layers == 1
        assert circuit.num_variational_layers == 1
        assert circuit.rotation_gate == "ry"
        assert circuit.block_width == 2
        assert circuit.block_depth == 1

        with pytest.raises(ValueError):
            KyriienkoEncodingCircuit(num_qubits=2, variational_arrangement="invalid")

        with pytest.raises(ValueError):
            KyriienkoEncodingCircuit(num_qubits=2, rotation_gate="invalid")

        with pytest.raises(ValueError):
            KyriienkoEncodingCircuit(num_qubits=1)

    def test_parameter_bounds(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=2)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)
        assert np.all(bounds == [-2.0 * np.pi, 2.0 * np.pi])

    def test_num_parameters(self):
        qubits = 2
        variational_layers = 1
        block_depth = 1
        circuit = KyriienkoEncodingCircuit(
            num_qubits=qubits,
            variational_arrangement="HEA",
            num_variational_layers=variational_layers,
            block_depth=block_depth,
        )
        assert circuit.num_parameters == 3 * qubits * variational_layers

        circuit = KyriienkoEncodingCircuit(
            num_qubits=qubits,
            variational_arrangement="ABA",
            num_variational_layers=variational_layers,
            block_depth=block_depth,
        )
        assert circuit.num_parameters == 3 * qubits * block_depth * variational_layers

    def test_get_params(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
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
        circuit = KyriienkoEncodingCircuit(num_qubits=2)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert qc.num_qubits == 2
        assert isinstance(qc, QuantumCircuit)

        with pytest.raises(EncodingSlotsMismatchError):
            circuit.get_circuit(features=np.array([0.3, 0.5, -0.5]), parameters=params)

    def test_minimal_fit(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=2)

        X_train = np.array([[-1.0, -0.8], [-0.6, -0.4], [-0.2, 0.0], [0.4, 0.6], [0.8, 1.0]])
        y_train = np.array([-0.6, -0.2, 0.2, 0.6, 1.0])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = KyriienkoEncodingCircuit(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)

    @pytest.mark.parametrize(
        "encoding_style,variational_arrangement,rotation_gate,num_qubits,num_features,enc_layers,var_layers",
        [
            ("chebyshev_tower", "HEA", "ry", 2, 1, 1, 1),
            ("chebyshev_tower", "ABA", "rx", 4, 1, 1, 2),
            ("chebyshev_sparse", "HEA", "rz", 3, 1, 1, 1),
            ("chebyshev_product", "HEA", "ry", 2, 1, 1, 1),
            ("chebyshev_product", "ABA", "rx", 4, 1, 1, 2),
        ],
    )
    def test_kyriienko_get_circuit_ground_truth(
        self,
        encoding_style,
        variational_arrangement,
        rotation_gate,
        num_qubits,
        num_features,
        enc_layers,
        var_layers,
    ):
        circuit = KyriienkoEncodingCircuit(
            num_qubits=num_qubits,
            num_features=num_features,
            num_encoding_layers=enc_layers,
            num_variational_layers=var_layers,
            variational_arrangement=variational_arrangement,
            rotation_gate=rotation_gate,
            encoding_style=encoding_style,
        )

        features = np.linspace(-0.9, 0.9, num_features)
        parameters = (
            np.linspace(-np.pi, np.pi, circuit.num_parameters)
            if circuit.num_parameters > 0
            else np.array([])
        )

        qc_actual = circuit.get_circuit(features=features, parameters=parameters)

        qc_expected = _build_expected_kyriienko_circuit(
            num_qubits=num_qubits,
            num_features=num_features,
            encoding_style=encoding_style,
            variational_arrangement=variational_arrangement,
            num_encoding_layers=enc_layers,
            num_variational_layers=var_layers,
            rotation_gate=rotation_gate,
            parameters=parameters,
            features=features,
            block_width=circuit.block_width,
            block_depth=circuit.block_depth,
            alpha=circuit.alpha,
        )

        assert_circuits_equal(qc_actual, qc_expected)
