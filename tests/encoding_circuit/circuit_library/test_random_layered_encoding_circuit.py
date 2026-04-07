import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn import Executor
from squlearn.encoding_circuit import RandomLayeredEncodingCircuit
from qiskit.circuit import ParameterVector

from squlearn.kernel.lowlevel_kernel import FidelityKernel
from squlearn.kernel import QGPR


class TestRandomLayeredEncodingCircuit:
    def test_random_layered_encoding_circuit_configuration(self):
        """
        The test checks if the circuit is generated correctly and uniquly for a given seed
        and set of parameters.
        """

        reference1 = (
            "     ┌─────────┐┌────────────┐┌────────────────┐\nq_0: ┤ Rz(π/4) ├┤ Ry(π*x[0]) ├┤ Rz("
            "atan(x[2])) ├\n     ├─────────┤├────────────┤├────────────────┤\nq_1: ┤ Rz(π/4) ├┤ Ry"
            "(π*x[1]) ├┤ Rz(atan(x[3])) ├\n     └─────────┘└────────────┘└────────────────┘"
        )
        reference2 = (
            "     ┌─────────┐┌─────────┐┌────────────────┐\nq_0: ┤ Rz(π/4) ├┤ Rx(π/2) ├┤ Rz(atan(x"
            "[0])) ├\n     ├─────────┤├─────────┤├────────────────┤\nq_1: ┤ Rz(π/4) ├┤ Rx(π/2) ├┤ "
            "Rz(atan(x[1])) ├\n     ├─────────┤├─────────┤├────────────────┤\nq_2: ┤ Rz(π/4) ├┤ Rx"
            "(π/2) ├┤ Rz(atan(x[2])) ├\n     └─────────┘└─────────┘└────────────────┘"
        )
        reference3 = (
            "     ┌───┐     ┌──────────┐            \nq_0: ┤ Z ├──■──┤ Rx(x[0]) ├────────────\n   "
            "  ├───┤┌─┴─┐└──────────┘┌──────────┐\nq_1: ┤ Z ├┤ X ├─────■──────┤ Rx(x[1]) ├\n     ├"
            "───┤└───┘   ┌─┴─┐    ├──────────┤\nq_2: ┤ Z ├────────┤ X ├────┤ Rx(x[2]) ├\n     └───"
            "┘        └───┘    └──────────┘"
        )

        pqc = RandomLayeredEncodingCircuit(num_qubits=2, max_num_layers=3)
        check_circuit1 = repr(pqc.draw("text", fold=-1, num_features=4))
        assert check_circuit1 == reference1

        pqc.set_params(num_qubits=3, max_num_layers=3)
        check_circuit2 = repr(pqc.draw("text", fold=-1, num_features=3))
        assert check_circuit2 == reference2

        pqc.set_params(seed=1234)
        check_circuit3 = repr(pqc.draw("text", fold=-1, num_features=3))
        assert check_circuit3 == reference3

    def test_init(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=42,
        )

        assert circuit.num_qubits == 2
        assert circuit.min_num_layers == 3
        assert circuit.max_num_layers == 12
        assert circuit.feature_probability == 0.5
        assert circuit.seed == 42

    def test_get_params(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=42,
        )
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": None,
            "num_qubits": 2,
            "min_num_layers": 3,
            "max_num_layers": 12,
            "feature_probability": 0.5,
            "seed": 42,
        }

    def test_set_params(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=42,
        )
        circuit.set_params(
            num_qubits=3,
            min_num_layers=2,
            max_num_layers=11,
            feature_probability=0.4,
        )

        assert circuit.num_qubits == 3
        assert circuit.min_num_layers == 2
        assert circuit.max_num_layers == 11
        assert circuit.feature_probability == 0.4

    def test_get_circuit(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=42,
        )
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

    def test_minimal_fit(self):
        circuit = RandomLayeredEncodingCircuit(
            num_qubits=2,
            min_num_layers=3,
            max_num_layers=12,
            feature_probability=0.5,
            seed=40,
        )

        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([5, 7, 9, 11, 13])

        kernel = FidelityKernel(encoding_circuit=circuit, executor=Executor())
        estimator = QGPR(quantum_kernel=kernel)

        estimator.fit(X_train, y_train)
        result = estimator.predict(X_train)

        assert np.allclose(result, y_train, atol=1e-3)

    def test_feature_consistency(self):
        circuit = RandomLayeredEncodingCircuit(num_qubits=4, num_features=3)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        with pytest.raises(ValueError):
            circuit.get_circuit(features, params)
