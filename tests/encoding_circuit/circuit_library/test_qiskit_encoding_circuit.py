import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit
from qc_executor import QuantumCircuit, Parameters

from squlearn.encoding_circuit import QiskitEncodingCircuit


class TestQiskitEncodingCircuit:
    def test_init_with_callable_returns_empty_mode(self):
        def make_circ(n_qubits=2):
            return QiskitQuantumCircuit(n_qubits)

        circuit = QiskitEncodingCircuit(make_circ, mode="auto", decompose=False, feature_label="x")
        assert circuit._mode == "empty"
        # get_circuit should return the underlying circuit unchanged for empty mode
        qc_out = circuit.get_circuit([], [])
        assert qc_out.qiskit_circuit is circuit._qiskit_circuit

    def test_mode_x_assigns_features_and_raises_on_wrong_length(self):
        params = Parameters("x", 2)
        qc = QiskitQuantumCircuit(2)
        qc.rz(params[0], 0)
        qc.rz(params[1], 1)

        enc = QiskitEncodingCircuit(qc, mode="x", feature_label="x")
        # constructor should set num_features and zero trainable parameters
        assert enc._mode == "x"
        assert enc._num_features == 2
        assert enc._num_parameters == 0

        # correct length -> returns a bound circuit without free parameters
        features = np.array([0.1, 0.2])
        qc_bound = enc.get_circuit(features=features, parameters=np.array([]))
        assert isinstance(qc_bound, QuantumCircuit)
        assert len(qc_bound.parameters) == 0

        with pytest.raises(ValueError):
            enc.get_circuit(features=np.array([0.1]), parameters=np.array([]))

    def test_mode_p_assigns_parameters_and_raises_on_wrong_length(self):
        params = Parameters("p", 3)
        qc = QiskitQuantumCircuit(3)
        qc.rz(params[0], 0)
        qc.rz(params[1], 1)
        qc.rz(params[2], 2)

        enc = QiskitEncodingCircuit(qc, mode="p", parameter_label="p")
        assert enc._mode == "p"
        assert enc._num_parameters == 3
        assert enc._num_features == 0

        # correct length -> returns bound circuit with no free parameters
        parameters = np.array([0.3, 0.4, 0.5])
        qc_bound = enc.get_circuit(features=np.array([]), parameters=parameters)
        assert isinstance(qc_bound, QuantumCircuit)
        assert len(qc_bound.parameters) == 0

        with pytest.raises(ValueError, match="does not match"):
            enc.get_circuit(features=np.array([]), parameters=np.array([0.1, 0.2]))

    def test_auto_mode_detects_only_features(self):
        x = Parameters("x", 2)
        qc = QiskitQuantumCircuit(2)
        qc.rz(x[0], 0)
        qc.rz(x[1], 1)

        enc = QiskitEncodingCircuit(qc, mode="auto", feature_label="x", parameter_label="p")
        assert enc._mode == "x"
        assert enc._num_features == 2
        assert enc._num_parameters == 0

        features = np.array([0.6, -0.6])
        qc_bound = enc.get_circuit(features=features, parameters=np.array([]))
        assert isinstance(qc_bound, QuantumCircuit)
        assert len(qc_bound.parameters) == 0

    def test_auto_mode_detects_only_parameters(self):
        p = Parameters("p", 2)
        qc = QiskitQuantumCircuit(2)
        qc.rz(p[0], 0)
        qc.rz(p[1], 1)

        enc = QiskitEncodingCircuit(qc, mode="auto", feature_label="x", parameter_label="p")
        assert enc._mode == "p"
        assert enc._num_parameters == 2
        assert enc._num_features == 0

        parameters = np.array([1.1, -1.1])
        qc_bound = enc.get_circuit(features=np.array([]), parameters=parameters)
        assert isinstance(qc_bound, QuantumCircuit)
        assert len(qc_bound.parameters) == 0

    def test_auto_mode_detects_features_and_parameters(self):
        p = Parameters("p", 1)
        x = Parameters("x", 1)
        qc = QiskitQuantumCircuit(2)
        qc.rz(p[0], 0)
        qc.rz(x[0], 1)

        enc = QiskitEncodingCircuit(qc, mode="auto", parameter_label="p", feature_label="x")
        assert enc._mode == "both"
        assert enc._num_parameters == 1
        assert enc._num_features == 1

        # get_circuit should map the first parameter to parameters arg and the second to features arg
        features = np.array([0.7])
        parameters = np.array([1.3])
        qc_bound = enc.get_circuit(features=features, parameters=parameters)
        assert isinstance(qc_bound, QuantumCircuit)
        assert len(qc_bound.parameters) == 0  # all assigned

    def test_auto_mode_raises_if_automatic_detection_fails(self):
        # create a circuit with parameters named 'a[0]' which won't match 'x' or 'p'
        a = Parameters("a", 1)
        qc = QiskitQuantumCircuit(1)
        qc.rz(a[0], 0)

        with pytest.raises(RuntimeError):
            QiskitEncodingCircuit(qc, mode="auto", parameter_label="p", feature_label="x")

    def test_invalid_mode_raises_value_error(self):
        qc = QiskitQuantumCircuit(1)
        with pytest.raises(ValueError):
            QiskitEncodingCircuit(qc, mode="not_a_mode")

    def test_get_params_returns_underlying_qiskit_circuit(self):
        p = Parameters("p", 1)
        qc = QiskitQuantumCircuit(1)
        qc.rz(p[0], 0)
        enc = QiskitEncodingCircuit(qc, mode="p", parameter_label="p")
        params = enc.get_params()
        assert "qiskit_circuit" in params
        assert params["qiskit_circuit"] is enc._qiskit_circuit
