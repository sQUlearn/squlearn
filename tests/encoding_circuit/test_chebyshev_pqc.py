import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn.encoding_circuit import ChebyshevPQC
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError


class TestChebyshevPQC:
    def test_init(self):
        circuit = ChebyshevPQC(num_features=2, num_qubits=2)
        assert circuit.num_features == 2
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

    def test_num_parameters_closed(self):
        features = 2
        qubits = 3
        layers = 1
        circuit = ChebyshevPQC(
            num_features=features, num_qubits=qubits, num_layers=layers, closed=True
        )
        assert circuit.num_parameters == 2 * qubits + qubits * layers + qubits * layers

    def test_num_parameters_none_closed(self):
        features = 2
        qubits = 3
        layers = 1
        circuit = ChebyshevPQC(
            num_features=features, num_qubits=qubits, num_layers=layers, closed=False
        )
        assert circuit.num_parameters == 2 * qubits + qubits * layers + (qubits - 1) * layers

    def test_parameter_bounds(self):
        circuit = ChebyshevPQC(num_features=2, num_qubits=2, num_layers=1)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)

    def test_generate_initial_parameters(self):
        circuit = ChebyshevPQC(num_qubits=2, num_features=2)
        params = circuit.generate_initial_parameters(seed=42)
        assert len(params) == circuit.num_parameters

    def test_feature_bounds(self):
        circuit = ChebyshevPQC(num_features=2, num_qubits=2, num_layers=1, nonlinearity="arccos")
        bounds = circuit.feature_bounds
        assert bounds.shape == (circuit.num_features, 2)
        assert bounds[0, 0] == -1.0
        assert bounds[0, 1] == 1.0

        circuit = ChebyshevPQC(num_features=2, num_qubits=2, num_layers=1, nonlinearity="arctan")
        bounds = circuit.feature_bounds
        assert bounds.shape == (circuit.num_features, 2)
        assert bounds[0, 0] == -np.inf
        assert bounds[0, 1] == np.inf

    def test_get_params(self):
        circuit = ChebyshevPQC(num_features=2, num_qubits=2, num_layers=1)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": 2,
            "num_qubits": 2,
            "num_layers": 1,
            "closed": True,
            "entangling_gate": "crz",
            "alpha": 4.0,
            "nonlinearity": "arccos",
        }

    def test_set_params(self):
        circuit = ChebyshevPQC(num_features=2, num_qubits=2, num_layers=1)
        circuit.set_params(
            num_features=3,
            num_qubits=3,
            num_layers=2,
            closed=False,
            alpha=3.0,
            nonlinearity="arctan",
        )
        assert circuit.num_features == 3
        assert circuit.num_qubits == 3
        assert circuit.num_layers == 2
        assert circuit.closed is False
        assert circuit.alpha == 3.0
        assert circuit.nonlinearity == "arctan"

        with pytest.raises(ValueError):
            circuit.set_params(
                num_features=3, num_qubits=3, num_layers=2, closed=False, nonlinearity="invalid"
            )

    def test_get_circuit(self):
        circuit = ChebyshevPQC(num_features=2, num_qubits=2, num_layers=1)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        used_params = {param for instruction in qc.data for param in instruction.operation.params}
        assert len(used_params) == len(params)

        with pytest.raises(EncodingSlotsMismatchError):
            ChebyshevPQC(num_features=1, num_qubits=1, num_layers=1).get_circuit(
                features=features, parameters=params
            )
