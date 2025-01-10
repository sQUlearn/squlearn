from qiskit import QuantumCircuit
from squlearn.encoding_circuit import ChebyshevRx
import pytest
import copy
import numpy as np

from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError


class TestChebyshevRx:
    def test_init(self):
        circuit = ChebyshevRx(num_qubits=2, num_features=2)
        assert circuit.num_features == 2
        assert circuit.num_qubits == 2
        assert circuit.num_layers == 1
        assert circuit.closed is False
        assert circuit.alpha == 4.0
        assert circuit.nonlinearity == "arccos"

        with pytest.raises(ValueError):
            ChebyshevRx(num_qubits=2, num_features=2, nonlinearity="invalid")

    def test_num_parameters(self):
        features = 2
        qubits = 2
        layers = 1
        circuit = ChebyshevRx(
            num_qubits=qubits, num_features=features, num_layers=layers, closed=True
        )
        assert circuit.num_parameters == 2 * qubits * layers

    def test_parameter_bounds(self):
        qubits = 2
        features = 2
        layers = 1
        circuit = ChebyshevRx(num_qubits=qubits, num_features=features, num_layers=layers)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)
        assert np.all(bounds[: qubits * layers, 0] == 0.0)
        assert np.all(bounds[: qubits * layers, 1] == circuit.alpha)
        assert np.all(bounds[qubits * layers :, 0] == -np.pi)
        assert np.all(bounds[qubits * layers :, 1] == np.pi)

    def test_feature_bounds(self):
        circuit = ChebyshevRx(num_features=2, num_qubits=2, num_layers=1, nonlinearity="arccos")
        bounds = circuit.feature_bounds
        assert bounds.shape == (circuit.num_features, 2)
        assert bounds[0, 0] == -1.0
        assert bounds[0, 1] == 1.0

        circuit = ChebyshevRx(num_features=2, num_qubits=2, num_layers=1, nonlinearity="arctan")
        bounds = circuit.feature_bounds
        assert bounds.shape == (circuit.num_features, 2)
        assert bounds[0, 0] == -np.inf
        assert bounds[0, 1] == np.inf

    def test_generate_initial_parameters(self):
        circuit = ChebyshevRx(num_qubits=2, num_features=2)
        params = circuit.generate_initial_parameters(seed=42)
        assert len(params) == circuit.num_parameters

    def test_get_params(self):
        circuit = ChebyshevRx(num_features=2, num_qubits=2, num_layers=1)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": 2,
            "num_qubits": 2,
            "num_layers": 1,
            "closed": False,
            "alpha": 4.0,
            "nonlinearity": "arccos",
        }

    def test_set_params(self):
        circuit = ChebyshevRx(num_features=2, num_qubits=2, num_layers=1)
        circuit.set_params(
            num_features=3,
            num_qubits=3,
            num_layers=2,
            closed=True,
            alpha=3.0,
            nonlinearity="arctan",
        )
        assert circuit.num_features == 3
        assert circuit.num_qubits == 3
        assert circuit.num_layers == 2
        assert circuit.closed is True
        assert circuit.alpha == 3.0
        assert circuit.nonlinearity == "arctan"

        with pytest.raises(ValueError):
            circuit.set_params(
                num_features=3, num_qubits=3, num_layers=2, closed=False, nonlinearity="invalid"
            )

    def test_get_circuit(self):
        circuit = ChebyshevRx(num_features=2, num_qubits=2, num_layers=1)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        used_params = {param for instruction in qc.data for param in instruction.operation.params}
        assert len(used_params) == len(params)

        with pytest.raises(EncodingSlotsMismatchError):
            ChebyshevRx(num_features=1, num_qubits=1, num_layers=1).get_circuit(
                features=features, parameters=params
            )

    def test_drawing_does_not_violate_circuit_parameters(self):
        circuit = ChebyshevRx(num_features=2, num_qubits=2, num_layers=1)

        params_with_features_before = copy.deepcopy(circuit.get_params())
        circuit.draw(output="mpl")
        params_with_features_after = copy.deepcopy(circuit.get_params())

        assert params_with_features_before == params_with_features_after

        # same but with num_features=None
        circuit = ChebyshevRx(num_features=2, num_qubits=2, num_layers=1)

        params_without_features_before = copy.deepcopy(circuit.get_params())
        circuit.draw(output="mpl")
        params_without_features_after = copy.deepcopy(circuit.get_params())

        assert params_without_features_before == params_without_features_after
