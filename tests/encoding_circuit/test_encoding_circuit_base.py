import pytest
import numpy as np
from qiskit import QuantumCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from squlearn.encoding_circuit.layered_encoding_circuit import LayeredEncodingCircuit


class MockCircuitBase(EncodingCircuitBase):
    def get_circuit(self, features, parameters):
        return QuantumCircuit(self.num_qubits)

    @property
    def num_parameters(self):
        return 2

    @property
    def num_encoding_slots(self):
        return 10


class TestEncodingCircuitBase:

    def test_init(self):
        with pytest.warns(DeprecationWarning):
            circuit = MockCircuitBase(num_qubits=2, num_features=2)
            assert circuit.num_qubits == 2
            assert circuit.num_features == 2

    def test_generate_initial_parameters(self):
        custom_circuit = MockCircuitBase(num_qubits=4)
        params = custom_circuit.generate_initial_parameters(seed=42, num_features=2)
        assert len(params) == 2
        assert (params >= -np.pi).all() and (params <= np.pi).all()

    def test_get_and_set_params(self):
        circuit = MockCircuitBase(num_qubits=4)
        params = circuit.get_params()
        assert params == {"num_qubits": 4, "num_features": None}

        circuit.set_params(num_qubits=5)
        assert circuit.num_qubits == 5

        with pytest.raises(ValueError):
            circuit.set_params(invalid_param=1)


class TestComposedEncodingCircuit:
    def test_add(self):
        circuit_1 = MockCircuitBase(num_qubits=4)
        circuit_2 = MockCircuitBase(num_qubits=4)
        circuit_3 = MockCircuitBase(num_qubits=3)
        circuit_composed = circuit_1 + circuit_2

        # check if the composed circuit has the correct number of qubits
        assert circuit_composed.num_qubits == 4
        assert circuit_composed.num_parameters == 4

        # check if the composed circuit has the correct number of parameters
        composed_params = circuit_composed.generate_initial_parameters(seed=42, num_features=None)
        assert len(composed_params) == 4
        assert (composed_params >= -np.pi).all() and (composed_params <= np.pi).all()

        # check if the composed circuit has the correct named parameters
        composed_named_params = circuit_composed.get_params()
        assert composed_named_params == {
            "ec1": circuit_1,
            "ec2": circuit_2,
            "ec1__num_features": None,
            "ec2__num_features": None,
            "num_qubits": 4,
        }

        # check if the composed circuit can be set with new parameters
        circuit_composed.set_params(num_qubits=5)
        assert circuit_composed.num_qubits == 5

        # unequal number of qubits
        with pytest.raises(ValueError):
            circuit_1 + circuit_3

        # invalid type
        with pytest.raises(ValueError):
            circuit_1 + "invalid"

    def test_get_circuit(self):
        circuit1 = LayeredEncodingCircuit(num_qubits=2)
        circuit2 = LayeredEncodingCircuit(num_qubits=2)

        circuit1.H()
        circuit2.H()

        composed_circuit = circuit1 + circuit2

        expected_circuit = QuantumCircuit(2)
        expected_circuit.h(range(2))
        expected_circuit.h(range(2))

        assert str(composed_circuit.get_circuit([], [])) == str(expected_circuit)

    def test_unequal_num_qubits_error(self):
        with pytest.raises(ValueError):
            MockCircuitBase(num_qubits=2) + MockCircuitBase(num_qubits=3)

    def test_num_encoding_slots(self):
        circuit1 = MockCircuitBase(num_qubits=4)
        circuit2 = MockCircuitBase(num_qubits=4)
        composed_circuit = circuit1 + circuit2
        assert (
            composed_circuit.num_encoding_slots
            == circuit1.num_encoding_slots + circuit2.num_encoding_slots
        )
