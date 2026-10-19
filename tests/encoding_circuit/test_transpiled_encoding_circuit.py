import pytest
from qc_executor import Parameters
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from squlearn.encoding_circuit import ChebyshevRx, TranspiledEncodingCircuit


class TestTranspiledEncodingCircuit:

    @pytest.fixture
    def circuit(self):
        return TranspiledEncodingCircuit(
            ChebyshevRx(3, 1), backend=FakeManilaV2(), num_features=1, initial_layout=[0, 1, 4]
        )

    def test_get_circuit_symbolic_is_repeatable(self, circuit):
        for _ in range(2):
            result = circuit.get_circuit(
                Parameters("x", 1), Parameters("p", circuit.num_parameters)
            )
            assert len(result.qiskit_circuit.parameters) == circuit.num_parameters + 1

    def test_get_circuit_numeric(self, circuit):
        values = [0.1] * circuit.num_parameters
        result = circuit.get_circuit([0.5], values)
        assert len(result.qiskit_circuit.parameters) == 0
