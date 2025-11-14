import numpy as np
import pytest
from squlearn.observables import SingleProbability
from qiskit.quantum_info import SparsePauliOp, Pauli


class TestSingleProbability:

    @pytest.fixture
    def ob(self) -> SingleProbability:
        return SingleProbability(num_qubits=3, qubit=1, one_state=True, parameterized=True)

    def test_init(self, ob: SingleProbability):
        assert ob.num_qubits == 3
        assert ob.one_state is True
        assert ob.parameterized is True
        assert ob.num_parameters == 1

    @pytest.mark.parametrize(
        "parmeterized, expected_num_params",
        [
            (True, 1),
            (False, 0),
        ],
    )
    def test_num_parameters(self, parmeterized, expected_num_params):
        ob = SingleProbability(num_qubits=2, qubit=0, one_state=False, parameterized=parmeterized)
        assert ob.num_parameters == expected_num_params

    def test_get_params(self, ob: SingleProbability):
        params = ob.get_params()
        assert params["num_qubits"] == 3
        assert params["qubit"] == 1
        assert params["one_state"] is True
        assert params["parameterized"] is True

    def test_get_pauli(self, ob: SingleProbability):
        pauli = ob.get_pauli(np.array([0.5]))
        assert isinstance(pauli, SparsePauliOp)
        assert pauli.num_qubits == 3
        assert pauli.paulis[0] == Pauli("III")
