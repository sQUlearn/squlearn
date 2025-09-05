import pytest
import numpy as np
from squlearn.observables import SummedProbabilities
from qiskit.quantum_info import SparsePauliOp


class TestSummedProbability:

    @pytest.fixture
    def ob(self) -> SummedProbabilities:
        return SummedProbabilities(
            num_qubits=2, one_state=True, full_sum=True, include_identity=True
        )

    def test_init(self, ob: SummedProbabilities):
        assert ob.num_qubits == 2
        assert ob.one_state is True
        assert ob.full_sum is True
        assert ob.include_identity is True

    @pytest.mark.parametrize(
        "num_qubits, include_identity, full_sum, expected",
        [
            (2, True, True, 3),
            (2, True, False, 2),
            (2, False, False, 1),
        ],
    )
    def test_num_parameters(self, num_qubits, full_sum, include_identity, expected):
        ob = SummedProbabilities(
            num_qubits=num_qubits,
            full_sum=full_sum,
            include_identity=include_identity,
        )
        assert ob.num_parameters == expected

    def test_get_params(self, ob: SummedProbabilities):
        params = ob.get_params()
        assert params["num_qubits"] == 2
        assert params["one_state"] is True
        assert params["full_sum"] is True
        assert params["include_identity"] is True

    def test_get_pauli(self, ob: SummedProbabilities):
        pauli = ob.get_pauli(np.array([0.5]))
        assert isinstance(pauli, SparsePauliOp)
