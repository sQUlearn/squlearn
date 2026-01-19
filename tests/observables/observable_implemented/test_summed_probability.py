import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from squlearn.observables import SummedProbabilities


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

    @pytest.mark.parametrize(
        "basis_state, expected_exp_val",
        [
            ("00", 6.0),
            ("01", 4.0),
            ("10", 3.0),
            ("11", 1.0),
        ],
    )
    def test_summed_probabilities_zero_state_returns_expected_expectation(
        self, basis_state, expected_exp_val
    ):
        """
        SummedProbabilities with one_state=False measures probability of |0> for each qubit.
        """
        num_qubits = 2
        # include_identity=True -> parameters: [I_coeff, p_q0, p_q1]
        ob = SummedProbabilities(
            num_qubits=num_qubits, one_state=False, full_sum=True, include_identity=True
        )

        # parameters: identity coeff = 1.0, per-qubit coeffs = 2.0, 3.0
        params = np.array([1.0, 2.0, 3.0])
        pauli = ob.get_pauli(parameters=params)

        # Prepare computational basis state according to the same ordering used in your previous test:
        qc = QuantumCircuit(num_qubits)
        for pos, bit in enumerate(basis_state):
            qubit_index = num_qubits - pos - 1
            if bit == "1":
                qc.x(qubit_index)

        state = Statevector.from_instruction(qc)

        # compute expectation by qiskit
        exp_val = float(state.expectation_value(pauli).real)

        assert np.isclose(exp_val, expected_exp_val)

    @pytest.mark.parametrize(
        "basis_state, expected_exp_val",
        [
            ("00", 0.5),
            ("01", 2.5),
            ("10", 3.5),
            ("11", 5.5),
        ],
    )
    def test_summed_probabilities_one_state_returns_expected_expectation(
        self, basis_state, expected_exp_val
    ):
        """
        SummedProbabilities with one_state=True measures probability of |1> for each qubit.
        """
        num_qubits = 2
        ob = SummedProbabilities(
            num_qubits=num_qubits, one_state=True, full_sum=True, include_identity=True
        )

        # parameters: identity coeff = 0.5, per-qubit coeffs = 2.0, 3.0 (values arbitrary)
        params = np.array([0.5, 2.0, 3.0])
        pauli = ob.get_pauli(parameters=params)

        qc = QuantumCircuit(num_qubits)
        for pos, bit in enumerate(basis_state):
            qubit_index = num_qubits - pos - 1
            if bit == "1":
                qc.x(qubit_index)

        state = Statevector.from_instruction(qc)
        exp_val = float(state.expectation_value(pauli).real)

        assert np.isclose(exp_val, expected_exp_val)
