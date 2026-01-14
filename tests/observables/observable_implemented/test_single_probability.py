import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn.observables import SingleProbability
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector


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

    @pytest.mark.parametrize("basis_state", ["00", "01", "10", "11"])
    @pytest.mark.parametrize("one_state", [False, True])
    def test_single_probability_returns_expected_expectation_value(self, basis_state, one_state):
        """
        Test that SingleProbability returns the correct probability on computational
        basis states.
        """
        num_qubits = 2
        measured_qubit = 0  # test probability on qubit 0

        ob = SingleProbability(
            num_qubits=num_qubits,
            qubit=measured_qubit,
            one_state=one_state,
            parameterized=False,
        )

        pauli = ob.get_pauli()

        # Prepare computational basis state
        qc = QuantumCircuit(num_qubits)
        for pos, bit in enumerate(basis_state):
            qubit_index = num_qubits - pos - 1
            if bit == "1":
                qc.x(qubit_index)

        state = Statevector.from_instruction(qc)

        # Expectation value from qiskit
        exp_val = state.expectation_value(pauli).real

        # Manual expectation value
        labels = list(pauli.paulis.to_labels())
        coeffs = pauli.coeffs

        expected_exp_val = 0.0
        for lbl, coeff in zip(labels, coeffs):
            term = 1.0
            for k, ch in enumerate(lbl):
                if ch == "I":
                    continue
                elif ch == "Z":
                    bit = basis_state[k]
                    term *= 1.0 if bit == "0" else -1.0
                else:
                    term *= 0.0
            expected_exp_val += float(coeff.real) * term

        assert np.isclose(exp_val, expected_exp_val)
