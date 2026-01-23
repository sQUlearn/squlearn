import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from squlearn.observables import SummedPaulis


class TestSummedPaulis:

    def test_init_raises_on_invalid_operator(self):
        with pytest.raises(ValueError):
            SummedPaulis(num_qubits=2, op_str=("Z", "A"))

    @pytest.mark.parametrize(
        "num_qubits, op_str, full_sum, include_identity, expected",
        [
            (3, ("Z",), True, True, 1 + 1 * 3),  # identity + Z full sum (3)
            (2, ("X", "Z"), True, False, 2 * 2),  # no identity, two ops full sum => 4
            (4, ("Y",), False, True, 1 + 1),  # identity + single param for Y (full_sum False)
            (3, ("X", "Z"), False, False, 2),  # two ops, no identity, full_sum False => 2
        ],
    )
    def test_num_parameters_counts(self, num_qubits, op_str, full_sum, include_identity, expected):
        ob = SummedPaulis(
            num_qubits=num_qubits,
            op_str=op_str,
            full_sum=full_sum,
            include_identity=include_identity,
        )
        assert ob.num_parameters == expected

    def test_get_params(self):
        ob = SummedPaulis(num_qubits=2, op_str=("Z", "X"), full_sum=False, include_identity=True)
        params = ob.get_params()
        assert params["op_str"] == ("Z", "X")
        assert params["full_sum"] is False
        assert params["include_identity"] is True

    def _gen_string(self, num_qubits, i, op_str):
        H = "I" * num_qubits
        return H[(i + 1) :] + op_str + H[:i]

    def test_get_pauli_full_sum_labels_and_coeffs(self):
        num_qubits = 2
        op_str = ("Z", "X")
        ob = SummedPaulis(
            num_qubits=num_qubits, op_str=op_str, full_sum=True, include_identity=True
        )

        nparams = ob.num_parameters
        params = np.arange(1, nparams + 1, dtype=float)
        pauli = ob.get_pauli(parameters=params)

        assert isinstance(pauli, SparsePauliOp)

        # build expected operator labels in same order as implementation:
        expected_ops = []
        # identity first if included
        if ob.include_identity:
            expected_ops.append("I" * num_qubits)
        # then each op_str over each qubit (full_sum True => distinct params per appended term)
        for op in op_str:
            for i in range(num_qubits):
                expected_ops.append(self._gen_string(num_qubits, i, op))

        labels = list(pauli.paulis.to_labels())
        assert labels == expected_ops

        # coefficients should equal the provided params in order
        coeffs = np.asarray(pauli.coeffs, dtype=float)
        assert np.allclose(coeffs, params)

    def test_get_pauli_non_full_sum_repeats_coeffs_per_op_str(self):
        # full_sum=False -> one parameter per op_str (per all qubits)
        num_qubits = 3
        op_str = ("Z", "X")
        ob = SummedPaulis(
            num_qubits=num_qubits, op_str=op_str, full_sum=False, include_identity=False
        )

        # with full_sum=False and include_identity=False, num_parameters == len(op_str)
        assert ob.num_parameters == len(op_str)

        params = np.array([10.0, 20.0])
        pauli = ob.get_pauli(parameters=params)

        labels = list(pauli.paulis.to_labels())
        # expected ordering: for each op_str in order, append that op for each qubit
        expected_ops = []
        for op in op_str:
            for i in range(num_qubits):
                expected_ops.append(self._gen_string(num_qubits, i, op))
        assert labels == expected_ops

        coeffs = np.asarray(pauli.coeffs, dtype=float)
        # expected coeffs: [10 repeated 3 times, then 20 repeated 3 times]
        expected_coeffs = np.concatenate(
            [np.full(num_qubits, params[i]) for i in range(len(params))]
        )
        assert np.allclose(coeffs, expected_coeffs)

    def test_include_identity_flag_affects_first_term(self):
        num_qubits = 2
        op_str = ("Z",)
        ob_with_id = SummedPaulis(
            num_qubits=num_qubits, op_str=op_str, full_sum=False, include_identity=True
        )
        ob_without_id = SummedPaulis(
            num_qubits=num_qubits, op_str=op_str, full_sum=False, include_identity=False
        )

        params_with = np.array(
            [1.0, 2.0]
        )  # first for identity, second for Z (since full_sum False)
        pa_with = ob_with_id.get_pauli(parameters=params_with)
        labels_with = list(pa_with.paulis.to_labels())
        assert labels_with[0] == "I" * num_qubits  # identity first

        params_without = np.array([2.0])
        pa_without = ob_without_id.get_pauli(parameters=params_without)
        labels_without = list(pa_without.paulis.to_labels())
        # first label should be the Z operator on qubit 0 (per generator order)
        assert labels_without[0] != "I" * num_qubits

    @pytest.mark.parametrize(
        "basis_state, expected_exp_val",
        [
            ("00", 6.0),
            ("01", 2.0),
            ("10", 0.0),
            ("11", -4.0),
        ],
    )
    def test_summed_paulis_returns_expected_expectation_value(self, basis_state, expected_exp_val):
        """Test that SummedPaulis returns expected expectation value on basis states."""
        num_qubits = 2

        # Observable: I + sum_i Z_i  (full_sum=True => separate parameter per qubit)
        ob = SummedPaulis(
            num_qubits=num_qubits,
            op_str="Z",
            full_sum=True,
            include_identity=True,
        )

        # Parameters:
        # [I_coeff, Z(q0)_coeff, Z(q1)_coeff]
        params = np.array([1.0, 2.0, 3.0])

        pauli = ob.get_pauli(parameters=params)

        # Prepare computational basis state
        qc = QuantumCircuit(num_qubits)
        for pos, bit in enumerate(basis_state):
            qubit_index = num_qubits - pos - 1
            if bit == "1":
                qc.x(qubit_index)

        state = Statevector.from_instruction(qc)

        # Expectation value from Qiskit
        exp_val = state.expectation_value(pauli).real

        assert np.isclose(exp_val, expected_exp_val)
