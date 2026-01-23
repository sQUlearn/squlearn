import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from squlearn.observables import SinglePauli


class TestSinglePauli:

    def test_init_raises_on_invalid_operator(self):
        with pytest.raises(ValueError):
            SinglePauli(num_qubits=2, qubit=0, op_str="A", parameterized=False)

    def test_num_parameters(self):
        sp_nonparam = SinglePauli(num_qubits=2, qubit=0, op_str="Z", parameterized=False)
        assert sp_nonparam.num_parameters == 0

        sp_param = SinglePauli(num_qubits=2, qubit=0, op_str="Z", parameterized=True)
        assert sp_param.num_parameters == 1

    def test_get_params(self):
        sp = SinglePauli(num_qubits=4, qubit=2, op_str="X", parameterized=True)
        params = sp.get_params()
        assert params["qubit"] == 2
        assert params["op_str"] == "X"
        assert params["parameterized"] is True

    def test_get_pauli_non_parameterized_label_and_coeffs(self):
        num_qubits = 3
        qubit = 1
        sp = SinglePauli(num_qubits=num_qubits, qubit=qubit, op_str="X", parameterized=False)

        pauli_op = sp.get_pauli(parameters=np.array([]))
        assert isinstance(pauli_op, SparsePauliOp)

        # build expected label using the same slicing logic as implementation
        H = "I" * num_qubits
        expected_label = H[(qubit + 1) :] + "X" + H[:qubit]
        labels = list(pauli_op.paulis.to_labels())
        assert labels == [expected_label]

        # non-parameterized coefficients default to 1
        coeffs = np.asarray(pauli_op.coeffs, dtype=float)
        assert np.allclose(coeffs, np.array([1.0]))

    def test_get_pauli_parameterized_with_numpy_parameter(self):
        num_qubits = 2
        qubit = 0
        value = 2.5
        sp = SinglePauli(num_qubits=num_qubits, qubit=qubit, op_str="Z", parameterized=True)

        pauli_op = sp.get_pauli(parameters=np.array([value]))
        assert isinstance(pauli_op, SparsePauliOp)

        labels = list(pauli_op.paulis.to_labels())
        H = "I" * num_qubits
        expected_label = H[(qubit + 1) :] + "Z" + H[:qubit]
        assert labels == [expected_label]

        coeffs = np.asarray(pauli_op.coeffs, dtype=float)
        assert np.allclose(coeffs, np.array([value]))

    def test_get_pauli_parameterized_with_parameter_vector(self):
        num_qubits = 4
        qubit = 3
        sp = SinglePauli(num_qubits=num_qubits, qubit=qubit, op_str="Y", parameterized=True)

        pvec = ParameterVector("p", 1)
        pauli_op = sp.get_pauli(parameters=pvec)
        assert isinstance(pauli_op, SparsePauliOp)

        coeffs = pauli_op.coeffs
        assert len(coeffs) == 1

    def test_get_pauli_raises_on_invalid_qubit(self):
        sp = SinglePauli(num_qubits=1, qubit=2, op_str="Z", parameterized=False)
        with pytest.raises(ValueError):
            sp.get_pauli(parameters=np.array([]))

    @pytest.mark.parametrize(
        "basis_state, qubit, parameterized, coeff, expected_exp_val",
        [
            ("00", 0, False, 1.0, 1.0),
            ("01", 0, False, 1.0, -1.0),
            ("10", 1, False, 1.0, -1.0),
            ("11", 1, False, 1.0, -1.0),
            ("00", 0, True, 2.5, 2.5),
            ("01", 0, True, 2.5, -2.5),
            ("10", 1, True, -1.3, 1.3),
            ("11", 1, True, -1.3, 1.3),
        ],
    )
    def test_single_pauli_z_expectation_value(
        self, basis_state, qubit, parameterized, coeff, expected_exp_val
    ):
        """Z Pauli gives ±coeff depending on computational basis state."""
        num_qubits = 2

        ob = SinglePauli(
            num_qubits=num_qubits,
            qubit=qubit,
            op_str="Z",
            parameterized=parameterized,
        )

        params = np.array([coeff]) if parameterized else np.array([])
        pauli = ob.get_pauli(parameters=params)

        # Prepare basis state
        qc = QuantumCircuit(num_qubits)
        for pos, bit in enumerate(basis_state):
            q = num_qubits - pos - 1
            if bit == "1":
                qc.x(q)

        state = Statevector.from_instruction(qc)
        exp_val = state.expectation_value(pauli).real

        assert np.isclose(exp_val, expected_exp_val)

    @pytest.mark.parametrize("op_str", ["X", "Y"])
    @pytest.mark.parametrize("basis_state", ["00", "01", "10", "11"])
    @pytest.mark.parametrize("parameterized", [False, True])
    def test_single_pauli_xy_zero_on_computational_basis(self, op_str, basis_state, parameterized):
        """X and Y Pauli operators have zero expectation on basis states."""
        num_qubits = 2
        qubit = 0

        ob = SinglePauli(
            num_qubits=num_qubits,
            qubit=qubit,
            op_str=op_str,
            parameterized=parameterized,
        )

        params = np.array([3.7]) if parameterized else np.array([])
        pauli = ob.get_pauli(parameters=params)

        qc = QuantumCircuit(num_qubits)
        for pos, bit in enumerate(basis_state):
            q = num_qubits - pos - 1
            if bit == "1":
                qc.x(q)

        state = Statevector.from_instruction(qc)
        exp_val = state.expectation_value(pauli).real

        assert np.isclose(exp_val, 0.0)
