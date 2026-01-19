import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from squlearn.observables import IsingHamiltonian


class TestIsingHamiltonian:

    def _expected_operator_sequence(self, num_qubits, I, Z, X, ZZ):
        """Helper: build expected operator strings in the same order as IsingHamiltonian.get_pauli."""

        def gen_double_ising_string(i, j):
            H = "I" * num_qubits
            H = H[i + 1 :] + "Z" + H[:i]
            if i != j:
                H = H[: num_qubits - j - 1] + "Z" + H[num_qubits - j :]
            return H

        def gen_single_ising_string(i, s):
            H = "I" * num_qubits
            H = H[i + 1 :] + s + H[:i]
            return H

        op_list = []

        if I == "S":
            op_list.append("I" * num_qubits)

        if Z in ("S", "F"):
            for i in range(num_qubits):
                op_list.append(gen_single_ising_string(i, "Z"))

        if X in ("S", "F"):
            for i in range(num_qubits):
                op_list.append(gen_single_ising_string(i, "X"))

        if ZZ in ("S", "F"):
            for i in range(num_qubits):
                for j in range(i):
                    op_list.append(gen_double_ising_string(i, j))

        return op_list

    def test_init_raises_on_invalid_term(self):
        with pytest.raises(ValueError):
            IsingHamiltonian(num_qubits=2, I="X", Z="N", X="N", ZZ="N")
        with pytest.raises(ValueError):
            IsingHamiltonian(num_qubits=2, I="N", Z="Y", X="N", ZZ="N")

    @pytest.mark.parametrize(
        "num_qubits,I,Z,X,ZZ,expected",
        [
            (2, "S", "N", "N", "N", 1),  # only identity S -> 1 param
            (3, "N", "F", "N", "N", 3),  # Z=F -> num_qubits params
            (4, "N", "N", "S", "N", 1),  # X=S -> 1 param
            (3, "N", "N", "N", "F", 3),  # ZZ=F for 3 qubits -> 3 pairs
            (3, "S", "S", "S", "S", None),  # compute expected by formula below
        ],
    )
    def test_num_parameters_counts(self, num_qubits, I, Z, X, ZZ, expected):
        ob = IsingHamiltonian(num_qubits=num_qubits, I=I, Z=Z, X=X, ZZ=ZZ)
        if expected is not None:
            assert ob.num_parameters == expected
        else:
            calc = 0
            if I == "S":
                calc += 1
            if Z == "S":
                calc += 1
            elif Z == "F":
                calc += num_qubits
            if X == "S":
                calc += 1
            elif X == "F":
                calc += num_qubits
            if num_qubits > 1:
                if ZZ == "S":
                    calc += 1
                elif ZZ == "F":
                    calc += (num_qubits * (num_qubits - 1)) // 2
            assert ob.num_parameters == calc

    def test_get_pauli_all_terms_and_coeffs_with_numpy_params(self):
        """Construct an Ising Hamiltonian with many terms and verify labels & coeffs mapping (modulo cycling)."""
        num_qubits = 3
        # choose variety: I='S' (1), Z='F' (3), X='F' (3), ZZ='F' (3 for 3 qubits) => total 10 terms
        ob = IsingHamiltonian(num_qubits=num_qubits, I="S", Z="F", X="F", ZZ="F")
        # expected op list using the same ordering as get_pauli
        expected_ops = self._expected_operator_sequence(num_qubits, "S", "F", "F", "F")
        assert len(expected_ops) > 0

        # prepare parameter vector shorter than number of ops to test cycling
        params = np.array([1.0, 2.0, 3.0])  # nparam = 3, will map cyclically over 10 ops
        pauli = ob.get_pauli(parameters=params)

        assert isinstance(pauli, SparsePauliOp)
        labels = list(pauli.paulis.to_labels())
        assert labels == expected_ops

        coeffs = np.asarray(pauli.coeffs, dtype=float)
        # expected coefficients constructed by cycling through params in the same order used by implementation
        nparam = len(params)
        expected_coeffs = np.array(
            [params[i % nparam] for i in range(len(expected_ops))], dtype=float
        )
        assert np.allclose(coeffs, expected_coeffs)

    def test_get_pauli_parameterized_with_parametervector_preserves_parameters(self):
        """When passing a ParameterVector, the coeffs should be Parameter objects in the same cyclical order."""
        num_qubits = 4
        ob = IsingHamiltonian(num_qubits=num_qubits, I="S", Z="F", X="N", ZZ="F")
        # build expected ops
        expected_ops = self._expected_operator_sequence(num_qubits, "S", "F", "N", "F")

        # make a ParameterVector shorter than the number of ops to ensure cycling
        nparams = 2
        pvec = ParameterVector("p", nparams)
        pauli = ob.get_pauli(parameters=pvec)

        assert isinstance(pauli, SparsePauliOp)
        labels = list(pauli.paulis.to_labels())
        assert labels == expected_ops

    def test_get_pauli_raises_if_no_terms(self):
        ob = IsingHamiltonian(num_qubits=3, I="N", Z="N", X="N", ZZ="N")
        with pytest.raises(ValueError):
            ob.get_pauli(parameters=np.array([1.0]))

    @pytest.mark.parametrize(
        "basis_state, expected_exp_val",
        [
            ("00", 8.0),
            ("01", -2.0),
            ("10", -2.0),
            ("11", 0.0),
        ],
    )
    def test_obeservable_retruns_expected_expectation_value(self, basis_state, expected_exp_val):
        """Test that the observable returns expected expectation value on a set of basis states."""
        num_qubits = 2
        ob = IsingHamiltonian(num_qubits=num_qubits, I="S", Z="S", X="N", ZZ="S")
        # parameters for I, Z, ZZ terms
        params = np.array([1.0, 2.0, 3.0])  # I coeff=1.0, Z coeff=2.0, ZZ coeff=3.0

        pauli = ob.get_pauli(parameters=params)

        # Prepare the requested computational basis state (left-to-right in the string
        # corresponds to qubits num_qubits-1 .. 0 as used by Pauli label strings).
        qc = QuantumCircuit(num_qubits)
        for pos, bit in enumerate(basis_state):
            # map string position to qubit index
            qubit_index = num_qubits - pos - 1
            if bit == "1":
                qc.x(qubit_index)

        state = Statevector.from_instruction(qc)

        # Compute expectation value using qiskit (the value under test)
        exp_val = state.expectation_value(pauli).real

        assert np.isclose(exp_val, expected_exp_val)
