import numpy as np
import pytest
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from squlearn.observables import CustomObservable


class TestCustomObservable:

    def test_init_accepts_valid_single_string(self):
        obs = CustomObservable(num_qubits=2, operator_string="ZI", parameterized=False)
        assert obs.num_qubits == 2
        assert isinstance(obs.operator_string, list)
        assert obs.operator_string == ["ZI"]

    def test_init_accepts_list_of_strings(self):
        obs = CustomObservable(num_qubits=2, operator_string=["ZI", "XX"], parameterized=False)
        assert obs.operator_string == ["ZI", "XX"]
        assert obs.num_qubits == 2

    def test_init_rejects_wrong_length_string(self):
        with pytest.raises(ValueError):
            CustomObservable(num_qubits=3, operator_string="ZI", parameterized=False)

    def test_init_rejects_invalid_characters(self):
        with pytest.raises(ValueError):
            CustomObservable(num_qubits=2, operator_string="ZA", parameterized=False)

    def test_num_parameters_property(self):
        obs_nonparam = CustomObservable(
            num_qubits=2, operator_string=["ZI", "XX"], parameterized=False
        )
        assert obs_nonparam.num_parameters == 0

        obs_param = CustomObservable(
            num_qubits=2, operator_string=["ZI", "XX"], parameterized=True
        )
        assert obs_param.num_parameters == 2

    def test_get_params(self):
        obs = CustomObservable(num_qubits=2, operator_string="ZI", parameterized=True)
        params = obs.get_params()
        assert "operator_string" in params
        assert params["operator_string"] == ["ZI"]
        assert "parameterized" in params
        assert params["parameterized"] is True

    def test_get_pauli_non_parameterized_returns_sparsepauliop(self):
        op_strings = ["ZI", "XX"]
        obs = CustomObservable(num_qubits=2, operator_string=op_strings, parameterized=False)

        pauli_op = obs.get_pauli()
        assert isinstance(pauli_op, SparsePauliOp)

        # number of terms
        assert pauli_op.size == len(op_strings)

        # paulis labels should match the supplied operator strings
        labels = list(pauli_op.paulis.to_labels())
        assert labels == op_strings

        coeffs = np.asarray(pauli_op.coeffs, dtype=float)
        assert np.allclose(coeffs, np.ones(len(op_strings)))

    def test_get_pauli_parameterized_with_parameter_vector_has_parameters(self):
        op_strings = ["ZII", "XXY", "YYY"]
        obs = CustomObservable(num_qubits=3, operator_string=op_strings, parameterized=True)

        # Create a ParameterVector of length 3 (one parameter per operator)
        pvec = ParameterVector("p", 3)
        pauli_op = obs.get_pauli(parameters=pvec)

        assert isinstance(pauli_op, SparsePauliOp)
        # The SparsePauliOp should expose the free parameters used in coeffs
        # and they should match the ParameterVector elements
        pauli_params = tuple(pauli_op.parameters)
        assert set(pauli_params) == set(pvec)

        assert pauli_op.size == len(op_strings)

    def test_get_pauli_parameterized_with_numpy_params_repeats_params(self):
        # If parameters is a numpy array shorter than operator list, the code repeats them modulo nparam
        op_strings = ["ZII", "XXX", "YYY"]
        obs = CustomObservable(num_qubits=3, operator_string=op_strings, parameterized=True)

        # single numeric parameter: should be applied to all operators
        params = np.array([2.5])
        pauli_op = obs.get_pauli(parameters=params)

        coeffs = np.asarray(pauli_op.coeffs, dtype=float)
        expected = np.array([2.5, 2.5, 2.5], dtype=float)
        assert np.allclose(coeffs, expected)

    def test_get_pauli_parameterized_with_numpy_params_cycle(self):
        # multiple numeric parameters should be assigned in order; if fewer than operators, should cycle
        op_strings = ["ZZII", "XXXX", "YYYY", "ZZZZ"]
        obs = CustomObservable(num_qubits=4, operator_string=op_strings, parameterized=True)

        params = np.array([1.0, 2.0])  # will be used as [1.0, 2.0, 1.0, 2.0]
        pauli_op = obs.get_pauli(parameters=params)

        coeffs = np.asarray(pauli_op.coeffs, dtype=float)
        expected = np.array([1.0, 2.0, 1.0, 2.0], dtype=float)
        assert np.allclose(coeffs, expected)
