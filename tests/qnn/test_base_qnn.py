"""Tests for BaseQNN"""

import pytest

import numpy as np

from squlearn import Executor
from squlearn.observables import IsingHamiltonian
from squlearn.encoding_circuit import ChebyshevPQC
from squlearn.optimizers import SLSQP
from squlearn.qnn import SquaredLoss
from squlearn.qnn.base_qnn import BaseQNN


class MockBaseQNN(BaseQNN):
    """Mock class for BaseQNN."""

    def _fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        pass


class TestBaseQNN:
    """Test class for BaseQNN."""

    @pytest.fixture(scope="module")
    def qnn_single_op(self) -> MockBaseQNN:
        """BaseQNN module with single operator."""
        np.random.seed(42)
        executor = Executor()
        pqc = ChebyshevPQC(num_qubits=4, num_features=1, num_layers=2)
        operator = IsingHamiltonian(num_qubits=4, I="S", Z="S", ZZ="S")
        loss = SquaredLoss()
        optimizer = SLSQP(options={"maxiter": 2})
        return MockBaseQNN(pqc, operator, executor, loss, optimizer)

    @pytest.fixture(scope="module")
    def qnn_multi_op(self) -> MockBaseQNN:
        """BaseQNN module with multiple operators."""
        np.random.seed(42)
        executor = Executor()
        pqc = ChebyshevPQC(num_qubits=4, num_features=1, num_layers=2)
        operator = [IsingHamiltonian(num_qubits=4, I="S", Z="S", ZZ="S") for _ in range(5)]
        loss = SquaredLoss()
        optimizer = SLSQP(options={"maxiter": 2})
        return MockBaseQNN(pqc, operator, executor, loss, optimizer)

    def test_set_params_invalid_param(self, qnn_single_op: MockBaseQNN):
        """
        Test if setting an invalid parameter raises a ValueError.

        Args:
            qnn_single_op (MockBaseQNN): The MockBaseQNN object to test.

        Returns:
            None
        """
        with pytest.raises(ValueError):
            qnn_single_op.set_params(invalid_param=3)

    def test_set_params_seed(self, qnn_single_op: MockBaseQNN):
        """
        Test `set_params` with `parameter_seed`.

        Args:
            qnn_single_op (MockBaseQNN): An instance of the `MockBaseQNN` class.

        Returns:
            None
        """
        qnn_single_op.set_params(parameter_seed=42)
        assert qnn_single_op.parameter_seed == 42

    def test_set_params_num_qubits_single_op(self, qnn_single_op: MockBaseQNN):
        """
        Test `set_params` with `num_qubits` for single operator.

        Args:
            qnn_single_op (MockBaseQNN): An instance of the `MockBaseQNN` class.

        Returns:
            None
        """
        qnn_single_op.set_params(num_qubits=5)
        assert qnn_single_op.encoding_circuit.num_qubits == 5
        assert qnn_single_op.operator.num_qubits == 5
        assert qnn_single_op._qnn.num_qubits == 5

    def test_set_params_num_qubits_multi_op(self, qnn_multi_op):
        """
        Test `set_params` with `num_qubits` for multiple operators.

        Args:
            qnn_multi_op (MockBaseQNN): An instance of the `MockBaseQNN` class.

        Returns:
            None
        """
        qnn_multi_op.set_params(num_qubits=5)
        assert qnn_multi_op.encoding_circuit.num_qubits == 5
        for operator in qnn_multi_op.operator:
            assert operator.num_qubits == 5
        assert qnn_multi_op._qnn.num_qubits == 5

    def test_set_params_encoding_circuit(self, qnn_single_op):
        """
        Test `set_params` for pqc parameters.

        Args:
            qnn_single_op (MockBaseQNN): An instance of the `MockBaseQNN` class.

        Returns:
            None
        """
        qnn_single_op.set_params(num_layers=3, closed=True)
        assert qnn_single_op.encoding_circuit.num_layers == 3
        assert qnn_single_op.encoding_circuit.closed
        assert qnn_single_op._qnn._pqc.get_params()["num_layers"] == 3
        assert qnn_single_op._qnn._pqc.get_params()["closed"]

    def test_set_params_single_operator(self, qnn_single_op):
        """
        Test `set_params` for single operator parameters.

        Args:
            qnn_single_op (MockBaseQNN): An instance of the `MockBaseQNN` class.

        Returns:
            None
        """
        qnn_single_op.set_params(X="S", Z="N")
        assert qnn_single_op.operator.X == "S"
        assert qnn_single_op.operator.Z == "N"
        assert qnn_single_op._qnn._observable.X == "S"
        assert qnn_single_op._qnn._observable.Z == "N"

    def test_set_params_multi_operator(self, qnn_multi_op):
        """
        Test `set_params` for multiple operator parameters.

        Args:
            qnn_multi_op (MockBaseQNN): An instance of the `MockBaseQNN` class.

        Returns:
            None
        """
        qnn_multi_op.set_params(op0__X="S", op3__Z="N")
        assert qnn_multi_op.operator[0].X == "S"
        assert qnn_multi_op.operator[3].Z == "N"
        assert qnn_multi_op._qnn._observable[0].X == "S"
        assert qnn_multi_op._qnn._observable[3].Z == "N"
