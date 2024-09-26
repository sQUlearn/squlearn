""" Tests for BaseQRC """

import pytest

import numpy as np

from squlearn import Executor
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from sklearn.linear_model import LinearRegression
from squlearn.qrc.base_qrc import BaseQRC


class MockBaseQRC(BaseQRC):
    """Mock class for BaseQRC."""

    def _initialize_ml_model(self) -> None:
        """Initialize the machine learning model, has to be implemented in the child class"""
        self._ml_model = LinearRegression()


class TestBaseQRC:
    """Test class for BaseQRC."""

    @pytest.fixture(scope="module")
    def qrc_class(self) -> MockBaseQRC:
        """BaseQRC module."""
        executor = Executor()
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=4, num_features=2)
        return MockBaseQRC(encoding_circuit, executor)

    def test_set_params_invalid_param(self, qrc_class: MockBaseQRC):
        """
        Test if setting an invalid parameter raises a ValueError.

        Args:
            qrc_class (MockBaseQRC): The MockBaseQRC object to test.

        Returns:
            None
        """
        with pytest.raises(ValueError):
            qrc_class.set_params(invalid_param=3)

    def test_set_params_seed(self, qrc_class: MockBaseQRC):
        """
        Test `set_params` with `parameter_seed`.

        Args:
            qrc_class (MockBaseQRC): An instance of the `MockBaseQRC` class.

        Returns:
            None
        """
        qrc_class.set_params(parameter_seed=42)
        assert qrc_class.parameter_seed == 42

    def test_set_params_num_qubits(self, qrc_class: MockBaseQRC):
        """
        Test `set_params` with `num_qubits` for QRC base class.

        Args:
            qrc_class (MockBaseQRC): An instance of the `MockBaseQRC` class.

        Returns:
            None
        """
        qrc_class.set_params(num_qubits=5)
        assert qrc_class.encoding_circuit.num_qubits == 5
        assert qrc_class._operators[0].num_qubits == 5
        assert qrc_class._qnn.num_qubits == 5

    def test_set_params_num_operators(self, qrc_class: MockBaseQRC):
        """
        Test `set_params` with `num_operators` for QRC base class

        Args:
            qrc_class (MockBaseQRC): An instance of the `MockBaseQRC` class.

        Returns:
            None
        """
        qrc_class.set_params(num_operators=12)
        assert len(qrc_class._operators) == 12
        assert qrc_class.num_operators == 12
        qrc_class.fit(np.array([[0.1, 0.2], [0.1, 0.2]]), np.array([0.3, 0.4]))
        assert qrc_class._ml_model.n_features_in_ == 12
