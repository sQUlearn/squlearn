"""Tests for QNNRegressor"""
import pytest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.expectation_operator import IsingHamiltonian
from squlearn.feature_map import ChebPQC
from squlearn.optimizers import SLSQP, Adam
from squlearn.qnn import QNNRegressor, SquaredLoss


class TestQNNRegressor:
    """Test class for QNNRegressor."""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        X, y = make_regression(n_samples=6, n_features=1, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    @pytest.fixture(scope="module")
    def qnn_regressor(self) -> QNNRegressor:
        """QNNRegressor module."""
        np.random.seed(42)
        executor = Executor("statevector_simulator")
        pqc = ChebPQC(num_qubits=4, num_features=1, num_layers=2)
        operator = IsingHamiltonian(num_qubits=4, I="S", Z="S", ZZ="S")
        loss = SquaredLoss()
        optimizer = SLSQP(options={"maxiter": 2})
        param_ini = np.random.rand(pqc.num_parameters)
        param_op_ini = np.random.rand(operator.num_parameters)
        return QNNRegressor(pqc, operator, executor, loss, optimizer, param_ini, param_op_ini)

    def test_params(self, qnn_regressor):
        """Asserts that all parameters are present in the QNNRegressor."""
        assert list(qnn_regressor.get_params().keys()) == [
            "batch_size",
            "epochs",
            "executor",
            "loss",
            "operator",
            "opt_param_op",
            "optimizer",
            "param_ini",
            "param_op_ini",
            "pqc",
            "shuffle",
            "variance",
        ]

    def test_predict_unfitted(self, qnn_regressor, data):
        """Tests concerning the unfitted QNNRegressor.

        Tests include
            - whether `_is_fitted` is False
            - whether a warning is raised
            - whether the prediction output is correct
        """
        X, y = data
        assert not qnn_regressor._is_fitted
        with pytest.warns(UserWarning, match="The model is not fitted."):
            y_pred = qnn_regressor.predict(X)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape

    def test_fit(self, qnn_regressor, data):
        """Tests concerning the fit function of the QNNRegressor.

        Tests include
            - whether `_is_fitted` is set True
            - whether `param` is updated
            - whether `param_op` is updated
        """
        X, y = data
        qnn_regressor.fit(X, y)
        assert qnn_regressor._is_fitted
        assert not np.allclose(qnn_regressor.param, qnn_regressor.param_ini)
        assert not np.allclose(qnn_regressor.param_op, qnn_regressor.param_op_ini)

    def test_partial_fit(self, qnn_regressor, data):
        """Tests concerning the partial_fit function of the QNNRegressor.

        Tests include
            - whether `param` is the same after two calls to fit
            - whether `param` is different after a call to partial_fit and a call to fit
        """
        X, y = data

        qnn_regressor.fit(X, y)
        param_1 = qnn_regressor.param
        qnn_regressor.partial_fit(X, y)
        param_2 = qnn_regressor.param
        qnn_regressor.fit(X, y)
        param_3 = qnn_regressor.param

        assert np.allclose(param_1, param_3)
        assert not np.allclose(param_2, param_3)

    def test_fit_minibtach(self, qnn_regressor, data):
        """Tests concerning fit with minibatch GD.

        Tests include
            - whether `_is_fitted` is True
            - whether `param` is updated
            - whether `param_op` is updated
        """
        X, y = data

        qnn_regressor.set_params(
            optimizer=Adam({"maxiter_total": 10, "maxiter": 2, "lr": 0.1}),
            batch_size=2,
            epochs=2,
            shuffle=True,
        )
        qnn_regressor.fit(X, y)

        assert qnn_regressor._is_fitted
        assert not np.allclose(qnn_regressor.param, qnn_regressor.param_ini)
        assert not np.allclose(qnn_regressor.param_op, qnn_regressor.param_op_ini)

    def test_predict(self, qnn_regressor, data):
        """Tests concerning the predict function of the QNNRegressor.

        Tests include
            - whether the prediction output is correct
        """
        X, y = data
        qnn_regressor.param = np.arange(0.1, 2.45, 0.1)
        qnn_regressor.param_op = np.arange(0.1, 0.35, 0.1)
        qnn_regressor._is_fitted = True
        y_pred = qnn_regressor.predict(X)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape
        assert np.allclose(
            y_pred,
            np.array([0.57240016, 1.20027015, 0.39182531, 1.24520867, -0.1851392, 1.24521491]),
        )
