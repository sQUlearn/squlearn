"""Tests for QNNRegressor"""

import io
import pytest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.observables import SummedPaulis
from squlearn.encoding_circuit import ChebyshevRx
from squlearn.optimizers import SLSQP, Adam
from squlearn.qnn import QNNRegressor, MeanSquaredError


class TestQNNRegressor:
    """Test class for QNNRegressor."""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_regression(n_samples=6, n_features=1, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    def get_qnn_regressor(self, framework: str = None) -> QNNRegressor:
        """QNNRegressor module."""
        random_device = np.random.default_rng(seed=30)
        if framework is None:
            executor = Executor()
        else:
            executor = Executor(framework)
        pqc = ChebyshevRx(num_qubits=2, num_layers=1)
        operator = SummedPaulis(num_qubits=2)
        loss = MeanSquaredError()
        optimizer = SLSQP(options={"maxiter": 2})
        param_ini = random_device.random(pqc.num_parameters)
        param_op_ini = random_device.random(operator.num_parameters)
        return QNNRegressor(pqc, operator, executor, loss, optimizer, param_ini, param_op_ini)

    @pytest.fixture(scope="module")
    def qnn_regressor_2out(self) -> QNNRegressor:
        """QNNRegressor module."""
        random_device = np.random.default_rng(seed=30)
        executor = Executor()
        pqc = ChebyshevRx(num_qubits=2, num_layers=1)
        operator = [SummedPaulis(num_qubits=2), SummedPaulis(num_qubits=2)]
        loss = MeanSquaredError()
        optimizer = SLSQP(options={"maxiter": 2})
        param_ini = random_device.random(pqc.num_parameters)
        param_op_ini = random_device.random(sum(op.num_parameters for op in operator))
        return QNNRegressor(pqc, operator, executor, loss, optimizer, param_ini, param_op_ini)

    def test_predict_unfitted(self, data):
        """Tests concerning the unfitted QNNRegressor.

        Tests include
            - whether `_is_fitted` is False
            - whether a RuntimeError is raised
        """
        X, _ = data
        qnn_regressor = self.get_qnn_regressor()
        assert not qnn_regressor._is_fitted
        with pytest.raises(RuntimeError, match="The model is not fitted."):
            qnn_regressor.predict(X)

    @pytest.mark.parametrize("framework", ["qiskit", "pennylane", "qulacs"])
    def test_fit(self, data, framework):
        """Tests concerning the fit function of the QNNRegressor.

        Tests include
            - whether `_is_fitted` is set True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        qnn_regressor = self.get_qnn_regressor(framework)
        qnn_regressor.fit(X, y)
        assert qnn_regressor._is_fitted
        assert not np.allclose(qnn_regressor.param, qnn_regressor.param_ini)
        assert not np.allclose(qnn_regressor.param_op, qnn_regressor.param_op_ini)

    def test_list_input(self, data):
        """Test concerning the fit function with list y.

        Tests include
            - whether `_is_fitted` is set True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        qnn_regressor = self.get_qnn_regressor()
        qnn_regressor.fit(X.tolist(), y.tolist())
        assert qnn_regressor._is_fitted
        assert not np.allclose(qnn_regressor.param, qnn_regressor.param_ini)
        assert not np.allclose(qnn_regressor.param_op, qnn_regressor.param_op_ini)

    def test_fit_2out(self, qnn_regressor_2out, data):
        """Tests concerning the fit function of the QNNRegressor for 2 outputs.

        Tests include
            - whether `_is_fitted` is set True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        y = np.array([y, y]).T
        qnn_regressor_2out.fit(X, y)
        assert qnn_regressor_2out._is_fitted
        assert not np.allclose(qnn_regressor_2out.param, qnn_regressor_2out.param_ini)
        assert not np.allclose(qnn_regressor_2out.param_op, qnn_regressor_2out.param_op_ini)

    def test_partial_fit(self, data):
        """Tests concerning the partial_fit function of the QNNRegressor.

        Tests include
            - whether `_param` is the same after two calls to fit
            - whether `_param` is different after a call to partial_fit and a call to fit
        """
        X, y = data
        qnn_regressor = self.get_qnn_regressor()
        qnn_regressor.fit(X, y)
        param_1 = qnn_regressor.param
        qnn_regressor.partial_fit(X, y)
        param_2 = qnn_regressor.param
        qnn_regressor.fit(X, y)
        param_3 = qnn_regressor.param

        assert np.allclose(param_1, param_3)
        assert not np.allclose(param_2, param_3)

    def test_fit_minibtach(self, data):
        """Tests concerning fit with mini-batch GD.

        Tests include
            - whether `_is_fitted` is True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        qnn_regressor = self.get_qnn_regressor()
        qnn_regressor._optimizer = Adam({"maxiter_total": 10, "maxiter": 2, "lr": 0.1})
        qnn_regressor.set_params(
            batch_size=2,
            epochs=2,
            shuffle=True,
        )
        qnn_regressor.fit(X, y)

        assert qnn_regressor._is_fitted
        assert not np.allclose(qnn_regressor.param, qnn_regressor.param_ini)
        assert not np.allclose(qnn_regressor.param_op, qnn_regressor.param_op_ini)

    def test_fit_minibtach_2out(self, qnn_regressor_2out, data):
        """Tests concerning fit with mini-batch GD for two outputs.

        Tests include
            - whether `_is_fitted` is True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        y = np.array([y, y]).T
        qnn_regressor_2out._optimizer = Adam({"maxiter_total": 10, "maxiter": 2, "lr": 0.1})
        qnn_regressor_2out.set_params(
            batch_size=2,
            epochs=2,
            shuffle=True,
        )
        qnn_regressor_2out.fit(X, y)

        assert qnn_regressor_2out._is_fitted
        assert not np.allclose(qnn_regressor_2out.param, qnn_regressor_2out.param_ini)
        assert not np.allclose(qnn_regressor_2out.param_op, qnn_regressor_2out.param_op_ini)

    @pytest.mark.parametrize("framework", ["qiskit", "pennylane", "qulacs"])
    def test_predict(self, data, framework):
        """Tests concerning the predict function of the QNNRegressor.

        Tests include
            - whether the prediction output is correct
        """
        X, y = data
        qnn_regressor = self.get_qnn_regressor(framework)
        qnn_regressor._param = np.linspace(0.1, 0.4, 4)
        qnn_regressor._param_op = np.linspace(0.1, 0.3, 3)
        qnn_regressor._is_fitted = True
        y_pred = qnn_regressor.predict(X)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape
        assert np.allclose(
            y_pred,
            np.array([0.50619332, 0.4905991, 0.51004432, 0.48826691, 0.5372742, 0.48826651]),
        )

    def test_set_params_and_fit(self, data):
        """
        Tests fit after changing parameters that alter the number of parameters of the pqc.

        Tests include
            - whether `_is_fitted` is True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        qnn_regressor = self.get_qnn_regressor()
        qnn_regressor.set_params(num_layers=3)
        qnn_regressor.fit(X, y)

        assert qnn_regressor._is_fitted
        assert len(qnn_regressor.param) != len(qnn_regressor.param_ini)
        assert not np.allclose(qnn_regressor.param_op, qnn_regressor.param_op_ini)

    @pytest.mark.parametrize("framework", ["qiskit", "pennylane", "qulacs"])
    def test_serialization(self, request, data, framework):
        """Tests concerning the serialization of the QNNRegressor."""

        X, y = data
        qnn_regressor = self.get_qnn_regressor(framework)
        qnn_regressor.fit(X, y)

        buffer = io.BytesIO()
        qnn_regressor.dump(buffer)

        predict_before = qnn_regressor.predict(X)

        buffer.seek(0)
        instance_loaded = QNNRegressor.load(buffer, Executor("qiskit"))
        predict_after = instance_loaded.predict(X)

        assert isinstance(instance_loaded, QNNRegressor)
        assert np.allclose(predict_before, predict_after, atol=1e-6)

        instance_loaded._is_fitted = False
        instance_loaded.fit(X, y)

        assert instance_loaded._is_fitted
        assert np.allclose(instance_loaded.param, qnn_regressor.param)
        assert np.allclose(instance_loaded.param_op, qnn_regressor.param_op)
