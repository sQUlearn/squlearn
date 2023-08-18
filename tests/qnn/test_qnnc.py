"""Tests for QNNClassifier"""
import pytest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.expectation_operator import IsingHamiltonian
from squlearn.feature_map import ChebPQC
from squlearn.optimizers import SLSQP, Adam
from squlearn.qnn import QNNClassifier, SquaredLoss


class TestQNNClassifier:
    """Test class for QNNClassifier."""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_blobs(n_samples=6, n_features=2, centers=2, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    @pytest.fixture(scope="module")
    def qnn_classifier(self) -> QNNClassifier:
        """QNNClassifier module."""
        np.random.seed(42)
        executor = Executor("statevector_simulator")
        pqc = ChebPQC(num_qubits=4, num_features=2, num_layers=2)
        operator = IsingHamiltonian(num_qubits=4, I="S", Z="S", ZZ="S")
        loss = SquaredLoss()
        optimizer = SLSQP(options={"maxiter": 2})
        param_ini = np.random.rand(pqc.num_parameters)
        param_op_ini = np.random.rand(operator.num_parameters)
        return QNNClassifier(pqc, operator, executor, loss, optimizer, param_ini, param_op_ini)

    def test_params(self, qnn_classifier):
        """Asserts that all parameters are present in the QNNClassifier."""
        assert list(qnn_classifier.get_params().keys()) == [
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

    def test_predict_unfitted(self, qnn_classifier, data):
        """Tests concerning the unfitted QNNClassifier.

        Tests include
            - whether `_is_fitted` is False
            - whether a RuntimeError is raised
        """
        X, _ = data
        assert not qnn_classifier._is_fitted
        with pytest.raises(RuntimeError, match="The model is not fitted."):
            qnn_classifier.predict(X)

    def test_fit(self, qnn_classifier, data):
        """Tests concerning the fit function of the QNNClassifier.

        Tests include
            - whether `_is_fitted` is set True
            - whether `param` is updated
            - whether `param_op` is updated
        """
        X, y = data
        qnn_classifier.fit(X, y)
        assert qnn_classifier._is_fitted
        assert not np.allclose(qnn_classifier.param, qnn_classifier.param_ini)
        assert not np.allclose(qnn_classifier.param_op, qnn_classifier.param_op_ini)

    def test_partial_fit(self, qnn_classifier, data):
        """Tests concerning the partial_fit function of the QNNClassifier.

        Tests include
            - whether `param` is the same after two calls to fit
            - whether `param` is different after a call to partial_fit and a call to fit
        """
        X, y = data

        qnn_classifier.fit(X, y)
        param_1 = qnn_classifier.param
        qnn_classifier.partial_fit(X, y)
        param_2 = qnn_classifier.param
        qnn_classifier.fit(X, y)
        param_3 = qnn_classifier.param

        assert np.allclose(param_1, param_3)
        assert not np.allclose(param_2, param_3)

    def test_fit_minibtach(self, qnn_classifier, data):
        """Tests concerning fit with mini-batch GD.

        Tests include
            - whether `_is_fitted` is True
            - whether `param` is updated
            - whether `param_op` is updated
        """
        X, y = data

        qnn_classifier.set_params(
            optimizer=Adam({"maxiter_total": 10, "maxiter": 2, "lr": 0.1}),
            batch_size=2,
            epochs=2,
            shuffle=True,
        )
        qnn_classifier.fit(X, y)

        assert qnn_classifier._is_fitted
        assert not np.allclose(qnn_classifier.param, qnn_classifier.param_ini)
        assert not np.allclose(qnn_classifier.param_op, qnn_classifier.param_op_ini)

    def test_predict(self, qnn_classifier, data):
        """Tests concerning the predict function of the QNNClassifier.

        Tests include
            - whether the prediction output is correct
        """
        X, y = data
        qnn_classifier.param = np.arange(0.1, 2.45, 0.1)
        qnn_classifier.param_op = np.arange(0.1, 0.35, 0.1)
        qnn_classifier._is_fitted = True
        y_pred = qnn_classifier.predict(X)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape
        assert np.allclose(y_pred, np.zeros_like(y))
