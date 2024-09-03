"""Tests for QNNClassifier"""

import pytest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

from squlearn import Executor
from squlearn.observables import SummedPaulis
from squlearn.encoding_circuit import ChebyshevPQC
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
        executor = Executor()
        pqc = ChebyshevPQC(num_qubits=2, num_features=2, num_layers=1)
        operator = SummedPaulis(num_qubits=2)
        loss = SquaredLoss()
        optimizer = SLSQP(options={"maxiter": 2})
        param_ini = np.random.rand(pqc.num_parameters)
        param_op_ini = np.random.rand(operator.num_parameters)
        return QNNClassifier(pqc, operator, executor, loss, optimizer, param_ini, param_op_ini)

    @pytest.fixture(scope="module")
    def qnn_classifier_2out(self) -> QNNClassifier:
        """QNNClassifier module."""
        executor = Executor()
        pqc = ChebyshevPQC(num_qubits=2, num_features=2, num_layers=1)
        operator = [SummedPaulis(num_qubits=2), SummedPaulis(num_qubits=2)]
        loss = SquaredLoss()
        optimizer = SLSQP(options={"maxiter": 2})
        return QNNClassifier(pqc, operator, executor, loss, optimizer, parameter_seed=0)

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
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        qnn_classifier.fit(X, y)
        assert qnn_classifier._is_fitted
        assert not np.allclose(qnn_classifier.param, qnn_classifier.param_ini)
        assert not np.allclose(qnn_classifier.param_op, qnn_classifier.param_op_ini)

    def test_list_input(self, qnn_classifier, data):
        """Test concerning the fit function with list y.

        Tests include
            - whether `_is_fitted` is set True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        qnn_classifier.fit(X.tolist(), y.tolist())
        assert qnn_classifier._is_fitted
        assert not np.allclose(qnn_classifier.param, qnn_classifier.param_ini)
        assert not np.allclose(qnn_classifier.param_op, qnn_classifier.param_op_ini)

    def test_fit_2out(self, qnn_classifier_2out, data):
        """Tests concerning the fit function of the QNNClassifier with 2 outputs.

        Tests include
            - whether `_is_fitted` is set True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        y = np.array([y, y]).T
        qnn_classifier_2out.fit(X, y)
        assert qnn_classifier_2out._is_fitted
        assert not np.allclose(qnn_classifier_2out.param, qnn_classifier_2out.param_ini)
        assert not np.allclose(qnn_classifier_2out.param_op, qnn_classifier_2out.param_op_ini)

    def test_partial_fit(self, qnn_classifier, data):
        """Tests concerning the partial_fit function of the QNNClassifier.

        Tests include
            - whether `_param` is the same after two calls to fit
            - whether `_param` is different after a call to partial_fit and a call to fit
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
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data

        qnn_classifier._optimizer = Adam({"maxiter_total": 10, "maxiter": 2, "lr": 0.1})
        qnn_classifier.set_params(
            batch_size=2,
            epochs=2,
            shuffle=True,
        )
        qnn_classifier.fit(X, y)

        assert qnn_classifier._is_fitted
        assert not np.allclose(qnn_classifier.param, qnn_classifier.param_ini)
        assert not np.allclose(qnn_classifier.param_op, qnn_classifier.param_op_ini)

    def test_fit_minibtach_2out(self, qnn_classifier_2out, data):
        """Tests concerning fit with mini-batch GD with 2 outputs.

        Tests include
            - whether `_is_fitted` is True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        y = np.array([y, y]).T

        qnn_classifier_2out._optimizer = Adam({"maxiter_total": 10, "maxiter": 2, "lr": 0.1})
        qnn_classifier_2out.set_params(
            batch_size=2,
            epochs=2,
            shuffle=True,
        )
        qnn_classifier_2out.fit(X, y)

        assert qnn_classifier_2out._is_fitted
        assert not np.allclose(qnn_classifier_2out.param, qnn_classifier_2out.param_ini)
        assert not np.allclose(qnn_classifier_2out.param_op, qnn_classifier_2out.param_op_ini)

    def test_predict(self, qnn_classifier, data):
        """Tests concerning the predict function of the QNNClassifier.

        Tests include
            - whether the prediction output is correct
        """
        X, y = data
        qnn_classifier._param = np.linspace(0.1, 0.7, 7)
        qnn_classifier._param_op = np.linspace(0.1, 0.3, 3)
        qnn_classifier._label_binarizer = LabelBinarizer()
        qnn_classifier._label_binarizer.fit(y)
        qnn_classifier._is_fitted = True
        y_pred = qnn_classifier.predict(X)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y.shape
        assert np.allclose(y_pred, np.zeros_like(y))

    def test_set_params_and_fit(self, qnn_classifier, data):
        """
        Tests fit after changing parameters that alter the number of parameters of the pqc.

        Tests include
            - whether `_is_fitted` is True
            - whether `_param` is updated
            - whether `_param_op` is updated
        """
        X, y = data
        qnn_classifier.set_params(num_layers=3)
        qnn_classifier.fit(X, y)

        assert qnn_classifier._is_fitted
        assert not np.allclose(qnn_classifier.param, qnn_classifier.param_ini)
        assert not np.allclose(qnn_classifier.param_op, qnn_classifier.param_op_ini)
