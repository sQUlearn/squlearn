"""Tests for QRCClassifier"""

import pytest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.qrc.qrc_classifier import QRCClassifier


class TestQRCClassifier:
    """Test class for QRCClassifier."""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_classification(n_samples=6, n_features=4, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    def qrc_classifier(self, ml_model: str) -> QRCClassifier:
        """QNNClassifier module."""
        pqc = HubregtsenEncodingCircuit(num_qubits=4, num_features=4, num_layers=1)
        return QRCClassifier(pqc, Executor(), ml_model=ml_model)

    @pytest.mark.parametrize(
        "ml_model",
        [
            "linear",
            "mlp",
            "kernel",
        ],
    )
    def test_fit_predict(self, data, ml_model):
        """Tests fit and predict methods of QRCClassifier."""
        X, y = data
        qrc_classifier = self.qrc_classifier(ml_model)

        if ml_model == "mlp":
            qrc_classifier.set_params(ml_model_options={"random_state": 0, "max_iter": 1000})

        qrc_classifier.fit(X, y)
        values = qrc_classifier.predict(X)

        referece_values = {
            "linear": np.array([0, 1, 0, 1, 1, 1]),
            "mlp": np.array([0, 1, 0, 1, 0, 1]),
            "kernel": np.array([0, 1, 0, 0, 0, 1]),
        }
        assert np.allclose(values, referece_values[ml_model], atol=1e-7)
