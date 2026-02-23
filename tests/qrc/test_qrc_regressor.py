"""Tests for QRCRegressor"""

import io
import pytest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from squlearn.qrc.qrc_regressor import QRCRegressor


class TestQRCRegressor:
    """Test class for QRCRegressor."""

    @pytest.fixture(scope="module")
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_regression(n_samples=6, n_features=1, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y

    def qrc_regressor(self, ml_model: str) -> QRCRegressor:
        """QNNRegressor module."""
        pqc = HubregtsenEncodingCircuit(num_qubits=2, num_features=1, num_layers=1)
        return QRCRegressor(pqc, Executor(), ml_model=ml_model)

    @pytest.mark.parametrize(
        "ml_model",
        [
            "linear",
            "mlp",
            "kernel",
        ],
    )
    def test_fit_predict(self, data, ml_model):
        """Tests fit and predict methods of QRCRegressor."""
        X, y = data
        qrc_regressor = self.qrc_regressor(ml_model)

        if ml_model == "mlp":
            qrc_regressor.set_params(ml_model_options={"random_state": 0, "max_iter": 1000})

        qrc_regressor.fit(X, y)
        values = qrc_regressor.predict(X)

        referece_values = {
            "linear": np.array(
                [2.88509523, -0.80308901, 3.76200899, -1.35995202, 8.84630756, -1.36004739]
            ),
            "mlp": np.array(
                [3.04943393, -0.87369101, 3.98814234, -1.42608296, 8.61361362, -1.42617601]
            ),
            "kernel": np.array(
                [2.54250631, 0.06183984, 3.13005284, -0.29406904, 6.04568976, -0.29412921]
            ),
        }

        assert np.allclose(values, referece_values[ml_model], atol=1e-7)

    @pytest.mark.parametrize(
        "ml_model",
        [
            "linear",
            "mlp",
            "kernel",
        ],
    )
    def test_serialization(self, data, ml_model):
        """Tests concerning the serialization of the QRCRegressor."""
        X, y = data
        qrc_regressor = self.qrc_regressor(ml_model)
        qrc_regressor.fit(X, y)

        buffer = io.BytesIO()
        qrc_regressor.dump(buffer)

        predict_before = qrc_regressor.predict(X)

        buffer.seek(0)
        instance_loaded = QRCRegressor.load(buffer, Executor())
        predict_after = instance_loaded.predict(X)

        assert isinstance(instance_loaded, QRCRegressor)
        assert np.allclose(predict_before, predict_after, atol=1e-6)
