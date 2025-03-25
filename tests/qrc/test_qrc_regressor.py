"""Tests for QRCRegressor"""

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
                [3.06585455, -0.96776248, 4.0395428, -1.52636054, 8.80042369, -1.52645433]
            ),
            "kernel": np.array(
                [2.87653068, -0.65406699, 3.74209315, -1.13894099, 8.15484098, -1.13902237]
            ),
        }

        assert np.allclose(values, referece_values[ml_model], atol=1e-7)
