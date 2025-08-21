import scipy as sp
import pytest
import numpy as np
import sympy

from squlearn.qnn import CrossEntropyLoss
from squlearn import Executor
from squlearn.encoding_circuit import ChebyshevPQC
from squlearn.observables import SummedPaulis
from squlearn.qnn.qnnr import QNNRegressor
from squlearn.optimizers import Adam, LBFGSB
from squlearn.qnn import QNNRegressor
from squlearn.qnn.util import get_lr_decay
from squlearn.qnn import ODELoss


class TestCrossEntropyLoss:

    @pytest.mark.parametrize(
        "value_dict, ground_truth, loss_value",
        [
            (
                {"f": [0.99, 0.01, 0.9, 0.1]},
                np.array([1, 0, 1, 0]),
                0.057705,
            ),
            (
                {"f": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},
                np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                0.6931472,
            ),
            (
                {"f": [[0.99, 0.01, 0.0], [0.0, 0.99, 0.01], [0.01, 0.0, 0.99]]},
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                0.0100503,
            ),
        ],
    )
    def test_value(self, value_dict, ground_truth, loss_value):
        cross_entropy_loss = CrossEntropyLoss(1e-8)
        assert np.isclose(
            cross_entropy_loss.value(value_dict, ground_truth=ground_truth), loss_value
        )

    @pytest.mark.parametrize(
        "value_dict, ground_truth, multiple_output, gradient_value",
        [
            (
                {"f": [0.99, 0.01, 0.9], "dfdp": [[0.1, 0.2, 0.3]]},
                np.array([1, 0, 1]),
                False,
                np.array([0.05993933, 0.11987865, 0.17981798]),
            ),
            (
                {
                    "f": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    "dfdp": [[0.1, 0.2, 0.3]],
                },
                np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                False,
                np.array([0.13333333, 0.26666667, 0.4]),
            ),
            (
                {
                    "f": [[0.99, 0.01, 0.0], [0.0, 0.99, 0.01], [0.01, 0.0, 0.99]],
                    "dfdp": [[[0.1, 0.2, 0.3]]],
                },
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                True,
                np.array([0.1, 0.2, 0.3]),
            ),
        ],
    )
    def test_gradient(self, value_dict, ground_truth, multiple_output, gradient_value):
        cross_entropy_loss = CrossEntropyLoss(1e-8)
        cross_entropy_loss.set_opt_param_op(False)
        assert np.allclose(
            cross_entropy_loss.gradient(
                value_dict, multiple_output=multiple_output, ground_truth=ground_truth
            ),
            gradient_value,
        )


class TestODELoss:

    def test_ode_loss(self):
        x, y, dydx = sympy.symbols("x y dydx")  # Define the symbols
        eq = dydx - y  # Define the differential equation

        ode_loss = ODELoss(
            eq,
            symbols_involved_in_ode=["x", "y", "dydx"],
            initial_values=[1],
            eta=10,
        )

        circuit = ChebyshevPQC(4, 1)
        observable = SummedPaulis(4)

        ode_regressor = QNNRegressor(
            circuit,
            observable,
            Executor("pennylane"),
            ode_loss,
            Adam(options={"maxiter": 3}),
        )

        x_numerical = np.linspace(0, 0.9, 3).reshape(-1, 1)
        ref_values = np.zeros(len(x_numerical))
        ode_regressor.fit(x_numerical, ref_values)

        assert np.allclose(
            ode_regressor.predict(x_numerical), np.array([1.62429362, 2.87454102, 1.46558265])
        )
