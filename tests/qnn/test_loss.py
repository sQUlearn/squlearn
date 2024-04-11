import pytest
import numpy as np

from squlearn.qnn import LogLoss


class TestLogLoss:

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
        log_loss = LogLoss(1e-8)
        assert np.isclose(log_loss.value(value_dict, ground_truth=ground_truth), loss_value)

    @pytest.mark.parametrize(
        "value_dict, ground_truth, multiple_output, gradient_value",
        [
            (
                {"f": [0.99, 0.01, 0.9], "dfdp": [[0.1, 0.2, 0.3]]},
                np.array([1, 0, 1]),
                False,
                np.array([-0.037037, -0.074074, -0.111111]),
            ),
            (
                {
                    "f": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    "dfdp": [[0.1, 0.2, 0.3]],
                },
                np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                False,
                np.zeros(3),
            ),
            (
                {
                    "f": [[0.99, 0.01, 0.0], [0.0, 0.99, 0.01], [0.01, 0.0, 0.99]],
                    "dfdp": [[[0.1, 0.2, 0.3]]],
                },
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                True,
                np.array([-0.101010, -0.202020, -0.303030]),
            ),
        ],
    )
    def test_gradient(self, value_dict, ground_truth, multiple_output, gradient_value):
        log_loss = LogLoss(1e-8)
        log_loss.set_opt_param_op(False)
        assert np.allclose(
            log_loss.gradient(
                value_dict, multiple_output=multiple_output, ground_truth=ground_truth
            ),
            gradient_value,
        )
