import pytest
import numpy as np

from squlearn.qnn import CrossEntropyLoss


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
