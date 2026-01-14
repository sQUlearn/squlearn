import numpy as np
import pytest
from squlearn.qnn.loss import VarianceLoss


class TestVarianceLoss:
    def test_properties_and_variance_method(self):
        loss = VarianceLoss(alpha=0.01)
        assert loss.loss_variance_available is True
        assert loss.loss_args_tuple == ("var",)
        assert loss.variance_args_tuple == tuple()
        assert loss.gradient_args_tuple == ("var", "dvardp", "dvardop")
        # variance() always returns 0.0
        assert (
            loss.variance(
                {},
            )
            == 0.0
        )

        loss._opt_param_op = False
        assert loss.gradient_args_tuple == ("var", "dvardp")

    def test_value_with_scalar_alpha(self):
        loss = VarianceLoss(alpha=0.1)
        value_dict = {"var": np.array([0.2, 0.3, 0.0])}
        expected = 0.1 * np.sum(value_dict["var"])
        out = loss.value(value_dict)
        assert np.allclose(out, expected)

    def test_value_with_callable_alpha_requires_iteration(self):
        loss = VarianceLoss(alpha=lambda it: 0.2 * it)
        value_dict = {"var": np.array([0.1, 0.1])}

        # missing iteration -> AttributeError
        with pytest.raises(AttributeError):
            loss.value(value_dict)

        # with iteration provided, works
        out = loss.value(value_dict, iteration=3)  # alpha = 0.6
        expected = 0.6 * np.sum(value_dict["var"])
        assert np.allclose(out, expected)

    def test_gradient_no_opt_param_op_simple(self):
        alpha = 0.5
        loss = VarianceLoss(alpha=alpha)

        value_dict = {
            "dfdp": np.ones((3, 2)),  # dfdp used only to check emptiness
            "dvardp": np.array([[1.0, 2.0], [0.5, 0.5], [0.0, 1.0]]),
            # dfdop/dvardop not needed here
        }
        loss._opt_param_op = False

        # expected: alpha * sum over samples axis=0
        expected = alpha * np.sum(value_dict["dvardp"], axis=0)
        grad = loss.gradient(value_dict)
        assert isinstance(grad, np.ndarray)
        assert np.allclose(grad, expected)

    def test_gradient_multiple_output_and_with_opt_param_op(self):
        alpha = 0.2
        loss = VarianceLoss(alpha=alpha)
        n_samples = 2
        n_outputs = 2
        n_params = 3
        n_param_op = 2

        dvardp = np.ones((n_samples, n_outputs, n_params)) * 2.0
        dvardop = np.ones((n_samples, n_outputs, n_param_op)) * 3.0

        value_dict = {
            "dfdp": np.ones((n_samples, n_params)),
            "dvardp": dvardp,
            "dfdop": np.ones((n_samples, n_param_op)),  # used only for emptiness check of op
            "dvardop": dvardop,
        }

        loss._opt_param_op = True

        expected_dp = alpha * np.sum(dvardp, axis=(0, 1))
        expected_dop = alpha * np.sum(dvardop, axis=(0, 1))

        dp, dop = loss.gradient(value_dict, multiple_output=True)
        assert np.allclose(dp, expected_dp)
        assert np.allclose(dop, expected_dop)

    def test_gradient_returns_empty_when_dfdp_empty_and_dfdop_empty(self):
        loss = VarianceLoss(alpha=1.0)
        # empty sample axis -> shape[0] == 0
        value_dict = {
            "dfdp": np.zeros((0, 2)),
            "dvardp": np.zeros((0, 2)),
            "dfdop": np.zeros((0, 1)),
            "dvardop": np.zeros((0, 1)),
        }
        loss._opt_param_op = False
        grad = loss.gradient(value_dict)
        assert isinstance(grad, np.ndarray)
        assert grad.size == 0

        loss._opt_param_op = True
        dp, dop = loss.gradient(value_dict)
        assert dp.size == 0
        assert dop.size == 0

    def test_gradient_callable_alpha_requires_iteration(self):
        loss = VarianceLoss(alpha=lambda it: 0.1 * it)
        value_dict = {
            "dfdp": np.ones((1, 1)),
            "dvardp": np.ones((1, 1)),
            "dfdop": np.ones((1, 1)),
            "dvardop": np.ones((1, 1)),
        }

        with pytest.raises(AttributeError):
            loss.gradient(value_dict)

        # with iteration provided
        grad = loss.gradient(value_dict, iteration=5)
        # alpha = 0.5, expected dp = alpha * sum axis=0 -> 0.5 * [1] = [0.5]
        assert np.allclose(grad if not loss._opt_param_op else grad[0], np.array([0.5]))
