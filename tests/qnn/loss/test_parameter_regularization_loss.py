import numpy as np
import pytest
from squlearn.qnn.loss import ParameterRegularizationLoss


class TestParameterRegularizationLoss:
    def test_init(self):
        loss = ParameterRegularizationLoss(
            alpha=0.01, mode="L1", parameter_list=[0, 2], parameter_operator_list=[1]
        )
        assert loss._alpha == 0.01
        assert loss._mode == "L1"
        assert loss._parameter_list == [0, 2]
        assert loss._parameter_operator_list == [1]

        with pytest.raises(ValueError):
            loss = ParameterRegularizationLoss(mode="not a mode")

    def test_value_and_gradient_l2_default(self):
        loss = ParameterRegularizationLoss()

        loss._opt_param_op = False

        params = np.array([1.0, 2.0])
        value_dict = {"param": params}

        # Value: alpha * sum(p^2) = 0.005 * (1 + 4) = 0.025
        expected_value = 0.005 * (1.0**2 + 2.0**2)
        actual_value = loss.value(value_dict)
        assert pytest.approx(expected_value, rel=1e-12) == actual_value

        # Gradient: alpha * 2 * p
        grad = loss.gradient(value_dict)
        expected_grad = 0.005 * 2.0 * params
        assert isinstance(grad, np.ndarray)
        assert np.allclose(grad, expected_grad)

    def test_value_and_gradient_l1(self):
        loss = ParameterRegularizationLoss(alpha=0.1, mode="L1")

        loss._opt_param_op = False

        params = np.array([-1.0, 2.0, 0.0])
        value_dict = {"param": params}

        # Value: alpha * sum(|p|) = 0.1 * (1 + 2 + 0) = 0.3
        expected_value = 0.1 * (1.0 + 2.0 + 0.0)
        assert pytest.approx(expected_value) == loss.value(value_dict)

        # Gradient: alpha * sign(p)
        expected_grad = 0.1 * np.sign(params)
        np.testing.assert_allclose(loss.gradient(value_dict), expected_grad)

    def test_parameter_list_selection_l2(self):
        # Only regularize index 1
        loss = ParameterRegularizationLoss(alpha=1.0, mode="L2", parameter_list=[1])

        loss._opt_param_op = False

        params = np.array([3.0, 4.0])
        value_dict = {"param": params}

        # Value should be 1.0 * (4^2) = 16
        assert pytest.approx(16.0) == loss.value(value_dict)

        # Gradient: only index 1 non-zero => [0, 2*4]
        expected_grad = np.array([0.0, 8.0])
        assert np.allclose(loss.gradient(value_dict), expected_grad)

    def test_operator_parameters_included_and_gradient_tuple(self):
        # Test including operator parameters and getting (d_p, d_op)
        loss = ParameterRegularizationLoss(
            alpha=0.5, mode="L2", parameter_list=None, parameter_operator_list=None
        )

        params = np.array([1.0, 2.0])
        params_op = np.array([0.5, -1.5])
        value_dict = {"param": params, "param_op": params_op}

        # Value: alpha * (sum params^2 + sum params_op^2)
        expected_value = 0.5 * (np.sum(params**2) + np.sum(params_op**2))
        assert pytest.approx(expected_value) == loss.value(value_dict)

        # Gradient: d_p = alpha*2*p ; d_op = alpha*2*p_op
        d_p_expected = 0.5 * 2.0 * params
        d_op_expected = 0.5 * 2.0 * params_op

        grad = loss.gradient(value_dict)
        assert isinstance(grad, tuple) and len(grad) == 2
        assert np.allclose(grad[0], d_p_expected)
        assert np.allclose(grad[1], d_op_expected)

    def test_parameter_operator_list_selection_and_l1(self):
        # Only regularize first operator parameter (L1) and all normal params (L1)
        loss = ParameterRegularizationLoss(
            alpha=2.0, mode="L1", parameter_list=None, parameter_operator_list=[0]
        )

        params = np.array([0.0, -2.0])
        params_op = np.array([3.0, 4.0])
        value_dict = {"param": params, "param_op": params_op}

        # Value: alpha * (sum |params| + |params_op[0]|)
        expected_value = 2.0 * (np.sum(np.abs(params)) + np.sum(np.abs(params_op)))
        assert pytest.approx(expected_value) == loss.value(value_dict)

        grad = loss.gradient(value_dict)
        # d_p = alpha * sign(params)
        expected_d_p = 2.0 * np.sign(params)
        # d_op only index 0: alpha * sign(3.0) = 2.0 * 1 = 2.0 for first, 0 for second
        expected_d_op = np.array([2.0, 0.0])

        assert isinstance(grad, tuple)
        assert np.allclose(grad[0], expected_d_p)
        assert np.allclose(grad[1], expected_d_op)

    def test_callable_alpha_requires_iteration_for_value_and_gradient(self):
        alpha_callable = lambda it: 0.2
        loss = ParameterRegularizationLoss(alpha=alpha_callable, mode="L2")
        params = np.array([1.0])
        value_dict = {"param": params}

        with pytest.raises(AttributeError):
            loss.value(value_dict)  # no iteration provided

        with pytest.raises(AttributeError):
            loss.gradient(value_dict)  # no iteration provided

    def test_callable_alpha_with_iteration_behaves_correctly(self):
        alpha_callable = lambda it: 0.1 * it
        loss = ParameterRegularizationLoss(alpha=alpha_callable, mode="L2")
        loss._opt_param_op = False
        params = np.array([2.0, 3.0])
        value_dict = {"param": params}

        # iteration=2 => alpha = 0.2 ; value = 0.2 * (4 + 9) = 0.2 * 13 = 2.6
        val = loss.value(value_dict, iteration=2)
        assert pytest.approx(2.6) == val

        # gradient: alpha*2*p => 0.2 * 2 * params = 0.4 * params
        expected_grad = 0.4 * params
        np.testing.assert_allclose(loss.gradient(value_dict, iteration=2), expected_grad)

    def test_variance_is_zero(self):
        loss = ParameterRegularizationLoss()
        params = np.array([1.0, 2.0])
        value_dict = {"param": params}
        assert loss.variance(value_dict) == 0.0
