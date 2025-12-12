import numpy as np
import pytest

from squlearn.qnn.loss import MeanSquaredError


class TestMeanSquaredError:
    def test_properties_and_requirements(self):
        loss = MeanSquaredError()
        assert loss.loss_variance_available is True
        assert loss.loss_args_tuple == ("f",)
        assert loss.variance_args_tuple == ("f", "var")
        assert loss.gradient_args_tuple == ("f", "dfdp", "dfdop")

        loss._opt_param_op = False
        assert loss.gradient_args_tuple == ("f", "dfdp")

        # missing ground_truth should raise for value/variance/gradient
        with pytest.raises(AttributeError):
            loss.value({"f": np.array([0.0])})
        with pytest.raises(AttributeError):
            loss.variance({"f": np.array([0.0]), "var": np.array([0.0])})
        with pytest.raises(AttributeError):
            loss.gradient({"f": np.array([0.0]), "dfdp": np.zeros((1, 1))})

        # weights provided should raise ValueError
        with pytest.raises(ValueError):
            loss.value(
                {"f": np.array([0.0])}, ground_truth=np.array([0.0]), weights=np.array([1.0])
            )
        with pytest.raises(ValueError):
            loss.variance(
                {"f": np.array([0.0]), "var": np.array([0.0])},
                ground_truth=np.array([0.0]),
                weights=np.array([1.0]),
            )
        with pytest.raises(ValueError):
            loss.gradient(
                {"f": np.array([0.0]), "dfdp": np.zeros((1, 1))},
                ground_truth=np.array([0.0]),
                weights=np.array([1.0]),
            )

    def test_value_basic_and_with_weights(self):
        loss = MeanSquaredError()
        f = np.array([1.0, 2.0, 3.0])
        gt = np.array([0.0, 2.0, 1.0])
        # default weights = ones
        value_dict = {"f": f}
        expected = np.sum((f - gt) ** 2) / 3.0
        out = loss.value(value_dict, ground_truth=gt)
        assert np.allclose(out, expected)

    def test_variance_basic_and_with_weights(self):
        loss = MeanSquaredError()
        f = np.array([1.0, 0.0])
        gt = np.array([0.0, 1.0])
        var = np.array([0.1, 0.2])
        value_dict = {"f": f, "var": var}

        # default weights ones
        diff_square = (f - gt) ** 2
        expected = np.sum(4 * diff_square * var) / 2.0
        out = loss.variance(value_dict, ground_truth=gt)
        assert np.allclose(out, expected)

    def test_gradient_single_output_no_opt_param_op(self):
        loss = MeanSquaredError()
        # 2 samples, 2 params
        f = np.array([1.0, 2.0])
        gt = np.array([0.0, 1.0])
        dfdp = np.array([[0.1, 0.2], [0.3, 0.4]])  # (n_samples, n_params)
        value_dict = {"f": f, "dfdp": dfdp}
        loss._opt_param_op = False

        weighted_diff = f - gt  # weights default ones
        expected = 2.0 * np.einsum("j,jk->k", weighted_diff, dfdp) / 2.0
        dp = loss.gradient(value_dict, ground_truth=gt)
        assert np.allclose(dp, expected)

    def test_gradient_single_output_with_opt_param_op(self):
        loss = MeanSquaredError()
        f = np.array([2.0, 0.0])
        gt = np.array([1.0, 0.0])
        dfdp = np.array([[1.0, 0.0], [0.0, 1.0]])
        dfdop = np.array([[0.5], [0.25]])
        value_dict = {"f": f, "dfdp": dfdp, "dfdop": dfdop}
        loss._opt_param_op = True

        weighted_diff = f - gt
        expected_dp = 2.0 * np.einsum("j,jk->k", weighted_diff, dfdp) / 2.0
        expected_dop = 2.0 * np.einsum("j,jk->k", weighted_diff, dfdop) / 2.0

        dp, dop = loss.gradient(value_dict, ground_truth=gt)
        assert np.allclose(dp, expected_dp)
        assert np.allclose(dop, expected_dop)

    def test_gradient_multiple_output_no_opt_param_op(self):
        loss = MeanSquaredError()

        f = np.array([[1.0, 0.5], [0.2, 0.8]])
        gt = np.array([[0.0, 1.0], [0.0, 1.0]])

        dfdp = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        value_dict = {"f": f, "dfdp": dfdp}
        loss._opt_param_op = False

        weighted_diff = f - gt
        expected = 2.0 * np.einsum("ij,ijk->k", weighted_diff, dfdp) / 2.0
        dp = loss.gradient(value_dict, ground_truth=gt, multiple_output=True)
        assert np.allclose(dp, expected)

    def test_gradient_multiple_output_with_opt_param_op(self):
        loss = MeanSquaredError()
        f = np.array([[1.0, 0.5], [0.2, 0.8]])
        gt = np.array([[0.0, 1.0], [0.0, 1.0]])
        dfdp = np.ones((2, 2, 1))
        dfdop = np.full((2, 2, 2), 0.5)
        value_dict = {"f": f, "dfdp": dfdp, "dfdop": dfdop}
        loss._opt_param_op = True

        weighted_diff = f - gt
        expected_dp = 2.0 * np.einsum("ij,ijk->k", weighted_diff, dfdp) / 2.0
        expected_dop = 2.0 * np.einsum("ij,ijk->k", weighted_diff, dfdop) / 2.0

        dp, dop = loss.gradient(value_dict, ground_truth=gt, multiple_output=True)
        assert np.allclose(dp, expected_dp)
        assert np.allclose(dop, expected_dop)

    def test_gradient_empty_dfdp_and_empty_dfdop(self):
        loss = MeanSquaredError()
        # shapes with zero samples
        value_dict = {
            "f": np.zeros((0,)),
            "dfdp": np.zeros((0, 2)),
            "dfdop": np.zeros((0, 1)),
        }
        loss._opt_param_op = False
        g = loss.gradient(value_dict, ground_truth=np.zeros((0,)))
        assert isinstance(g, np.ndarray)
        assert g.size == 0

        loss._opt_param_op = True
        dp, dop = loss.gradient(value_dict, ground_truth=np.zeros((0,)))
        assert dp.size == 0
        assert dop.size == 0
