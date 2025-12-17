import numpy as np
import pytest
import sympy as sp
from squlearn.qnn.loss import ODELoss


def make_order1_loss(initial_value=0.0, eta=1.0):
    x, f, dfdx = sp.symbols("x f dfdx")
    ode_expr = dfdx - f
    loss = ODELoss(
        ode_functional=ode_expr,
        symbols_involved_in_ode=[x, f, dfdx],
        initial_values=[initial_value],
        eta=eta,
    )
    return loss


def make_order2_loss(initial_values=(0.0, 0.0), eta=1.0):
    x, f, dfdx, dfdxdx = sp.symbols("x f dfdx dfdxdx")
    ode_expr = dfdxdx - f
    loss = ODELoss(
        ode_functional=ode_expr,
        symbols_involved_in_ode=[x, f, dfdx, dfdxdx],
        initial_values=list(initial_values),
        eta=eta,
    )
    return loss


class TestODELoss:
    @pytest.mark.parametrize(
        ("symbol_string", "initial_values", "id"),
        [
            ("x dx", [], 1),
            ("x y dxdy", [1], 2),
            ("x y z dxdydz", [1, 2], 3),
        ],
    )
    def test_loss_args_tuple(self, symbol_string, initial_values, id):
        symbols = sp.symbols(symbol_string)
        ode = ODELoss(
            symbols_involved_in_ode=symbols,
            initial_values=initial_values,
            ode_functional=sp.Function("f")(*symbols),
        )

        if id == 1:
            assert ode.loss_args_tuple == ("f",)
        if id == 2:
            assert ode.loss_args_tuple == ("f", "dfdx")
        if id == 3:
            assert ode.loss_args_tuple == ("f", "dfdx", "dfdxdx")

    @pytest.mark.parametrize(
        ("symbol_string", "initial_values", "id"),
        [
            ("x dx", [], 1),
            ("x y dxdy", [1], 2),
            ("x y z dxdydz", [1, 2], 3),
        ],
    )
    def test_gradient_args_tuple(self, symbol_string, initial_values, id):
        symbols = sp.symbols(symbol_string)
        ode = ODELoss(
            symbols_involved_in_ode=symbols,
            initial_values=initial_values,
            ode_functional=sp.Function("f")(*symbols),
        )

        if id == 1:
            assert ode.gradient_args_tuple == ("f", "dfdp", "dfdop")
        if id == 2:
            assert ode.gradient_args_tuple == ("f", "dfdx", "dfdp", "dfdxdp", "dfdop", "dfdopdx")
        if id == 3:
            assert ode.gradient_args_tuple == (
                "f",
                "dfdx",
                "dfdxdx",
                "dfdp",
                "dfdxdp",
                "dfdxdxdp",
                "dfdop",
                "dfdopdx",
                "dfdopdxdx",
            )

        ode.set_opt_param_op(False)
        if id == 1:
            assert ode.gradient_args_tuple == ("f", "dfdp")
        if id == 2:
            assert ode.gradient_args_tuple == ("f", "dfdx", "dfdp", "dfdxdp")
        if id == 3:
            assert ode.gradient_args_tuple == ("f", "dfdx", "dfdxdx", "dfdp", "dfdxdp", "dfdxdxdp")

    def test_verify_size_of_ivp_with_order_of_ODE(self):
        symbols = sp.symbols("x y dxdy")
        with pytest.raises(ValueError):
            ODELoss(
                symbols_involved_in_ode=symbols,
                initial_values=[1, 2],
                ode_functional=sp.Function("f")(*symbols),
            )

        symbols = sp.symbols("x y z w dxdydzdw")
        with pytest.raises(ValueError):
            ODELoss(
                symbols_involved_in_ode=symbols,
                initial_values=[1, 2, 3],
                ode_functional=sp.Function("f")(*symbols),
            )

    def test_value_order1(self):
        loss = make_order1_loss(initial_value=0.0, eta=2.0)
        # choose values such that F = dfdx - f = 0 for each sample -> functional_loss = 0
        value_dict = {
            "x": np.array([[0.0], [1.0]]),
            "f": np.array([1.0, 2.0]),
            "dfdx": np.array([[1.0], [2.0]]),
        }
        # _ODE_functional(value_dict) = zeros -> functional_loss 0
        # initial value loss = eta * (f0 - init)^2 = 2.0 * (1.0 - 0.0)^2 = 2.0
        ground_truth = np.zeros((2,))  # zeros as ground truth
        out = loss.value(value_dict, ground_truth=ground_truth)
        assert np.allclose(out, 2.0)

    def test_value_order2(self):
        loss = make_order2_loss(initial_values=(0.0, 0.0), eta=2.0)
        # choose values such that F = dfdxdx - f = 0 for each sample -> functional_loss = 0
        value_dict = {
            "x": np.array([[0.0], [1.0]]),
            "f": np.array([1.0, 2.0]),
            "dfdx": np.array([[1.0], [2.0]]),
            "dfdxdx": np.array([[[1.0]], [[2.0]]]),  # shape (n_samples, 1, 1)
        }
        # functional_loss == 0
        # initial value losses: eta*(f0-init0)^2 + eta*(dfdx0-init1)^2
        # with eta=2.0, f0=1.0, dfdx0=1.0, initial_values=(0,0) => 2*1 + 2*1 = 4.0
        ground_truth = np.zeros((2,))
        out = loss.value(value_dict, ground_truth=ground_truth)
        assert np.allclose(out, 4.0)

    def test_gradient_order1(
        self,
    ):
        loss = make_order1_loss(initial_value=0.0, eta=1.0)

        value_dict = {
            "x": np.array([[0.0], [1.0]]),
            "f": np.array([1.0, 2.0]),
            "dfdx": np.array([[1.0], [2.0]]),
            "dfdp": np.array([[0.1, 0.2], [0.3, 0.4]]),  # (n_samples, n_params)
            "dfdxdp": np.array([[[0.0, 0.0]], [[0.0, 0.0]]]),  # (n_samples,1,n_params)
            "dfdop": np.array([[0.5], [0.6]]),  # (n_samples, n_param_op)
            "dfdopdx": np.array([[[0.0]], [[0.0]]]),  # (n_samples,1,n_param_op)
        }
        ground_truth = np.zeros((2,))

        # no operator params -> _opt_param_op False -> returns only d_p
        loss._opt_param_op = False
        d_p = loss.gradient(value_dict, ground_truth=ground_truth)
        # Since _ODE_functional(value_dict) == 0 -> weighted_diff zeros and corresponding term 0
        # Only boundary term: 2 * eta * (f0 - init) * dfdp[0] = 2 * 1 * 1.0 * [0.1, 0.2]
        expected_dp = 2.0 * np.array([0.1, 0.2])
        assert np.allclose(d_p, expected_dp)

        # with operator params enabled, return tuple (d_p, d_op)
        loss._opt_param_op = True
        d_p2, d_op2 = loss.gradient(value_dict, ground_truth=ground_truth)
        expected_dp2 = expected_dp  # same as before
        expected_dop2 = 2.0 * np.array([0.5])  # 2 * eta * (f0 - init) * df dop[0]
        assert np.allclose(d_p2, expected_dp2)
        assert np.allclose(d_op2, expected_dop2)

    def test_gradient_order2(self):
        loss = make_order2_loss(initial_values=(0.0, 0.0), eta=1.0)

        # build value_dict so functional part (F) is zero -> weighted_diff zeros -> only boundary terms remain
        value_dict = {
            "x": np.array([[0.0], [1.0]]),
            "f": np.array([1.0, 2.0]),
            "dfdx": np.array([[1.0], [2.0]]),
            "dfdxdx": np.array([[[1.0]], [[2.0]]]),  # matches f -> F = dfdxdx - f == 0
            # parameter derivatives: (n_samples, n_params)
            "dfdp": np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]),
            # (n_samples, 1, n_params)
            "dfdxdp": np.array([[[0.01, 0.02, 0.03]], [[0.0, 0.0, 0.0]]]),
            # (n_samples, 1, 1, n_params) for second derivative wrt params
            "dfdxdxdp": np.zeros((2, 1, 1, 3)),
            # operator params: choose single operator param (n_param_op = 1)
            "dfdop": np.array([[0.5], [0.6]]),  # (n_samples, n_param_op)
            "dfdopdx": np.zeros((2, 1, 1)),  # (n_samples,1,n_param_op)
            "dfdopdxdx": np.zeros((2, 1, 1, 1)),  # (n_samples,1,1,n_param_op)
        }
        ground_truth = np.zeros((2,))

        # no operator params -> _opt_param_op False -> returns only d_p
        loss._opt_param_op = False
        d_p = loss.gradient(value_dict, ground_truth=ground_truth)
        # boundary contributions:
        # 2 * eta * (f0 - init0) * dfdp[0]  -> 2 * 1 * 1.0 * [0.1,0.2,0.3] = [0.2,0.4,0.6]
        # 2 * eta * sum(dfdx0 - init1) * dfdxdp[0,0,:] -> 2 * 1 * 1.0 * [0.01,0.02,0.03] = [0.02,0.04,0.06]
        # total = [0.22, 0.44, 0.66]
        expected_dp = np.array([0.22, 0.44, 0.66])
        assert np.allclose(d_p, expected_dp)

        # with operator params enabled, return tuple (d_p, d_op)
        loss._opt_param_op = True
        d_p2, d_op2 = loss.gradient(value_dict, ground_truth=ground_truth)
        expected_dp2 = expected_dp
        # expected operator gradient: 2 * eta * (f0 - init0) * dfdop[0] = 2 * 1 * 1.0 * [0.5] = [1.0]
        expected_dop2 = 2.0 * np.array([0.5])
        assert np.allclose(d_p2, expected_dp2)
        assert np.allclose(d_op2, expected_dop2)

    def test_gradient_handles_empty_dfdp_and_empty_dfdop(self):
        loss = make_order1_loss(initial_value=0.0)
        # empty dfdp/d fdop arrays (shape[0]==0) should produce empty grad arrays
        value_dict = {
            "x": np.zeros((0, 1)),
            "f": np.zeros((0,)),
            "dfdx": np.zeros((0, 1)),
            "dfdp": np.zeros((0, 2)),
            "dfdxdp": np.zeros((0, 1, 2)),
            "dfdop": np.zeros((0, 1)),
            "dfdopdx": np.zeros((0, 1, 1)),
        }
        loss._opt_param_op = False
        grad = loss.gradient(value_dict, ground_truth=np.zeros((0,)))
        assert grad.size == 0

        loss._opt_param_op = True
        d_p, d_op = loss.gradient(value_dict, ground_truth=np.zeros((0,)))
        assert d_p.size == 0
        assert d_op.size == 0
