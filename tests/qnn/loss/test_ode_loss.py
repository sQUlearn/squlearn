import numpy as np
import pytest
import sympy as sp
from squlearn.qnn.loss import ODELoss


def make_order1_loss(initial_value=0.0, eta=1.0, boundary_handling="pinned"):
    x, f, dfdx = sp.symbols("x f dfdx")
    ode_expr = dfdx - f
    loss = ODELoss(
        ODE_functional=ode_expr,
        symbols_involved_in_ODE=[x, f, dfdx],
        initial_values=[initial_value],
        eta=eta,
        boundary_handling=boundary_handling,
    )
    return loss


def make_order2_loss(initial_values=(0.0, 0.0), eta=1.0, boundary_handling="pinned"):
    x, f, dfdx, dfdxdx = sp.symbols("x f dfdx dfdxdx")
    ode_expr = dfdxdx - f
    loss = ODELoss(
        ODE_functional=ode_expr,
        symbols_involved_in_ODE=[x, f, dfdx, dfdxdx],
        initial_values=list(initial_values),
        eta=eta,
        boundary_handling=boundary_handling,
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
            symbols_involved_in_ODE=symbols,
            initial_values=initial_values,
            ODE_functional=sp.Function("f")(*symbols),
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
            symbols_involved_in_ODE=symbols,
            initial_values=initial_values,
            ODE_functional=sp.Function("f")(*symbols),
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
                symbols_involved_in_ODE=symbols,
                initial_values=[1, 2],
                ODE_functional=sp.Function("f")(*symbols),
            )

        symbols = sp.symbols("x y z w dxdydzdw")
        with pytest.raises(ValueError):
            ODELoss(
                symbols_involved_in_ODE=symbols,
                initial_values=[1, 2, 3],
                ODE_functional=sp.Function("f")(*symbols),
            )

    def test_ansatz_to_floating_boundary_ansatz_order1_shifts_values_and_gradients(self):
        loss = make_order1_loss(initial_value=0.0)

        value_dict = {
            "f": np.array([1.5, 2.0, 3.0]),  # f0 = 1.5 -> bias = 1.5
            "dfdx": np.array([[0.1], [0.2], [0.3]]),  # shape (n_samples,1)
            "dfdp": np.array([[0.2, 0.4], [0.5, 0.6], [0.7, 0.8]]),  # (n_samples, n_params)
            "dfdop": np.array([[0.9], [1.0], [1.1]]),  # (n_samples, n_param_op)
        }

        # test without param_op first
        loss._opt_param_op = False
        vd = {k: v.copy() for k, v in value_dict.items()}
        out = loss._ansatz_to_floating_boundary_ansatz(vd, gradient_calculation=True)

        # f should be shifted by f0 (1.5) -> new first element 0.0
        assert np.allclose(out["f"][0], 0.0)
        # whole f shifted
        assert np.allclose(out["f"], np.array([0.0, 0.5, 1.5]))

        # dfdp should be shifted by its first row
        expected_dfdp = value_dict["dfdp"] - value_dict["dfdp"][0]
        assert np.allclose(out["dfdp"], expected_dfdp)

        # when _opt_param_op True, dfdop should be shifted
        loss._opt_param_op = True
        vd2 = {k: v.copy() for k, v in value_dict.items()}
        out2 = loss._ansatz_to_floating_boundary_ansatz(vd2, gradient_calculation=True)
        expected_dfdop = value_dict["dfdop"] - value_dict["dfdop"][0]
        assert np.allclose(out2["dfdop"], expected_dfdop)

    def test_ansatz_to_floating_boundary_ansatz_order2_shifts_values_and_gradients(self):
        loss = make_order2_loss(initial_values=(0.0, 0.5))

        value_dict = {
            "f": np.array([2.0, 3.0]),  # f0 = 2.0 -> bias = 2.0
            "dfdx": np.array([[1.5], [1.6]]),
            "dfdxdx": np.array(
                [
                    [
                        [0.1],
                    ],
                    [[0.2]],
                ]
            ).reshape(2, 1, 1),
            "dfdp": np.array([[0.3], [0.4]]),  # shape (n_samples, n_params)
            "dfdxdp": np.array([[[0.3]], [[0.4]]]),  # (n_samples,1,n_params)
            "dfdop": np.array([[0.7], [0.8]]),
            "dfdxdop": np.array([[[0.7]], [[0.8]]]),
        }

        loss._opt_param_op = True
        vd = {k: v.copy() for k, v in value_dict.items()}
        out = loss._ansatz_to_floating_boundary_ansatz(vd, gradient_calculation=True)

        # f shifted by 2.0
        assert np.allclose(out["f"], np.array([0.0, 1.0]))
        # dfdx shifted by dfdx0 - initial_values[1] (1.5 - 0.5 = 1.0)
        assert np.allclose(
            out["dfdx"].flatten(), np.array([0.5, 0.6])
        )  # original [1.5,1.6] - 1.0 -> [0.5,0.6]
        # dfdp first row should become zero after subtracting its first row
        assert np.allclose(out["dfdp"][0], np.zeros_like(out["dfdp"][0]))
        # df dop shifted similarly
        assert np.allclose(out["dfdop"][0], np.zeros_like(out["dfdop"][0]))

    def test_value_pinned_boundary_order1_and_floating_raises(self):
        loss = make_order1_loss(initial_value=0.0, eta=2.0, boundary_handling="pinned")
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

        # floating boundary handling should raise NotImplementedError
        loss_floating = make_order1_loss(initial_value=0.0, boundary_handling="floating")
        with pytest.raises(NotImplementedError):
            loss_floating.value(value_dict, ground_truth=ground_truth)

    def test_gradient_order1_pinned_boundary_returns_boundary_contribution_only_and_operator_tuple(
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
