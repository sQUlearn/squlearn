import numpy as np
import sympy as sp
import pytest

from squlearn.kernel.loss import ODELoss


def make_identity_kernel_tensor(n_samples: int, order: int):
    """Return a simple kernel_tensor: list of identity matrices length order+1."""
    return [np.eye(n_samples) for _ in range(order + 1)]


class TestODELoss:
    def test_sympy_expr_basic_loss(self):
        # ODE: dfdx - f = 0
        x, f, dfdx = sp.symbols("x f dfdx")
        expr = dfdx - f

        # single sample
        n = 1
        initial_values = [2.5]  # choose to match f(0) so initial penalty = 0

        # Create ODELoss (sympy case -> symbols required)
        ode = ODELoss(expr, symbols_involved_in_ode=[x, f, dfdx], initial_values=initial_values)

        # kernel_tensor identity for orders 0 and 1
        kernel_tensor = make_identity_kernel_tensor(n_samples=n, order=1)

        # choose parameters: alpha0, alpha1  (shape (n+1,1))
        # With kernel=I and alpha1=2.0, alpha0=0.5:
        # f = I @ [alpha1] + alpha0 = 2 + 0.5 = 2.5
        # df = I @ [alpha1] = 2.0
        params = np.array([[0.5], [2.0]])

        data = np.array([[0.0]])

        # compute loss: residual = df - f = 2.0 - 2.5 = -0.5 -> squared = 0.25
        # initial penalty: f(0) - initial_values = 2.5 - 2.5 = 0 -> total loss_value = 0.25
        loss_value = ode.compute(params, data, kernel_tensor=kernel_tensor)
        assert pytest.approx(0.25, rel=1e-12) == float(loss_value)

    def test_callable_with_symbols_basic_loss(self):
        # same ODE but as callable; provide symbols as well for sanity checks
        x, f, dfdx = sp.symbols("x f dfdx")

        def ode_callable(x_vals, f_vals, df_vals):
            return df_vals - f_vals

        initial_values = [2.5]
        ode = ODELoss(
            ode_callable,
            symbols_involved_in_ode=[x, f, dfdx],
            initial_values=initial_values,
            ode_order=1,
        )

        kernel_tensor = make_identity_kernel_tensor(n_samples=1, order=1)
        params = np.array([[0.5], [2.0]])
        data = np.array([[0.0]])

        loss_value = ode.compute(params, data, kernel_tensor=kernel_tensor)
        assert pytest.approx(0.25, rel=1e-12) == float(loss_value)

    def test_callable_without_symbols_requires_ode_order(self):
        # callable without symbols: must pass ode_order
        def ode_callable(x_vals, f_vals, df_vals):
            return df_vals - f_vals

        # passing ode_order explicitly
        ode = ODELoss(
            ode_callable, symbols_involved_in_ode=None, initial_values=[2.5], ode_order=1
        )

        kernel_tensor = make_identity_kernel_tensor(n_samples=1, order=1)
        params = np.array([[0.5], [2.0]])
        data = np.array([[0.0]])

        loss_value = ode.compute(params, data, kernel_tensor=kernel_tensor)
        assert pytest.approx(0.25, rel=1e-12) == float(loss_value)

    def test_sympy_expr_without_symbols_raises(self):
        x, f, dfdx = sp.symbols("x f dfdx")
        expr = dfdx - f

        # symbols_involved_in_ode is required for sympy.Expr
        with pytest.raises(ValueError):
            ODELoss(expr, symbols_involved_in_ode=None, initial_values=[1.0])

    def test_initial_values_length_mismatch_raises(self):
        # sympy expr implies order=1, but provide initial_values length != 1
        x, f, dfdx = sp.symbols("x f dfdx")
        expr = dfdx - f

        with pytest.raises(ValueError):
            ODELoss(expr, symbols_involved_in_ode=[x, f, dfdx], initial_values=[1.0, 2.0])

    def test_callable_symbols_and_ode_order_mismatch_raises(self):
        # callable with symbols provided but ode_order inconsistent should raise
        def ode_callable(x_vals, f_vals):
            return f_vals  # dummy

        # symbols indicate order = len(symbols)-2 = 1, but ode_order set to 2 -> mismatch
        x, f, dfdx = sp.symbols("x f dfdx")
        with pytest.raises(ValueError):
            ODELoss(
                ode_callable,
                symbols_involved_in_ode=[x, f, dfdx],
                initial_values=[0.0],
                ode_order=2,
            )
