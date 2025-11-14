import numpy as np

from squlearn.optimizers import LBFGSB, SLSQP, SPSA
from squlearn.optimizers.optimizer_base import OptimizerResult


def quad(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.dot(x, x))


def quad_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return 2.0 * x


class TestWrappedOptimizers:
    def test_slsqp_minimize_with_grad(self):
        options = {"maxiter": 200, "tol": 1e-9}

        opt = SLSQP(options=options)
        x0 = np.array([2.0])
        res = opt.minimize(fun=quad, x0=x0, grad=quad_grad)

        assert isinstance(res, OptimizerResult)
        assert np.linalg.norm(res.x) < 1e-6
        assert res.fun < 1e-8

    def test_slsqp_bounds_respected(self):
        options = {"maxiter": 200}
        opt = SLSQP(options=options)
        x0 = np.array([2.0])
        bounds = np.array([[-1.0, 1.0]])

        res = opt.minimize(fun=quad, x0=x0, grad=quad_grad, bounds=bounds)

        x_final = np.asarray(res.x).flatten()
        assert -1.0 - 1e-12 <= x_final[0] <= 1.0 + 1e-12

    def test_lbfgsb_minimize_with_grad(self):
        options = {"maxiter": 200}

        opt = LBFGSB(options=options)
        x0 = np.array([2.0])
        res = opt.minimize(fun=quad, x0=x0, grad=quad_grad)

        assert isinstance(res, OptimizerResult)
        assert np.linalg.norm(res.x) < 1e-6
        assert res.fun < 1e-8

    def test_lbfgsb_without_grad_uses_scipy_approx(self):
        # If grad=None, scipy should approximate and optimization still proceed
        options = {"maxiter": 200}
        opt = LBFGSB(options=options)
        x0 = np.array([2.0])

        res = opt.minimize(fun=quad, x0=x0, grad=None)

        assert np.linalg.norm(res.x) < 1e-6
        assert res.fun < 1e-8

    def test_spsa_minimize(self):
        options = {"maxiter": 50}
        opt = SPSA(options=options)

        x0 = np.array([1.0])
        res = opt.minimize(fun=quad, x0=x0, grad=None)

        assert isinstance(res, OptimizerResult)
        assert np.linalg.norm(res.x) < 1e-3
        assert res.fun < 1e-6
