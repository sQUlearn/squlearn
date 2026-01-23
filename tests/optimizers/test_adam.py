import numpy as np

from squlearn.optimizers import Adam


def quad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.dot(x, x)


def quad_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return 2.0 * x


def shifted_quartic(x):
    # global minima at 2 and -3
    return (x[0] - 2) ** 2 * (x[0] + 3) ** 2


def shifted_quartic_grad(x):
    return np.array([2 * (x[0] - 2) * (x[0] + 3) ** 2 + 2 * (x[0] + 3) * (x[0] - 2) ** 2])


class TestAdamOptimizer:
    def test_minimize_quadratic_with_grad(self):
        options = {"lr": 0.1, "tol": 1e-8, "maxiter": 100}
        opt = Adam(options=options)

        x0 = np.array([2.0])
        res = opt.minimize(fun=quad, x0=x0, grad=quad_grad)

        x_final = np.asarray(res.x).flatten()
        assert np.linalg.norm(x_final) < 1e-2
        assert res.fun < 1e-2

    def test_minimize_quadratic_without_grad_uses_finite_diff(self):
        options = {"lr": 0.05, "tol": 1e-6, "maxiter": 100, "eps": 1e-3}
        opt = Adam(options=options)

        x0 = np.array([1.5])
        res = opt.minimize(fun=quad, x0=x0, grad=None)

        x_final = np.asarray(res.x).flatten()
        assert np.linalg.norm(x_final) < 1e-2
        assert res.fun < 1e-2

    def test_bounds_clipping(self):
        options = {"lr": 0.1, "tol": 1e-8, "maxiter": 100}
        opt = Adam(options=options)

        x0 = np.array([2.0])
        bounds = np.array([[-1.0, 1.0]])  # shape (n_params, 2)
        res = opt.minimize(fun=shifted_quartic, x0=x0, grad=shifted_quartic_grad, bounds=bounds)

        x_final = np.asarray(res.x).flatten()
        assert -1.0 - 1e-12 <= x_final[0] <= 1.0 + 1e-12
        assert res.fun >= 0.0

    def test_reset_resets_internal_state(self):
        options = {"lr": 0.1, "maxiter": 2}
        opt = Adam(options=options)

        x0 = np.array([1.0])
        _ = opt.minimize(fun=quad, x0=x0, grad=quad_grad)

        opt.reset()
        assert opt.m is None
        assert opt.v is None
        assert opt.x is None

    def test_lr_callable(self):
        def lr_callable(it):
            return 0.1 if it < 5 else 0.01

        options = {"lr": lr_callable, "tol": 1e-6, "maxiter": 100}
        opt = Adam(options=options)

        x0 = np.array([1.0])
        res = opt.minimize(fun=quad, x0=x0, grad=quad_grad)

        x_final = np.asarray(res.x).flatten()
        assert np.linalg.norm(x_final) < 1e-2
        assert res.fun < 1e-2
