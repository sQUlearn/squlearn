import numpy as np
import pytest

from squlearn.optimizers import FiniteDiffGradient, StochasticPerturbationGradient


def quad_fun(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2)


def quad_grad(x: np.ndarray) -> np.ndarray:
    return 2.0 * x


def sin_quad(x: np.ndarray) -> np.ndarray:
    return np.sum(np.sin(x) + (x - 1.0) ** 2)


def sin_quad_grad(x: np.ndarray) -> np.ndarray:
    return np.cos(x) + 2.0 * (x - 1.0)


class TestFiniteDiffGradient:

    def test_invalid_formula_raises(self):
        with pytest.raises(ValueError):
            FiniteDiffGradient(fun=quad_fun, eps=1e-3, formula="unknown")

    @pytest.mark.parametrize(
        "formula, eps",
        [
            ("central", 1e-6),
            ("five-point", 1e-6),
            ("forward", 1e-6),
            ("backwards", 1e-6),
        ],
    )
    @pytest.mark.parametrize(
        "fun, grad",
        [
            (quad_fun, quad_grad),
            (sin_quad, sin_quad_grad),
        ],
    )
    def test_gradient_matches_analytic_for_various_formulas(self, formula, eps, fun, grad):
        # test-point (3D) used for all objective functions
        x = np.array([1.234, -0.75, 2.0])
        fd = FiniteDiffGradient(fun=fun, eps=eps, formula=formula)
        g_approx = fd.gradient(x)
        g_true = grad(x)

        assert np.allclose(g_approx, g_true)

    def test_non_1d_input_raises(self):
        fd = FiniteDiffGradient(fun=quad_fun, eps=1e-6, formula="central")
        with pytest.raises(ValueError):
            fd.gradient(np.array([[1.0, 2.0]]))


class TestStochasticPerturbationGradient:

    # Reuse module-level functions for SPG as well
    @pytest.mark.parametrize(
        "fun, x, eps, seed",
        [
            (quad_fun, np.array([0.1, -0.2, 1.0]), 1e-3, 42),
            (sin_quad, np.array([0.1, -0.2, 1.0]), 1e-3, 42),
        ],
    )
    def test_gradient_matches_manual_rng_calculation(self, fun, x, eps, seed):
        spg = StochasticPerturbationGradient(fun=fun, eps=eps, seed=seed)
        g_est = spg.gradient(x)

        # Recreate the random vector deterministically and compute expected value
        rng = np.random.default_rng(seed=seed)
        r = rng.random(len(x))
        f1 = fun(x + eps * r)
        f2 = fun(x - eps * r)
        expected = (f1 - f2) / (2.0 * eps * r)

        assert np.allclose(g_est, expected, atol=1e-8, rtol=1e-6)

    @pytest.mark.parametrize(
        "fun, x, seed",
        [
            (quad_fun, np.array([0.3, -0.4]), 7),
            (sin_quad, np.array([0.3, -0.4]), 7),
        ],
    )
    def test_set_eps_changes_estimate(self, fun, x, seed):
        spg = StochasticPerturbationGradient(fun=fun, eps=1e-3, seed=seed)
        g1 = spg.gradient(x)

        spg.set_eps(1e-2)
        g2 = spg.gradient(x)

        assert g1.shape == g2.shape
        assert not np.allclose(g1, g2)

    def test_non_1d_input_raises(self):
        spg = StochasticPerturbationGradient(fun=quad_fun, eps=1e-3, seed=0)
        with pytest.raises(ValueError):
            spg.gradient(np.array([[0.1, 0.2]]))
