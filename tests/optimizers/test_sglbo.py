"""Tests for the SGLBO optimizer."""

import numpy as np

from squlearn.optimizers.sglbo import SGLBO


# define the function to be optimized
def quadratic(x):
    return np.sum(x**2)


class TestSGLBO:
    """Test class for SGLBO optimizer."""

    def test_minimize(self):
        """Test the minimize method of SGLBO."""
        x0 = np.array([1.0, 2.0])

        optimizer = SGLBO()

        result = optimizer.minimize(quadratic, x0)

        assert np.isclose(result.fun, 0.0, atol=1e-6)

    def test_step(self):
        """Test the step method of SGLBO."""

        def gradient_function(x):
            return 2 * x

        x0 = np.array([2.0])

        optimizer = SGLBO()
        optimizer.x = x0
        optimizer.func = quadratic

        result = optimizer.step(x=optimizer.x, grad=gradient_function(optimizer.x))

        assert result < x0
