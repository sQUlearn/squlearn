"""Tests for the symbolic-aware NumPy replacements."""

import numpy as np
import pytest
import sympy as sp
from qc_executor import Parameters

from squlearn.util.symbolic_math import arccos, arctan, symbolic_aware


class TestSymbolicAware:
    X = Parameters("x", 1)

    @pytest.mark.parametrize(
        "numpy_function, sympy_function",
        [
            (np.sin, sp.sin),
            (np.cos, sp.cos),
            (np.tan, sp.tan),
            (np.arcsin, sp.asin),
            (np.arccos, sp.acos),
            (np.arctan, sp.atan),
            (np.exp, sp.exp),
            (np.log, sp.log),
            (np.sqrt, sp.sqrt),
        ],
    )
    def test_symbolic_input_uses_the_sympy_function(self, numpy_function, sympy_function):
        assert symbolic_aware(numpy_function)(self.X[0]) == sympy_function(self.X[0])

    @pytest.mark.parametrize("numpy_function", [np.sin, np.arccos, np.arctan, np.exp])
    def test_numeric_input_is_left_to_numpy(self, numpy_function):
        wrapped = symbolic_aware(numpy_function)
        scalar, array = 0.3, np.array([0.1, 0.5, 0.9])

        assert isinstance(wrapped(scalar), np.floating)
        assert wrapped(scalar) == numpy_function(scalar)
        np.testing.assert_array_equal(wrapped(array), numpy_function(array))

    def test_numpy_would_fail_on_a_parameter(self):
        """The reason the wrapper exists: NumPy looks for an ``.arccos()`` method."""
        with pytest.raises(TypeError, match="arccos"):
            np.arccos(self.X[0])

    def test_function_without_a_counterpart_is_returned_unchanged(self):
        assert symbolic_aware(np.floor) is np.floor

    def test_prebuilt_helpers(self):
        assert arccos(self.X[0]) == sp.acos(self.X[0])
        assert arctan(self.X[0]) == sp.atan(self.X[0])
        assert arccos(0.5) == np.arccos(0.5)
        assert arctan(0.5) == np.arctan(0.5)

    def test_expression_input(self):
        expression = 2 * self.X[0] + 1
        assert arccos(expression) == sp.acos(expression)
