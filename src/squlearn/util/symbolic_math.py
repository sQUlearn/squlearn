"""Symbolic-aware replacements for NumPy's elementwise math functions.

``qc_executor.parameters.Parameter`` is a SymPy symbol, not Qiskit's old
``ParameterExpression``, so it has no ``.arccos()``/``.sin()``/... method for
NumPy's ufunc dispatch to call (NumPy resolves ``np.arccos(x)`` on an
object-dtype input by looking for ``x.arccos()``). qc_executor's migration
guide documents the fix for this class of break: "build expressions with
SymPy rather than Qiskit" (``sp.sin(x[0])``, not the old ``x[0].sin()``).

The functions here use the SymPy equivalent for symbolic input and the plain
NumPy function - unchanged - for numeric input, since SymPy's functions do not
accept a NumPy array.
"""

import numpy as np
import sympy as sp

__all__ = ["symbolic_aware", "arccos", "arctan", "_NUMPY_TO_SYMPY"]

#: NumPy function -> SymPy function, for the functions Qiskit's parameter
#: expressions used to provide as methods.
_NUMPY_TO_SYMPY = {
    np.sin: sp.sin,
    np.cos: sp.cos,
    np.tan: sp.tan,
    np.arcsin: sp.asin,
    np.arccos: sp.acos,
    np.arctan: sp.atan,
    np.sinh: sp.sinh,
    np.cosh: sp.cosh,
    np.tanh: sp.tanh,
    np.exp: sp.exp,
    np.log: sp.log,
    np.sqrt: sp.sqrt,
}


def symbolic_aware(function):
    """Return ``function`` extended to accept SymPy expressions.

    Args:
        function: A NumPy function such as ``np.arccos``.

    Returns:
        A function that calls the SymPy counterpart for symbolic input and
        ``function`` itself otherwise. Anything without a known counterpart is
        returned unchanged.
    """
    sympy_function = _NUMPY_TO_SYMPY.get(function)
    if sympy_function is None:
        return function

    def wrapped(x, *args, **kwargs):
        if isinstance(x, sp.Basic):
            return sympy_function(x)
        return function(x, *args, **kwargs)

    wrapped.__name__ = getattr(function, "__name__", "wrapped")
    wrapped.__doc__ = function.__doc__
    return wrapped


arccos = symbolic_aware(np.arccos)
arctan = symbolic_aware(np.arctan)
