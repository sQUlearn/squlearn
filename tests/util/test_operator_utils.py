"""Tests for square_operator."""

import itertools
import random

import numpy as np
import pytest
import sympy as sp
from qc_executor import Parameters, QuantumOperator

from squlearn.util.operator_utils import square_operator


def _terms(operator):
    return {
        label: complex(coefficient) for label, coefficient in zip(operator.paulis, operator.coeffs)
    }


def _bind(operator, values):
    return {
        label: complex(sp.sympify(coefficient).xreplace(values))
        for label, coefficient in zip(operator.paulis, operator.coeffs)
    }


class TestSquareOperator:
    def test_numeric_operator_matches_compose(self):
        operator = QuantumOperator(["ZI", "IZ", "XX"], [0.5, -1.0, 2.0])
        expected = operator.compose(operator).simplify()

        assert _terms(square_operator(operator)) == pytest.approx(_terms(expected))

    @pytest.mark.parametrize("num_qubits, num_terms", [(1, 3), (2, 5), (3, 8), (3, 12)])
    def test_symbolic_operator_matches_the_bound_square(self, num_qubits, num_terms):
        """Squaring while symbolic, then binding, equals binding, then squaring."""
        rng = random.Random(num_terms)
        labels = rng.sample(
            ["".join(t) for t in itertools.product("IXYZ", repeat=num_qubits)], num_terms
        )
        parameters = Parameters("p_op", num_terms)
        values = np.random.default_rng(num_terms).random(num_terms)
        binding = dict(zip(parameters, (float(v) for v in values)))

        squared = square_operator(QuantumOperator(labels, list(parameters)))
        bound = QuantumOperator(labels, list(values))
        expected = bound.compose(bound).simplify()

        got = _bind(squared, binding)
        want = _terms(expected)
        for label in set(got) | set(want):
            assert got.get(label, 0) == pytest.approx(want.get(label, 0), abs=1e-10)
        assert squared.is_parametrized

    def test_shared_pauli_strings_are_merged(self):
        p = Parameters("p", 2)
        squared = square_operator(QuantumOperator(["Z", "Z"], list(p)))

        assert squared.paulis == ["I"]
        assert sp.expand(squared.coeffs[0] - (p[0] + p[1]) ** 2) == 0

    def test_terms_that_cancel_are_dropped(self):
        """(X Y - Y X terms) cancel: (a X + b Y)^2 has no X Y contribution left."""
        p = Parameters("p", 2)
        squared = square_operator(QuantumOperator(["X", "Y"], list(p)))

        assert squared.paulis == ["I"]

    def test_mixed_symbolic_and_numeric_coefficients(self):
        p = Parameters("p", 1)
        squared = square_operator(QuantumOperator(["ZI", "IZ"], [p[0], 0.5]))
        got = _bind(squared, {p[0]: 2.0})
        want = _terms(
            QuantumOperator(["ZI", "IZ"], [2.0, 0.5])
            .compose(QuantumOperator(["ZI", "IZ"], [2.0, 0.5]))
            .simplify()
        )

        assert got == pytest.approx(want)
