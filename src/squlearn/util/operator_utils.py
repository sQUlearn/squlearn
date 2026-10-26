"""Helpers for qc_executor operators."""

import sympy as sp
from qc_executor.base.operator_base import QuantumOperatorBase

__all__ = ["square_operator", "multiply_operators"]


def multiply_operators(
    left: QuantumOperatorBase, right: QuantumOperatorBase
) -> QuantumOperatorBase:
    """Return ``left`` composed with ``right``, simplified - also for symbolic coefficients.

    ``QuantumOperatorBase.compose`` and ``simplify`` refuse operators whose coefficients
    still hold free parameters ("bind the parameters first"). A parametrized observable
    is only bound at evaluation time, though, so its product is needed while it is still
    symbolic. Writing the operators as ``sum_i c_i P_i`` and ``sum_j d_j Q_j``, the product is
    ``sum_ij c_i d_j (P_i Q_j)``: the products of Pauli strings and their phases are
    numeric, only the coefficients ``c_i d_j`` are symbolic, so this composes the bare
    Pauli structure and keeps the coefficients on the side.

    Args:
        left: The left operator.
        right: The right operator.

    Returns:
        The product operator, of the type of ``left``, with equal Pauli strings merged and
        terms whose coefficient vanishes dropped.
    """
    if not (left.is_parametrized or right.is_parametrized):
        return left.compose(right).simplify()

    left_coeffs = [sp.sympify(c) for c in left.coeffs]
    right_coeffs = [sp.sympify(c) for c in right.coeffs]
    num_right = len(right_coeffs)

    # One term per ordered pair (i, j), row-major, each with the numeric phase of P_i Q_j.
    left_structure = type(left)(left.paulis, [1.0] * len(left_coeffs))
    right_structure = type(left)(right.paulis, [1.0] * num_right)
    products = left_structure.compose(right_structure)
    product_labels, phases = products.paulis, products.coeffs_array

    merged: dict = {}
    for pair, (label, phase) in enumerate(zip(product_labels, phases)):
        i, j = divmod(pair, num_right)
        merged[label] = merged.get(label, 0) + complex(phase) * left_coeffs[i] * right_coeffs[j]

    kept = {}
    for label, coefficient in merged.items():
        coefficient = sp.expand(coefficient)
        if coefficient != 0:
            kept[label] = coefficient
    if not kept:
        return type(left)(num_qubits=left.num_qubits)
    return type(left)(list(kept), list(kept.values()))


def square_operator(operator: QuantumOperatorBase) -> QuantumOperatorBase:
    """Return ``operator`` composed with itself, simplified - also for symbolic coefficients.

    See :func:`multiply_operators`.

    Args:
        operator: The operator to square.

    Returns:
        The squared operator, of the same type.
    """
    return multiply_operators(operator, operator)
