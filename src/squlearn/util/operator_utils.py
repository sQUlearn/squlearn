"""Helpers for qc_executor operators."""

import sympy as sp
from qc_executor.base.operator_base import QuantumOperatorBase

__all__ = ["square_operator"]


def square_operator(operator: QuantumOperatorBase) -> QuantumOperatorBase:
    """Return ``operator`` composed with itself, simplified - also for symbolic coefficients.

    ``QuantumOperatorBase.compose`` and ``simplify`` refuse operators whose coefficients
    still hold free parameters ("bind the parameters first"). A parametrized observable
    is only bound at evaluation time, though, so its square is needed while it is still
    symbolic. Writing the operator as ``sum_i c_i P_i``, the square is
    ``sum_ij c_i c_j (P_i P_j)``: the products of Pauli strings and their phases are
    numeric, only the coefficients ``c_i c_j`` are symbolic, so this composes the bare
    Pauli structure and keeps the coefficients on the side.

    Args:
        operator: The operator to square.

    Returns:
        The squared operator, of the same type, with equal Pauli strings merged and
        terms whose coefficient vanishes dropped.
    """
    if not operator.is_parametrized:
        return operator.compose(operator).simplify()

    labels = operator.paulis
    coeffs = [sp.sympify(c) for c in operator.coeffs]
    num_terms = len(labels)

    # One term per ordered pair (i, j), row-major, each with the numeric phase of P_i P_j.
    structure = type(operator)(labels, [1.0] * num_terms)
    products = structure.compose(structure)
    product_labels, phases = products.paulis, products.coeffs_array

    merged: dict = {}
    for pair, (label, phase) in enumerate(zip(product_labels, phases)):
        i, j = divmod(pair, num_terms)
        merged[label] = merged.get(label, 0) + complex(phase) * coeffs[i] * coeffs[j]

    kept = {}
    for label, coefficient in merged.items():
        coefficient = sp.expand(coefficient)
        if coefficient != 0:
            kept[label] = coefficient
    if not kept:
        return type(operator)(num_qubits=operator.num_qubits)
    return type(operator)(list(kept), list(kept.values()))
