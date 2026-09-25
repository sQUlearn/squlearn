"""Helpers for building qc_executor circuits."""

from typing import Optional, Sequence

from qc_executor.base.circuit_base import QuantumCircuitBase

__all__ = ["compose_keep_parameters"]


def compose_keep_parameters(
    circuit: QuantumCircuitBase,
    other: QuantumCircuitBase,
    qubits: Optional[Sequence[int]] = None,
    clbits: Optional[Sequence[int]] = None,
) -> QuantumCircuitBase:
    """Append ``other`` onto ``circuit`` in place, leaving every parameter name as it is.

    ``QuantumCircuitBase.compose`` re-indexes the parameters of both circuits into one
    vector as soon as they share a parameter. Parameters are interned by name, so a shared
    parameter is the *same* one: sQUlearn's layered circuits reuse the feature ``x[0]`` in
    every layer and bind ``x`` and ``p`` by name, which the re-indexing destroys. This
    appends the instructions as they are.

    Args:
        circuit: The circuit to append to, modified in place.
        other: The circuit to append.
        qubits: Where the qubits of ``other`` land, by default the identity.
        clbits: Where the classical bits of ``other`` land, by default the identity.

    Returns:
        ``circuit``, to allow chaining.
    """
    if qubits is None:
        qubits = range(other.num_qubits)
    circuit.ir.extend(other.ir, [int(q) for q in qubits], clbits)
    return circuit
