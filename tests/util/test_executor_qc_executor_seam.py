"""Tests for the seam between sQUlearn's Executor and qc_executor's QiskitExecutor."""

import pytest

from qc_executor.qiskit import QiskitExecutor

from squlearn import Executor


@pytest.fixture
def executor():
    return Executor("qasm_simulator", shots=100, seed=0)


def test_estimator_backend_property_returns_executor_backend(executor):
    assert executor.get_estimator().backend is executor.backend


def test_sampler_backend_property_returns_executor_backend(executor):
    assert executor.get_sampler().backend is executor.backend


def test_qiskit_executor_resolves_backend_from_injected_estimator(executor):
    """qc_executor.QiskitExecutor reads the injected primitive's ``backend``
    property to recover the underlying backend - required for ISA
    transpilation and for building its own counterpart sampler."""
    qc_exec = QiskitExecutor(backend=executor.get_estimator(), shots=100, seed=0)

    assert qc_exec._backend is executor.backend
    assert qc_exec._sampler is not None
    # Aer is not IBM Quantum hardware, so no ISA transpilation is required.
    assert qc_exec._isa_transpile is False
