"""Tests for the seam between sQUlearn's Executor and qc_executor's QiskitExecutor."""

import numpy as np
import pytest

from qc_executor import Parameters
from qc_executor.qiskit import QiskitExecutor

from squlearn import Executor
from squlearn.encoding_circuit import ParamZFeatureMap
from squlearn.observables import SummedPaulis
from squlearn.qnn.lowlevel_qnn import LowLevelQNN


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


def test_as_qc_executor_injects_estimator_not_raw_backend():
    executor = Executor("statevector_simulator")
    qc_exec = executor.as_qc_executor

    assert qc_exec._backend is executor.backend
    assert qc_exec._sampler is not None


def test_as_qc_executor_is_cached_across_accesses():
    executor = Executor("statevector_simulator")
    assert executor.as_qc_executor is executor.as_qc_executor


def test_as_qc_executor_rebuilds_after_set_backend():
    from qiskit_aer import Aer

    executor = Executor("statevector_simulator")
    first = executor.as_qc_executor
    executor.set_backend(Aer.get_backend("aer_simulator"))
    assert executor.as_qc_executor is not first


def test_native_qnn_evaluation_matches_direct_statevector_path():
    """Expectation values evaluated through the injected-primitive seam
    (as_qc_executor) must match qc_executor's direct statevector path
    bit-for-bit."""
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(3)
    rng = np.random.default_rng(3)
    x = rng.random((4, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    seam_qnn = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    seam = seam_qnn.evaluate(x, param, param_op, "f", "dfdx", "dfdp", "dfdop")

    x_p = Parameters("x", 2)
    p_p = Parameters("p", pqc.num_parameters)
    p_op_p = Parameters("p_op", obs.num_parameters)
    circuit = pqc.get_circuit(x_p, p_p)
    operator = obs.get_operator(p_op_p)
    direct = QiskitExecutor(backend="statevector")

    for i, x_i in enumerate(x):
        expected = direct.expectation_value(
            circuit, operator, x=x_i.tolist(), p=param.tolist(), p_op=param_op.tolist()
        )
        assert np.isclose(seam["f"][i], expected, atol=1e-12)
