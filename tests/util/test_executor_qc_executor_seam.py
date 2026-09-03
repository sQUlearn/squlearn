"""Tests for the seam between sQUlearn's Executor and qc_executor's QiskitExecutor."""

import numpy as np
import pytest

from qc_executor import Parameters
from qc_executor.qiskit import QiskitExecutor

from squlearn import Executor
from squlearn.encoding_circuit import ParamZFeatureMap
from squlearn.observables import SummedPaulis
from squlearn.qnn.lowlevel_qnn import LowLevelQNN
from squlearn.util.execution.parallel_estimator import ParallelEstimatorV2


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
    qc_executor = QiskitExecutor(backend=executor.get_estimator(), shots=100, seed=0)

    assert qc_executor._backend is executor.backend
    assert qc_executor._sampler is not None
    # Aer is not IBM Quantum hardware, so no ISA transpilation is required.
    assert qc_executor._isa_transpile is False


def test_executor_owns_exactly_one_qc_executor_instance():
    """The Executor no longer builds a second, separate qc_executor instance
    for the native evaluation path (the old as_qc_executor / _qc_backend_executor
    split) - self._qc_executor is the single instance used everywhere."""
    executor = Executor("qasm_simulator")
    assert executor._qc_executor is not None
    assert executor._qc_executor.backend is executor.backend


def test_qc_executor_rebuilds_after_set_backend():
    from qiskit_aer import Aer

    executor = Executor("statevector_simulator")
    first = executor._qc_executor
    executor.set_backend(Aer.get_backend("aer_simulator"))
    assert executor._qc_executor is not first


def test_native_qnn_evaluation_matches_direct_statevector_path():
    """Expectation values and their first-order derivatives evaluated through
    Executor.expectation_value/expectation_value_derivatives (the native
    evaluation path used by LowLevelQNNUnified) must match qc_executor's
    direct statevector path bit-for-bit."""
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
        call_kwargs = dict(x=x_i.tolist(), p=param.tolist(), p_op=param_op.tolist())

        expected_f = direct.expectation_value(circuit, operator, **call_kwargs)
        assert np.isclose(seam["f"][i], expected_f, atol=1e-12)

        for key, derivative_param in (("dfdx", "x"), ("dfdp", "p"), ("dfdop", "p_op")):
            expected = direct.expectation_value_derivatives(
                circuit, operator, derivative_param, **call_kwargs
            )
            assert np.allclose(seam[key][i], np.asarray(expected).reshape(-1), atol=1e-12)


def test_estimator_property_is_the_raw_undecorated_primitive(executor):
    """Executor.estimator (unlike get_estimator()) is qc_executor's own raw
    primitive - no retry/caching/parallelization wrapping."""
    assert executor.estimator is executor._qc_executor.raw_estimator


def test_get_estimator_is_the_decorated_qc_executor_primitive(executor):
    """get_estimator() returns qc_executor's own decorated primitive
    (built via the primitive_wrapper hook, see Executor._decorate_primitive) -
    not a separately constructed sQUlearn-side object."""
    assert executor.get_estimator() is executor._qc_executor.estimator


def test_decorate_primitive_updates_inner_estimator_and_returns_stable_wrapper():
    """_decorate_primitive is qc_executor's primitive_wrapper: called for
    every primitive qc_executor (re)builds. It must update
    self._inner_estimator (the actual execution target for
    estimator_run_v1/v2) to the fresh raw primitive, and its returned
    wrapper must be an ExecutorEstimatorV2 resolving back through this same
    Executor - so a caller holding it survives future rebuilds."""
    from squlearn.util.executor import ExecutorEstimatorV2

    executor = Executor("statevector_simulator")

    class _StubEstimator:
        def run(self, pubs, *, precision=None):
            return "swapped"

    stub = _StubEstimator()
    wrapper = executor._decorate_primitive(stub, "estimator")

    assert executor._inner_estimator is stub
    assert isinstance(wrapper, ExecutorEstimatorV2)
    assert wrapper._executor is executor


def test_qpu_parallelization_wraps_the_inner_estimator_not_the_raw_one():
    """QPU parallelization must wrap the execution target (_inner_estimator),
    while .estimator stays the plain raw primitive and get_estimator() stays
    a stable wrapper resolving through the Executor - not a ParallelEstimator
    itself."""
    executor = Executor("qasm_simulator", shots=100, seed=0, qpu_parallelization=2)

    assert isinstance(executor._inner_estimator, ParallelEstimatorV2)
    assert not isinstance(executor.estimator, ParallelEstimatorV2)

    first = executor.get_estimator()
    second = executor.get_estimator()
    assert first is second  # stable identity across repeated access


def test_statevector_exact_estimator_uses_qc_executors_statevector_alias():
    """Exact statevector simulation (shots=None) is delegated to
    qc_executor's own "statevector" string convention (StatevectorEstimator/
    StatevectorSampler) rather than sQUlearn constructing anything itself -
    qc_executor's own backend for that instance is None (no Aer object
    needed), even though Executor.backend keeps a real Aer object for API
    stability (is_statevector, backend_name, ...)."""
    executor = Executor("statevector_simulator")

    assert executor._qc_executor.backend is None
    assert executor.backend is not None  # sQUlearn's own, kept for API stability
    assert executor.is_statevector is True
