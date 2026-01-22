import gc
import weakref
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pennylane as qml
import pytest
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import BackendEstimator, BackendSampler, Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer
from qiskit_ibm_runtime import IBMBackend, Session

from squlearn.util import Executor
from squlearn.util.executor import (
    BaseEstimatorV1,
    BaseEstimatorV2,
    BaseSamplerV1,
    BaseSamplerV2,
    SessionContextMisuseWarning,
)
from squlearn.util.pennylane import PennyLaneCircuit


class TestExecutor:
    @pytest.fixture(scope="module")
    def ExecutorSampler(self) -> Executor:
        """Executor with Sampler initialization."""
        return Executor(Sampler(), seed=0)

    @pytest.fixture(scope="module")
    def ExecutorEstimator(self) -> Executor:
        """Executor with Estimator initialization."""
        return Executor(Estimator(), seed=0)

    @pytest.fixture(scope="module")
    def ExecutorStatevector(self) -> Executor:
        """Executor with statevector_simulator initialization."""
        return Executor("statevector_simulator", seed=0)

    @pytest.fixture(scope="module")
    def ExecutorQasm(self) -> Executor:
        """Executor with qasm_simulator initialization."""
        return Executor("qasm_simulator", seed=0)

    @pytest.fixture(scope="module")
    def ExecutorBackendSampler(self) -> Executor:
        """Executor with BackendSampler initialization."""
        return Executor(BackendSampler(Aer.get_backend("aer_simulator")), seed=0)

    @pytest.fixture(scope="module")
    def ExecutorBackendEstimator(self) -> Executor:
        """Executor with BackendEstimator initialization."""
        return Executor(BackendEstimator(Aer.get_backend("aer_simulator")), seed=0)

    @pytest.fixture(scope="module")
    def ExecutorPennyLane(self) -> Executor:
        """Executor with PennyLane initialization."""
        return Executor("pennylane", seed=0)

    @pytest.fixture(scope="module")
    def ExecutorPennyLaneShots(self) -> Executor:
        """Executor with PennyLane initialization."""
        return Executor("pennylane", seed=0, shots=1024)

    @pytest.fixture(scope="module")
    def ExecutorPennyLaneDevice(self) -> Executor:
        """Executor with PennyLane initialization."""
        return Executor(qml.device("lightning.qubit", wires=2), seed=0)

    @pytest.fixture(scope="module")
    def ExecutorParallelSampler(self) -> Executor:
        """Executor with Sampler initialization."""
        return Executor(Sampler(), seed=0, qpu_parallelization=3)

    @pytest.fixture(scope="module")
    def ExecutorParallelEstimator(self) -> Executor:
        """Executor with Estimator initialization."""
        return Executor(Estimator(), seed=0, qpu_parallelization=3)

    @pytest.fixture(scope="module")
    def simple_circuit(self):
        """Creates a simple circuit for testing."""
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        return qc

    @pytest.fixture(scope="module")
    def parameterized_circuit(self):
        x = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.ry(x[0], 0)
        qc.ry(x[1], 1)
        return qc

    @pytest.fixture(scope="module")
    def observable(self):
        """Creates a simple observable for testing."""
        return SparsePauliOp("ZZ")

    @pytest.mark.parametrize(
        "executor_str",
        [
            "ExecutorSampler",
            "ExecutorEstimator",
            "ExecutorStatevector",
            "ExecutorQasm",
            "ExecutorBackendSampler",
            "ExecutorBackendEstimator",
            "ExecutorParallelSampler",
            "ExecutorParallelEstimator",
            "ExecutorPennyLane",
            "ExecutorPennyLaneShots",
            "ExecutorPennyLaneDevice",
        ],
    )
    def test_shots(self, executor_str, request):
        """Tests of the default shots and the set_shots method work."""

        executor = request.getfixturevalue(executor_str)

        assert_dict = {
            "ExecutorSampler": None,
            "ExecutorEstimator": None,
            "ExecutorStatevector": None,
            "ExecutorQasm": 1024,
            "ExecutorBackendSampler": 1024,
            "ExecutorBackendEstimator": 1024,
            "ExecutorParallelSampler": None,
            "ExecutorParallelEstimator": None,
            "ExecutorPennyLane": None,
            "ExecutorPennyLaneShots": 1024,
            "ExecutorPennyLaneDevice": None,
        }

        assert executor.shots == assert_dict[executor_str]
        executor.set_shots(1234)
        assert executor.shots == 1234
        assert executor.get_shots() == 1234

    @pytest.mark.parametrize(
        "executor_str",
        [
            "ExecutorSampler",
            "ExecutorEstimator",
            "ExecutorStatevector",
            "ExecutorQasm",
            "ExecutorBackendSampler",
            "ExecutorBackendEstimator",
            "ExecutorParallelSampler",
            "ExecutorParallelEstimator",
        ],
    )
    def test_sampler(self, executor_str, request, simple_circuit):
        """Tests the Executor Sampler Primitive"""

        assert_dict = {
            "ExecutorSampler": {3: 1.0},
            "ExecutorEstimator": {3: 1.0},
            "ExecutorStatevector": {3: 1.0},
            "ExecutorQasm": {3: 1.0},
            "ExecutorBackendSampler": {3: 1.0},
            "ExecutorBackendEstimator": {3: 1.0},
            "ExecutorParallelSampler": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
            "ExecutorParallelEstimator": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
        }

        executor = request.getfixturevalue(executor_str)
        executor.set_shots(100)
        circuit = simple_circuit.measure_all(inplace=False)
        sampler = executor.get_sampler()
        if isinstance(sampler, BaseSamplerV1):
            res = sampler.run(circuit).result()
            assert res.metadata[0]["shots"] == 100
            assert res.quasi_dists[0] == assert_dict[executor_str]
        else:
            res = sampler.run([(circuit,)]).result()
            assert np.isclose(res[0].metadata["shots"], 100, 1)
            assert all(
                np.isclose(value / 100, assert_dict[executor_str][key], 1 / 100)
                for key, value in res[0].data.meas.get_int_counts().items()
            )

    @pytest.mark.parametrize(
        "executor_str",
        [
            "ExecutorSampler",
            "ExecutorEstimator",
            "ExecutorStatevector",
            "ExecutorQasm",
            "ExecutorBackendSampler",
            "ExecutorBackendEstimator",
            "ExecutorParallelSampler",
            "ExecutorParallelEstimator",
        ],
    )
    def test_executor(self, executor_str, request, simple_circuit, observable):
        """Tests the Executor Estimator Primitive"""

        assert_dict = {
            "ExecutorSampler": np.array([1.0]),
            "ExecutorEstimator": np.array([1.0]),
            "ExecutorStatevector": np.array([1.0]),
            "ExecutorQasm": np.array([1.0]),
            "ExecutorBackendSampler": np.array([1.0]),
            "ExecutorBackendEstimator": np.array([1.0]),
            "ExecutorParallelSampler": np.array([1.0]),
            "ExecutorParallelEstimator": np.array([1.0]),
        }

        executor = request.getfixturevalue(executor_str)
        executor.set_shots(100)
        estimator = executor.get_estimator()
        if isinstance(estimator, BaseEstimatorV1):
            res = estimator.run(simple_circuit, observable).result()
            assert res.metadata[0]["shots"] == 100
            assert res.values[0] == assert_dict[executor_str]
        else:
            res = estimator.run([(simple_circuit, observable)]).result()
            assert np.isclose(res[0].metadata["target_precision"], 0.1, 0.01)
            assert np.isclose(res[0].data.evs, assert_dict[executor_str], 0.1)

    @pytest.mark.parametrize(
        "executor_str",
        [
            "ExecutorPennyLane",
            "ExecutorPennyLaneShots",
            "ExecutorPennyLaneDevice",
        ],
    )
    def test_pennylane_probs(self, executor_str, request, simple_circuit):
        """Tests the PennyLane execution of a circuit with the probs return type."""

        assert_dict = {
            "ExecutorPennyLane": np.array([0.0, 0.0, 0.0, 1.0]),
            "ExecutorPennyLaneShots": np.array([0.0, 0.0, 0.0, 1.0]),
            "ExecutorPennyLaneDevice": np.array([0.0, 0.0, 0.0, 1.0]),
        }

        executor = request.getfixturevalue(executor_str)
        circuit = PennyLaneCircuit(simple_circuit, "probs")

        res = executor.pennylane_execute(circuit)
        assert np.allclose(assert_dict[executor_str], res)

    @pytest.mark.parametrize(
        "executor_str",
        [
            "ExecutorPennyLane",
            "ExecutorPennyLaneShots",
            "ExecutorPennyLaneDevice",
        ],
    )
    def test_pennylane_observable(self, executor_str, request, simple_circuit, observable):
        """Tests the PennyLane execution of a circuit with an observable return type."""

        assert_dict = {
            "ExecutorPennyLane": 1.0,
            "ExecutorPennyLaneShots": 1.0,
            "ExecutorPennyLaneDevice": 1.0,
        }

        executor = request.getfixturevalue(executor_str)
        circuit = PennyLaneCircuit(simple_circuit, observable)

        res = executor.pennylane_execute(circuit)
        assert np.allclose(assert_dict[executor_str], res)

    @pytest.mark.parametrize(
        "executor_str",
        [
            "ExecutorPennyLane",
            "ExecutorPennyLaneShots",
            "ExecutorPennyLaneDevice",
        ],
    )
    def test_pennylane_probs_batched(self, executor_str, request, parameterized_circuit):
        """Tests the batched PennyLane execution of a circuit with the probs return type."""

        assert_dict = {
            "ExecutorPennyLane": np.array([0.0, 0.0, 0.0, 1.0]),
            "ExecutorPennyLaneShots": np.array([0.0, 0.0, 0.0, 1.0]),
            "ExecutorPennyLaneDevice": np.array([0.0, 0.0, 0.0, 1.0]),
        }

        executor = request.getfixturevalue(executor_str)
        circuit = [PennyLaneCircuit(parameterized_circuit, "probs")] * 4

        res = executor.pennylane_execute_batched(
            circuit, [([np.pi, np.pi],), ([np.pi, np.pi],), ([np.pi, np.pi],), ([np.pi, np.pi],)]
        )
        assert np.allclose(assert_dict[executor_str], res)

    @pytest.mark.parametrize(
        "executor_str",
        [
            "ExecutorPennyLane",
            "ExecutorPennyLaneShots",
            "ExecutorPennyLaneDevice",
        ],
    )
    def test_pennylane_observable_batched(
        self, executor_str, request, parameterized_circuit, observable
    ):
        """Tests the batched PennyLane execution of a circuit with an observable return type."""

        assert_dict = {
            "ExecutorPennyLane": 1.0,
            "ExecutorPennyLaneShots": 1.0,
            "ExecutorPennyLaneDevice": 1.0,
        }

        executor = request.getfixturevalue(executor_str)
        circuit = [PennyLaneCircuit(parameterized_circuit, observable)] * 4

        res = executor.pennylane_execute_batched(
            circuit, [([np.pi, np.pi],), ([np.pi, np.pi],), ([np.pi, np.pi],), ([np.pi, np.pi],)]
        )
        assert np.allclose(assert_dict[executor_str], res)


class TestExecutorCleanup:
    """
    Tests for the automatic cleanup of the session when the Executor is garbage collected.
    """

    @pytest.fixture(scope="function")
    def ibm_backend(self):
        """
        Fixture providing a mock IBMBackend object.
        """
        backend = MagicMock(spec=IBMBackend, name="ibm_kingston")
        backend.name = "ibm_kingston"
        backend.configuration = MagicMock(backend_name="ibm_kingston")
        yield backend
        backend.reset_mock()
        del backend

    @pytest.fixture(scope="function")
    def mock_session(self, ibm_backend):
        """
        Fixture providing a mock Session object.
        """
        mock_session = MagicMock(spec=Session)
        mock_session.backend.return_value = ibm_backend
        mock_session.service.backend.return_value = ibm_backend
        yield mock_session
        mock_session.reset_mock()
        del mock_session

    def test_non_ibm_no_cleanup(self):
        """
        Verifies that no cleanup is performed when the backend is not an IBMBackend.
        """
        executor = Executor("pennylane")
        assert not hasattr(executor, "_finalizer")
        del executor
        gc.collect()

    @patch.object(Executor, "_cleanup_session")
    def test_finalizer_triggers_cleanup(self, mock_cleanup, ibm_backend, mock_session):
        """
        Verifies that the finalizer triggers the cleanup of the session.
        """
        with patch("squlearn.util.executor.Session", return_value=mock_session):
            executor = Executor(ibm_backend)
            with pytest.warns(SessionContextMisuseWarning):
                executor.create_session()
            del executor
            gc.collect()
            mock_cleanup.assert_called_once()

    @patch.object(Executor, "_cleanup_session")
    def test_preexisting_session_cleanup(self, mock_cleanup, mock_session):
        """
        Verifies that the cleanup of the session is performed even if it was created before the Executor object.
        """
        executor = Executor(mock_session)
        del executor
        gc.collect()
        mock_cleanup.assert_called_once()

    def test_cleanup_logic(self, mock_session, ibm_backend):
        """
        Verifies that _cleanup_session calls close_session.
        """
        with patch("squlearn.util.executor.Session", return_value=mock_session):
            executor = Executor(ibm_backend)
            with pytest.warns(SessionContextMisuseWarning):
                executor.create_session()
            # pylint: disable=protected-access
            session_ref = weakref.ref(executor._session)
            Executor._cleanup_session(session_ref)  # Direct call
            mock_session.close.assert_called_once()
            executor = None
            Executor._cleanup_session(session_ref)

    def test_normal_creation_deletion_closes_session(self, ibm_backend, mock_session):
        """
        Verifies that the session is closed after normal creation and deletion of the Executor object.
        """
        mock_session.reset_mock()
        with patch("squlearn.util.executor.Session", return_value=mock_session):
            mock_session.close.assert_not_called()
            executor = Executor(ibm_backend)
            with pytest.warns(SessionContextMisuseWarning):
                executor.create_session()
            del executor
            gc.collect()
            mock_session.close.assert_called_once()

    def test_preexisting_session_is_closed_on_deletion(self, mock_session):
        """
        Verifies that the session is closed after deletion of the Executor object when a pre-existing session is provided.
        """
        executor = Executor(mock_session)
        del executor
        gc.collect()
        mock_session.close.assert_called_once()

    def test_context_manager_closes_session(self, ibm_backend, mock_session, recwarn):
        """
        Verifies that the context manager closes the session.
        """
        with patch("squlearn.util.executor.Session", return_value=mock_session):
            with Executor(ibm_backend) as executor:
                executor.create_session()
            mock_session.close.assert_called_once()
            for warning in recwarn.list:
                assert not isinstance(warning, SessionContextMisuseWarning)

    def test_context_manager_exception_closes_session(self, ibm_backend, mock_session, recwarn):
        """
        Verifies that the context manager closes the session even when an exception is raised.
        """
        with patch("squlearn.util.executor.Session", return_value=mock_session):
            try:
                with Executor(ibm_backend) as executor:
                    executor.create_session()
                    # pylint: disable=pointless-statement
                    1 / 0
            except ZeroDivisionError:
                pass
            mock_session.close.assert_called_once()
            for warning in recwarn.list:
                assert not isinstance(warning, SessionContextMisuseWarning)

    def test_python_side_failure_closes_session(self, ibm_backend, mock_session):
        """
        Verifies that the session is closed even when a Python-side failure (e.g. OOM failure) occurs.
        """
        with patch("squlearn.util.executor.Session", return_value=mock_session):
            try:
                executor = Executor(ibm_backend)
                with pytest.warns(SessionContextMisuseWarning):
                    executor.create_session()
                raise MemoryError("OOM")
            except MemoryError:
                pass
            finally:
                del executor
            gc.collect()
            mock_session.close.assert_called_once()
