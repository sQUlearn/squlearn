import gc
import weakref
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from packaging import version
import pennylane as qml
import pytest
from qiskit import __version__ as qiskit_version
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer
from qiskit_ibm_runtime import IBMBackend, Session

QISKIT_SMALLER_1_1 = version.parse(qiskit_version) < version.parse("1.1.0")
QISKIT_SMALLER_2_0 = version.parse(qiskit_version) < version.parse("2.0.0")

if QISKIT_SMALLER_1_1:
    from qiskit.primitives import (
        BackendEstimator,
        BackendSampler,
        Estimator,
        Sampler,
        StatevectorEstimator,
        StatevectorSampler,
    )

    QISKIT_PRIMITIVES = {
        "BackendEstimator": BackendEstimator(Aer.get_backend("aer_simulator")),
        "BackendSampler": BackendSampler(Aer.get_backend("aer_simulator")),
        "Estimator": Estimator(),
        "Sampler": Sampler(),
        "StatevectorEstimator": StatevectorEstimator(),
        "StatevectorSampler": StatevectorSampler(),
    }
elif QISKIT_SMALLER_2_0:
    from qiskit.primitives import (
        BackendEstimator,
        BackendEstimatorV2,
        BackendSampler,
        BackendSamplerV2,
        Estimator,
        Sampler,
        StatevectorEstimator,
        StatevectorSampler,
    )

    QISKIT_PRIMITIVES = {
        "BackendEstimator": BackendEstimator(Aer.get_backend("aer_simulator")),
        "BackendEstimatorV2": BackendEstimatorV2(backend=Aer.get_backend("aer_simulator")),
        "BackendSampler": BackendSampler(Aer.get_backend("aer_simulator")),
        "BackendSamplerV2": BackendSamplerV2(backend=Aer.get_backend("aer_simulator")),
        "Estimator": Estimator(),
        "Sampler": Sampler(),
        "StatevectorEstimator": StatevectorEstimator(),
        "StatevectorSampler": StatevectorSampler(),
    }
else:
    from qiskit.primitives import (
        BackendEstimatorV2,
        BackendSamplerV2,
        StatevectorEstimator,
        StatevectorSampler,
    )

    QISKIT_PRIMITIVES = {
        "BackendEstimatorV2": BackendEstimatorV2(backend=Aer.get_backend("aer_simulator")),
        "BackendSamplerV2": BackendSamplerV2(backend=Aer.get_backend("aer_simulator")),
        "StatevectorEstimator": StatevectorEstimator(),
        "StatevectorSampler": StatevectorSampler(),
    }

from squlearn.util import Executor
from squlearn.util.executor import (
    BaseEstimatorV1,
    BaseEstimatorV2,
    BaseSamplerV1,
    BaseSamplerV2,
    SessionContextMisuseWarning,
)
from squlearn.util.pennylane import PennyLaneCircuit
from squlearn.util.qulacs import QulacsCircuit
from squlearn.util.qulacs.qulacs_execution import (
    qulacs_evaluate,
    qulacs_evaluate_statevector,
    qulacs_evaluate_probabilities,
)


@pytest.mark.parametrize("qpu_parallelization", [None, 3])
class TestExecutorQiskit:

    execution_dict = {
        "StatevectorSimulator": "statevector_simulator",
        "QasmSimulator": "qasm_simulator",
    }
    execution_dict.update(QISKIT_PRIMITIVES)

    def setup_executor(self, execution, qpu_parallelization=None) -> Executor:
        """Basic Executor initialization."""
        return Executor(execution, seed=0, qpu_parallelization=qpu_parallelization)

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
        "execution_key",
        execution_dict.keys(),
    )
    def test_shots(self, execution_key, qpu_parallelization):
        """Tests of the default shots and the set_shots method work."""

        executor = self.setup_executor(
            self.execution_dict[execution_key], qpu_parallelization=qpu_parallelization
        )

        assert_dict = {
            "StatevectorSimulator": None,
            "QasmSimulator": 1024,
            "BackendEstimator": 1024,
            "BackendEstimatorV2": 4096,
            "BackendSampler": 1024,
            "BackendSamplerV2": 1024,
            "Estimator": None,
            "Sampler": None,
            "StatevectorEstimator": None,
            "StatevectorSampler": 1024,
        }

        assert executor.shots == assert_dict[execution_key]
        original_shots = executor.get_shots()
        executor.set_shots(1234)
        assert executor.shots == 1234
        assert executor.get_shots() == 1234
        executor.set_shots(original_shots)  # Reset to original shots for other tests

    @pytest.mark.parametrize(
        "execution_key",
        execution_dict.keys(),
    )
    def test_sampler(self, execution_key, qpu_parallelization, simple_circuit):
        """Tests the Executor Sampler Primitive"""

        assert_dict = {
            "StatevectorSimulator": {3: 1.0},
            "QasmSimulator": {3: 1.0},
            "BackendEstimator": {3: 1.0},
            "BackendEstimatorV2": {3: 1.0},
            "BackendSampler": {3: 1.0},
            "BackendSamplerV2": {3: 1.0},
            "Estimator": {3: 1.0},
            "Sampler": {3: 1.0},
            "StatevectorEstimator": {3: 1.0},
            "StatevectorSampler": {3: 1.0},
        }

        executor = self.setup_executor(
            self.execution_dict[execution_key], qpu_parallelization=qpu_parallelization
        )
        executor.set_shots(100)
        circuit = simple_circuit.measure_all(inplace=False)
        sampler = executor.get_sampler()
        if isinstance(sampler, BaseSamplerV1):
            res = sampler.run(circuit).result()
            assert res.metadata[0]["shots"] == 100
            assert res.quasi_dists[0] == assert_dict[execution_key]
        else:
            res = sampler.run([(circuit,)]).result()
            # In Qiskit 1.1+, metadata structure changed and may not contain 'shots'
            if "shots" in res[0].metadata:
                assert np.isclose(res[0].metadata["shots"], 100, 1)
            assert all(
                np.isclose(value / 100, assert_dict[execution_key][key], 1 / 100)
                for key, value in res[0].data.meas.get_int_counts().items()
            )

    @pytest.mark.parametrize(
        "execution_key",
        execution_dict.keys(),
    )
    def test_executor(self, execution_key, qpu_parallelization, simple_circuit, observable):
        """Tests the Executor Estimator Primitive"""

        assert_dict = {
            "StatevectorSimulator": np.array([1.0]),
            "QasmSimulator": np.array([1.0]),
            "BackendEstimator": np.array([1.0]),
            "BackendEstimatorV2": np.array([1.0]),
            "BackendSampler": np.array([1.0]),
            "BackendSamplerV2": np.array([1.0]),
            "Estimator": np.array([1.0]),
            "Sampler": np.array([1.0]),
            "StatevectorEstimator": np.array([1.0]),
            "StatevectorSampler": np.array([1.0]),
        }

        executor = self.setup_executor(
            self.execution_dict[execution_key], qpu_parallelization=qpu_parallelization
        )
        executor.set_shots(100)
        estimator = executor.get_estimator()
        if isinstance(estimator, BaseEstimatorV1):
            res = estimator.run(simple_circuit, observable).result()
            assert res.metadata[0]["shots"] == 100
            assert res.values[0] == assert_dict[execution_key]
        else:
            res = estimator.run([(simple_circuit, observable)]).result()
            # Check target_precision only if it exists in metadata (newer Qiskit versions may not include it)
            if "target_precision" in res[0].metadata:
                assert np.isclose(res[0].metadata["target_precision"], 0.1, 0.01)
            assert np.isclose(res[0].data.evs, assert_dict[execution_key], 0.1)


class TestExecutorPennyLane:

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
            "ExecutorPennyLane",
            "ExecutorPennyLaneShots",
            "ExecutorPennyLaneDevice",
        ],
    )
    def test_shots(self, executor_str, request):
        """Tests of the default shots and the set_shots method work."""

        executor = request.getfixturevalue(executor_str)

        assert_dict = {
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


class TestExecutorQulacs:

    @pytest.fixture(scope="module")
    def ExecutorQulacs(self) -> Executor:
        """Executor with Qulacs initialization."""
        return Executor("qulacs", seed=0)

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

    def test_shots(self, ExecutorQulacs):
        """Tests of the default shots and the set_shots method work."""

        executor = ExecutorQulacs
        assert executor.shots is None

    @pytest.mark.parametrize(
        "qulacs_execution_func",
        [
            qulacs_evaluate,
            qulacs_evaluate_statevector,
            qulacs_evaluate_probabilities,
        ],
    )
    def test_qulacs_evaluate_simple(
        self, qulacs_execution_func, ExecutorQulacs, simple_circuit, observable
    ):
        """Tests the Qulacs execution of a circuit with an observable return type."""

        assert_dict = {
            "qulacs_evaluate": 1.0,
            "qulacs_evaluate_probabilities": np.array([0.0, 0.0, 0.0, 1.0]),
            "qulacs_evaluate_statevector": np.array([0.0, 0.0, 0.0, 1.0]),
        }

        executor = ExecutorQulacs
        circuit = QulacsCircuit(simple_circuit, observable)

        res = executor.qulacs_execute(qulacs_execution_func, circuit)
        assert np.allclose(assert_dict[qulacs_execution_func.__name__], res)

    @pytest.mark.parametrize(
        "qulacs_execution_func",
        [
            qulacs_evaluate,
            qulacs_evaluate_statevector,
            qulacs_evaluate_probabilities,
        ],
    )
    def test_qulacs_evaluate_parameterized(
        self, qulacs_execution_func, ExecutorQulacs, parameterized_circuit, observable
    ):
        """Tests the Qulacs execution of a circuit with an observable return type."""

        assert_dict = {
            "qulacs_evaluate": 1.0,
            "qulacs_evaluate_probabilities": np.array([0.0, 0.0, 0.0, 1.0]),
            "qulacs_evaluate_statevector": np.array([0.0, 0.0, 0.0, 1.0]),
        }

        executor = ExecutorQulacs
        circuit = QulacsCircuit(parameterized_circuit, observable)

        res = executor.qulacs_execute(qulacs_execution_func, circuit, x=[np.pi, np.pi])
        assert np.allclose(assert_dict[qulacs_execution_func.__name__], res)


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
