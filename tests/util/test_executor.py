import pytest
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator, Sampler, BackendEstimator, BackendSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer

import pennylane as qml

from squlearn.util import Executor
from squlearn.util.executor import BaseEstimatorV1, BaseEstimatorV2, BaseSamplerV1, BaseSamplerV2
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
        circuit = PennyLaneCircuit(simple_circuit, "probs", executor)

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
        circuit = PennyLaneCircuit(simple_circuit, observable, executor)

        res = executor.pennylane_execute(circuit)
        assert np.allclose(assert_dict[executor_str], res)
