import pytest
import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.primitives import Estimator, Sampler, BackendEstimator, BackendSampler
from qiskit.quantum_info import SparsePauliOp

from squlearn.util import Executor


class TestExecutor:
    @pytest.fixture(scope="module")
    def ExecutorSampler(self) -> Executor:
        """Executor with Sampler initialization."""
        return Executor(Sampler(), primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorEstimator(self) -> Executor:
        """Executor with Estimator initialization."""
        return Executor(Estimator(), primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorStatevector(self) -> Executor:
        """Executor with statevector_simulator initialization."""
        return Executor("statevector_simulator", primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorQasm(self) -> Executor:
        """Executor with qasm_simulator initialization."""
        return Executor("qasm_simulator", primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorBackendSampler(self) -> Executor:
        """Executor with BackendSampler initialization."""
        return Executor(BackendSampler(Aer.get_backend("qasm_simulator")), primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorBackendEstimator(self) -> Executor:
        """Executor with BackendEstimator initialization."""
        return Executor(BackendEstimator(Aer.get_backend("qasm_simulator")), primitive_seed=0)

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
        }

        executor = request.getfixturevalue(executor_str)
        executor.set_shots(100)
        circuit = simple_circuit.measure_all(inplace=False)
        res = executor.get_sampler().run(circuit).result()
        assert res.metadata[0]["shots"] == 100
        assert res.quasi_dists[0] == assert_dict[executor_str]

    @pytest.mark.parametrize(
        "executor_str",
        [
            "ExecutorSampler",
            "ExecutorEstimator",
            "ExecutorStatevector",
            "ExecutorQasm",
            "ExecutorBackendSampler",
            "ExecutorBackendEstimator",
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
        }

        executor = request.getfixturevalue(executor_str)
        executor.set_shots(100)
        res = executor.get_estimator().run(simple_circuit, observable).result()
        assert res.metadata[0]["shots"] == 100
        assert np.allclose(assert_dict[executor_str], res.values[0])
