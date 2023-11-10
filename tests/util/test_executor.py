import pytest
import numpy as np

from qiskit import Aer,QuantumCircuit
from qiskit.primitives import Estimator, Sampler, BackendEstimator, BackendSampler
from qiskit.quantum_info import SparsePauliOp

from squlearn.util import Executor

class TestExecutor:

    @pytest.fixture(scope="module")
    def ExecutorSampler(self) -> Executor:
        """Executor with Sampler initilization."""
        return Executor(Sampler(),primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorEstimator(self) -> Executor:
        """Executor with Estimator initilization."""
        return Executor(Estimator(),primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorStatevector(self) -> Executor:
        """Executor with statevector_simulator initilization."""
        return Executor("statevector_simulator",primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorQasm(self) -> Executor:
        """Executor with qasm_simulator initilization."""
        return Executor("qasm_simulator",primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorBackendSampler(self) -> Executor:
        """Executor with Sampler initilization."""
        return Executor(BackendSampler(Aer.get_backend("qasm_simulator")),primitive_seed=0)

    @pytest.fixture(scope="module")
    def ExecutorBackendEstimator(self) -> Executor:
        """Executor with Estimator initilization."""
        return Executor(BackendEstimator(Aer.get_backend("qasm_simulator")),primitive_seed=0)

    @pytest.fixture(scope="module")
    def simple_circuit(self):
        """Executor with Estimator initilization."""
        qc = QuantumCircuit(2)
        qc.h([0,1])
        return qc

    @pytest.fixture(scope="module")
    def observable(self):
        return SparsePauliOp("ZZ")

    @pytest.mark.parametrize("executor_str", ["ExecutorSampler", "ExecutorEstimator",
                                            "ExecutorStatevector", "ExecutorQasm",
                                            "ExecutorBackendSampler", "ExecutorBackendEstimator"])
    def test_shots(self, executor_str, request):

        executor = request.getfixturevalue(executor_str)

        assert_dict = {"ExecutorSampler" : None,
        "ExecutorEstimator" : None,
        "ExecutorStatevector" : None,
        "ExecutorQasm" : 1024,
        "ExecutorBackendSampler" : 1024,
        "ExecutorBackendEstimator" : 1024}

        assert executor.shots == assert_dict[executor_str]
        executor.set_shots(1234)
        assert executor.shots == 1234
        assert executor.get_shots() == 1234

    @pytest.mark.parametrize("executor_str", ["ExecutorSampler", "ExecutorEstimator",
                                            "ExecutorStatevector", "ExecutorQasm",
                                            "ExecutorBackendSampler", "ExecutorBackendEstimator"])
    def test_sampler(self, executor_str, request, simple_circuit):

        assert_dict = {"ExecutorSampler" : {0: 0.26, 1: 0.22, 2: 0.2, 3: 0.32},
        "ExecutorEstimator" : {0: 0.26, 1: 0.22, 2: 0.2, 3: 0.32},
        "ExecutorStatevector" : {0: 0.26, 1: 0.22, 2: 0.2, 3: 0.32},
        "ExecutorQasm" : {3: 0.24, 2: 0.29, 1: 0.28, 0: 0.19},
        "ExecutorBackendSampler" : {3: 0.24, 2: 0.29, 1: 0.28, 0: 0.19},
        "ExecutorBackendEstimator" : {3: 0.24, 2: 0.29, 1: 0.28, 0: 0.19}}

        executor = request.getfixturevalue(executor_str)
        executor.set_shots(100)
        circuit = simple_circuit.measure_all(inplace=False)
        res = executor.get_sampler().run(circuit).result()
        assert res.metadata[0]["shots"] == 100
        assert res.quasi_dists[0] == assert_dict[executor_str]

    @pytest.mark.parametrize("executor_str", ["ExecutorSampler", "ExecutorEstimator",
                                            "ExecutorStatevector", "ExecutorQasm",
                                            "ExecutorBackendSampler", "ExecutorBackendEstimator"])
    def test_executor(self, executor_str, request, simple_circuit,observable):

        assert_dict = {"ExecutorSampler" : np.array([0.03455842]),
        "ExecutorEstimator" : np.array([0.03455842]),
        "ExecutorStatevector" : np.array([0.03455842]),
        "ExecutorQasm" : np.array([0.04]),
        "ExecutorBackendSampler" : np.array([0.04]),
        "ExecutorBackendEstimator" : np.array([0.04])}

        executor = request.getfixturevalue(executor_str)
        executor.set_shots(100)
        res = executor.get_estimator().run(simple_circuit,observable).result()
        assert res.metadata[0]["shots"] == 100
        assert np.allclose(assert_dict[executor_str],res.values[0])
