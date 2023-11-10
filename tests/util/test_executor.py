import pytest
import numpy as np

from qiskit import Aer
from qiskit.primitives import Estimator, Sampler, BackendEstimator, BackendSampler

from squlearn.util import Executor

class TestExecutor:

    @pytest.fixture(scope="module")
    def ExecutorSampler(self) -> Executor:
        """Executor with Sampler initilization."""
        return Executor(Sampler())

    @pytest.fixture(scope="module")
    def ExecutorEstimator(self) -> Executor:
        """Executor with Estimator initilization."""
        return Executor(Estimator())

    @pytest.fixture(scope="module")
    def ExecutorStatevector(self) -> Executor:
        """Executor with statevector_simulator initilization."""
        return Executor("statevector_simulator")

    @pytest.fixture(scope="module")
    def ExecutorQasm(self) -> Executor:
        """Executor with qasm_simulator initilization."""
        return Executor("qasm_simulator")

    @pytest.fixture(scope="module")
    def ExecutorBackendSampler(self) -> Executor:
        """Executor with Sampler initilization."""
        return Executor(BackendSampler(Aer.get_backend("qasm_simulator")))

    @pytest.fixture(scope="module")
    def ExecutorBackendEstimator(self) -> Executor:
        """Executor with Estimator initilization."""
        return Executor(BackendEstimator(Aer.get_backend("qasm_simulator")))

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