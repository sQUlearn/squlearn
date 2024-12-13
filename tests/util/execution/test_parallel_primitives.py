import pytest
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from squlearn.util import Executor
from squlearn.util.executor import BaseEstimatorV1, BaseEstimatorV2, BaseSamplerV1, BaseSamplerV2


class TestParallelExecutor:

    @pytest.mark.parametrize("parallel_mode", ["auto", 2])
    def test_parallel_sampler(self, parallel_mode):
        """
        Script for testing parallel execution of the sampler primitive.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        qc.measure_all()
        executor = Executor(FakeManilaV2(), seed=0, shots=10000, qpu_parallelization=parallel_mode)
        sampler = executor.get_sampler()
        if isinstance(sampler, BaseSamplerV1):
            result = sampler.run(qc).result()
            assert result.metadata[0]["shots"] == 10000
        else:
            result = sampler.run([(qc,)]).result()
            assert result[0].metadata["shots"] == 10000

    @pytest.mark.parametrize("parallel_mode", ["auto", 2])
    def test_parallel_estimator(self, parallel_mode):
        """
        Script for testing parallel execution of the estimator primitive.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        obs = SparsePauliOp("ZZ")
        executor = Executor(FakeManilaV2(), seed=0, shots=10000, qpu_parallelization=parallel_mode)
        estimator = executor.get_estimator()
        if isinstance(estimator, BaseEstimatorV1):
            result = estimator.run(qc, obs).result()
            assert result.metadata[0]["shots"] == 10000
        else:
            result = estimator.run([(qc, obs)]).result()
            assert result[0].metadata["shots"] == 10000
