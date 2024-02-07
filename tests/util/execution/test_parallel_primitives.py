import pytest
import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.primitives import Estimator, Sampler, BackendEstimator, BackendSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from squlearn.util import Executor


class TestParallelExecutor:
    @pytest.mark.parametrize("parallel_mode", ["auto", 2])
    def test_parallel_sampler(self, parallel_mode):
        """
        Script for testing parallel execution of the sampler primitive.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        qc.measure_all()
        executor = Executor(FakeManilaV2(), shots=100000, qpu_parallelization=parallel_mode)
        sampler = executor.get_sampler()
        result = sampler.run(qc).result()
        assert round(result.quasi_dists[0][0], 1) == 0.0
        assert round(result.quasi_dists[0][1], 1) == 0.0
        assert round(result.quasi_dists[0][2], 1) == 0.1
        assert round(result.quasi_dists[0][3], 1) == 0.8
        assert result.metadata[0]["shots"] == 100000

    @pytest.mark.parametrize("parallel_mode", ["auto", 2])
    def test_parallel_estimator(self, parallel_mode):
        """
        Script for testing parallel execution of the estimator primitive.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        obs = SparsePauliOp("ZZ")
        executor = Executor(FakeManilaV2(), shots=100000, qpu_parallelization=parallel_mode)
        estimator = executor.get_estimator()
        result = estimator.run(qc, obs).result()
        assert round(result.values[0], 1) == 0.7
        assert result.metadata[0]["shots"] == 100000
