import pytest
import numpy as np

from qiskit import QuantumCircuit
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
        executor = Executor(FakeManilaV2(), seed=0, shots=10000, qpu_parallelization=parallel_mode)
        sampler = executor.get_sampler()
        result = sampler.run(qc).result()
        assert result.metadata[0]["shots"] == 10000

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
        result = estimator.run(qc, obs).result()
        assert result.metadata[0]["shots"] == 10000
