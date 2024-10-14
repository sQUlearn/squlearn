import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeBelemV2, FakeAthensV2
from squlearn.util import Executor
from squlearn.qnn import QNNRegressor
from squlearn.observables import SummedPaulis
from squlearn.encoding_circuit import ChebyshevTower
from squlearn.optimizers import Adam
from squlearn.qnn.loss import SquaredLoss
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel, QKRR

import pytest


class TestBackendAutoSelection:

    @pytest.mark.parametrize("use_hqaa", [True, False])
    def test_auto_select_circuit(self, use_hqaa):
        """
        Test for auto selection of the backend for a given circuit.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        obs = SparsePauliOp("ZZ")
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=1000)
        qc2, info = executor.select_backend(qc, use_hqaa=use_hqaa)
        assert str(executor.backend_name) == "fake_manila"
        sampler = executor.get_sampler()
        result = sampler.run(qc.measure_all(inplace=False)).result()
        assert result.metadata[0]["shots"] == 1000
        estimator = executor.get_estimator()
        result = estimator.run(qc, obs).result()
        assert result.metadata[0]["shots"] == 1000

    @pytest.mark.parametrize("use_hqaa", [True, False])
    def test_auto_select_circuit_parallel(self, use_hqaa):
        """
        Test for auto selection of the backend for a given circuit.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        obs = SparsePauliOp("ZZ")
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=10000, qpu_parallelization=2)
        qc2, info = executor.select_backend(qc, use_hqaa=use_hqaa)
        assert str(executor.backend_name) == "fake_manila"
        sampler = executor.get_sampler()
        result = sampler.run(qc.measure_all(inplace=False)).result()
        assert result.metadata[0]["shots"] == 10000
        estimator = executor.get_estimator()
        result = estimator.run(qc, obs).result()
        assert result.metadata[0]["shots"] == 10000

    def test_auto_select_qnn(self):
        """
        Test for auto selection of the backend for a given circuit.
        """
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=10000)
        pqc = ChebyshevTower(2, 1, 2)
        obs = SummedPaulis(2)
        param = []
        param_op = np.ones(obs.num_parameters)
        qnn = QNNRegressor(
            pqc,
            obs,
            executor,
            SquaredLoss(),
            Adam(),
            pretrained=True,
            param_ini=param,
            param_op_ini=param_op,
        )
        qnn.predict(np.array([[0.25], [0.75]]))
        assert str(executor.backend_name) == "fake_manila"

    def test_auto_select_fidelity_kernel(self):
        """
        Test for auto selection of the backend for a given circuit.
        """
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=10000)
        pqc = ChebyshevTower(2, 1, 2)
        fqk = FidelityKernel(pqc, executor)
        qkrr = QKRR(fqk)
        qkrr.fit(np.array([[0.25], [0.75]]), np.array([0.25, 0.75]))
        qkrr.predict(np.array([[0.25], [0.75]]))

        assert str(executor.backend_name) == "fake_manila"

    def test_auto_select_projected_kernel(self):
        """
        Test for auto selection of the backend for a given circuit.
        """
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=10000)
        pqc = ChebyshevTower(2, 1, 2)
        fqk = ProjectedQuantumKernel(pqc, executor)
        qkrr = QKRR(fqk)
        qkrr.fit(np.array([[0.25], [0.75]]), np.array([0.25, 0.75]))
        qkrr.predict(np.array([[0.25], [0.75]]))
        assert str(executor.backend_name) == "fake_manila"
