import numpy as np

from packaging import version
from qiskit.circuit import QuantumCircuit
from qiskit import __version__ as qiskit_version
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeBelemV2, FakeAthensV2
from squlearn.util import Executor
from squlearn.qnn import QNNRegressor
from squlearn.observables import SummedPaulis
from squlearn.encoding_circuit import ChebyshevTower
from squlearn.optimizers import Adam
from squlearn.qnn.loss import SquaredLoss
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel, QKRR
from squlearn.util.executor import BaseEstimatorV1, BaseEstimatorV2, BaseSamplerV1, BaseSamplerV2

import pytest

QISKIT_SMALLER_1_0 = version.parse(qiskit_version) < version.parse("1.0.0")
QISKIT_SMALLER_1_2 = version.parse(qiskit_version) < version.parse("1.2.0")


class TestBackendAutoSelection:
    """Test for auto selection of the backend for a given circuit."""

    if QISKIT_SMALLER_1_0:
        mark = pytest.mark.parametrize(
            "mode, use_hqaa, true_backend",
            [
                ("quality", True, "fake_athens"),
                ("speed", True, "fake_belem"),
                ("quality", False, "fake_manila"),
                ("speed", False, "fake_belem"),
            ],
        )
    else:
        mark = pytest.mark.parametrize(
            "mode, use_hqaa, true_backend",
            [
                ("quality", True, "fake_manila"),
                ("speed", True, "fake_belem"),
                ("quality", False, "fake_manila"),
                ("speed", False, "fake_belem"),
            ],
        )

    @mark
    def test_auto_select_circuit(self, mode, true_backend, use_hqaa):
        """
        Test for auto selection of the backend for a given circuit.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        obs = SparsePauliOp("ZZ")
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=1000)
        _, _ = executor.select_backend(qc, mode=mode, use_hqaa=use_hqaa)
        assert str(executor.backend_name) == true_backend

        sampler = executor.get_sampler()
        if isinstance(sampler, BaseSamplerV1):
            result = sampler.run(qc.measure_all(inplace=False)).result()
            assert result.metadata[0]["shots"] == 1000
        elif isinstance(sampler, BaseSamplerV2):
            result = sampler.run([(qc.measure_all(inplace=False),)]).result()
            assert result[0].metadata["shots"] == 1000
        else:
            raise ValueError("Unknown sampler type")

        estimator = executor.get_estimator()
        if isinstance(estimator, BaseEstimatorV1):
            result = estimator.run(qc, obs).result()
            assert result.metadata[0]["shots"] == 1000
        elif isinstance(estimator, BaseEstimatorV2):
            result = estimator.run([(qc, obs)]).result()
            assert result[0].metadata["shots"] == 1001
        else:
            raise ValueError("Unknown estimator type")

    if QISKIT_SMALLER_1_0:
        mark = pytest.mark.parametrize(
            "mode, use_hqaa, true_backend",
            [
                ("quality", True, "fake_manila"),
                ("speed", True, "fake_belem"),
                ("quality", False, "fake_belem"),
                ("speed", False, "fake_belem"),
            ],
        )
    else:
        mark = pytest.mark.parametrize(
            "mode, use_hqaa, true_backend",
            [
                ("quality", True, "fake_manila"),
                ("speed", True, "fake_belem"),
                ("quality", False, "fake_manila"),
                ("speed", False, "fake_belem"),
            ],
        )

    @mark
    def test_auto_select_circuit_parallel(self, mode, true_backend, use_hqaa):
        """
        Test for auto selection of the backend for a given circuit.
        """
        qc = QuantumCircuit(2)
        qc.x([0, 1])
        obs = SparsePauliOp("ZZ")
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=10000, qpu_parallelization=2)
        _, _ = executor.select_backend(qc, mode=mode, use_hqaa=use_hqaa)
        assert str(executor.backend_name) == true_backend
        sampler = executor.get_sampler()

        if isinstance(sampler, BaseSamplerV1):
            result = sampler.run(qc.measure_all(inplace=False)).result()
            assert result.metadata[0]["shots"] == 10000
        elif isinstance(sampler, BaseSamplerV2):
            result = sampler.run([(qc.measure_all(inplace=False),)]).result()
            assert result[0].metadata["shots"] == 10000
        else:
            raise ValueError("Unknown sampler type")

        estimator = executor.get_estimator()
        if isinstance(estimator, BaseEstimatorV1):
            result = estimator.run(qc, obs).result()
            assert result.metadata[0]["shots"] == 10000
        elif isinstance(estimator, BaseEstimatorV2):
            result = estimator.run([(qc, obs)]).result()
            assert result[0].metadata["shots"] == 10000
        else:
            raise ValueError("Unknown estimator type")

    if QISKIT_SMALLER_1_0:
        mark = pytest.mark.parametrize(
            "mode, true_backend",
            [
                ("quality", "fake_manila"),
                ("speed", "fake_belem"),
                ("quality_hqaa", "fake_athens"),
                ("speed_hqaa", "fake_belem"),
            ],
        )
    else:
        mark = pytest.mark.parametrize(
            "mode, true_backend",
            [
                ("quality", "fake_manila"),
                ("speed", "fake_belem"),
                ("quality_hqaa", "fake_manila"),
                ("speed_hqaa", "fake_belem"),
            ],
        )

    @mark
    def test_auto_select_qnn(self, mode, true_backend):
        """
        Test for auto selection of the backend for a given circuit.
        """
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=10000, auto_backend_mode=mode)
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
        assert str(executor.backend_name) == true_backend

    @mark
    def test_auto_select_fidelity_kernel(self, mode, true_backend):
        """
        Test for auto selection of the backend for a given circuit.
        """
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=10000, auto_backend_mode=mode)
        pqc = ChebyshevTower(2, 1, 2)

        if QISKIT_SMALLER_1_2:
            fqk = FidelityKernel(pqc, executor)
            qkrr = QKRR(fqk)
            qkrr.fit(np.array([[0.25], [0.75]]), np.array([0.25, 0.75]))
            qkrr.predict(np.array([[0.25], [0.75]]))
            assert str(executor.backend_name) == true_backend
        else:
            with pytest.raises(ValueError):
                fqk = FidelityKernel(pqc, executor)

    @mark
    def test_auto_select_projected_kernel(self, mode, true_backend):
        """
        Test for auto selection of the backend for a given circuit.
        """
        backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
        executor = Executor(backends, seed=0, shots=10000, auto_backend_mode=mode)
        pqc = ChebyshevTower(2, 1, 2)
        fqk = ProjectedQuantumKernel(pqc, executor)
        qkrr = QKRR(fqk)
        qkrr.fit(np.array([[0.25], [0.75]]), np.array([0.25, 0.75]))
        qkrr.predict(np.array([[0.25], [0.75]]))
        assert str(executor.backend_name) == true_backend
