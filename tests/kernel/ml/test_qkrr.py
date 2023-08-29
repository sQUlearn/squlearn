"""Tests for QKRR"""
import pytest
import numpy as np
from unittest.mock import MagicMock

from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from squlearn import Executor
from squlearn.feature_map import HZCRxCRyCRz
from squlearn.kernel import QKRR
from squlearn.kernel.matrix import ProjectedQuantumKernel, FidelityKernel

class TestQKRR:
    """Test class for QKRR"""

    @pytest.fixture(scope="module") 
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Test data module."""
        # pylint: disable=unbalanced-tuple-unpacking
        X, y = make_regression(n_samples=6, n_features=2, random_state=42)
        scl = MinMaxScaler((0.1, 0.9))
        X = scl.fit_transform(X, y)
        return X, y
    
    @pytest.fixture(scope="module")
    def qkrr_fidelity(self) -> QKRR:
        """QKRR module with FidelityKernel."""
        np.random.seed(42) # why?
        executor = Executor("statevector_simulator")
        feature_map = HZCRxCRyCRz(num_qubits=3, num_features=2, num_layers=2)
        kernel = FidelityKernel(
            feature_map=feature_map,
            executor=executor,
            regularization="thresholding",
            mit_depol_noise="msplit"
        )
        return QKRR(quantum_kernel=kernel, alpha=1.e-6)
    
    @pytest.fixture(scope="module")
    def qkrr_pqk(self) -> QKRR:
        """QKRR module with ProjectedQuantumKernel."""
        np.random.seed(42) #why?
        executor = Executor("statevector_simulator")
        feature_map = HZCRxCRyCRz(num_qubits=3, num_features=2, num_layers=2)
        kernel = ProjectedQuantumKernel(
            feature_map=feature_map,
            executor=executor,
            regularization="thresholding"
        )
        return QKRR(quantum_kernel=kernel, alpha=1.e-6)
    
    def test_that_qkrr_params_are_present(self):
        """Asserts that all classical parameters are present in the QKRR."""
        qkrr_instance = QKRR(quantum_kernel=MagicMock())
        assert list(qkrr_instance.get_params(deep=False).keys()) == [
            "quantum_kernel",
            "alpha"
        ]

    @pytest.mark.parametrize("qkrr", ["qkrr_fidelity", "qkrr_pqk"])
    def test_predict_unfitted(self, qkrr, request, data):
        """Tests concerning the unfitted QKRR.
        
        Tests include
            - whether a NotFittedError is raised
        """
        qkrr_instance = request.getfixturevalue(qkrr)
        X, _ = data
        with pytest.raises(NotFittedError):
            qkrr_instance.predict(X)