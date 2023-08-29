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
        """QKRR module with FidelityKernel"""
        
        return 0
    
    @pytest.fixture(scope="module")
    def qkrr_pqk(self) -> QKRR:
        return 0