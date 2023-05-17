import numpy as np

from typing import Sequence
from sklearn.preprocessing._data import _handle_zeros_in_scale
from .kernel_loss_base import KernelLossBase
from ..matrix import KernelMatrixBase


class TargetAlignment(KernelLossBase):
    def __init__(self, quantum_kernel: KernelMatrixBase, sigma=0.0):
        super().__init__(quantum_kernel)
        self._sigma = sigma

    def compute(
        self, parameter_values: Sequence[float], data: np.ndarray, labels: np.ndarray
    ) -> float:
        # Bind training parameters
        self._quantum_kernel.assign_parameters(parameter_values)
        # Training parameters are handeled differently when using QuantumInstance and PQK.
        # This is checked here. @ JSL?

        # Get estimated kernel matrix
        kmatrix = self._quantum_kernel.evaluate(data)
        # regularize ->  TODO: Check if necessary
        kmatrix = kmatrix + self._sigma * np.eye(kmatrix.shape[0])

        labels_mean = np.mean(labels)
        labels_std = _handle_zeros_in_scale(np.std(labels, axis=0))
        labels = (labels - labels_mean) / labels_std
        kmatrix_opt = labels @ labels.T
        numerator = np.multiply(kmatrix_opt, kmatrix)
        denominator = np.multiply(kmatrix, kmatrix)
        alignment = np.sum(numerator) / (kmatrix.shape[0] * np.sqrt(np.sum(denominator)))
        return -alignment
