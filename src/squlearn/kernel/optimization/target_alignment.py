import numpy as np

from typing import Sequence
from .kernel_loss_base import KernelLossBase
from ..matrix.kernel_matrix_base import KernelMatrixBase


class TargetAlignment(KernelLossBase):
    def __init__(self, quantum_kernel: KernelMatrixBase, sigma=0.0):
        super().__init__(quantum_kernel)
        self._sigma = sigma

    def compute(
        self,
        parameter_values: Sequence[float],
        data: np.ndarray,
        labels: np.ndarray,
        rescale_class_labels=True,
    ) -> float:
        # Bind training parameters
        self._quantum_kernel.assign_parameters(parameter_values)

        # Get estimated kernel matrix
        kmatrix = self._quantum_kernel.evaluate(data)
        if rescale_class_labels:
            nplus = np.count_nonzero(np.array(labels) == 1)
            nminus = len(labels) - nplus
            _Y = np.array([y / nplus if y == 1 else y / nminus for y in labels])
        else:
            _Y = np.array(labels)

        T = np.outer(_Y, _Y)
        inner_product = np.sum(kmatrix * T)
        norm = np.sqrt(np.sum(kmatrix * kmatrix) * np.sum(T * T))
        alignment = inner_product / norm
        return -alignment
