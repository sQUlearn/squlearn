import scipy
import numpy as np

from typing import Sequence
from .kernel_loss_base import KernelLossBase
from ..matrix import KernelMatrixBase


class NLL(KernelLossBase):
    def __init__(self, quantum_kernel: KernelMatrixBase, sigma=0.0):
        super().__init__(quantum_kernel)
        self._sigma = sigma

    # ProjectedQuantumKernel might cause errors since its not present in original KernelLoss
    def compute(
        self, parameter_values: Sequence[float], data: np.ndarray, labels: np.ndarray
    ) -> float:
        # Bind training parameters
        self._quantum_kernel.assign_parameters(parameter_values)

        # TODO: implement the equivalent for the pqk here @JSL?
        # get estimated kernel matrix
        kmatrix = self._quantum_kernel.evaluate(data)
        # ensure invertability -> TODO: check this step
        kmatrix = kmatrix + self._sigma * np.eye(kmatrix.shape[0])

        # Cholesky decomposition since numerically more stable
        L = scipy.linalg.cholesky(kmatrix, lower=True)
        S1 = scipy.linalg.solve_triangular(L, labels, lower=True)
        S2 = scipy.linalg.solve_triangular(L.T, S1, lower=False)
        neg_log_lh = (
            np.sum(np.log(np.diagonal(L)))
            + 0.5 * labels.T @ S2
            + 0.5 * len(data) * np.log(2.0 * np.pi)
        )
        neg_log_lh = neg_log_lh.reshape(-1)

        return neg_log_lh
