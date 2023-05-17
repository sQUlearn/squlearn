import numpy as np

from ..matrix import KernelMatrixBase


class KernelMlBase:
    def __init__(self, quantum_kernel: KernelMatrixBase) -> None:
        self._quantum_kernel = quantum_kernel

    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError()

    def predict(self, x: np.ndarray):
        raise NotImplementedError()
