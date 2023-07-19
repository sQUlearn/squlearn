from ..matrix.kernel_matrix_base import KernelMatrixBase


class KernelLossBase:
    def __init__(self, quantum_kernel: KernelMatrixBase) -> None:
        self._quantum_kernel = quantum_kernel

    def compute(self):
        raise NotImplementedError
