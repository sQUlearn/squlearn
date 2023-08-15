from ..matrix.kernel_matrix_base import KernelMatrixBase


class KernelLossBase:
    """
    Empty parent class for a kernel loss function.

    Args:
        quantum_kernel (KernelMatrixBase) : Specified quantum kernel object (either FQK or PQK)
    """

    def __init__(self, quantum_kernel: KernelMatrixBase) -> None:
        self._quantum_kernel = quantum_kernel

    def compute(self):
        """
        Empty function for holding the actual implementation for computing the respective
        kernel loss function.

        Args:
            self :
                Holds the function itself
        """
        raise NotImplementedError
