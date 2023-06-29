from ..matrix.kernel_matrix_base import KernelMatrixBase

from sklearn.svm import SVR


class QSVR(SVR):
    """
    Quantum Support Vector Regression

    This class is a wrapper of sklearn.svm.SVR. It uses a quantum kernel matrix
    to replace the kernel matrix in the sklearn.svm.SVR class.

    Args:
        quantum_kernel: The quantum kernel matrix to be used in the SVR.
        **kwargs: Other parameters that are passed to sklearn.svm.SVR.

    Attributes:
        quantum_kernel: The quantum kernel matrix to be used in the SVR.
    """

    def __init__(self, *, quantum_kernel: KernelMatrixBase, **kwargs) -> None:
        self.quantum_kernel = quantum_kernel
        super().__init__(kernel=self.quantum_kernel.evaluate, **kwargs)

    @classmethod
    def _get_param_names(cls):
        names = SVR._get_param_names()
        names.remove("kernel")
        names.remove("gamma")
        names.remove("degree")
        names.remove("coef0")
        return sorted(names + ["quantum_kernel"])
