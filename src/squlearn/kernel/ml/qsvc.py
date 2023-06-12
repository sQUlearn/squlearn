from ..matrix import KernelMatrixBase

from sklearn.svm import SVC


class QSVC(SVC):
    """
    Quantum Support Vector Classification

    This class is a wrapper of sklearn.svm.SVC. It uses a quantum kernel matrix
    to replace the kernel matrix in the sklearn.svm.SVC class.

    Args:
        quantum_kernel: The quantum kernel matrix to be used in the SVC.
        **kwargs: Other parameters that are passed to sklearn.svm.SVC.

    Attributes:
        quantum_kernel: The quantum kernel matrix to be used in the SVC.

    """

    def __init__(self, *, quantum_kernel: KernelMatrixBase, **kwargs) -> None:
        self.quantum_kernel = quantum_kernel
        super().__init__(kernel=self.quantum_kernel.evaluate, **kwargs)

    @classmethod
    def _get_param_names(cls):
        names = SVC._get_param_names()
        names.remove("kernel")
        names.remove("gamma")
        names.remove("degree")
        names.remove("coef0")
        return sorted(names + ["quantum_kernel"])
