from ..matrix.kernel_matrix_base import KernelMatrixBase

from sklearn.svm import SVR


class QSVR(SVR):
    """
    Quantum Support Vector Regression

    This class is a wrapper of sklearn.svm.SVR. It uses a quantum kernel matrix
    to replace the kernel matrix in the sklearn.svm.SVR class.
    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    for additional information about the standard SVR parameters.
    The sklearn SVR has kernel specific arguments that are omitted here because they do not apply to the quantum
    kernels. These are

    - kernel
    - gamma
    - degree
    - coef0

    Args:
        quantum_kernel: The quantum kernel matrix to be used in the SVC. Either a fidelity
            quantum kernel (FQK) or projected quantum kernel (PQK) must be provided.
        **kwargs: Other parameters that are passed to sklearn.svm.SVR

    See Also
    --------
        squlearn.kernel.ml.QSVC

    **Example**

    .. code-block::

        import numpy as np

        from sklearn.model_selection import train_test_split

        from squlearn import Executor
        from squlearn.feature_map import QEKFeatureMap
        from squlearn.kernel.ml.qsvr import QSVR
        from squlearn.kernel.matrix import ProjectedQuantumKernel

        feature_map = QEKFeatureMap(num_qubits=2, num_features=1, num_layers=2)
        kernel = ProjectedQuantumKernel(
            feature_map, executor=Executor("statevector_simulator"), initial_parameters=np.random.rand(feature_map.num_parameters))

        X = np.linspace(0, np.pi, 100)
        y = np.sin(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        qsvc = QSVR(quantum_kernel=kernel)
        qsvc.fit(X_train, y_train)
        print(f"The score on the test set is {qsvc.score(X_test, y_test)}")

    Methods:
    --------
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
