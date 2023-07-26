from typing import Union

from ..matrix.kernel_matrix_base import KernelMatrixBase

from sklearn.svm import SVC


class QSVC(SVC):
    """
    Quantum Support Vector Classification

    This class is a wrapper of sklearn.svm.SVC. It uses a quantum kernel matrix
    to replace the kernel matrix in the sklearn.svm.SVC class.
    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    for additional information about the standard SVC parameters.
    The sklearn SVC has kernel specific arguments that are omitted here because they do not apply
    to the quantum kernels. These are

    - kernel
    - gamma
    - degree
    - coef0

    Args:
        quantum_kernel (KernelMatrixBase): The quantum kernel matrix to be used in the SVC. Either
            a fidelity quantum kernel (FQK) or projected quantum kernel (PQK) must be provided.
        regularization (Union[str, None], default=None):
            Option for choosing different regularization techniques ('thresholding' or 'tikhonov')
            after Ref. [3] for the training kernel matrix, prior to  solving the linear system
            in the ``fit()``-procedure.
        **kwargs: Other parameters that are passed to sklearn.svm.SVC

    See Also
    --------
        squlearn.kernel.ml.QSVR : Quantum Support Vector Regression

    **Example**

    .. code-block::

        import numpy as np

        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split

        from squlearn import Executor
        from squlearn.feature_map import QEKFeatureMap
        from squlearn.kernel.ml.qsvc import QSVC
        from squlearn.kernel.matrix import ProjectedQuantumKernel

        feature_map = QEKFeatureMap(num_qubits=2, num_features=2, num_layers=2)
        kernel = ProjectedQuantumKernel(
            feature_map,
            executor=Executor("statevector_simulator"),
            initial_parameters=np.random.rand(feature_map.num_parameters)
        )

        X, y = make_moons(n_samples=100, noise=0.3, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        qsvc = QSVC(quantum_kernel=kernel)
        qsvc.fit(X_train, y_train)
        print(f"The score on the test set is {qsvc.score(X_test, y_test)}")

    Methods:
    --------
    """

    def __init__(
        self,
        *,
        quantum_kernel: KernelMatrixBase,
        regularization: Union[str, None] = None,
        **kwargs,
    ) -> None:
        self.quantum_kernel = quantum_kernel
        super().__init__(
            kernel=lambda x, y: self.quantum_kernel.evaluate(x, y, regularization=regularization),
            **kwargs,
        )

    @classmethod
    def _get_param_names(cls):
        names = SVC._get_param_names()
        names.remove("kernel")
        names.remove("gamma")
        names.remove("degree")
        names.remove("coef0")
        return sorted(names + ["quantum_kernel"])
