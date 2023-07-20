""" Quantum Gaussian process classifier"""
import numpy as np
from ..matrix.kernel_matrix_base import KernelMatrixBase
from .kernel_util import kernel_wrapper
from sklearn.gaussian_process import GaussianProcessClassifier


class QGPC(GaussianProcessClassifier):
    """
    Quantum Gaussian process classification (QGPC), that extends the scikit-learn
    `sklearn.gaussian_process.GaussianProcessClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html>`.
    GaussianProcessClassifier class
    to use a quantum kernel.

    This class shows how to use a quantum kernel for QGPC. The class inherits its methods
    like ``fit`` and ``predict`` from scikit-learn, see the example below.
    Read more in the
    `scikit-learn user guide <https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process>`_.

    Args:
        quantum_kernel (KernelMatrixBase): The quantum kernel matrix to be used for the Gaussian process
                (either a fidelity quantum kernel (FQK) or projected quantum kernel (PQK) must be provided)

    See Also
    --------
        squlearn.kernel.ml.QSVC : Quantum Support Vector classification.

    **Example**

    .. code-block::

        from sklearn.datasets import load_iris
        from squlearn import Executor
        from squlearn.feature_map import QEKFeatureMap
        from squlearn.kernel.matrix import FidelityKernel
        from squlearn.kernel.ml import QGPC
        X, y = load_iris(return_X_y=True)

        fmap = QEKFeatureMap(num_qubits=X.shape[1], num_features=X.shape[1], num_layers=2)
        q_kernel = FidelityKernel(feature_map=fmap, executor=Executor("statevector_simulator"))
        q_kernel.assign_parameters(np.random.rand(fmap.num_parameters))
        qgpc_ansatz = QGPC(quantum_kernel=q_kernel)
        qgpc_ansatz.fit(X, y)
        qgpc_ansatz.score(X, y)
            0.98...
        qgpc_ansatz.predict_proba(X[:2,:])
            array([[0.85643716, 0.07037611, 0.07318673],
            [0.80314475, 0.09988938, 0.09696586]])

    Methods:
    --------
    """

    def __init__(self, quantum_kernel: KernelMatrixBase, **kwargs):
        self._quantum_kernel = quantum_kernel
        super().__init__(**kwargs)
        self.kernel = kernel_wrapper(quantum_kernel)

    @property
    def quantum_kernel(self) -> KernelMatrixBase:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: KernelMatrixBase):
        """Sets quantum kernel"""
        self._quantum_kernel = quantum_kernel
        self.kernel = kernel_wrapper(quantum_kernel)
