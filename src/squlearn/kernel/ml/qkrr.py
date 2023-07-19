"""Quantum Kernel Ridge Regressor"""
from ..matrix.kernel_matrix_base import KernelMatrixBase

import scipy
import numpy as np
from typing import Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin

from .kernel_util import regularize_kernel, tikhonov_regularization

# from ..kernel_util import KernelRegularizer, DepolarizingNoiseMitigation


class QKRR(BaseEstimator, RegressorMixin):
    """
    Quantum Kernel Ridge Regression

    This class implements the Quantum Kernel Ridge Regression analogous to KRR [1] in sklearn
    but is not a wrapper.
    Read more about the theoretical background of KRR in, e.g., the
    `scikit-learn user guide <https://scikit-learn.org/stable/modules/kernel_ridge.html#kernel-ridge>`_.

    Args:
        quantum_kernel (KernelMatrixBase) :
            The quantum kernel matrix to be used in the KRR pipeline (either a fidelity
            quantum kernel (FQK) or projected quantum kernel (PQK) must be provided)
        alpha (Union[float, np.ndarray], default=1.0e-6) :
            Hyperparameter for the regularization strength; must be a positive float. This
            regularization improves the conditioning of the problem and assure the solvability
            of the resulting linear system. Larger values specify stronger regularization, cf.,
            e.g., Ref. [2]
        regularize  (Union[str, None], default=None) :
            Option for choosing different regularization techniques ('thresholding' or 'tikhonov')
            after Ref. [3] for the training kernel matrix, prior to  solving the linear system
            in the ``fit()``-procedure.

    See Also
    --------
        squlearn.kernel.ml.QGPR : Quantum Gaussian Process regression.
        squlearn.kernel.ml.QSVR : Quantum Support Vector regression.

    References
    -----------
        [1] Kevin P. Murphy "Machine Learning: A Probabilistic Perspective", The MIT Press
        chapter 14.4.3, pp. 493-493

        [2] https://en.wikipedia.org/wiki/Ridge_regression

        [3] T. Hubregtsen et al., "Training Quantum Embedding Kernels on Near-Term Quantum Computers",
        `arXiv:2105.02276v1 (2021) <https://arxiv.org/pdf/2105.02276.pdf>`_.

    **Example**

    .. code-block::

        from squlearn import Executor
        from squlearn.feature_map import ChebPQC
        from squlearn.kernel.matrix import ProjectedQuantumKernel
        from squlearn.kernel.ml import QKRR

        fmap = ChebPQC(num_qubits=4, num_features=1, num_layers=2)
        q_kernel_pqk = ProjectedQuantumKernel(
            feature_map=fmap,
            executor=Executor("statevector_simulator"),
            measurement="XYZ",
            outer_kernel="gaussian",
            initial_parameters=param,
            gamma=2.0
        )
        qkrr_pqk = QKRR(quantum_kernel=q_kernel_pqk, alpha=1e-5)
        qkrr_pqk.fit(x_train.reshape(-1, 1), y_train)
        y_pred_pqk = qkrr_pqk.predict(x.reshape(-1, 1))

    Methods:
    --------
    """

    def __init__(
        self,
        quantum_kernel: Optional[KernelMatrixBase] = None,
        alpha: Union[float, np.ndarray] = 1.0e-6,
        regularize: Union[str, None] = None,
    ) -> None:
        self._quantum_kernel = quantum_kernel  # May be worth to set FQK as default here?
        self.alpha = alpha
        self._regularize = regularize
        self.x_train = None
        self.k_testtrain = None
        self.k_train = None
        self.dual_coeff_ = None
        self.num_qubits = self._quantum_kernel.num_qubits

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the Quantum Kernel Ridge regression model. Depending on whether ``regularize``
        is set, the training kernel matrix is pre-processed accordingly prior to the
        actual fitting step is performed. The respective solution of the QKRR problem
        is obtained by solving the linear system using scipy's Cholesky decomposition for
        providing numercial stability

        Args:
            x_train (np.ndarray) : Training data of shape (n_samples, n_features)
            y_train (np.ndarray) : Target values or labels of shape (n_samples,)

        Returns:
            self :
                Returns the instance itself.
        """
        self.x_train = x_train
        self.k_train = self._quantum_kernel.evaluate(x=self.x_train)  # set up kernel matrix
        # check if regularize argument is set and define corresponding method
        if self._regularize is not None:
            if self._regularize == "thresholding":
                self.k_train = regularize_kernel(self.k_train)
            elif self._regularize == "tikhonov":
                self.k_train = tikhonov_regularization(self.k_train)

        self.k_train = self.k_train + self.alpha * np.eye(self.k_train.shape[0])

        # Cholesky decomposition for providing numerical stability
        try:
            L = scipy.linalg.cholesky(self.k_train, lower=True)
            self.dual_coeff_ = scipy.linalg.cho_solve((L, True), y_train)
        except np.linalg.LinAlgError:
            print("Increase regularization parameter alpha")

        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict using the Quantum Kernel Ridge model.

        Args:
            x_test (np.ndarray) : Samples of data of shape (n_samples, n_features) on which QKRR
                model makes predictions.

        Returns:
            np.ndarray :
                Returns predicted labels (at x_test) of shape (n_samples,)
        """
        if self.k_train is None:
            raise ValueError("The fit() method has to be called beforehand.")

        self.k_testtrain = self._quantum_kernel.evaluate(x_test, self.x_train)
        prediction = np.dot(self.k_testtrain, self.dual_coeff_)
        return prediction

    # All scikit-learn estimators have get_params and set_params
    # (cf. https://scikit-learn.org/stable/developers/develop.html)
    def get_params(self, deep: bool = True):
        return {
            "quantum_kernel": self._quantum_kernel,
            "alpha": self.alpha,
            "regularize": self._regularize,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


######
# BACKUP FOR DOCUMENTATION
# Attributes:
#         _regularize (Union[str, None]) :
#             Applied regularization technique
#         x_train (np.ndarray) :
#             Training data, which is also required for the prediction step of shape +
#             (n_samples, n_features)
#         k_train (np.ndarray) :
#             Training Kernel matrix, which is a square-symmetric matrix of shape (n_train, natrain)
#             required for the ``fit()`` step
#         k_testtrain (np.ndarray) :
#             Test-training matrix of shape (n_test, n_train) required for prediction step
#         dual_coeff_ (np.ndarray) :
#             Representation of weight vectors in kernel space
#         num_qubits (int) :
#             Number of qubits
