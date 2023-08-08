"""Quantum Kernel Ridge Regressor"""
from ..matrix.kernel_matrix_base import KernelMatrixBase

import scipy
import numpy as np
from typing import Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin

from ..matrix.regularization import thresholding_regularization, tikhonov_regularization


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

    See Also
    --------
        squlearn.kernel.ml.QGPR : Quantum Gaussian Process regression.
        squlearn.kernel.ml.QSVR : Quantum Support Vector regression.

    References
    -----------
        [1] Kevin P. Murphy "Machine Learning: A Probabilistic Perspective", The MIT Press
        chapter 14.4.3, pp. 493-493

        [2] https://en.wikipedia.org/wiki/Ridge_regression


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
        **kwargs
    ) -> None:
        self._quantum_kernel = quantum_kernel
        self.alpha = alpha
        self.x_train = None
        self.k_testtrain = None
        self.k_train = None
        self.dual_coeff_ = None

        # Apply kwargs to quantum kernel set_params
        valid_params = self.get_params().keys()
        set_params_dict = {}
        for key, value in kwargs.items():
            if key in valid_params:
                set_params_dict[key] = value
        if len(set_params_dict) > 0:
            self.set_params(**set_params_dict)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the Quantum Kernel Ridge regression model. Depending on whether ``regularization``
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
        params = dict()
        params["quantum_kernel"] = self._quantum_kernel
        params["alpha"] = self.alpha
        if deep:
            params.update(self._quantum_kernel.get_params(deep=True))
        else:
            params.update(self._quantum_kernel.get_params(deep=False))
        return params

    def set_params(self, **params):
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        param_dict = {}
        for key, value in params.items():
            # if key in valid_params:
            if key in self._quantum_kernel.get_params().keys():
                param_dict[key] = value
        self._quantum_kernel.set_params(**param_dict)

        if "alpha" in params.keys():
            self.alpha = params["alpha"]

        self.__init__(self._quantum_kernel, self.alpha)

        return self
