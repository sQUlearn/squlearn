"""Quantum Kernel Ridge Regressor"""

from ..matrix.kernel_matrix_base import KernelMatrixBase

import scipy
import numpy as np
from typing import Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin

from ..matrix.regularization import thresholding_regularization, tikhonov_regularization


class QKRR(BaseEstimator, RegressorMixin):
    r"""
    Quantum Kernel Ridge Regression.

    This class implements the Quantum Kernel Ridge Regression analogous to KRR [1] in scikit-learn
    but is not a wrapper.
    Read more about the theoretical background of KRR in, e.g., the
    `scikit-learn user guide <https://scikit-learn.org/stable/modules/kernel_ridge.html#kernel-ridge>`_.
    Additional arguments can be set via ``**kwargs``.

    Args:
        quantum_kernel (Optional[Union[KernelMatrixBase, str]]) :
            The quantum kernel matrix to be used in the KRR pipeline (either a fidelity
            quantum kernel (FQK) or projected quantum kernel (PQK) must be provided). By
            setting quantum_kernel="precomputed", X is assumed to be a kernel matrix
            (train and test-train). This is particularly useful when storing quantum kernel
            matrices from real backends to numpy arrays.
        alpha (Union[float, np.ndarray], default=1.0e-6) :
            Hyperparameter for the regularization strength; must be a positive float. This
            regularization improves the conditioning of the problem and assure the solvability
            of the resulting linear system. Larger values specify stronger regularization, cf.,
            e.g., Ref. [2]
        **kwargs: Keyword arguments for the quantum kernel matrix, possible arguments can be obtained
            by calling ``get_params()``. Can be used to set for example the number of qubits
            (``num_qubits=``), or (if supported) the number of layers (``num_layers=``)
            of the underlying encoding circuit.

    Attributes:
    -----------
        dual_coeff\_ : (np.ndarray) :
            Array containing the weight vector in kernel space
        k_train (np.ndarray) :
            Training kernel matrix of shape (n_train, n_train) which is available after calling the fit procedure
        k_testtrain (np.ndarray) :
            Kernel matrix of shape (n_test, n_train) which is evaluated at the predict step

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
        from squlearn.encoding_circuit import ChebyshevPQC
        from squlearn.kernel.matrix import ProjectedQuantumKernel
        from squlearn.kernel.ml import QKRR

        enc_circ = ChebyshevPQC(num_qubits=4, num_features=1, num_layers=2)
        q_kernel_pqk = ProjectedQuantumKernel(
            encoding_circuit=enc_circ,
            executor=Executor(),
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
        quantum_kernel: Optional[Union[KernelMatrixBase, str]] = None,
        alpha: Union[float, np.ndarray] = 1.0e-6,
        **kwargs,
    ) -> None:
        self._quantum_kernel = quantum_kernel
        self.alpha = alpha
        self.X_train = None
        self.k_testtrain = None
        self.k_train = None
        self.dual_coeff_ = None

        # Apply kwargs to set_params
        update_params = self.get_params().keys() & kwargs.keys()
        if update_params:
            self.set_params(**{key: kwargs[key] for key in update_params})

    def fit(self, X, y):
        """
        Fit the Quantum Kernel Ridge regression model.

        Depending on whether ``regularization`` is set, the training kernel matrix is pre-processed
        accordingly prior to the actual fitting step is performed. The respective solution of the
        QKRR problem is obtained by solving the linear system using scipy's Cholesky decomposition
        for providing numerical stability.

        Args:
            X: array-like or sparse matrix of shape (n_samples, n_features)
                If quantum_kernel == "precomputed" this is instead a precomputed training kernel
                matrix of shape (n_samples, n_samples).
            y: array-like of shape (n_samples,)
                Target values or labels

        Return:
            Returns an instance of self.
        """

        X, y = self._validate_data(
            X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )

        self.X_train = X

        if isinstance(self._quantum_kernel, str):
            if self._quantum_kernel == "precomputed":
                self.k_train = X
            else:
                raise ValueError("Unknown quantum kernel: {}".format(self._quantum_kernel))
        elif isinstance(self._quantum_kernel, KernelMatrixBase):
            # check if quantum kernel is trainable
            if self._quantum_kernel.is_trainable:
                self._quantum_kernel.run_optimization(self.X_train, y)

            self.k_train = self._quantum_kernel.evaluate(x=self.X_train)  # set up kernel matrix
        else:
            raise ValueError(
                "Unknown type of quantum kernel: {}".format(type(self._quantum_kernel))
            )

        self.k_train = self.k_train + self.alpha * np.eye(self.k_train.shape[0])

        # Cholesky decomposition for providing numerical stability
        try:
            L = scipy.linalg.cholesky(self.k_train, lower=True)
            self.dual_coeff_ = scipy.linalg.cho_solve((L, True), y)
        except np.linalg.LinAlgError:
            print("Increase regularization parameter alpha")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Quantum Kernel Ridge model.

        Args:
            X (np.ndarray) : Samples of data of shape (n_samples, n_features) on which QKRR
                model makes predictions. If quantum_kernel == "precomputed" this is instead a
                precomputed (test-train) kernel matrix of shape (n_samples, n_samples_fitted),
                where n_samples_fitted is the number of samples used in the fitting.

        Returns:
            np.ndarray :
                Returns predicted labels (at X) of shape (n_samples,)
        """
        if self.k_train is None:
            raise ValueError("The fit() method has to be called beforehand.")

        X = self._validate_data(X, accept_sparse=("csr", "csc"), reset=False)

        if isinstance(self._quantum_kernel, str):
            if self._quantum_kernel == "precomputed":
                self.k_testtrain = X
        elif isinstance(self._quantum_kernel, KernelMatrixBase):
            self.k_testtrain = self._quantum_kernel.evaluate(X, self.X_train)
        else:
            raise ValueError(
                "Unknown type of quantum kernel: {}".format(type(self._quantum_kernel))
            )

        prediction = np.dot(self.k_testtrain, self.dual_coeff_)
        return prediction

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyperparameters and their values of the QKRR method.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyperparameters and values.
        """
        params = {
            "quantum_kernel": self._quantum_kernel,
            "alpha": self.alpha,
        }

        if deep and isinstance(self._quantum_kernel, KernelMatrixBase):
            params.update(self._quantum_kernel.get_params(deep=deep))
        return params

    def set_params(self, **params) -> None:
        """
        Sets value of the encoding circuit hyperparameters.

        Args:
            params: Hyperparameters and their values, e.g. ``num_qubits=2``.
        """
        valid_params = self.get_params()
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        # Set parameters of the QKRR
        self_params = self.get_params(deep=False).keys() & params.keys()
        for key in self_params:
            try:
                setattr(self, key, params[key])
            except AttributeError:
                setattr(self, "_" + key, params[key])

        if isinstance(self._quantum_kernel, KernelMatrixBase):
            # Set parameters of the Quantum Kernel and its underlying objects
            quantum_kernel_params = self._quantum_kernel.get_params().keys() & params.keys()
            if quantum_kernel_params:
                self._quantum_kernel.set_params(
                    **{key: params[key] for key in quantum_kernel_params}
                )
        return self
