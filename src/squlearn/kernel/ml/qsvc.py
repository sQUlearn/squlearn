from ..matrix.kernel_matrix_base import KernelMatrixBase

from sklearn.svm import SVC
from typing import Union, Optional


class QSVC(SVC):
    """
    Quantum Support Vector Classification

    This class is a wrapper of :class:`sklearn.svm.SVC`. It uses a quantum kernel matrix
    to replace the kernel matrix in the :class:`sklearn.svm.SVC` class.
    The parameters of the parent class can be adjusted via ``**kwargs``. See the documentation
    there for additional information about the standard SVC parameters.
    The scikit-learn SVC has kernel specific arguments that are omitted here because they do not
    apply to the quantum kernels. These are

        - `kernel`
        - `gamma`
        - `degree`
        - `coef0`

    Args:
        quantum_kernel (Union[KernelMatrixBase, str]): The quantum kernel matrix to be used in the SVC. Either
            a fidelity quantum kernel (FQK) or projected quantum kernel (PQK) must be provided. By
            setting quantum_kernel="precomputed", X is assumed to be a kernel matrix
            (train and test-train). This is particularly useful when storing quantum kernel
            matrices from real backends to numpy arrays.
        **kwargs: Possible arguments can be
            obtained by calling ``get_params()``. Notable examples are parameters of the
            :class:`sklearn.svm.SVC` class such as the regularization parameters ``C``
            (float, default=1.0). Additionally, properties of the underlying encoding circuit can be
            adjusted via kwargs such as the number of qubits (``num_qubits``), or (if supported)
            the number of layers (``num_layers``).

    See Also
    --------
        squlearn.kernel.ml.QSVR : Quantum Support Vector Regression

    **Example**

    .. code-block::

        import numpy as np

        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split

        from squlearn import Executor
        from squlearn.encoding_circuit import HubregtsenEncodingCircuit
        from squlearn.kernel.ml.qsvc import QSVC
        from squlearn.kernel.matrix import ProjectedQuantumKernel

        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=2, num_features=2, num_layers=2)
        kernel = ProjectedQuantumKernel(
            encoding_circuit,
            executor=Executor(),
            initial_parameters=np.random.rand(encoding_circuit.num_parameters)
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
        quantum_kernel: Optional[Union[KernelMatrixBase, str]] = None,
        **kwargs,
    ) -> None:
        self.quantum_kernel = quantum_kernel

        if isinstance(self.quantum_kernel, KernelMatrixBase):
            # Apply kwargs to set_params of quantum kernel
            quantum_kernel_update_params = self.quantum_kernel.get_params().keys() & kwargs.keys()
            if quantum_kernel_update_params:
                self.quantum_kernel.set_params(
                    **{key: kwargs[key] for key in quantum_kernel_update_params}
                )
                # remove quantum_kernel_kwargs for SVR initialization
                for key in quantum_kernel_update_params:
                    kwargs.pop(key, None)

            super().__init__(
                kernel=self.quantum_kernel.evaluate,
                **kwargs,
            )
        else:
            super().__init__(kernel="precomputed", **kwargs)

    @classmethod
    def _get_param_names(cls):
        names = SVC._get_param_names()
        names.remove("kernel")
        names.remove("gamma")
        names.remove("degree")
        names.remove("coef0")
        return names

    def fit(self, X, y):
        """
        Fit the QSVC model according to the given training data.

        Args:
            X (array-like, sparse matrix):  Training data of shape (n_samples, n_features)
                                            or (n_samples, n_samples)
                                            Training vectors, where `n_samples` is the number of samples
                                            and `n_features` is the number of features.
                                            For kernel="precomputed", the expected shape of X is
                                            (n_samples, n_samples).

            y (array-like): Lables with shape (n_samples,)

            sample_weight (array-like): Weights of shape (n_samples,), default=None
                                        Per-sample weights. Rescale C per sample. Higher weights
                                        force the classifier to put more emphasis on these points.

        Return:
            Returns an instance of self.
        """
        if self.quantum_kernel.is_trainable:
            self.quantum_kernel.run_optimization(X, y)
        return super().fit(X, y)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the QSVC class.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = dict()

        # get parameters from the parent SVC class
        for key in self._get_param_names():
            params[key] = getattr(self, key)

        # add qsvc specific parameters
        params["quantum_kernel"] = self.quantum_kernel
        if deep and isinstance(self.quantum_kernel, KernelMatrixBase):
            params.update(self.quantum_kernel.get_params(deep=deep))
        return params

    def set_params(self, **params) -> None:
        """
        Sets value of the QSVC hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``.
        """
        valid_params = self.get_params(deep=True).keys()
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        self_params = self.get_params(deep=False).keys() & params.keys()
        for key in self_params:
            try:
                setattr(self, key, params[key])
            except AttributeError:
                setattr(self, "_" + key, params[key])

        # Set parameters of the Quantum Kernel and its underlying objects
        if isinstance(self.quantum_kernel, KernelMatrixBase):
            quantum_kernel_params = self.quantum_kernel.get_params().keys() & params.keys()
            if quantum_kernel_params:
                self.quantum_kernel.set_params(
                    **{key: params[key] for key in quantum_kernel_params}
                )
        return self
