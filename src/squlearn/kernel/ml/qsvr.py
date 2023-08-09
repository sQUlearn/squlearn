from ..matrix.kernel_matrix_base import KernelMatrixBase

from sklearn.svm import SVR


class QSVR(SVR):
    """
    Quantum Support Vector Regression

    This class is a wrapper of sklearn.svm.SVR. It uses a quantum kernel matrix
    to replace the kernel matrix in the sklearn.svm.SVR class.
    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    for additional information about the standard SVR parameters.
    The sklearn SVR has kernel specific arguments that are omitted here because they do not apply
    to the quantum kernels. These are

    - kernel
    - gamma
    - degree
    - coef0

    Args:
        quantum_kernel (KernelMatrixBase): The quantum kernel matrix to be used in the SVC. Either
            a fidelity quantum kernel (FQK) or projected quantum kernel (PQK) must be provided.
        **kwargs: Keyword arguments for the quantum kernel matrix, possible arguments can be
            obtained by calling ``get_params()``. Can be used to set for example the number of
            qubits (``num_qubits=``), or (if supported) the number of layers (``num_layers=``)
            of the underlying feature map.

    See Also
    --------
        squlearn.kernel.ml.QSVC : Quantum Support Vector Classification

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
            feature_map,
            executor=Executor("statevector_simulator"),
            initial_parameters=np.random.rand(feature_map.num_parameters))

        X = np.linspace(0, np.pi, 100)
        y = np.sin(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        qsvc = QSVR(quantum_kernel=kernel)
        qsvc.fit(X_train, y_train)
        print(f"The score on the test set is {qsvc.score(X_test, y_test)}")

    Methods:
    --------
    """

    def __init__(
        self,
        *,
        quantum_kernel: KernelMatrixBase,
        **kwargs,
    ) -> None:
        self.quantum_kernel = quantum_kernel

        # Apply kwargs to set_params of quantum kernel
        valid_params_quantum_kernel = self.quantum_kernel.get_params(deep=True)
        set_quantum_kernel_params_dict = {}
        for key, value in kwargs.items():
            if key in valid_params_quantum_kernel:
                set_quantum_kernel_params_dict[key] = value

        if len(set_quantum_kernel_params_dict) > 0:
            self.quantum_kernel.set_params(**set_quantum_kernel_params_dict)

        # remove quantum_kernel_kwargs for SVR initialization
        for key in set_quantum_kernel_params_dict:
            kwargs.pop(key, None)

        super().__init__(
            kernel=self.quantum_kernel.evaluate,
            **kwargs,
        )

    @classmethod
    def _get_param_names(cls):
        names = SVR._get_param_names()
        names.remove("kernel")
        names.remove("gamma")
        names.remove("degree")
        names.remove("coef0")
        return names

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the QSVR class.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = dict()

        # get parameters from the parent SVR class
        for key in self._get_param_names():
            params[key] = getattr(self, key)

        # add qsvr specific parameters
        params["quantum_kernel"] = self.quantum_kernel
        if deep:
            params.update(self.quantum_kernel.get_params(deep=deep))
        return params

    def set_params(self, **params) -> None:
        """
        Sets value of the QSVR hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. num_qubits=2.
        """
        valid_params = self.get_params(deep=True)
        valid_params_qsvr = self.get_params(deep=False)
        valid_params_quantum_kernel = self.quantum_kernel.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

            # Set parameters of the QSVR
            if key in valid_params_qsvr:
                try:
                    setattr(self, key, value)
                except:
                    setattr(self, "_" + key, value)

        # Set parameters of the Quantum Kernel and its underlying objects
        param_dict = {}
        for key, value in params.items():
            if key in valid_params_quantum_kernel:
                param_dict[key] = value
        if len(param_dict) > 0:
            self.quantum_kernel.set_params(**param_dict)
