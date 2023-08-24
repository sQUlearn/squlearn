""" Quantum Gaussian process classifier"""
from ..matrix.kernel_matrix_base import KernelMatrixBase
from squlearn.kernel.matrix.kernel_util import kernel_wrapper
from sklearn.gaussian_process import GaussianProcessClassifier


class QGPC(GaussianProcessClassifier):
    """
    Quantum Gaussian process classification (QGPC), that extends the scikit-learn
    `sklearn.gaussian_process.GaussianProcessClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html>`_.
    GaussianProcessClassifier class
    to use a quantum kernel.

    This class shows how to use a quantum kernel for QGPC. The class inherits its methods
    like ``fit`` and ``predict`` from scikit-learn, see the example below.
    Read more in the
    `scikit-learn user guide
    <https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process>`_.

    Args:
        quantum_kernel (KernelMatrixBase): The quantum kernel matrix to be used for the GP
                (either a fidelity quantum kernel (FQK)
                or projected quantum kernel (PQK) must be provided)

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

    def __init__(self, quantum_kernel: KernelMatrixBase, **kwargs) -> None:
        self._quantum_kernel = quantum_kernel
        # Apply kwargs to set_params of quantum kernel
        valid_params_quantum_kernel = self._quantum_kernel.get_params(deep=True)
        set_quantum_kernel_params_dict = {}
        for key, value in kwargs.items():
            if key in valid_params_quantum_kernel:
                set_quantum_kernel_params_dict[key] = value

        if len(set_quantum_kernel_params_dict) > 0:
            self._quantum_kernel.set_params(**set_quantum_kernel_params_dict)

        # remove quantum_kernel_kwargs for QGPC initialization
        for key in set_quantum_kernel_params_dict:
            kwargs.pop(key, None)

        super().__init__(**kwargs)
        self.kernel = kernel_wrapper(self._quantum_kernel)

    @classmethod
    def _get_param_names(cls):
        names = GaussianProcessClassifier._get_param_names()
        names.remove("kernel")
        names.remove("warm_start")
        return names

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the QGPC class.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = dict()

        # get parameters from the parent GPC class
        for key in self._get_param_names():
            params[key] = getattr(self, key)

        # add qgpc specific parameters
        params["quantum_kernel"] = self._quantum_kernel
        if deep:
            params.update(self._quantum_kernel.get_params(deep=deep))
        return params

    def set_params(self, **params) -> None:
        """
        Sets value of the QGPC hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. num_qubits=2.
        """
        valid_params = self.get_params(deep=True)
        valid_params_qgpc = self.get_params(deep=False)
        valid_params_quantum_kernel = self._quantum_kernel.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

            # Set parameters of the QGPC
            if key in valid_params_qgpc:
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
            self._quantum_kernel.set_params(**param_dict)
        return self

    @property
    def quantum_kernel(self) -> KernelMatrixBase:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: KernelMatrixBase):
        """Sets quantum kernel"""
        self._quantum_kernel = quantum_kernel
        self.kernel = kernel_wrapper(quantum_kernel)
