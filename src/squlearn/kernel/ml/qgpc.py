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
    Additional arguments can be set via ``**kwargs``.

    Args:
        quantum_kernel (Union[KernelMatrixBase, str]): The quantum kernel matrix to be used for the GP
                (either a fidelity quantum kernel (FQK)
                or projected quantum kernel (PQK) must be provided)
        **kwargs: Keyword arguments for the quantum kernel matrix, possible arguments can be obtained
            by calling ``get_params()``. Can be used to set for example the number of qubits
            (``num_qubits=``), or (if supported) the number of layers (``num_layers=``)
            of the underlying encoding circuit.

    See Also
    --------
        squlearn.kernel.ml.QSVC : Quantum Support Vector classification.

    **Example**

    .. code-block::

        from sklearn.datasets import load_iris
        from squlearn import Executor
        from squlearn.encoding_circuit import HubregtsenEncodingCircuit
        from squlearn.kernel.matrix import FidelityKernel
        from squlearn.kernel.ml import QGPC
        X, y = load_iris(return_X_y=True)

        enc_circ = HubregtsenEncodingCircuit(num_qubits=X.shape[1], num_features=X.shape[1], num_layers=2)
        q_kernel = FidelityKernel(encoding_circuit=enc_circ, executor=Executor())
        q_kernel.assign_parameters(np.random.rand(enc_circ.num_parameters))
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

        print("self.quantum_kernel", self.quantum_kernel)
        print("kwargs", kwargs)

        quantum_kernel_update_params = self.quantum_kernel.get_params().keys() & kwargs.keys()
        if quantum_kernel_update_params:
            self.quantum_kernel.set_params(
                **{key: kwargs[key] for key in quantum_kernel_update_params}
            )
            # remove quantum_kernel_kwargs for SVR initialization
            for key in quantum_kernel_update_params:
                kwargs.pop(key, None)

        super().__init__(**kwargs)
        self.kernel = kernel_wrapper(self._quantum_kernel)

    @classmethod
    def _get_param_names(cls):
        names = GaussianProcessClassifier._get_param_names()
        names.remove("kernel")
        names.remove("warm_start")
        return names

    def fit(self, X, y):
        """Fit Gaussian process classification model.

        Args:
            X : array-like of shape (n_samples, n_features) or list of object
                Feature vectors or other representations of training data.

            y : array-like of shape (n_samples,)
                Target values, must be binary.

        Return:
            Returns an instance of self.
        """
        if self._quantum_kernel.is_trainable:
            self._quantum_kernel.run_optimization(X, y)
        return super().fit(X, y)

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
        quantum_kernel_params = self._quantum_kernel.get_params().keys() & params.keys()
        if quantum_kernel_params:
            self._quantum_kernel.set_params(**{key: params[key] for key in quantum_kernel_params})
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
