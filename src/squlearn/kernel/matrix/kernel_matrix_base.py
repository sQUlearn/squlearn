from typing import Union
import numpy as np

from ...feature_map.feature_map_base import FeatureMapBase
from ...util.executor import Executor


class KernelMatrixBase:
    """
    Base class for defining quantum kernels.

    Args:
        feature_map (FeatureMapBase) :
            PQC feature map
        executor (Executor) :
            Executor object
        initial_parameters (Union[np.ndarray, None], default=None) :
            Initial parameters of the PQC feature map
    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
        executor: Executor,
        initial_parameters: Union[np.ndarray, None] = None,
    ) -> None:
        self._feature_map = feature_map
        self._num_qubits = self._feature_map.num_qubits
        self._executor = executor
        self._parameters = initial_parameters

    @property
    def feature_map(self) -> FeatureMapBase:
        """
        Returns the feature map from which the kernel matrix is constructed
        """
        return self._feature_map

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used in the definition of the feature map."""
        return self._num_qubits

    @property
    def num_features(self) -> int:
        """Returns the feature dimension of the feature map"""
        return self._feature_map.num_features

    @property
    def parameters(self) -> np.ndarray:
        """
        Returns the numeric values of the trainable parameters assigned to the
        feature map as np.ndarray
        """
        return self._parameters

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters of the feature map."""
        return self._feature_map.num_parameters

    def evaluate(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Computes and the quantum kernel matrix.

        Args:
            x (np.ndarray) :
                Vecotr of training or test data for which the kernel matrix is evaluated
            y (np.ndarray, default=None) :
                Vecotr of training or test data for which the kernel matrix is evaluated

        Returns:
            Returns the quantum kernel matrix as 2D numpy array.
        """
        raise NotImplementedError()

    def evaluate_pairwise(self, x: np.ndarray, y: np.ndarray = None) -> float:
        """
        Computes the quantum kernel matrix.

        Args:
            x (np.ndarray) :
                Vecotr of training or test data for which the kernel matrix is evaluated
            y (np.ndarray, default=None) :
                Vecotr of training or test data for which the kernel matrix is evaluated
        """
        if y is not None:
            return self.evaluate([x], [y])[0, 0]
        else:
            return self.evaluate([x], None)[0, 0]

    def assign_parameters(self, parameters):
        """
        Fix the training parameters of the feature map to numerical values

        Args:
            parameters (np.ndarray) :
                Array contraining numerical values to be assigned to the trainable parameters
                of the feature map
        """
        self._parameters = parameters

    def evaluate_with_parameters(
        self, x: np.ndarray, y: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        """
        Computes the quantum kernel matrix with assigned parameters

        Args:
            x (np.ndarray) :
                Vecotr of training or test data for which the kernel matrix is evaluated
            y (np.ndarray) :
                Vecotr of training or test data for which the kernel matrix is evaluated
            parameters (np.ndarray) :
                Array contraining numerical values to be assigned to the trainable parameters
                of the feature map
        """
        self.assign_parameters(parameters)
        return self.evaluate(x, y)

    def __add__(self, x):
        """
        Overwrites the a + b function, such that the addition of
        quantum kernels returns the composition of both quantum kernels.

        Number of  features have to be equal in both feature maps!

        Args:
            self (KernelMatrixBase): first quantum kernel
            x (KernelMatrixBase): second quantum kernel

        Returns:
            Returns the composed feature map as special class _ComposedKernelMatrix
        """
        return _ComposedKernelMatrix(self, x, "+")

    def __mul__(self, x):
        """
        Overwrites the a * b function, such that the multiplication of
        quantum kernels returns the composition of both quantum kernels.

        Number of  features have to be equal in both feature maps!

        Args:
            self (KernelMatrixBase): first quantum kernel
            x (KernelMatrixBase): second quantum kernel

        Returns:
            Returns the composed feature map as special class _ComposedKernelMatrix
        """
        return _ComposedKernelMatrix(self, x, "*")

    def __sub__(self, x):
        """
        Overwrites the a - b function, such that the subtraction of
        quantum kernels returns the composition of both quantum kernels.

        Number of  features have to be equal in both feature maps!

        Args:
            self (KernelMatrixBase): first quantum kernel
            x (KernelMatrixBase): second quantum kernel

        Returns:
            Returns the composed feature map as special class _ComposedKernelMatrix
        """
        return _ComposedKernelMatrix(self, x, "-")

    def __div__(self, x):
        """
        Overwrites the a / b function, such that the division of
        quantum kernels returns the composition of both quantum kernels.

        Number of  features have to be equal in both feature maps!

        Args:
            self (KernelMatrixBase): first quantum kernel
            x (KernelMatrixBase): second quantum kernel

        Returns:
            Returns the composed feature map as special class _ComposedKernelMatrix
        """
        return _ComposedKernelMatrix(self, x, "/")

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the kernel method.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        raise NotImplementedError()

    def set_params(self, **params):
        """
        Sets value of the fidelity kernel hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. num_qubits=2
        """
        raise NotImplementedError()


class _ComposedKernelMatrix(KernelMatrixBase):
    """
    Special class for composed kernel matrices

    Args:
        km1 (KernelMatrixBase) :
            first kernel matrix
        km2 (KernelMatrixBase) :
            second kernel matrix
    """

    def __init__(self, km1: KernelMatrixBase, km2: KernelMatrixBase, composition: str = "*"):
        if km1.num_features != km2.num_features:
            raise ValueError("Feature dimension is not equal in both feature maps.")

        self._km1 = km1
        self._km2 = km2
        self._composition = composition

    @property
    def feature_map(self) -> FeatureMapBase:
        """
        Returns the feature map from which the kernel matrix is constructed
        """
        raise RuntimeError("The feature map is not available for composed kernel matrices")

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used in the definition of the kernel matrix.

        Raises an RuntimeError if the number of qubits is not equal in both kernel matrices.
        """
        if self._km1.num_qubits == self._km2.num_qubits:
            return self._km1.num_qubits
        raise RuntimeError("The number of qubits is not available for composed kernel matrices")

    @property
    def num_features(self) -> int:
        """The feature dimension for the composed kernel matrix"""
        return self._km1.num_features

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters corresponding to the composed kernel matrix"""
        return self._km1.num_parameters + self._km2.num_parameters

    @property
    def parameters(self) -> np.ndarray:
        """
        The numeric values of the trainable parameters assigned to the
        feature map as np.ndarray
        """
        return np.concatenate((self._km1.parameters, self._km2.parameters))

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the composed kernel method.

        Hyper-parameter names are prefixed by ``km1__`` or ``km2__`` depending on
        which kernel matrix they belong to.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = dict(km1=self._km1, km2=self._km2)
        if deep:
            deep_items = self._km1.get_params().items()
            params.update(("km1__" + k, val) for k, val in deep_items)
            deep_items = self._km2.get_params().items()
            params.update(("km2__" + k, val) for k, val in deep_items)

        if self._km1.get_params()["num_qubits"] == self._km2.get_params()["num_qubits"]:
            params["num_qubits"] = self._km1.get_params()["num_qubits"]

        return params

    def set_params(self, **params):
        """
        Sets value of the composed kernel hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. num_qubits=2
        """
        valid_params = self.get_params()
        km1_dict = {}
        km2_dict = {}
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )
            if key.startswith("km1__"):
                km1_dict[key[5:]] = value
            elif key.startswith("km2__"):
                km2_dict[key[5:]] = value

            if key == "num_qubits":
                km1_dict["num_qubits"] = value
                km2_dict["num_qubits"] = value

        self._km1.set_params(**km1_dict)
        self._km2.set_params(**km2_dict)

    def evaluate(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Computes and the composed quantum kernel matrix.

        Args:
            x (np.ndarray) :
                Vecotr of training or test data for which the kernel matrix is evaluated
            y (np.ndarray, default=None) :
                Vecotr of training or test data for which the kernel matrix is evaluated

        Returns:
            Returns the quantum kernel matrix as 2D numpy array.
        """
        K1 = self._km1.evaluate(x, y)
        K2 = self._km2.evaluate(x, y)

        if self._composition == "*":
            return np.multiply(K1, K2)
        elif self._composition == "/":
            return np.divide(K1, K2)
        elif self._composition == "+":
            return np.add(K1, K2)
        elif self._composition == "-":
            return np.subtract(K1, K2)
        else:
            raise ValueError("Unknown composition: ", self._composition)

    def assign_parameters(self, parameters):
        self._km1.assign_parameters(parameters[: self._km1.num_parameters])
        self._km2.assign_parameters(parameters[self._km1.num_parameters :])
