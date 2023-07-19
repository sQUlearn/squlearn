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
        self._num_features = self._feature_map.num_features
        self._num_parameters = self._feature_map.num_parameters
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
        return self._num_features

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
        return self._num_parameters

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

    def get_params(self, deep=True):
        return {
            "feature_map": self._feature_map,
            "executor": self._executor,
            "initial_parameters": self._parameters,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, "_" + param, value)
        return self


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
        """Returns RuntimeError since number of qubit is not meaningful for composed kernel matrices"""
        raise RuntimeError("The number of qubits is not available for composed kernel matrices")

    @property
    def num_features(self) -> int:
        """Returns feature dimension for the composed kernel matrix"""
        return self._km1.num_features

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters corresponding to the composed kernel matrix"""
        return self._km1.num_parameters + self._km2.num_parameters

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
