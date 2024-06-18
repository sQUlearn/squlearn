import abc
from typing import Union
import numpy as np
import copy

from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..util import Executor


class LowLevelQNNBase(abc.ABC):
    """Base class for low-level QNNs.

    Args:
        pqc (EncodingCircuitBase): The parameterized quantum circuit.
        observable (Union[ObservableBase, list]): The observable(s) to measure.
        executor (Executor): The executor for the quantum circuit.
    """

    def __init__(
        self,
        parameterized_quantum_circuit: EncodingCircuitBase,
        observable: Union[ObservableBase, list],
        executor: Executor,
    ) -> None:
        self._pqc = copy.copy(parameterized_quantum_circuit)
        self._observable = copy.copy(observable)
        self._executor = executor

    @abc.abstractmethod
    def set_params(self, **params) -> None:
        """Set the parameters of the QNN."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_params(self, deep: bool = True) -> dict:
        """Get the parameters of the QNN."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC"""
        raise NotImplementedError

    @property
    def num_operator(self) -> int:
        """Return the number outputs"""
        raise NotImplementedError

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        raise NotImplementedError

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
        *values: Union[
            str,
            ParameterVector,
            ParameterVectorElement,
            tuple,
        ],
    ) -> dict:
        """General function for evaluating the output of derivatives of the QNN.

        Evaluation works for given combination of
        input features `x` and parameters `param` and `param_op`.
        The function includes caching of results

        If `x`, `param`, and/or `param_op` are given as a nested list
        (for example multiple sets of parameters),
        the values are returned in a nested list.

        Args:
            x (np.ndarray): Values of the input feature data.
            param (np.ndarray): Parameter values of the PQC parameters
            param_op (np.ndarray): Parameter values of the operator parameters
            values : Derivatives (or values) of the QNN that are evaluated. Higher order
                     derivatives are given as tuples of parameters or vectors.

        Results:
            Returns a dictionary with the computed values.
            The keys of the dictionary are given by the entries in the values tuple

        """
        raise NotImplementedError

    def gradient(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ):
        """Return the gradient wrt. trainable parameters of the QNN.

        Args:
            x (Union[float, np.ndarray]): Input data.
            param (Union[float, np.ndarray]): Parameters of the PQC.
            param_obs (Union[float, np.ndarray]): Parameters of the observable.

        Returns:
            np.ndarray: Gradient of the QNN.

        """
        return np.concatenate(
            (
                self.evaluate(x, param, param_obs, "dfdp")["dfdp"],
                self.evaluate(x, param, param_obs, "dfdop")["dfdop"],
            ),
            axis=None,
        )

    def __call__(
        self,
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ):
        """
        Function for evaluating the QNN.

        Args:
            x (Union[float, np.ndarray]): Input data.
            param (Union[float, np.ndarray]): Parameters of the PQC.
            param_obs (Union[float, np.ndarray]): Parameters of the observable.

        Returns:
            np.ndarray: The output of the QNN.

        """
        return self.evaluate(x, param, param_obs, "f")["f"]
