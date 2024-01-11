from typing import Any, Union, List, Tuple
import numpy as np

from ..observables.observable_base import ObservableBase
from ..observables.observable_derivatives import (
    ObservableDerivatives,
)

from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..encoding_circuit.encoding_circuit_derivatives import (
    EncodingCircuitDerivatives,
)

from ..util import Executor


class LowLevelQNNBase:

    def __init__(self,
                 pqc: EncodingCircuitBase,
                 observable: Union[ObservableBase, list],
                 executor: Executor,
                 ) -> None:

        self._pqc = pqc
        self._observable = observable
        self._executor = executor

        self._initilize_derivative()


    def _initilize_derivative(self):
        """Initializes the derivative classes"""

        num_qubits_operator = 0
        if isinstance(self._observable, list):
            for i,obs in enumerate(self._observable):
                self._observable[i].set_map(self._pqc.qubit_map, self._pqc.num_physical_qubits)
                num_qubits_operator = max(num_qubits_operator, obs.num_qubits)
        else:
            self._observable.set_map(self._pqc.qubit_map, self._pqc.num_physical_qubits)
            num_qubits_operator = self._observable.num_qubits

        self._observable_derivatives = ObservableDerivatives(self._observable, True)
        self._pqc_derivatives = EncodingCircuitDerivatives(self._pqc, True)

        if self._pqc.num_virtual_qubits != num_qubits_operator:
            raise ValueError("Number of Qubits are not the same!")
        else:
            self._num_qubits = self._pqc.num_virtual_qubits

        if self._executor.optree_executor == "sampler":
            # In case of the sampler primitive, X and Y Pauli matrices have to be treated extra
            # This can be very inefficient!
            operator_string = str(self._observable)
            if "X" in operator_string or "Y" in operator_string:
                self._split_paulis = True
                print(
                    "The observable includes X and Y gates, consider switching"
                    + " to the Estimator primitive for a faster performance!"
                )
            else:
                self._split_paulis = False
        else:
            self._split_paulis = False

    def set_params(self, **params) -> None:
        raise NotImplementedError

    def get_params(self) -> dict:
        raise NotImplementedError

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN"""
        return self._num_qubits

    @property
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC"""
        return self._pqc_derivatives.num_features

    @property
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC"""
        return self._pqc_derivatives.num_parameters

    @property
    def num_operator(self) -> int:
        """Return the number outputs"""
        return self._observable_derivatives.num_operators

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        return self._observable_derivatives.num_parameters

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        return self._observable_derivatives.multiple_output

    @property
    def parameters(self):
        """Return the parameter vector of the PQC."""
        return self._pqc_derivatives.parameter_vector

    @property
    def features(self):
        """Return the feature vector of the PQC."""
        return self._pqc_derivatives.feature_vector

    @property
    def parameters_operator(self):
        """Return the parameter vector of the cost operator."""
        return self._observable_derivatives.parameter_vector

    def evaluate(
        self,
        values,  # TODO: data type definition missing Union[str,Expec,tuple,...]
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ) -> dict:
        raise NotImplementedError

    def gradient(self,
                 x: Union[float, np.ndarray],
                 param: Union[float, np.ndarray],
                 param_obs: Union[float, np.ndarray]):

        return np.concatenate((self.evaluate("dfdp",x, param, param_obs)["dfdp"],
                               self.evaluate("dfdop",x, param, param_obs)["dfdop"]),axis=None)

    def __call__(self,
                 x: Union[float, np.ndarray],
                 param: Union[float, np.ndarray],
                 param_obs: Union[float, np.ndarray]):

        return self.evaluate("f",x, param, param_obs)["f"]

