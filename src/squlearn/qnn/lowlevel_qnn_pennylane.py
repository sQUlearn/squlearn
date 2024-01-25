import abc
from typing import Union
import numpy as np

from qiskit.circuit import ParameterVector

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..util import Executor

from ..util.pennylane import PennyLaneCircuit, PennyLaneDevice

from .lowlevel_qnn_base import LowLevelQNNBase


class LowLevelQNNPennyLane(LowLevelQNNBase):

    def __init__(self,
                 pqc: EncodingCircuitBase,
                 observable: Union[ObservableBase, list],
                 executor: Executor,
                 ) -> None:

        super().__init__(pqc, observable, executor)
        #self._pqc = pqc
        #self._observable = observable
        #self._executor = executor

        self._device = PennyLaneDevice()

        self._x = ParameterVector("x", self._pqc.num_features)
        self._param = ParameterVector("param", self._pqc.num_parameters)
        self._param_obs = ParameterVector("param_obs", self._observable.num_parameters)

        self._qiskit_circuit = self._pqc.get_circuit(self._x, self._param)
        self._qiskit_observable = self._observable.get_operator(self._param_obs)

        self._pennylane_circuit = PennyLaneCircuit(self._device, self._qiskit_circuit, self._qiskit_observable)

    def draw(self,**kwargs):

        return self._pennylane_circuit.draw(**kwargs)


    #@abc.abstractmethod
    def set_params(self, **params) -> None:
        pass

    #@abc.abstractmethod
    def get_params(self) -> dict:
        pass

    @property
    #@abc.abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits of the QNN"""
        pass

    @property
    #@abc.abstractmethod
    def num_features(self) -> int:
        """Return the dimension of the features of the PQC"""
        pass

    @property
    #@abc.abstractmethod
    def num_parameters(self) -> int:
        """Return the number of trainable parameters of the PQC"""
        pass

    @property
    def num_operator(self) -> int:
        """Return the number outputs"""
        pass

    @property
    def num_parameters_observable(self) -> int:
        """Return the number of trainable parameters of the expectation value operator"""
        pass

    @property
    def multiple_output(self) -> bool:
        """Return true if multiple outputs are used"""
        pass

    #@abc.abstractmethod
    def evaluate(
        self,
        values,  # TODO: data type definition missing Union[str,Expec,tuple,...]
        x: Union[float, np.ndarray],
        param: Union[float, np.ndarray],
        param_obs: Union[float, np.ndarray],
    ) -> dict:

        if values=="f" or values ==("f",):
            return {"f":np.array(self._pennylane_circuit(x=x, param=param,param_obs=param_obs))}


