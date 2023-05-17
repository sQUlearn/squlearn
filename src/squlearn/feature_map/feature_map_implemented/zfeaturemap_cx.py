import numpy as np
from typing import Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from ..feature_map_base import FeatureMapBase


class ZFeatureMap_CX(FeatureMapBase):
    """
    Creates Qiskit's ZFeatureMap with additional CNOT gates between the default layers.

    The number of qubits and the number of features have to be the same!
    """

    def __init__(self, num_qubits: int, num_features: int, reps: int = 2) -> None:
        super().__init__(num_qubits, num_features)
        self._reps = reps

        if self._num_features != self._num_qubits:
            raise ValueError(
                "The number of qubits and the number of features have to be the same!"
            )

    @property
    def num_parameters(self) -> int:
        return self._num_qubits * self._reps

    @property
    def num_layers(self) -> int:
        return self._reps

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        if self._num_features != len(features):
            raise ValueError("Wrong number of features!")

        circuit = QuantumCircuit(self._num_qubits)
        ioff = 0
        for _ in range(self._reps):
            for i in range(self._num_qubits):
                circuit.h(i)
                circuit.p(parameters[ioff] * features[i], i)
                ioff += 1
            if self._reps % 2 == 0:
                for j in range(self._num_qubits - 1):
                    circuit.cx(j, j + 1)
            else:
                for j in range(1, self._num_qubits - 1, 2):
                    circuit.cx(j, j + 1)

        return circuit
