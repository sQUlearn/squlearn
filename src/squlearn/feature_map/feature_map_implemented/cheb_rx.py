import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..feature_map_base import FeatureMapBase


class ChebRx(FeatureMapBase):
    """
    ChebRx feature map

    -[-Rx(p*arcos(x))-Rx(p)-c-----]-
    -[-Rx(p*arcos(x))-Rx(p)-X-c---]-
    -[-Rx(p*arcos(x))-Rx(p)---X-c-]-
    -[-Rx(p*arcos(x))-Rx(p)-----X-]-

    [] = is repeated by num_layers

    Args:
        num_qubits (int): Number of qubits of the ChebRx feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of repetitions (default: 1)
    """

    def __init__(self, num_qubits: int, num_features: int, num_layers: int = 1) -> None:
        super().__init__(num_qubits, num_features)
        self.num_layers = num_layers

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters of the ChebRx feature map."""
        return 2 * self.num_qubits * self.num_layers

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the ChebRx feature map

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the circuit in Qiskit's QuantumCircuit format
        """
        nfeature = len(features)
        nparam = len(parameters)
        QC = QuantumCircuit(self.num_qubits)
        ioff = 0
        for ilayer in range(self.num_layers):
            # Chebyshev feature map
            for i in range(self.num_qubits):
                QC.rx(self.__phi_map(parameters[ioff % nparam], features[i % nfeature]), i)
                ioff = ioff + 1
            # Trafo
            for i in range(self.num_qubits):
                QC.rx(parameters[ioff % nparam], i)
                ioff = ioff + 1
            QC = self.__entangle_layer(QC)

        return QC

    def __entangle_layer(self, QC: QuantumCircuit) -> QuantumCircuit:
        """
        Creation of a simple nearest neighbor entangling layer
        """
        for i in range(0, self.num_qubits - 1, 2):
            QC.cx(i, i + 1)
        for i in range(1, self.num_qubits - 1, 2):
            QC.cx(i, i + 1)
        return QC

    def __phi_map(self, a, x):
        """Helper function for returning a*arccos(x)"""
        return a * np.arccos(x)
