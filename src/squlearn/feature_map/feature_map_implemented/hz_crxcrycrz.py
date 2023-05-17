import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..feature_map_base import FeatureMapBase


class HZCRxCRyCRz(FeatureMapBase):
    """
    HZ encoding followed by controlled Rx, Ry Rz rotations:

    -[-H-Rz(x)-c-----c-----c-----------------------Rx(p)-Ry(p)-Rz(p)-]-
    -[-H-Rz(x)-Rx(p)-Ry(p)-Rz(p)-c-----c-----c-----|-----|-----|-----]-
    -[-H-Rz(x)-c-----c-----c-----Rx(p)-Ry(p)-Rz(p)-|-----|-----|-----]-
    -[-H-Rz(x)-Rx(p)-Ry(p)-Rz(p)-------------------c-----c-----c-----]-
        [] = repeated by num_layers

    Args:
        num_qubits (int): Number of qubits of the HZCRxCRyCRz feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
        closed (bool): If true, the last and the first qubit are entangled;
                       not necessarily hardware efficient! (default: true)
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        num_layers: int = 1,
        closed: bool = True,
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.num_layers = num_layers
        self.closed = closed

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters of the HZCRxCRyCRz feature map."""
        num_param = 3 * (self.num_qubits - 1) * self.num_layers
        if self.closed:
            num_param += 3 * self.num_qubits * self.num_layers
        return num_param

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the HZCRxCRyCRz feature map

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
            # First ZZ-feature map
            QC.h(range(self.num_qubits))
            for i in range(self.num_qubits):
                QC.rz(features[i % nfeature], i)

            for i in range(0, self.num_qubits - 1, 2):
                QC.crx(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                ioff = ioff + 1
                QC.cry(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                ioff = ioff + 1
                QC.crz(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                ioff = ioff + 1

            if self.num_qubits > 2:
                if self.closed:
                    istop = self.num_qubits
                else:
                    istop = self.num_qubits - 1

                for i in range(1, istop, 2):
                    QC.crx(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                    ioff = ioff + 1
                    QC.cry(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                    ioff = ioff + 1
                    QC.crz(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                    ioff = ioff + 1

        return QC
