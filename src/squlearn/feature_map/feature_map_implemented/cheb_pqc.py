import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..feature_map_base import FeatureMapBase


class ChebPQC(FeatureMapBase):
    """
    Cheb PQC feature map

    -Ry(p)-[-Rx(p*arcos(x))-c-----------Rz(p)-]-Ry(p)-
    -Ry(p)-[-Rx(p*arcos(x))-Rz(p)-c-----|-----]-Ry(p)-
    -Ry(p)-[-Rx(p*arcos(x))-c-----Rz(p)-|-----]-Ry(p)-
    -Ry(p)-[-Rx(p*arcos(x))-Rz(p)-------c ----]-Ry(p)-
                  repeated by num_layers

    Args:
        num_qubits (int): Number of qubits of the ChebPQC feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers of the arcos encoding and the two qubit manipulation (default: 1)
        closed (bool): If true, the last and the first qubit are entangled;
                       not necessarily hardware efficient! (default: true)
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        num_layers: int = 1,
        closed: bool = True,
        entangling_gate: str = "crz",
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.num_layers = num_layers
        self.closed = closed
        self.entangling_gate = entangling_gate
        if self.entangling_gate not in ("crz", "rzz"):
            raise ValueError("Unknown value for entangling_gate: ", entangling_gate)

    @property
    def num_parameters(self) -> int:
        """Returns th e numbeself)r of trainable parameters of the Cheb PQC feature map."""
        num_param = 2 * self.num_qubits + self.num_qubits * self.num_layers
        if self.closed:
            num_param += self.num_qubits * self.num_layers
        else:
            num_param += (self.num_qubits - 1) * self.num_layers
        return num_param

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the Cheb PQC feature map

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

        if self.entangling_gate == "crz":
            egate = QC.crz
        elif self.entangling_gate == "rzz":
            egate = QC.rzz
        else:
            raise ValueError("Unknown entangling gate")

        for i in range(self.num_qubits):
            QC.ry(parameters[ioff % nparam], i)
            ioff = ioff + 1

        for ilayer in range(self.num_layers):
            # Chebyshev feature map
            for i in range(self.num_qubits):
                QC.rx(self.__phi_map(parameters[ioff % nparam], features[i % nfeature]), i)
                ioff = ioff + 1

            for i in range(0, self.num_qubits, 2):
                egate(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                ioff = ioff + 1

            if self.num_qubits > 2:
                if self.closed:
                    istop = self.num_qubits
                else:
                    istop = self.num_qubits - 1
                for i in range(1, istop, 2):
                    egate(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                    ioff = ioff + 1

        for i in range(self.num_qubits):
            QC.ry(parameters[ioff % nparam], i)
            ioff = ioff + 1

        return QC

    def __phi_map(self, a, x):
        """Helper function for returning a*arccos(x)"""
        return a * np.arccos(x)

    def get_cheb_indices(self, flatten: bool = True):
        """
        Returns a nested list of the indices of the parameters involved
        in the arccos encoding.
        The outer list is connected to the layers of the feature map.
        """
        cheb_index = []
        ioff = self.num_qubits
        for ilayer in range(self.num_layers):
            cheb_index_layer = []
            for i in range(self.num_qubits):
                cheb_index_layer.append(ioff)
                ioff = ioff + 1

            for i in range(0, self.num_qubits, 2):
                ioff = ioff + 1

            if self.num_qubits > 2:
                if self.closed:
                    istop = self.num_qubits
                else:
                    istop = self.num_qubits - 1
                for i in range(1, istop, 2):
                    ioff = ioff + 1
            if flatten:
                cheb_index += cheb_index_layer
            else:
                cheb_index.append(cheb_index_layer)
        return cheb_index
