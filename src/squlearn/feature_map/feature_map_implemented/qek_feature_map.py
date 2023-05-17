import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..feature_map_base import FeatureMapBase


class QEKFeatureMap(FeatureMapBase):

    """
    Creates the QEK feature map as presented in Reference http://arxiv.org/pdf/2105.02276v1

    The structure of the whole QEK feature map reads:
    0-H-Layer1-Layer2-Rz(x) ...

    A single layer is constructed as:

    -Rz(x0)-Ry(theta[i+0])-------x-------------------------------------CRz(theta[i+5])-
    -Rz(x1)-Ry(theta[i+1])-CRz(theta[i+3])--------------x--------------------|---------
    -Rz(x0)-Ry(theta[i+2])------------------------CRz(theta[i+4])------------x---------

    The layers are finalized by a last layer of encoding the data, since otherwise the last parameterized gates are redundant

    Args:
        num_qubits (int): Number of Qubits of the feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default:1)

    """

    def __init__(self, num_qubits: int, num_features: int, num_layers: int = 1) -> None:
        super().__init__(num_qubits, num_features)

        self.num_layers = num_layers

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters of the feature map."""
        return 2 * self.num_qubits * self.num_layers

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Generates and returns the circuit of the QEK feature map

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the QEK circuit in qiskit QuantumCircuit format
        """

        if self._num_features != len(features):
            raise ValueError("Wrong number of features in supplied vector")

        if self.num_parameters != len(parameters):
            raise ValueError("Wrong number of parameters in supplied vector")

        QC = QuantumCircuit(self.num_qubits)
        ioff = 0

        QC.h(range(self.num_qubits))

        # Loops through the layers
        for ilayer in range(self.num_layers):
            # Loops through the data encoding gates
            n_feature_loop = int(np.ceil(self.num_features / self.num_qubits))
            for i in range(n_feature_loop * self.num_qubits):
                if (i // self.num_qubits) % 2 == 0:
                    QC.rz(features[i % self.num_features], i % self.num_qubits)
                else:
                    QC.rx(features[i % self.num_features], i % self.num_qubits)

            # Single theta Ry gates
            for i in range(self.num_qubits):
                QC.ry(parameters[ioff], i)
                ioff = ioff + 1

            # Entangled theta CRZ gates
            for i in range(self.num_qubits):
                QC.crz(parameters[ioff], i, (i + 1) % self.num_qubits)
                ioff = ioff + 1

        # Repeat encoding finally to make the previous rotations not redundant
        n_feature_loop = int(np.ceil(self.num_features / self.num_qubits))
        for i in range(n_feature_loop * self.num_qubits):
            if int(np.ceil(i / self.num_qubits)) % 2 == 0:
                QC.rz(features[i % self.num_features], i % self.num_qubits)
            else:
                QC.rx(features[i % self.num_features], i % self.num_qubits)

        return QC
