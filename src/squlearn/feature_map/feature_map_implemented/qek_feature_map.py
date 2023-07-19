import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..feature_map_base import FeatureMapBase


class QEKFeatureMap(FeatureMapBase):

    """
    Creates the QEK feature map as presented in Reference http://arxiv.org/pdf/2105.02276v1

    **Example for 4 qubits, a 2 dimensional feature vector, 2 layers:**

    .. plot::

       from squlearn.feature_map import QEKFeatureMap
       pqc = QEKFeatureMap(4, 2, 2)
       plt = pqc.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt

    The encoding can be optionally repeated at the end to make the previous rotations not
    redundant in a fidelity kernel setting.
    The circuit is closed by default, i.e. the last qubit is entangled with the first one.

    Args:
        num_qubits (int): Number of qubits of the feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default:1)
        closed (bool): If true, the last and the first qubit are entangled;
                       not necessarily hardware efficient! (default: true)
        final_encoding (bool): If True, the encoding is repeated at the end (default: False)
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        num_layers: int = 1,
        closed: bool = True,
        final_encoding=False,
    ) -> None:
        super().__init__(num_qubits, num_features)

        self.num_layers = num_layers
        self.closed = closed
        self.final_encoding = final_encoding

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the QEK feature map."""
        num_param = self.num_qubits * self.num_layers
        if self.num_qubits > 2:
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
        Generates and returns the circuit of the QEK feature map

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            Returns the QEK circuit in qiskit QuantumCircuit format
        """

        nfeatures = len(features)
        nparam = len(parameters)

        QC = QuantumCircuit(self.num_qubits)
        ioff = 0

        QC.h(range(self.num_qubits))

        # Loops through the layers
        for ilayer in range(self.num_layers):
            # Loops through the data encoding gates
            n_feature_loop = int(np.ceil(self.num_features / self.num_qubits))
            for i in range(n_feature_loop * self.num_qubits):
                if (i // self.num_qubits) % 2 == 0:
                    QC.rz(features[i % nfeatures], i % self.num_qubits)
                else:
                    QC.rx(features[i % nfeatures], i % self.num_qubits)

            # Single theta Ry gates
            for i in range(self.num_qubits):
                QC.ry(parameters[ioff % nparam], i)
                ioff = ioff + 1

            # Entangled theta CRZ gates
            if self.num_qubits > 2:
                if self.closed:
                    istop = self.num_qubits
                else:
                    istop = self.num_qubits - 1

                for i in range(istop):
                    QC.crz(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                    ioff = ioff + 1

        if self.final_encoding:
            # Repeat encoding finally to make the previous rotations not redundant
            n_feature_loop = int(np.ceil(self.num_features / self.num_qubits))
            for i in range(n_feature_loop * self.num_qubits):
                if int(np.ceil(i / self.num_qubits)) % 2 == 0:
                    QC.rz(features[i % nfeatures], i % self.num_qubits)
                else:
                    QC.rx(features[i % nfeatures], i % self.num_qubits)

        return QC
