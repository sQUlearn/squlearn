import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..feature_map_base import FeatureMapBase


class YZ_CX_FeatureMap(FeatureMapBase):
    """
    Creates the YZ-CX Feature Map from Reference https://arxiv.org/abs/2108.01039v3

    **Example for 4 qubits, a 4 dimensional feature vector, 2 layers and c = 2.0:**

    .. plot::

       from squlearn.feature_map import YZ_CX_FeatureMap
       pqc = YZ_CX_FeatureMap(4, 4, 2, c=2.0)
       plt = pqc.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt

    One combination of Ry and Rz is considered as a single layer.

    Args:
        num_qubits (int): Number of qubits of the YZ-CX Feature Map feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
        c (float): Prefactor :math:`c` for rescaling the data (default: 1.0)
    """

    def __init__(
        self, num_qubits: int, num_features: int, num_layers: int = 1, c: float = 1.0
    ) -> None:
        super().__init__(num_qubits, num_features)

        self._num_layers = num_layers
        self._c = c

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the YZ-CX Feature Map feature map."""
        return 2 * self.num_qubits * self._num_layers

    @property
    def num_layers(self) -> int:
        """The number of layers of the YZ-CX Feature Map feature map."""
        return self._num_layers

    @property
    def c(self) -> int:
        """The prefactor :math:`c` of the YZ-CX Feature Map feature map."""
        return self._c

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Return the circuit of the YZ-CX feature map.

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

        Return:
            Returns the circuit in qiskit format.
        """

        nfeature = len(features)
        nparam = len(parameters)

        # Creates the layers of the feature map
        QC = QuantumCircuit(self.num_qubits)
        ioff = 0
        for ilayer in range(self.num_layers):
            for i in range(self.num_qubits):
                QC.ry(parameters[ioff % nparam] + self.c * features[i % nfeature], i)
                ioff = ioff + 1
                QC.rz(parameters[ioff % nparam] + self.c * features[i % nfeature], i)
                ioff = ioff + 1
            # Entangling layer depends on odd or even layer
            if ilayer % 2 == 0:
                for i in range(0, self.num_qubits - 1, 2):
                    QC.cx(i, i + 1)
            else:
                for i in range(1, self.num_qubits - 1, 2):
                    QC.cx(i, i + 1)
        return QC
