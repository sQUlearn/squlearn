import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..feature_map_base import FeatureMapBase


class ChebRx(FeatureMapBase):
    """
    Simple Chebyshev feature map build from  Rx gates

    **Example for 4 qubits, a 2 dimensional feature vector and 2 layers:**

    .. plot::

       from squlearn.feature_map import ChebRx
       pqc = ChebRx(4, 2, 2)
       plt = pqc.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt

    Args:
        num_qubits (int): Number of qubits of the ChebRx feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
        closed (bool): If true, the last and the first qubit are entangled (default: false)
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        num_layers: int = 1,
        closed: bool = False,
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.num_layers = num_layers
        self.closed = closed

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the ChebRx feature map."""
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

        def entangle_layer(QC: QuantumCircuit) -> QuantumCircuit:
            """Creation of a simple nearest neighbor entangling layer"""
            for i in range(0, self.num_qubits - 1, 2):
                QC.cx(i, i + 1)

            if self.num_qubits > 2:
                if self.closed:
                    istop = self.num_qubits
                else:
                    istop = self.num_qubits - 1
                for i in range(1, istop - 1, 2):
                    QC.cx(i, i + 1)

            return QC

        def mapping(a, x):
            """Helper function for returning a*arccos(x)"""
            return a * np.arccos(x)

        nfeature = len(features)
        nparam = len(parameters)

        QC = QuantumCircuit(self.num_qubits)
        ioff = 0
        for ilayer in range(self.num_layers):
            # Chebyshev feature map
            for i in range(self.num_qubits):
                QC.rx(mapping(parameters[ioff % nparam], features[i % nfeature]), i)
                ioff = ioff + 1
            # Trafo
            for i in range(self.num_qubits):
                QC.rx(parameters[ioff % nparam], i)
                ioff = ioff + 1
            QC = entangle_layer(QC)

        return QC

    def get_cheb_indices(self, flatten: bool = True):
        """
        Function that returns the indices of the parameters involved in the Chebyshev encoding.

        Args:
            flatten (bool): If true, the indices are returned as a flat list, otherwise
                            as a list of lists, where the outer list corresponds to the layers
                            (default: True)
        """
        cheb_index = []
        ioff = 0
        for ilayer in range(self.num_layers):
            cheb_index_layer = []
            for i in range(self.num_qubits):
                cheb_index_layer.append(ioff)
                ioff = ioff + 1

            for i in range(self.num_qubits):
                ioff = ioff + 1

            if flatten:
                cheb_index += cheb_index_layer
            else:
                cheb_index.append(cheb_index_layer)
        return cheb_index
