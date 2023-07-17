import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..feature_map_base import FeatureMapBase


class ChebPQC(FeatureMapBase):
    """
    Chebyshev Feature Map from Reference https://arxiv.org/abs/2306.01639

    The feature map consists of three elements:

    #. Basis change in the form of a trainable rotation around the y-axis at start and end.

    #. Non-linear encoding of the features via the Chebyshev polynomials from Rx gates.
       The degree of the Chebyshev polynomials is optimized during training.

    #. Parameterized two-qubit controlled or RZZ rotations

    2. and 3. form a layer that can be repeated multiple times.

    **Example for 4 qubits, a 2 dimensional feature vector and 2 layers:**

    .. plot::

       from squlearn.feature_map import ChebPQC
       pqc = ChebPQC(4, 2, 2)
       plt = pqc.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt

    The entangling gate can be chosen between ``crz`` and ``rzz``.
    The latter is more hardware efficient.
    Also, the entangling between the first and the last qubit can be switched off via
    the ``closed`` parameter to avoid swap gates.

    Args:
        num_qubits (int): Number of qubits of the ChebPQC feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers of the Chebyshev encoding and the two qubit
                          manipulation (default: 1)
        closed (bool): If false, the last and the first qubit are not entangled (default: True)
        entangling_gate (str): Entangling gate to use. Either ``crz``
                               or ``rzz`` (default: ``crz``)
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
        """The number of trainable parameters of the ChebPQC feature map."""
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
        Returns the circuit of the ChebPQC feature map

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained

        Return:
            Returns the circuit in Qiskit's QuantumCircuit format
        """

        def phi_map(a, x):
            """Helper function for returning a*arccos(x)"""
            return a * np.arccos(x)

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

        # Basis change at the beginning
        for i in range(self.num_qubits):
            QC.ry(parameters[ioff % nparam], i)
            ioff = ioff + 1

        for ilayer in range(self.num_layers):
            # Chebyshev feature map
            for i in range(self.num_qubits):
                QC.rx(phi_map(parameters[ioff % nparam], features[i % nfeature]), i)
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

    def get_cheb_indices(self, flatten: bool = True):
        """
        Function that returns the indices of the parameters involved in the Chebyshev encoding.

        Args:
            flatten (bool): If true, the indices are returned as a flat list, otherwise
                            as a list of lists, where the outer list corresponds to the layers
                            (default: True)
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
