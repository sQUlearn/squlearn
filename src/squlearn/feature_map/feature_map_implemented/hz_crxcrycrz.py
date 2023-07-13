import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..feature_map_base import FeatureMapBase


class HZCRxCRyCRz(FeatureMapBase):
    """
    Featuremap with HZ encoding followed by controlled Rx, Ry Rz rotations.

    **Example for 4 qubits, a 2 dimensional feature vector and 1 layer:**

    .. plot::

       from squlearn.feature_map import HZCRxCRyCRz
       pqc = HZCRxCRyCRz(4, 2, 1)
       plt = pqc.draw(style={'fontsize':15,'subfontsize ': 15})
       plt.tight_layout()
       plt

    The circuit is repeated for the number of layers.
    The circuit is closed by default, i.e. the last qubit is entangled with the first one.
    The encoding can be optionally repeated at the end to make the previous rotations not
    redundant in a fidelity kernel setting.

    Args:
        num_qubits (int): Number of qubits of the HZCRxCRyCRz feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
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

        if self.num_qubits < 2:
            raise ValueError("HZCRxCRyCRz requires at least two qubits.")

        self.num_layers = num_layers
        self.closed = closed
        self.final_encoding = final_encoding

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the HZCRxCRyCRz feature map."""
        num_param = 3 * (self.num_qubits - 1) * self.num_layers
        if self.closed:
            num_param += 3 * self.num_layers
        return num_param

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the HZCRxCRyCRz feature map

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained.
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained.

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

            if self.closed:
                istop = self.num_qubits
            else:
                istop = self.num_qubits - 1

            for i in range(0, istop, 2):
                QC.crx(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                ioff = ioff + 1
                QC.cry(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                ioff = ioff + 1
                QC.crz(parameters[ioff % nparam], i, (i + 1) % self.num_qubits)
                ioff = ioff + 1

            if self.num_qubits >= 2:
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

        if self.final_encoding:
            for i in range(self.num_qubits):
                QC.rz(features[i % nfeature], i)
        return QC
