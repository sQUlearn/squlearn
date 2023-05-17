import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..feature_map_base import FeatureMapBase


class YZ_CX_FeatureMap(FeatureMapBase):
    """
    Creates the YZ-CX PQC feature map from Reference https://arxiv.org/pdf/2108.01039v1.pdf

    The structure of the feature map is

    -Ry(c*x0+theta_y0,0)-Rz(c*x0+theta_z0,0)-.-Ry(c*x0+theta_y0,1)-Rz(c*x0+theta_z0,1)--
    -Ry(c*x1+theta_y1,0)-Rz(c*x1+theta_z1,0)-x-Ry(c*x1+theta_y1,1)-Rz(c*x1+theta_z1,1)-.
    -Ry(c*x2+theta_y2,0)-Rz(c*x2+theta_z2,0)-.-Ry(c*x2+theta_y2,1)-Rz(c*x2+theta_z2,1)-x
    -Ry(c*x3+theta_y3,0)-Rz(c*x3+theta_z3,0)-X-Ry(c*x3+theta_y3,1)-Rz(c*x3+theta_z3,1)--

    The number of qubits and the number of features have to be the same!

    One combination of Ry and Rz is considered as a single layer
    """

    def __init__(
        self, num_qubits: int, num_features: int, num_layers: int = 1, c: float = 1.0
    ) -> None:
        super().__init__(num_qubits, num_features)

        self._num_layers = num_layers
        self._c = c

        if self._num_features != self._num_qubits:
            raise ValueError(
                "The number of qubits and the number of features have to be the same!"
            )

    @property
    def num_parameters(self) -> int:
        return 2 * self._num_qubits * self._num_layers

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def c(self) -> int:
        return self._c

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Return the circuit of the YZ-CX feature map

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the circuit in qiskit format
        """

        if self._num_features != len(features):
            raise ValueError("Wrong number of features in supplied vector")

        if self.num_parameters != len(parameters):
            raise ValueError("Wrong number of parameters in supplied vector")

        # Creates the layers of the feature map
        QC = QuantumCircuit(self._num_qubits)
        ioff = 0
        for ilayer in range(self._num_layers):
            for i in range(self._num_qubits):
                QC.ry(parameters[ioff] + self._c * features[i], i)
                ioff = ioff + 1
                QC.rz(parameters[ioff] + self._c * features[i], i)
                ioff = ioff + 1
            # Entangling layer depends on odd or even layer
            if ilayer % 2 == 0:
                for i in range(0, self._num_qubits - 1, 2):
                    QC.cx(i, i + 1)
            else:
                for i in range(1, self._num_qubits - 1, 2):
                    QC.cx(i, i + 1)
        return QC
