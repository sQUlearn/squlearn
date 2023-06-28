import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..feature_map_base import FeatureMapBase


class ChebyshevTower(FeatureMapBase):
    """
    Creates a featurmap based on Chebyshev polynomials created by Ry(arcos(x))

    **Example for 4 qubits, a 2 dimensional feature vector and 2 Chebychev terms per feature:**

    .. plot::

       from squlearn.feature_map import ChebyshevTower
       pqc = ChebyshevTower(4, 2, 2)
       plt = pqc.draw(style={'fontsize':15,'subfontsize ': 15})
       plt.tight_layout()
       plt

    The structure of the Chebyshev feature map reads:
    0-Layer1-Layer2- ...

    Args:
        num_qubits (int): Number of Qubits
        num_features (int): Dimension of feature vector
        n_chebyshev (int): Number of Chebyshev terms per feature
        alpha (float): scaling factor of Chebyshev tower
        num_layers (int): Number of layers
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        n_chebyshev: int,
        alpha: float = 1.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__(num_qubits, num_features)

        self.n_chebyshev = n_chebyshev
        self.alpha = alpha
        self.num_layers = num_layers

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters of the feature map."""
        return 0

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray] = None,
    ) -> QuantumCircuit:
        """
        Generates and returns the circuit of the Chebyshev feature map
        """

        if self.num_features != len(features):
            raise ValueError("Wrong number of features")

        if parameters is not None:
            if len(parameters) != 0:
                raise ValueError("No parameters are needed!")

        # Create the circuit
        QC = QuantumCircuit(self.num_qubits)
        for ilayer in range(self.num_layers):
            QC.h(range(self.num_qubits))
            # Loops through the data encoding gates
            ifeature = 0
            for i in range(self.num_features):
                for icheb in range(self.n_chebyshev):
                    QC.ry(
                        self._phi_map(features[i], icheb + 1),
                        ifeature % self.num_qubits,
                    )
                    ifeature += 1
            # Entangling layer
            if ilayer + 1 < self.num_layers:
                QC = self._entangle_layer(QC)

        return QC

    def _entangle_layer(self, QC: QuantumCircuit):
        """
        Creation of a simple NN entangling layer

        Args:
            QC (QuantumCircuit): Quantum circuit

        Returns:
            Returns the Quantum circuit with a NN entangling layer added
        """
        for i in range(0, self.num_qubits - 1, 2):
            QC.cx(i, i + 1)
        for i in range(
            1,
            self.num_qubits - 1,
            2,
        ):
            QC.cx(i, i + 1)
        return QC

    def _phi_map(self, x, i):
        """
        Non-linear mapping for x: alpha*i*arccos(x)

        Args:
            x: x for arcos
            i: factor i

        Returns:
            alpha*i*arccos(x)
        """
        return self.alpha * i * np.arccos(x)
