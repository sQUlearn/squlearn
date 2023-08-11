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
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

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
        alpha: float = 4.0,
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.num_layers = num_layers
        self.closed = closed
        self.alpha = alpha

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the ChebRx feature map."""
        return 2 * self.num_qubits * self.num_layers

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the ChebRx feature map."""
        bounds = np.zeros((self.num_parameters, 2))
        ioff = 0
        for ilayer in range(self.num_layers):
            # Chebyshev feature map
            for i in range(self.num_qubits):
                bounds[ioff] = [0.0, self.alpha]
                ioff = ioff + 1
            # Trafo
            for i in range(self.num_qubits):
                bounds[ioff] = [-np.pi, np.pi]
                ioff = ioff + 1
        return bounds

    def generate_initial_parameters(self, seed: Union[int, None] = None) -> np.ndarray:
        """
        Generates random parameters for the ChebRx feature map.

        Args:
            seed (Union[int,None]): Seed for the random number generator (default: None)

        Return:
            The randomly generated parameters
        """
        param = super().generate_initial_parameters(seed)

        if len(param) > 0:
            index = self.get_cheb_indices(False)
            p = np.linspace(0.01, self.alpha, self.num_qubits)
            for i in index:
                param[i] = p

        return param

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the ChebRx feature map

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["num_layers"] = self.num_layers
        params["closed"] = self.closed
        return params

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
                elif self.num_qubits % 2 == 1:
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
