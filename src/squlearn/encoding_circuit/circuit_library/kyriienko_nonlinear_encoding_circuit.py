import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..encoding_circuit_base import EncodingCircuitBase
from ..circuit_library.chebyshev_tower import ChebyshevTower
from ..circuit_library.hardware_efficient_rzrxrz import HEE_rzrxrz
from ..layered_encoding_circuit import LayeredEncodingCircuit, Layer


class KyriienkoEncodingCircuit(EncodingCircuitBase):
    r"""
    Collection of encoding circuits introduced by Kyriienko et al. [1], to solve differential equations.
    The encoding circuits are build from nonlinear encoding circuits that map the function of interest to a quantum state and variational circuits that are used to solve the differential equation.


    **Example for 4 qubits, a 2 dimensional feature vector, 1 encoding layer, 1 variational layer, variational arrangement ABA and Chebyshev tower encoding:**

    .. plot::

        from squlearn.encoding_circuit import KyriienkoEncodingCircuit
        pqc = KyriienkoEncodingCircuit(4, num_features=2, num_encoding_layers=1, num_variational_layers=1, variational_arrangement="ABA")
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})



    The encoding gate and the scaling factor can be adjusted by parameters.
    It is also possible to change the indexing of the features.

    Args:
        num_qubits (int): Number of qubits of the encoding circuit
        encoding_style (str): Style of the encoding. Options are "chebyshev_tower" (default) and "chebyshev_sparse" (see reference [1], Equation 14 and 15)
        variational_arrangement (str): Arrangement of the variational layers. Options are "HEA" (default) and "ABA" (see reference [1], section IIIB)
        num_encoding_layers (int): Number of encoding layers (default: 1)
        num_variational_layers (int): Number of variational layers (default: 1)
        rotation_gate (str): Rotation gate to use. Either ``rx``, ``ry`` or ``rz`` (default: ``ry``)
        num_features (int): Dimension of the feature vector (default: 1)
        block_width (int): Only necessary for arrangement `ABA`. Width (vertical) of each blocks for the ABA arrangement (default: 2), also refered as Nb in the paper. Must be a divisor of the number of qubits
        block_depth (int): Only necessary for arrangement `ABA`. Depth (horizontal) of each blocks for the ABA arrangement (default: 1), also refered as b in the paper.

    References
    ----------
    [1]: O. Kyriienko et al., "Solving nonlinear differential equations with differentiable quantum circuits",
    `arXiv:2011.10395 (2021). <https://arxiv.org/pdf/2011.10395>`_
    """

    def __init__(
        self,
        num_qubits: int,
        encoding_style: str = "chebyshev_tower",
        variational_arrangement: str = "HEA",
        num_encoding_layers: int = 1,
        num_variational_layers: int = 1,
        rotation_gate: str = "ry",
        num_features: int = 1,
        block_width: int = 2,
        block_depth: int = 1,
    ) -> None:
        super().__init__(num_qubits, num_features)

        self.num_encoding_layers = num_encoding_layers
        self.num_variational_layers = num_variational_layers
        self.variational_arrangement = variational_arrangement
        self.encoding_style = encoding_style
        self.block_width = block_width
        self.block_depth = block_depth
        self.alpha = 2.0
        self.rotation_gate = rotation_gate
        self.num_chebyshev = self.num_qubits
        self.tower = True

        if self.variational_arrangement not in ("HEA", "ABA"):
            raise ValueError("Arrangement must be either 'HEA' or 'ABA'")

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the variational circuit."""
        return self._get_EncodingCircuitBase().num_parameters

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Chebyshev Tower encoding

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """

        return self._get_EncodingCircuitBase().get_params()

    def _get_EncodingCircuitBase(self) -> EncodingCircuitBase:
        """
        Generates and returns the circuit of the Kyriienko encoding circuit

        Args:
            features (Union[ParameterVector, np.ndarray]): The features to encode
            parameters (Union[ParameterVector, np.ndarray]): The parameters of the encoding circuit

        Returns:
            QuantumCircuit: The encoding circuit
        """
        if self.encoding_style in ("chebyshev_tower", "chebyshev_sparse"):
            if self.encoding_style == "chebyshev_sparse":
                self.tower = False
            circuit = ChebyshevTower(
                num_qubits=self.num_qubits,
                num_features=self.num_features,
                num_chebyshev=self.num_chebyshev,  # num_chebyshev = num_qubits for chebyshev_tower
                alpha=self.alpha,
                hadamard_start=False,
                rotation_gate=self.rotation_gate,
                num_layers=self.num_encoding_layers,
                tower=self.tower,
            )
        elif self.encoding_style == "chebyshev_product":
            circuit = LayeredEncodingCircuit(
                num_qubits=self.num_qubits, num_features=self.num_features
            )
            layer = Layer(circuit)
            {"rx": layer.Rx, "ry": layer.Ry, "rz": layer.Rz}[self.rotation_gate](
                "x", encoding=np.arcsin
            )
            circuit.add_layer(layer, num_layers=self.num_encoding_layers)

        circuit += HEE_rzrxrz(
            self.num_qubits,
            self.num_variational_layers,
            self.variational_arrangement,
            self.block_width,
            self.block_depth,
        )

        return circuit

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Generates and returns the circuit of the Kyriienko encoding circuit

        Args:
            features (Union[ParameterVector, np.ndarray]): The features to encode
            parameters (Union[ParameterVector, np.ndarray]): The parameters of the encoding circuit

        Returns:
            QuantumCircuit: The encoding circuit
        """

        return self._get_EncodingCircuitBase().get_circuit(features, parameters)
