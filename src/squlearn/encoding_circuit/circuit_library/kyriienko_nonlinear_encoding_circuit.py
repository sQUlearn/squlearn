import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit

from ..encoding_circuit_base import EncodingCircuitBase
from ..layered_encoding_circuit import LayeredEncodingCircuit, Layer


class KyriienkoEncodingCircuit(EncodingCircuitBase):
    r"""
    Collection of encoding circuits introduced by Kyriienko et al. in reference [1].

    The following circuits are implemented:

    * ``chebyshev_tower`` encoding (Eq. 15),
    * ``chebyshev_sparse`` encoding (Eq. 14)
    * ``chebyshev_product`` encoding (Eq. 5).

    Each encoding circuit is followed by a variational circuit as defined in reference [1],
    RZ-RX-RZ layers followed by entangling layers. Two arrangements are possible:

    * ``HEA``: Hardware Efficient Ansatz, with consecutive entangling layers
      (See Figure 5a or Section IIIB in [1])
    * ``ABA``: Alternating Block Ansatz with consecutive shifted entangling layers in
      each block (See Figure 5b or Section IIIB in [1])

    **Example: 4 qubits, a 2 dimensional feature vector, 1 encoding layer, 2 variational layers, variational arrangement ABA and Chebyshev tower encoding:**

    .. plot::

        from squlearn.encoding_circuit import KyriienkoEncodingCircuit
        pqc = KyriienkoEncodingCircuit(4, 1, num_encoding_layers=1, num_variational_layers=2,
                                       variational_arrangement="ABA")
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})


    Args:
        num_qubits (int): Number of qubits of the encoding circuit
        encoding_style (str): Style of the encoding. Options are ``'chebyshev_tower'`` (default),
                              ``'chebyshev_sparse'`` and ``'chebyshev_product'``
                              (see reference [1], Equation  15, 14 and 5 respectively)
        variational_arrangement (str): Arrangement of the variational layers. Options are
                                       ``'HEA'`` (default) and ``'ABA'`` (see reference [1],
                                       section IIIB)
        num_encoding_layers (int): Number of encoding layers (default: 1)
        num_variational_layers (int): Number of variational layers (default: 1)
        rotation_gate (str): Rotation gate to use. Either ``'rx'``, ``'ry'`` or
                             ``'rz'`` (default: ``'ry'`` as in reference [1])
        num_features (int): Dimension of the feature vector (default: 1)
        block_width (int): Only necessary for arrangement ``'ABA'``. Width (vertical) of each
                           blocks for the ABA arrangement (default: 2), also refered as Nb in
                           the paper. Must be a divisor of the number of qubits
        block_depth (int): Only necessary for arrangement ``'ABA'``. Depth (horizontal) of each
                           blocks for the ABA arrangement (default: 1), also refered as b in
                           the paper.

    References
    ----------
    [1]: O. Kyriienko et al., "Solving nonlinear differential equations with differentiable
    quantum circuits", `arXiv:2011.10395 (2021). <https://arxiv.org/pdf/2011.10395>`_
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int = 1,
        encoding_style: str = "chebyshev_tower",
        variational_arrangement: str = "HEA",
        num_encoding_layers: int = 1,
        num_variational_layers: int = 1,
        rotation_gate: str = "ry",
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
        self.tower = True if encoding_style == "chebyshev_tower" else False

        if self.variational_arrangement not in ("HEA", "ABA"):
            raise ValueError("Arrangement must be either 'HEA' or 'ABA'")

        if self.num_qubits < 2:
            raise ValueError("Variational circuit HEE_rzrxrz requires at least two qubits.")

        if self.rotation_gate not in ("rx", "ry", "rz"):
            raise ValueError("Rotation gate must be either 'rx', 'ry' or 'rz'")

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the HEE_rzrxrz encoding circuit."""
        return np.array([[-2.0 * np.pi, 2.0 * np.pi]] * self.num_parameters)

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the variational circuit."""
        if self.variational_arrangement == "HEA":
            num_param = 3 * self.num_qubits * self.num_variational_layers
        elif self.variational_arrangement == "ABA":
            num_param = 3 * self.num_qubits * self.block_depth * self.num_variational_layers
        return num_param

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Kyriienko encoding circuit

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["num_encoding_layers"] = self.num_encoding_layers
        params["num_variational_layers"] = self.num_variational_layers
        params["variational_arrangement"] = self.variational_arrangement
        params["encoding_style"] = self.encoding_style
        params["block_width"] = self.block_width
        params["block_depth"] = self.block_depth
        params["rotation_gate"] = self.rotation_gate

        return params

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

        def mapping(x, i):
            """Non-linear mapping for x: alpha*i*arccos(x)"""
            return self.alpha * i * np.arccos(x)

        def variational_gate_block(
            QC,
            num_layers,
            num_qubits,
            shift_parameter=0,
            index_offset=0,
            variational_arrangement="HEA",
        ):
            """
            Implements an Encoding circuit with layers of RZRXRZ followed by entangling layers.

            """
            qubit_starting_index = shift_parameter
            num_qubits += shift_parameter
            for layer in range(num_layers):
                for i in range(qubit_starting_index, num_qubits):
                    if i >= self.num_qubits:
                        i -= self.num_qubits
                    QC.rz(parameters[index_offset], i)
                    QC.rx(parameters[index_offset + 1], i)
                    QC.rz(parameters[index_offset + 2], i)
                    index_offset += 3
                if variational_arrangement == "HEA":
                    for start in (0, 1):
                        for i in range(start, num_qubits - 1, 2):
                            QC.cx(i, i + 1)

                elif variational_arrangement == "ABA":
                    for start in (qubit_starting_index, qubit_starting_index + 1):
                        for i in range(start, num_qubits - 1, 2):
                            if i + 1 < self.num_qubits:
                                QC.cx(i, i + 1)
                            else:
                                QC.cx(i, 0)
            return QC, index_offset

        nfeature = len(features)

        if self.encoding_style == "chebyshev_product":
            QC = LayeredEncodingCircuit(num_qubits=self.num_qubits, num_features=self.num_features)
            layer = Layer(QC)
            {"rx": layer.Rx, "ry": layer.Ry, "rz": layer.Rz}[self.rotation_gate](
                "x", encoding=np.arcsin
            )
            QC.add_layer(layer, num_layers=self.num_encoding_layers)

            QC = QC.get_circuit(features, [])
        elif self.encoding_style in ("chebyshev_tower", "chebyshev_sparse"):
            QC = QuantumCircuit(self.num_qubits)
            for layer in range(self.num_encoding_layers):
                index_offset_encoding = 0
                iqubit = 0
                icheb = 1

                inner = self.num_features
                outer = self.num_chebyshev

                for outer_ in range(outer):
                    for inner_ in range(inner):
                        {"rx": QC.rx, "ry": QC.ry, "rz": QC.rz}[self.rotation_gate](
                            mapping(features[index_offset_encoding % nfeature], icheb),
                            iqubit % self.num_qubits,
                        )
                        iqubit += 1
                        index_offset_encoding += 1
                    icheb = 1 + icheb if self.tower else 1

        index_offset_variational = 0
        if self.variational_arrangement == "HEA":
            QC = variational_gate_block(QC, self.num_variational_layers, self.num_qubits)[0]
        elif self.variational_arrangement == "ABA":
            if self.num_qubits % self.block_width != 0:
                raise ValueError(
                    f"block_width =  {self.block_width} must be a divisor of the num_qubits = {self.num_qubits}."
                )

            number_of_blocks = int(np.ceil(self.num_qubits / self.block_width))  # vertical blocks
            shifting_factor = np.floor(self.block_width / 2)
            for layer in range(self.num_variational_layers):
                if layer % 2 == 0:  # even layer
                    for block in range(number_of_blocks):
                        QC, index_offset_variational = variational_gate_block(
                            QC,
                            self.block_depth,
                            self.block_width,
                            int(block * self.block_width),
                            index_offset_variational,
                            variational_arrangement="ABA",
                        )
                else:
                    for block in range(number_of_blocks):
                        if (
                            shifting_factor + block * self.block_width + self.block_width
                            <= self.num_qubits
                        ):
                            QC, index_offset_variational = variational_gate_block(
                                QC,
                                self.block_depth,
                                self.block_width,
                                int(shifting_factor + block * self.block_width),
                                index_offset_variational,
                                variational_arrangement="ABA",
                            )
                        else:  # if the block is out of the range of the qubits, we need to wrap around
                            QC, index_offset_variational = variational_gate_block(
                                QC,
                                self.block_depth,
                                self.block_width,
                                int(shifting_factor + block * self.block_width),
                                index_offset_variational,
                                variational_arrangement="ABA",
                            )
                QC.barrier()

        return QC
