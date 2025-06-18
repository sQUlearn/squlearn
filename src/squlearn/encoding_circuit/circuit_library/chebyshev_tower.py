from __future__ import annotations

import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit

from squlearn.util.data_preprocessing import extract_num_features

from ..encoding_circuit_base import EncodingCircuitBase


class ChebyshevTower(EncodingCircuitBase):
    r"""
    A feature-map that is based on the Chebyshev Tower encoding.

    **Example for 4 qubits, a 2 dimensional feature vector, 2 Chebyshev terms per feature,
    and 2 layers:**

    .. plot::

        from squlearn.encoding_circuit import ChebyshevTower
        pqc = ChebyshevTower(4, 2, num_layers=2)
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10}, num_features=2)
        plt.tight_layout()

    The encoding gate and the scaling factor can be adjusted by parameters.
    It is also possible to change the indexing of the features.

    Args:
        num_qubits (int): Number of qubits of the ChebyshevTower encoding circuit
        num_chebyshev (int): Number of Chebyshev tower terms per feature dimension
        num_features (int): Dimension of the feature vector (default: None)
        alpha (float): Scaling factor of Chebyshev tower
        num_layers (int): Number of layers
        rotation_gate (str): Rotation gate to use. Either ``rx``, ``ry`` or ``rz`` (default: ``ry``)
        hadamard_start (bool): If true, the circuit starts with a layer of Hadamard gates
                               (default: True)
        arrangement (str): Arrangement of the layers, either ``block`` or ``alternating``.
                          ``block``: The features are stacked together, ``alternating``:
                          The features are placed alternately (default: ``block``).
        nonlinearity (str): Mapping function to use for the feature encoding. Either ``arccos``
                            or ``arctan`` (default: ``arccos``)
    """

    def __init__(
        self,
        num_qubits: int,
        num_chebyshev: int,
        num_features: int = None,
        alpha: float = 1.0,
        num_layers: int = 1,
        rotation_gate: str = "ry",
        hadamard_start: bool = True,
        arrangement: str = "block",
        nonlinearity: str = "arccos",
    ) -> None:
        super().__init__(num_qubits, num_features)

        self.num_chebyshev = num_chebyshev
        self.alpha = alpha
        self.num_layers = num_layers
        self.rotation_gate = rotation_gate
        self.hadamard_start = hadamard_start
        self.arrangement = arrangement
        self.nonlinearity = nonlinearity

        if self.rotation_gate not in ("rx", "ry", "rz"):
            raise ValueError("Rotation gate must be either 'rx', 'ry' or 'rz'")

        if self.arrangement not in ("block", "alternating"):
            raise ValueError("Arrangement must be either 'block' or 'alternating'")

        if self.nonlinearity not in ("arccos", "arctan"):
            raise ValueError(
                f"Unknown value for nonlinearity: {self.nonlinearity}."
                " Possible values are 'arccos' and 'arctan'"
            )

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the Chebyshev Tower encoding (equal 0 here)."""
        return 0

    @property
    def feature_bounds(self) -> np.ndarray:
        """
        The bounds of the features of the ChebyshevTower encoding circuit.

        To get the bounds for a specific number of features, use get_feature_bounds().
        """
        if self.nonlinearity == "arccos":
            return np.array([-1.0, 1.0])
        elif self.nonlinearity == "arctan":
            return np.array([-np.inf, np.inf])

    @property
    def num_encoding_slots(self) -> int:
        """The number of encoding slots of the Chebyshev Tower encoding."""
        return self.num_qubits * self.num_layers

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Chebyshev Tower encoding

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["num_chebyshev"] = self.num_chebyshev
        params["alpha"] = self.alpha
        params["num_layers"] = self.num_layers
        params["rotation_gate"] = self.rotation_gate
        params["hadamard_start"] = self.hadamard_start
        params["arrangement"] = self.arrangement
        params["nonlinearity"] = self.nonlinearity
        return params

    def set_params(self, **kwargs) -> ChebyshevTower:
        """
        Sets value of the encoding circuit hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``.
        """
        if "nonlinearity" in kwargs and kwargs["nonlinearity"] not in (
            "arccos",
            "arctan",
        ):
            raise ValueError(
                f"Unknown value for nonlinearity: {kwargs['nonlinearity']}."
                " Possible values are 'arccos' and 'arctan'"
            )
        return super().set_params(**kwargs)

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray] = None,
    ) -> QuantumCircuit:
        """
        Generates and returns the circuit of the Chebyshev encoding circuit.

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained

        Return:
            Returns the circuit in Qiskit's QuantumCircuit format
        """

        if self.rotation_gate not in ("rx", "ry", "rz"):
            raise ValueError("Rotation gate must be either 'rx', 'ry' or 'rz'")

        if self.arrangement not in ("block", "alternating"):
            raise ValueError("Arrangement must be either 'block' or 'alternating'")

        num_features = extract_num_features(features)
        self._check_feature_encoding_slots(num_features, self.num_encoding_slots)
        self._check_feature_consistency(features)

        def entangle_layer(QC: QuantumCircuit):
            """Creation of a simple NN entangling layer"""
            for i in range(0, self.num_qubits - 1, 2):
                QC.cx(i, i + 1)
            for i in range(1, self.num_qubits - 1, 2):
                QC.cx(i, i + 1)
            return QC

        if self.nonlinearity == "arccos":

            def mapping(x, i):
                """Non-linear mapping for x: alpha*i*arccos(x)"""
                return self.alpha * i * np.arccos(x)

        elif self.nonlinearity == "arctan":

            def mapping(x, i):
                """Non-linear mapping for x: alpha*i*arctan(x)"""
                return self.alpha * i * np.arctan(x)

        QC = QuantumCircuit(self.num_qubits)

        if self.hadamard_start:
            QC.h(range(self.num_qubits))

        for layer in range(self.num_layers):
            index_offset = 0
            iqubit = 0
            icheb = 1
            # Loops through the data encoding gates
            if self.arrangement == "block":
                outer = num_features
                inner = self.num_chebyshev
            elif self.arrangement == "alternating":
                inner = num_features
                outer = self.num_chebyshev
            else:
                raise ValueError("Arrangement must be either 'block' or 'alternating'")

            for outer_ in range(outer):
                for inner_ in range(inner):
                    if self.rotation_gate.lower() == "rx":
                        QC.rx(
                            mapping(features[index_offset % num_features], icheb),
                            iqubit % self.num_qubits,
                        )
                    elif self.rotation_gate.lower() == "ry":
                        QC.ry(
                            mapping(features[index_offset % num_features], icheb),
                            iqubit % self.num_qubits,
                        )
                    elif self.rotation_gate.lower() == "rz":
                        QC.rz(
                            mapping(features[index_offset % num_features], icheb),
                            iqubit % self.num_qubits,
                        )
                    else:
                        raise ValueError(
                            "Rotation gate {} not supported".format(self.rotation_gate)
                        )
                    iqubit += 1
                    if self.arrangement == "block":
                        icheb += 1
                    elif self.arrangement == "alternating":
                        index_offset += 1

                if self.arrangement == "block":
                    index_offset += 1
                    icheb = 1
                elif self.arrangement == "alternating":
                    icheb += 1

            # Entangling layer
            if layer + 1 < self.num_layers:
                QC = entangle_layer(QC)

        return QC
