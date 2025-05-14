from __future__ import annotations

import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit

from squlearn.util.data_preprocessing import extract_num_features

from ..encoding_circuit_base import EncodingCircuitBase


class ChebyshevRx(EncodingCircuitBase):
    """
    Simple Chebyshev encoding circuit build from  Rx gates

    **Example for 4 qubits, a 2 dimensional feature vector and 2 layers:**

    .. plot::

        from squlearn.encoding_circuit import ChebyshevRx
        pqc = ChebyshevRx(4, 2)
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10}, num_features=2)
        plt.tight_layout()

    Args:
        num_qubits (int): Number of qubits of the ChebyshevRx encoding circuit
        num_layers (int): Number of layers (default: 1)
        num_features (int): Dimension of the feature vector (default: None)
        closed (bool): If true, the last and the first qubit are entangled (default: false)
        alpha (float): Maximum value of the Chebyshev Tower initial parameters, i.e. parameters
                       that appear in the arccos encoding. (default: 4.0)
        nonlinearity (str): Mapping function to use for the feature encoding. Either ``arccos``
                            or ``arctan`` (default: ``arccos``)
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 1,
        num_features: int = None,
        closed: bool = False,
        alpha: float = 4.0,
        nonlinearity: str = "arccos",
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.num_layers = num_layers
        self.closed = closed
        self.alpha = alpha
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ("arccos", "arctan"):
            raise ValueError(
                f"Unknown value for nonlinearity: {self.nonlinearity}."
                " Possible values are 'arccos' and 'arctan'"
            )

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the ChebyshevRx encoding circuit."""
        return 2 * self.num_qubits * self.num_layers

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the ChebyshevRx encoding circuit."""
        bounds = np.zeros((self.num_parameters, 2))
        index_offset = 0
        for layer in range(self.num_layers):
            # Chebyshev encoding circuit
            for i in range(self.num_qubits):
                bounds[index_offset] = [0.0, self.alpha]
                index_offset += 1
            # Trafo
            for i in range(self.num_qubits):
                bounds[index_offset] = [-np.pi, np.pi]
                index_offset += 1
        return bounds

    def generate_initial_parameters(
        self, num_features: int, seed: Union[int, None] = None
    ) -> np.ndarray:
        """
        Generates random parameters for the ChebyshevPQC encoding circuit

        Args:
            num_features (int): Number of features of the input data
            seed (Union[int,None]): Seed for the random number generator (default: None)

        Return:
            The randomly generated parameters
        """
        param = super().generate_initial_parameters(num_features, seed)

        if len(param) > 0:
            index = self.get_cheb_indices(False)
            features_per_qubit = int(np.ceil(self.num_qubits / num_features))
            p = np.linspace(0.01, self.alpha, features_per_qubit)

            for index2 in index:
                for i, ii in enumerate(index2):
                    param[ii] = p[i % features_per_qubit]

        return param

    @property
    def feature_bounds(self) -> np.ndarray:
        """
        The bounds of the features of the ChebyshevPQC encoding circuit.

        To get the bounds for a specific number of features, use get_feature_bounds().
        """
        if self.nonlinearity == "arccos":
            return np.array([-1.0, 1.0])
        elif self.nonlinearity == "arctan":
            return np.array([-np.inf, np.inf])

    @property
    def num_encoding_slots(self) -> int:
        """The number of encoding slots of the ChebyshevRx encoding circuit."""
        return self.num_layers * self.num_qubits

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the ChebyshevRx encoding circuit

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["num_layers"] = self.num_layers
        params["closed"] = self.closed
        params["alpha"] = self.alpha
        params["nonlinearity"] = self.nonlinearity
        return params

    def set_params(self, **kwargs) -> ChebyshevRx:
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

    def get_cheb_indices(self, flatten: bool = True):
        """
        Function that returns the indices of the parameters involved in the Chebyshev encoding.

        Args:
            flatten (bool): If true, the indices are returned as a flat list, otherwise
                            as a list of lists, where the outer list corresponds to the layers
                            (default: True)
        """
        cheb_index = []
        index_offset = 0
        for layer in range(self.num_layers):
            cheb_index_layer = []
            for i in range(self.num_qubits):
                cheb_index_layer.append(index_offset)
                index_offset += 1

            for i in range(self.num_qubits):
                index_offset += 1

            if flatten:
                cheb_index += cheb_index_layer
            else:
                cheb_index.append(cheb_index_layer)
        return cheb_index

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the ChebyshevRx encoding circuit

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the circuit in Qiskit's QuantumCircuit format
        """
        num_features = extract_num_features(features)
        num_params = len(parameters)
        self._check_feature_encoding_slots(num_features, self.num_encoding_slots)
        self._check_feature_consistency(features)

        def entangle_layer(QC: QuantumCircuit) -> QuantumCircuit:
            """Creation of a simple nearest neighbor entangling layer"""
            for i in range(0, self.num_qubits + self.closed - 1, 2):
                QC.cx(i, (i + 1) % self.num_qubits)

            if self.num_qubits > 2:
                for i in range(1, self.num_qubits + self.closed - 1, 2):
                    QC.cx(i, (i + 1) % self.num_qubits)

            return QC

        if self.nonlinearity == "arccos":

            def mapping(a, x):
                """Helper function for returning a*arccos(x)"""
                return a * np.arccos(x)

        elif self.nonlinearity == "arctan":

            def mapping(a, x):
                """Helper function for returning a*arctan(x)"""
                return a * np.arctan(x)

        QC = QuantumCircuit(self.num_qubits)
        index_offset = 0
        feature_offset = 0
        for _ in range(self.num_layers):
            # Chebyshev encoding circuit
            for i in range(self.num_qubits):
                QC.rx(
                    mapping(
                        parameters[index_offset % num_params],
                        features[feature_offset % num_features],
                    ),
                    i,
                )
                index_offset += 1
                feature_offset += 1
            # Trafo
            for i in range(self.num_qubits):
                QC.rx(parameters[index_offset % num_params], i)
                index_offset += 1
            QC = entangle_layer(QC)

        return QC
