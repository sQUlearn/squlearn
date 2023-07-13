import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit


class FeatureMapBase:
    """
    Feature map base class

    Args:
        num_qubits (int): Number of Qubits of the feature map
        num_features (int): Dimension of the feature vector
    """

    def __init__(self, num_qubits: int, num_features: int) -> None:
        self._num_qubits = num_qubits
        self._num_features = num_features

    @property
    def num_qubits(self) -> int:
        """The number of qubits of the feature map."""
        return self._num_qubits

    @property
    def num_features(self) -> int:
        """The dimension of the features in the feature map."""
        return self._num_features

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the feature map."""
        return 0

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Return the circuit feature map (has to be overwritten, otherwise a NotImplementedError is thrown)

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the circuit in qiskit QuantumCircuit format
        """

        raise NotImplementedError()

    def draw(
        self,
        feature_label: str = "x",
        parameter_label: str = "p",
        decompose: bool = False,
        **kwargs,
    ) -> None:
        """
        Draws the feature map circuit using the QuantumCircuit.draw() function.

        Args:
            feature_label (str): Label for the feature vector (default:"x").
            parameter_label (str): Label for the parameter vector (default:"p").
            decompose (bool): If True, the circuit is decomposed before printing (default: False).
            kwargs: Additional arguments from Qiskit's QuantumCircuit.draw() function.

        Returns:
            Returns the circuit in qiskit QuantumCircuit.draw() format
        """

        feature_vec = ParameterVector(feature_label, self.num_features)
        parameters_vec = ParameterVector(parameter_label, self.num_parameters)

        circ = self.get_circuit(feature_vec, parameters_vec)
        if decompose:
            circ = circ.decompose()

        return circ.draw(**kwargs)

    def __mul__(self, x):
        return self.__add__(x)

    def __add__(self, x):
        """
        Overwrites the a + b function, such that the addition of
        feature maps returns the composition of both feature maps.

        Number of qubits and features have to be equal in both feature maps!
        The special function and properties of the both feature maps are lost
        by this composition.

        Args:
            self (FeatureMapBase): right / first feature map
            x (FeatureMapBase): left / second feature map

        Returns:
            Returns the composed feature map as special class ComposedFeatureMap
        """

        if not isinstance(x, FeatureMapBase):
            raise ValueError("Only the addition with other feature maps is allowed!")

        class ComposedFeatureMap(FeatureMapBase):
            """
            Special class for composed feature maps.

            Args:
                fm1 (FeatureMapBase): right / first feature map
                fm2 (FeatureMapBase): left / second feature map
            """

            def __init__(self, fm1: FeatureMapBase, fm2: FeatureMapBase):
                if fm1.num_qubits != fm2.num_qubits:
                    raise ValueError("Number of qubits is not equal in both feature maps.")

                super().__init__(fm1.num_qubits, max(fm1.num_features, fm2.num_features))

                self._fm1 = fm1
                self._fm2 = fm2

            @property
            def num_parameters(self) -> int:
                """Returns the number of trainable parameters of composed feature map.
                Is equal to the sum of both trainable parameters
                """
                return self._fm1.num_parameters + self._fm2.num_parameters

            def get_circuit(
                self,
                features: Union[ParameterVector, np.ndarray],
                parameters: Union[ParameterVector, np.ndarray],
            ) -> QuantumCircuit:
                """
                Returns the circuit of the composed feature maps

                Args:
                    features Union[ParameterVector,np.ndarray]: Input vector of the features
                        from which the gate inputs are obtained
                    param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                        from which the gate inputs are obtained

                Return:
                    Returns the circuit of the composed feature maps in qiskit QuantumCircuit format
                """

                circ1 = self._fm1.get_circuit(
                    features[: self._fm1.num_features], parameters[: self._fm1.num_parameters]
                )
                circ2 = self._fm2.get_circuit(
                    features[: self._fm2.num_features], parameters[self._fm1.num_parameters :]
                )

                return circ1.compose(circ2, range(self._fm1.num_qubits))

        return ComposedFeatureMap(self, x)
