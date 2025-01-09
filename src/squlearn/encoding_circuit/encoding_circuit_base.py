from __future__ import annotations
from abc import ABC, abstractmethod
import warnings

import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit


class EncodingCircuitBase(ABC):
    """
    Encoding circuit base class

    Args:
        num_qubits (int): Number of Qubits of the encoding circuit
        num_features (int): Dimension of the feature vector
    """

    def __init__(self, num_qubits: int, num_features: int = None) -> None:
        self._num_qubits = num_qubits
        self._num_features = num_features

    @property
    def num_qubits(self) -> int:
        """The number of qubits of the encoding circuit."""
        return self._num_qubits

    @property
    def num_features(self) -> int:
        """The dimension of the features in the encoding circuit."""
        return self._num_features

    @num_features.setter
    def num_features(self, value: int):
        self._num_features = value

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the encoding circuit."""
        return 0

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the encoding circuit.

        Default bounds are [-pi,pi] for each parameter.
        """
        return np.array([[-np.pi, np.pi]] * self.num_parameters)

    @property
    def feature_bounds(self) -> np.ndarray:
        """The bounds of the features of the encoding circuit.

        Default bounds are [-pi,pi] for each feature.
        """
        return np.array([[-np.pi, np.pi]] * self.num_features)

    @property
    def num_encoding_slots(self) -> int:
        """The number of encoding slots of the encoding circuit."""
        raise NotImplementedError()

    def generate_initial_parameters(self, seed: Union[int, None] = None) -> np.ndarray:
        """
        Generates random parameters for the encoding circuit

        Args:
            seed (Union[int,None]): Seed for the random number generator (default: None)

        Return:
            The randomly generated parameters
        """
        if self.num_parameters == 0:
            return np.array([])
        r = np.random.RandomState(seed)
        bounds = self.parameter_bounds
        return r.uniform(low=bounds[:, 0], high=bounds[:, 1])

    @abstractmethod
    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Return the encoding circuit and check the matching of the encoding slots with the provided features (has to be overwritten, otherwise a NotImplementedError is thrown)

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
        output: str = None,
        feature_label: str = "x",
        parameter_label: str = "p",
        decompose: bool = False,
        **kwargs,
    ) -> None:
        """
        Draws the encoding circuit circuit using the QuantumCircuit.draw() function.

        Args:
            feature_label (str): Label for the feature vector (default:"x").
            parameter_label (str): Label for the parameter vector (default:"p").
            decompose (bool): If True, the circuit is decomposed before printing (default: False).
            kwargs: Additional arguments from Qiskit's QuantumCircuit.draw() function.

        Returns:
            Returns the circuit in qiskit QuantumCircuit.draw() format
        """

        if self.num_features is None:
            warnings.warn(
                f"`num_features` is not set. Falling back to `num_encoding_slots` ({self.num_encoding_slots}).",
                UserWarning,
            )
            # set the number of features to the number of encoding slots
            num_features = self.num_encoding_slots
        else:
            num_features = self.num_features

        feature_vec = ParameterVector(feature_label, num_features)
        parameters_vec = ParameterVector(parameter_label, self.num_parameters)

        circ = self.get_circuit(feature_vec, parameters_vec)
        if decompose:
            circ = circ.decompose()

        return circ.draw(output, **kwargs)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the encoding circuit.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        param = {}
        param["num_qubits"] = self._num_qubits
        param["num_features"] = self._num_features
        return param

    def set_params(self, **params) -> EncodingCircuitBase:
        """
        Sets value of the encoding circuit hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )
            try:
                setattr(self, key, value)
            except:
                setattr(self, "_" + key, value)

        return self

    def _check_feature_encoding_slots(
        self, features: Union[ParameterVector, np.ndarray], num_encoding_slots: int
    ) -> None:
        """
        Checks if the number of features fits the available encoding slots.

        Args:
            features (Union[ParameterVector, np.ndarray]): The input features.
            num_encoding_slots (int): The number of available encoding slots.

        Raises:
            EncodingSlotsMismatchError: If the number of features exceeds the number of encoding slots.
        """
        num_features = len(features)

        if num_features > num_encoding_slots:
            raise EncodingSlotsMismatchError(num_encoding_slots, num_features)

    def __mul__(self, x):
        return self.__add__(x)

    def __add__(self, x):
        """
        Overwrites the a + b function, such that the addition of
        encoding circuits returns the composition of both encoding circuits.

        Number of qubits and features have to be equal in both encoding circuits!
        The special function and properties of the both encoding circuits are lost
        by this composition.

        Args:
            self (EncodingCircuitBase): right / first encoding circuit
            x (EncodingCircuitBase): left / second encoding circuit

        Returns:
            Returns the composed encoding circuit as special class ComposedEncodingCircuit
        """

        if not isinstance(x, EncodingCircuitBase):
            raise ValueError("Only the addition with other encoding circuits is allowed!")

        class ComposedEncodingCircuit(EncodingCircuitBase):
            """
            Special class for composed encoding circuits.

            Args:
                num_qubits: num qubits for both encoding circuits (necessary for scikit-learn interface)
                ec1 (EncodingCircuitBase): right / first encoding circuit
                ec2 (EncodingCircuitBase): left / second encoding circuit
            """

            def __init__(
                self, num_qubits: int, ec1: EncodingCircuitBase, ec2: EncodingCircuitBase
            ):

                if ec1.num_qubits != num_qubits:
                    ec1.set_params(num_qubits=num_qubits)
                if ec2.num_qubits != num_qubits:
                    ec2.set_params(num_qubits=num_qubits)

                super().__init__(ec1.num_qubits, max(ec1.num_features, ec2.num_features))

                self.ec1 = ec1
                self.ec2 = ec2

            @classmethod
            def create_from_encoding_circuits(
                cls, ec1: EncodingCircuitBase, ec2: EncodingCircuitBase
            ):
                """
                Create a composed encoding circuit from two encoding circuits.

                Args:
                    ec1 (EncodingCircuitBase): right / first encoding circuit
                    ec2 (EncodingCircuitBase): left / second encoding circuit

                Returns:
                    ComposedEncodingCircuit: Composed encoding circuit
                """
                if ec1.num_qubits != ec2.num_qubits:
                    raise ValueError("Number of qubits is not equal in both encoding circuits.")

                return cls(ec1.num_qubits, ec1, ec2)

            @property
            def num_parameters(self) -> int:
                """Returns the number of trainable parameters of composed encoding circuit.

                Is equal to the sum of both trainable parameters.
                """
                return self.ec1.num_parameters + self.ec2.num_parameters

            @property
            def parameter_bounds(self) -> np.ndarray:
                """Returns the bounds of the trainable parameters of composed encoding circuit.

                Is equal to the sum of both bounds.
                """
                return np.concatenate(
                    (self.ec1.parameter_bounds, self.ec2.parameter_bounds), axis=0
                )

            @property
            def feature_bounds(self) -> np.ndarray:
                """Returns the bounds of the features of composed encoding circuit.

                Is equal to the maximum and minimum of both bounds.
                """

                feature_bounds1 = self.ec1.feature_bounds
                feature_bounds2 = self.ec2.feature_bounds
                feature_bounds_values = np.zeros((self.num_features, 2))

                min_num_feature = min(self.ec1.num_features, self.ec2.num_features)

                if self.ec1.num_features == self.num_features:
                    feature_bounds_values = self.ec1.feature_bounds

                if self.ec2.num_features == self.num_features:
                    feature_bounds_values = self.ec2.feature_bounds

                for i in range(min_num_feature):
                    feature_bounds_values[i, 0] = min(feature_bounds1[i, 0], feature_bounds2[i, 0])
                    feature_bounds_values[i, 1] = max(feature_bounds1[i, 1], feature_bounds2[i, 1])

                return feature_bounds_values

            def generate_initial_parameters(self, seed: Union[int, None] = None) -> np.ndarray:
                """
                Generates random parameters for the composed encoding circuit

                Args:
                    seed (Union[int,None]): Seed for the random number generator

                Return:
                    Returns the randomly generated parameters
                """

                return np.concatenate(
                    (
                        self.ec1.generate_initial_parameters(seed),
                        self.ec2.generate_initial_parameters(seed),
                    ),
                    axis=0,
                )

            def get_params(self, deep: bool = True) -> dict:
                """
                Returns hyper-parameters and their values of the composed encoding circuit.

                Hyper-parameter names are prefixed by ``ec1__`` or ``ec2__`` depending on
                which encoding circuit they belong to.

                Args:
                    deep (bool): If True, also the parameters for
                                 contained objects are returned (default=True).

                Return:
                    Dictionary with hyper-parameters and values.
                """
                params = dict(ec1=self.ec1, ec2=self.ec2)
                if deep:
                    deep_items = self.ec1.get_params().items()
                    for k, val in deep_items:
                        if k != "num_qubits":
                            params["ec1__" + k] = val
                    deep_items = self.ec2.get_params().items()
                    for k, val in deep_items:
                        if k != "num_qubits":
                            params["ec2__" + k] = val

                params["num_qubits"] = self.ec1.get_params()["num_qubits"]

                return params

            def set_params(self, **params) -> None:
                """
                Sets value of the composed kernel hyper-parameters.

                Args:
                    params: Hyper-parameters and their values, e.g. ``num_qubits=2``
                """
                valid_params = self.get_params()
                ec1_dict = {}
                ec2_dict = {}
                for key, value in params.items():
                    if key not in valid_params:
                        raise ValueError(
                            f"Invalid parameter {key!r}. "
                            f"Valid parameters are {sorted(valid_params)!r}."
                        )
                    if key.startswith("ec1__"):
                        ec1_dict[key[5:]] = value
                    elif key.startswith("ec2__"):
                        ec2_dict[key[5:]] = value

                    if key == "num_qubits":
                        ec1_dict["num_qubits"] = value
                        ec2_dict["num_qubits"] = value
                        self._num_qubits = value

                if len(ec1_dict) > 0:
                    self.ec1.set_params(**ec1_dict)
                if len(ec2_dict) > 0:
                    self.ec2.set_params(**ec2_dict)

            def get_circuit(
                self,
                features: Union[ParameterVector, np.ndarray],
                parameters: Union[ParameterVector, np.ndarray],
            ) -> QuantumCircuit:
                """
                Returns the circuit of the composed encoding circuits

                Args:
                    features Union[ParameterVector,np.ndarray]: Input vector of the features
                        from which the gate inputs are obtained
                    param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                        from which the gate inputs are obtained

                Return:
                    Returns the circuit of the composed encoding circuits in qiskit QuantumCircuit format
                """

                circ1 = self.ec1.get_circuit(
                    features[: self.ec1.num_features], parameters[: self.ec1.num_parameters]
                )
                circ2 = self.ec2.get_circuit(
                    features[: self.ec2.num_features], parameters[self.ec1.num_parameters :]
                )

                return circ1.compose(circ2, range(self.ec1.num_qubits))

        return ComposedEncodingCircuit.create_from_encoding_circuits(self, x)


class EncodingSlotsMismatchError(Exception):
    """Exception raised when the number of encoding slots does not match the number of features."""

    def __init__(self, num_slots, num_features):
        self.num_slots = num_slots
        self.num_features = num_features
        self.message = (
            f"Encoding slots ({num_slots}) do not match the number of features ({num_features})."
        )
        super().__init__(self.message)
