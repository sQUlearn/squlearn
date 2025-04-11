from __future__ import annotations

import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit


class EncodingCircuitBase:
    """
    Encoding circuit base class

    Args:
        num_qubits (int): Number of Qubits of the encoding circuit
        num_features (int): Dimension of the feature vector
    """

    def __init__(self, num_qubits: int, num_features: int) -> None:
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

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Return the circuit encoding circuit

        Has to be overwritten, otherwise a NotImplementedError is thrown

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

        feature_vec = ParameterVector(feature_label, self.num_features)
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

    def inverse(self):
        """
        Returns the inverse of the encoding circuit.

        Returns:
            The inverse of the encoding circuit
        """

        class InvertedEncodingCircuit(EncodingCircuitBase):
            def __init__(self, encoding_circuit: EncodingCircuitBase):
                super().__init__(encoding_circuit.num_qubits, encoding_circuit.num_features)
                self._encoding_circuit = encoding_circuit

            @property
            def num_parameters(self) -> int:
                """Returns the number of trainable parameters of the encoding circuit."""
                return self._encoding_circuit.num_parameters

            @property
            def parameter_bounds(self) -> np.ndarray:
                """Returns the bounds of the trainable parameters of the encoding circuit."""
                return self._encoding_circuit.parameter_bounds

            @property
            def feature_bounds(self) -> np.ndarray:
                """Returns the bounds of the features of the encoding circuit."""
                return self._encoding_circuit.feature_bounds

            def generate_initial_parameters(self, seed: Union[int, None] = None) -> np.ndarray:
                """
                Generates random parameters for the encoding circuit

                Args:
                    seed (Union[int,None]): Seed for the random number generator

                Return:
                    Returns the randomly generated parameters
                """
                return self._encoding_circuit.generate_initial_parameters(seed)

            def get_circuit(
                self,
                features: Union[ParameterVector, np.ndarray],
                parameters: Union[ParameterVector, np.ndarray],
            ) -> QuantumCircuit:
                """
                Returns the inverse circuit of the encoding circuit

                Args:
                    features Union[ParameterVector,np.ndarray]: Input vector of the features
                        from which the gate inputs are obtained
                    param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                        from which the gate inputs are obtained

                Return:
                    Returns the inverse circuit of the encoding circuit in qiskit QuantumCircuit
                    format
                """
                circ = self._encoding_circuit.get_circuit(features, parameters)
                return circ.inverse()

        return InvertedEncodingCircuit(self)

    def __mul__(self, x):
        return self.__add__(x)

    def __add__(self, x):
        return self.compose(x, concatenate_features=False, concatenate_parameters=True)

    def compose(self, x, concatenate_features=True, concatenate_parameters=False):
        """
        Composition of encoding circuits with options for handling features and parameters

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
                num_qubits (int): num qubits for both encoding circuits
                    (necessary for scikit-learn interface)
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
                if concatenate_parameters:
                    return self.ec1.num_parameters + self.ec2.num_parameters
                else:
                    return max(self.ec1.num_parameters, self.ec2.num_parameters)

            @property
            def num_features(self) -> int:
                """The dimension of the features in the encoding circuit."""
                if concatenate_features:
                    return self.ec1.num_features + self.ec2.num_features
                else:
                    return max(self.ec1.num_features, self.ec2.num_features)

            @property
            def parameter_bounds(self) -> np.ndarray:
                """Returns the bounds of the trainable parameters of composed encoding circuit.

                Is equal to the sum of both bounds.
                """
                if concatenate_parameters:
                    return np.concatenate(
                        (self.ec1.parameter_bounds, self.ec2.parameter_bounds), axis=0
                    )
                else:
                    # We compare self.ec1.parameter_bounds and self.ec2.parameter_bounds,
                    # we return a new array,
                    # with the shape of the largest array, and the minimum values of the two arrays
                    # for the first column (lower bound),
                    # and the maximum values of the two arrays for the second column (upper bound)

                    # Extend parameter bounds to the shape of the largest array with np.pad
                    parameter_bounds1_extended = np.pad(
                        self.ec1.parameter_bounds,
                        ((0, self.num_parameters - self.ec1.num_parameters), (0, 0)),
                        constant_values=np.nan,
                    )
                    parameter_bounds2_extended = np.pad(
                        self.ec2.parameter_bounds,
                        ((0, self.num_parameters - self.ec2.num_parameters), (0, 0)),
                        constant_values=np.nan,
                    )

                    # Compute the minimum lower bound values for the first column and maximum upper
                    # bounds for the second column
                    parameter_bounds3_first_col = np.nanmin(
                        np.array(
                            [parameter_bounds1_extended[:, 0], parameter_bounds2_extended[:, 0]]
                        ),
                        axis=0,
                    )
                    parameter_bounds3_second_col = np.nanmax(
                        np.array(
                            [parameter_bounds1_extended[:, 1], parameter_bounds2_extended[:, 1]]
                        ),
                        axis=0,
                    )
                    # Stack the results into the final array
                    return np.column_stack(
                        (parameter_bounds3_first_col, parameter_bounds3_second_col)
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

                if concatenate_parameters:
                    return np.concatenate(
                        (
                            self.ec1.generate_initial_parameters(seed),
                            self.ec2.generate_initial_parameters(seed),
                        ),
                        axis=0,
                    )
                else:
                    if self.num_parameters == 0:
                        return np.array([])
                    r = np.random.RandomState(seed)
                    bounds = self.parameter_bounds
                    return r.uniform(low=bounds[:, 0], high=bounds[:, 1])

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
                    Returns the circuit of the composed encoding circuits in qiskit QuantumCircuit
                    format
                """
                if concatenate_features:
                    features_c1, features_c2 = (
                        features[: self.ec1.num_features],
                        features[self.ec1.num_features :],
                    )
                else:
                    features_c1, features_c2 = (
                        features[: self.ec1.num_features],
                        features[: self.ec2.num_features],
                    )

                if concatenate_parameters:
                    parameters_c1, parameters_c2 = (
                        parameters[: self.ec1.num_parameters],
                        parameters[self.ec1.num_parameters :],
                    )
                else:
                    parameters_c1, parameters_c2 = (
                        parameters[: self.ec1.num_parameters],
                        parameters[: self.ec1.num_parameters],
                    )

                circ1 = self.ec1.get_circuit(features_c1, parameters_c1)
                circ2 = self.ec2.get_circuit(
                    features_c2,
                    parameters_c2,  # Only line that changes, to include the new features
                )

                return circ1.compose(circ2, range(self.ec1.num_qubits))

        return ComposedEncodingCircuit.create_from_encoding_circuits(self, x)
