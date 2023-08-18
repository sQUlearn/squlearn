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

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the feature map.

        Default bounds are [-pi,pi] for each parameter.
        """
        return np.array([[-np.pi, np.pi]] * self.num_parameters)

    @property
    def feature_bounds(self) -> np.ndarray:
        """The bounds of the features of the feature map.

        Default bounds are [-pi,pi] for each feature.
        """
        return np.array([[-np.pi, np.pi]] * self.num_features)

    def generate_initial_parameters(self, seed: Union[int, None] = None) -> np.ndarray:
        """
        Generates random parameters for the feature map

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
        output: str = None,
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

        return circ.draw(output, **kwargs)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the feature map.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        param = {}
        param["num_qubits"] = self._num_qubits
        return param

    def set_params(self, **params) -> None:
        """
        Sets value of the feature map hyper-parameters.

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

        return None

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

                Is equal to the sum of both trainable parameters.
                """
                return self._fm1.num_parameters + self._fm2.num_parameters

            @property
            def parameter_bounds(self) -> np.ndarray:
                """Returns the bounds of the trainable parameters of composed feature map.

                Is equal to the sum of both bounds.
                """
                return np.concatenate(
                    (self._fm1.parameter_bounds, self._fm2.parameter_bounds), axis=0
                )

            @property
            def feature_bounds(self) -> np.ndarray:
                """Returns the bounds of the features of composed feature map.

                Is equal to the maximum and minimum of both bounds.
                """

                feature_bounds1 = self._fm1.feature_bounds
                feature_bounds2 = self._fm2.feature_bounds
                feature_bounds_values = np.zeros((self.num_features, 2))

                min_num_feature = min(self._fm1.num_features, self._fm2.num_features)

                if self._fm1.num_features == self.num_features:
                    feature_bounds_values = self._fm1.feature_bounds

                if self._fm2.num_features == self.num_features:
                    feature_bounds_values = self._fm2.feature_bounds

                for i in range(min_num_feature):
                    feature_bounds_values[i, 0] = min(feature_bounds1[i, 0], feature_bounds2[i, 0])
                    feature_bounds_values[i, 1] = max(feature_bounds1[i, 1], feature_bounds2[i, 1])

                return feature_bounds_values

            def generate_initial_parameters(self, seed: Union[int, None] = None) -> np.ndarray:
                """
                Generates random parameters for the composed feature map

                Args:
                    seed (Union[int,None]): Seed for the random number generator

                Return:
                    Returns the randomly generated parameters
                """

                return np.concatenate(
                    (
                        self._fm1.generate_initial_parameters(seed),
                        self._fm2.generate_initial_parameters(seed),
                    ),
                    axis=0,
                )

            def get_params(self, deep: bool = True) -> dict:
                """
                Returns hyper-parameters and their values of the composed feature map.

                Hyper-parameter names are prefixed by ``fm1__`` or ``fm2__`` depending on
                which feature map they belong to.

                Args:
                    deep (bool): If True, also the parameters for
                                 contained objects are returned (default=True).

                Return:
                    Dictionary with hyper-parameters and values.
                """
                params = dict(fm1=self._fm1, fm2=self._fm2)
                if deep:
                    deep_items = self._fm1.get_params().items()
                    for k, val in deep_items:
                        if k != "num_qubits":
                            params["fm1__" + k] = val
                    deep_items = self._fm2.get_params().items()
                    for k, val in deep_items:
                        if k != "num_qubits":
                            params["fm2__" + k] = val

                params["num_qubits"] = self._fm1.get_params()["num_qubits"]

                return params

            def set_params(self, **params) -> None:
                """
                Sets value of the composed kernel hyper-parameters.

                Args:
                    params: Hyper-parameters and their values, e.g. ``num_qubits=2``
                """
                valid_params = self.get_params()
                fm1_dict = {}
                fm2_dict = {}
                for key, value in params.items():
                    if key not in valid_params:
                        raise ValueError(
                            f"Invalid parameter {key!r}. "
                            f"Valid parameters are {sorted(valid_params)!r}."
                        )
                    if key.startswith("fm1__"):
                        fm1_dict[key[5:]] = value
                    elif key.startswith("fm2__"):
                        fm2_dict[key[5:]] = value

                    if key == "num_qubits":
                        fm1_dict["num_qubits"] = value
                        fm2_dict["num_qubits"] = value

                if len(fm1_dict) > 0:
                    self._fm1.set_params(**fm1_dict)
                if len(fm2_dict) > 0:
                    self._fm2.set_params(**fm2_dict)

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
