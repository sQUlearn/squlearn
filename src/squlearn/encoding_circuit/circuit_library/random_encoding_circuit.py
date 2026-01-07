import numpy as np
from typing import Union
import random

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qiskit.circuit.library import (
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    TGate,
    CXGate,
    CYGate,
    CZGate,
    CHGate,
    SwapGate,
    RXGate,
    RYGate,
    RZGate,
    PhaseGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
)

from squlearn.util.data_preprocessing import extract_num_features

from ..encoding_circuit_base import EncodingCircuitBase

default_gate_weights = {
    "x": 0.5,
    "y": 0.5,
    "z": 0.5,
    "h": 0.5,
    "cx": 0.5,
    "cy": 0.5,
    "cz": 0.5,
    "ch": 0.5,
    "rx": 1.0,
    "ry": 1.0,
    "rz": 1.0,
    "crx": 0.75,
    "cry": 0.75,
    "crz": 0.75,
    "rxx": 0.75,
    "ryy": 0.75,
    "rzz": 0.75,
    "rzx": 0.75,
}

default_encoding_weights = {
    "p": 0.5,
    "x": 0.5,
    "p_times_x": 0.5,
    "pi_times_x": 0.5,
    "arctan_x": 0.5,
    "p_times_arctan_x": 0.5,
    "arccos_x": 0.0,
    "p_times_arccos_x": 0.0,
}

# Available gates with number of qubits and parameterized
available_gates = {
    "x": (XGate, 1, False),
    "y": (YGate, 1, False),
    "z": (ZGate, 1, False),
    "h": (HGate, 1, False),
    "s": (SGate, 1, False),
    "t": (TGate, 1, False),
    "cx": (CXGate, 2, False),
    "cy": (CYGate, 2, False),
    "cz": (CZGate, 2, False),
    "ch": (CHGate, 2, False),
    "swap": (SwapGate, 2, False),
    "rx": (RXGate, 1, True),
    "ry": (RYGate, 1, True),
    "rz": (RZGate, 1, True),
    "p": (PhaseGate, 1, True),
    "cp": (CPhaseGate, 2, True),
    "crx": (CRXGate, 2, True),
    "cry": (CRYGate, 2, True),
    "crz": (CRZGate, 2, True),
    "rxx": (RXXGate, 2, True),
    "ryy": (RYYGate, 2, True),
    "rzz": (RZZGate, 2, True),
    "rzx": (RZXGate, 2, True),
}

# Different encodings for the parameters and features
available_encodings = {
    "p": (lambda x, p: p, "p"),
    "x": (lambda x, p: x, "x"),
    "p_times_x": (lambda x, p: x * p, "px"),
    "pi_times_x": (lambda x, p: np.pi * x, "x"),
    "arctan_x": (lambda x, p: np.arctan(x), "x"),
    "p_times_arctan_x": (lambda x, p: p * np.arctan(x), "px"),
    "arccos_x": (lambda x, p: np.arccos(x), "x"),
    "p_times_arccos_x": (lambda x, p: p * np.arccos(x), "px"),
}
# List of encodings with features
feature_encodings = [e for e in available_encodings.keys() if "x" in available_encodings[e][1]]
# List of encodings with parameters
parameter_encodings = [e for e in available_encodings.keys() if "p" in available_encodings[e][1]]
# List of parameterized gates
parameterized_gates = [k for k in available_gates.keys() if available_gates[k][2]]


class RandomEncodingCircuit(EncodingCircuitBase):
    r"""
    Random parameterized encoding circuit with randomly picked gates, qubits and feature encodings.

    The random ciruit generation picks gates from a large set of gates
    (both parameterized and non-parameterized) and places them on randomly drawn qubits.
    Parameterized gates can have different randomly picked encodings for the features and
    parameters.

    The weights for picking certain gates and encodings can be adjusted with the
    ``gate_weights`` and ``encoding_weights`` dictionaries.
    Default values are set in the ``default_gate_weights`` and ``default_encoding_weights``
    dictionaries. In the default values, the parameterized gates are more likely to be drawn.

    Every circuit is uniquly defined by the seed, and the input parameters of the encoding
    circuit. This allows for example for searching the optimal circuit by a gridsearch over
    different seeds.

    The random circuit generation enforces, that every feature is encoded at least once
    in the circuit. It also tries to keep the single features evenly distributed over the gates.

    **Example for 4 qubits and a 6 dimensional feature vector**

    .. plot::

        from squlearn.encoding_circuit import RandomEncodingCircuit
        pqc = RandomEncodingCircuit(num_qubits=4, seed = 2)
        pqc._gen_random_config(num_features=6, seed=2)
        plt = pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10}, num_features=6)
        plt.tight_layout()

    Args:
        num_qubits (int): Number of qubits of the encoding circuit
        num_features (int): Dimension of the feature vector (default: None)
        seed (int): Seed for the random number generator (default: 0)
        min_gates (int): Minimum number of gates in the circuit (default: 10)
        max_gates (int): Maximum number of gates in the circuit (default: 50)
        gate_weights (dict): Dictionary with the weights for the gates
                             (default: default_gate_weights)
        encoding_weights (dict): Dictionary with the weights for the encodings
                                 default: default_encoding_weights)
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int = None,
        seed: int = 0,
        min_gates: int = 10,
        max_gates: int = 50,
        gate_weights: dict = default_gate_weights,
        encoding_weights: dict = default_encoding_weights,
    ):
        super().__init__(num_qubits, num_features)
        self.min_gates = min_gates
        self.max_gates = max_gates
        self.encoding_weights = encoding_weights
        self.gate_weights = gate_weights
        self.seed = seed
        self.available_feature_gates = None
        self._num_parameters = None
        self._is_config_available = False

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        r"""
        Returns the random encoding circuit.

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
            parameters (Union[ParameterVector,np.ndarray]): Input vector of the parameters

        Return:
            Returns the random encoding circuit in qiskit QuantumCircuit format
        """
        num_features = extract_num_features(features)
        self._gen_random_config(num_features=num_features, seed=self.seed)

        self._check_feature_consistency(features)

        qc = QuantumCircuit(self.num_qubits)
        feature_counter = 0
        parameter_counter = 0
        parameterized_gate_counter = 0
        for igate, gate in enumerate(self._picked_gates):
            is_parameterized = available_gates[gate][2]
            qubits_arg = self._picked_qubits[igate]
            # switch between parameterized and non-parameterized gates
            if is_parameterized:
                encoding = available_encodings[self._picked_encodings[parameterized_gate_counter]][
                    0
                ]
                qc.append(
                    available_gates[gate][0](
                        encoding(
                            features[self._feature_indices[feature_counter]],
                            parameters[parameter_counter % self.num_parameters],
                        ),
                    ),
                    list(qubits_arg),
                )
                if self._picked_encodings[parameterized_gate_counter] in feature_encodings:
                    feature_counter += 1
                if self._picked_encodings[parameterized_gate_counter] in parameter_encodings:
                    parameter_counter = (parameter_counter + 1) % self.num_parameters
                parameterized_gate_counter += 1
            else:
                qc.append(available_gates[gate][0](), list(qubits_arg))

        return qc

    @property
    def included_gates(self) -> list:
        """Returns the list of gates from which the random circuit is drawn."""
        return available_gates.keys()

    def _gen_random_config(self, num_features: int, seed: int):
        """
        Generates a random configuration for the random encoding circuit.

        Args:
            num_features (int): Number of features of the input data.
            seed (int): Seed for the random number generator.
        """

        random.seed(seed)

        # Determine number of gates in the random circuit
        min_gates = max(self.min_gates, num_features)
        max_gates = max(self.min_gates + 1, self.max_gates, num_features)
        self._num_gates = random.randint(min_gates, max_gates)

        # Determine the probability of each gate to be drawn in the random selection
        # Dictionary with all gates and probability 0
        gate_weights = dict(zip(available_gates.keys(), [0.0] * len(available_gates.keys())))

        # Update the dictionary with the inputted gate_weights
        if set(self.gate_weights.keys()) - set(gate_weights.keys()):
            raise ValueError("Inputted gate_weights contains not supported keys!")
        gate_weights.update(self.gate_weights)
        # Pick the gates with the given probabilities
        self._picked_gates = random.choices(
            list(gate_weights.keys()), weights=list(gate_weights.values()), k=self._num_gates
        )

        # Pick the encodings for parameterized gates
        # Get the number of parameterized gates
        num_param_gates = sum([1 for gate in self._picked_gates if gate in parameterized_gates], 0)
        encoding_weights = dict(
            zip(available_encodings.keys(), [0.0] * len(available_encodings.keys()))
        )
        if set(self.encoding_weights.keys()) - set(encoding_weights.keys()):
            raise ValueError("Inputted encoding_weights contains not supported keys!")
        encoding_weights.update(self.encoding_weights)
        self._picked_encodings = random.choices(
            list(encoding_weights.keys()),
            weights=list(encoding_weights.values()),
            k=num_param_gates,
        )

        num_feature_gates = sum([1 for p in self._picked_encodings if p in feature_encodings], 0)

        # In case there are not enough gates with features, add additional gates with features
        if num_feature_gates < num_features:
            probabilities_feature = [gate_weights[gate] for gate in parameterized_gates]
            extra_gates = random.choices(
                parameterized_gates,
                weights=probabilities_feature,
                k=num_features - num_feature_gates,
            )
            feature_encoding_probabilities = [encoding_weights[gate] for gate in feature_encodings]
            extra_encodings = random.choices(
                feature_encodings,
                weights=feature_encoding_probabilities,
                k=num_features - num_feature_gates,
            )
            if len(extra_gates) != 0:
                self._picked_encodings += extra_encodings
                self._picked_gates += extra_gates
                random.shuffle(self._picked_gates)
                random.shuffle(self._picked_encodings)
            else:
                raise RuntimeError("No additional gates with features found!")

        # If the number of gates is larger than max_gates
        # remove first non-parameterized gates, than parameterized gates with parameters (not features)
        if len(self._picked_gates) > max_gates:
            while len(self._picked_gates) > max_gates:
                popped_gate = False
                # Try to remove a non-parameterized gate first
                for i, p in enumerate(self._picked_gates):
                    if p not in parameterized_gates:
                        self._picked_gates.pop(i)
                        popped_gate = True
                        break
                # No non-parameterized gates found -> remove a purely parameterized gate
                if not popped_gate:
                    for i in range(len(self._picked_gates)):
                        if self._picked_encodings[i] == "p":
                            self._picked_gates.pop(i)
                            self._picked_encodings.pop(i)
                            popped_gate = True
                            break
                if not popped_gate:
                    break
            # Shuffle gates again
            random.shuffle(self._picked_gates)

        # Random list for feature indices which keeps blocks of all features together
        # keeps the features evenly distributed, even if there are few gates
        # e.g. [2,1,3,4,4,3,1,2,3,1,4,3]
        self._num_gates = len(self._picked_gates)
        self._feature_indices = sum(
            [random.sample(range(num_features), k=num_features) for i in range(self._num_gates)],
            [],
        )

        # Qubit asignments for the gates
        self._picked_qubits = [
            tuple(random.sample(range(self.num_qubits), available_gates[gate][1]))
            for gate in self._picked_gates
        ]
        self._num_parameters = sum(
            [1 for p in self._picked_encodings if p in parameter_encodings], 0
        )

        self.available_feature_gates = num_feature_gates
        self._is_config_available = True

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the random encoding circuit."""
        if self._num_parameters is None:
            raise ValueError(
                "No random configuration available. Generate one with _gen_random_config."
            )
        return self._num_parameters

    @property
    def num_encoding_slots(self) -> float:
        """The number of encoding slots of the random encoding circuit (equal to np.inf)."""
        return np.inf

    def generate_initial_parameters(
        self, num_features: int, seed: Union[int, None] = None
    ) -> np.ndarray:
        """
        Generates random parameters for the RandomEncodingCircuit.

        Args:
            num_features (int): Number of features of the input data
            seed (Union[int,None]): Seed for the random number generator (default: None)

        Return:
            The randomly generated parameters
        """
        # the random configuration must be generated before generating the parameters
        self._gen_random_config(num_features=num_features, seed=self.seed)
        return super().generate_initial_parameters(num_features, seed)

    def get_params(self, deep: bool = True) -> dict:
        r"""
        Returns hyper-parameters and their values of the random encoding circuit.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        param = super().get_params()
        param["seed"] = self.seed
        param["min_gates"] = self.min_gates
        param["max_gates"] = self.max_gates
        param["encoding_weights"] = self.encoding_weights
        param["gate_weights"] = self.gate_weights
        return param

    def set_params(self, **params):
        r"""
        Sets value of the random encoding circuit hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``.
        """
        super().set_params(**params)

        # Reset the random configuration
        self._is_config_available = False
        return self
