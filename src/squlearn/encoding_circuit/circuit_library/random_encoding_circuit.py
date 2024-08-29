import numpy as np
from typing import Union
import random

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..encoding_circuit_base import EncodingCircuitBase

default_gate_weights = {
    "x": 0.25,
    "y": 0.25,
    "z": 0.25,
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
    "pi_times_x" : 0.5,
    "arctan_x": 0.5,
    "p_times_arctan_x": 0.5,
    "arccos_x": 0.0,
    "p_times_arccos_x": 0.0,
}


class RandomEncodingCircuit(EncodingCircuitBase):

    def __init__(
        self,
        num_qubits,
        num_features,
        min_gates=10,
        max_gates=50,
        gate_weights=default_gate_weights,
        encoding_weights=default_encoding_weights,
        seed=0,
    ):
        super().__init__(num_qubits, num_features)
        self.min_gates = min_gates
        self.max_gates = max_gates
        self.encoding_weights = encoding_weights
        self.gate_weights = gate_weights
        self.seed = seed
        self._quantum_circuit = self._gen_circle()

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        exchange_dict_x = dict(zip(self._x, features))
        exchange_dict_p = dict(zip(self._p, parameters))
        exchange_both = exchange_dict_x
        exchange_both.update(exchange_dict_p)
        return self._quantum_circuit.assign_parameters(exchange_both, inplace=False)

    def _gen_circle(self):

        random.seed(self.seed)
        qc = QuantumCircuit(self.num_qubits)
        gates = {
            "x": (qc.x, 1, False),
            "y": (qc.y, 1, False),
            "z": (qc.z, 1, False),
            "h": (qc.h, 1, False),
            "s": (qc.s, 1, False),
            "t": (qc.t, 1, False),
            "cx": (qc.cx, 2, False),
            "cy": (qc.cy, 2, False),
            "cz": (qc.cz, 2, False),
            "ch": (qc.ch, 2, False),
            "swap": (qc.swap, 2, False),
            "rx": (qc.rx, 1, True),
            "ry": (qc.ry, 1, True),
            "rz": (qc.rz, 1, True),
            "p": (qc.p, 1, True),
            "cp": (qc.cp, 2, True),
            "crx": (qc.crx, 2, True),
            "cry": (qc.cry, 2, True),
            "crz": (qc.crz, 2, True),
            "rxx": (qc.rxx, 2, True),
            "ryy": (qc.ryy, 2, True),
            "rzz": (qc.rzz, 2, True),
            "rzx": (qc.rzx, 2, True),
        }

        encodings = {
            "p": (lambda x, p: p, "p"),
            "x": (lambda x, p: x, "x"),
            "p_times_x": (lambda x, p: x * p, "px"),
            "pi_times_x": (lambda x, p: np.pi * x, "x"),
            "arctan_x": (lambda x, p: np.arctan(x), "x"),
            "p_times_arctan_x": (lambda x, p: p * np.arctan(x), "px"),
            "arccos_x": (lambda x, p: np.arccos(x), "x"),
            "p_times_arccos_x": (lambda x, p: p * np.arccos(x), "px"),
        }
        feature_encodings = [e for e in encodings.keys() if "x" in encodings[e][1]]
        parameter_encodings = [e for e in encodings.keys() if "p" in encodings[e][1]]

        # Build list of parameterized gates
        parameterized_gates = [k for k in gates.keys() if gates[k][2]]

        # Determine number of gates in the random circuit
        min_gates = max(self.min_gates, self.num_features)
        max_gates = max(self.min_gates + 1, self.max_gates, self.num_features)
        self.num_gates = random.randint(min_gates, max_gates)

        # Determine the probability of each gate to be drawn in the random selection
        gate_weights_ = dict(
            zip(gates.keys(), [0.0] * len(gates.keys()))
        )  # Dictionary with all gates and probability 0
        additional_keys = set(self.gate_weights.keys()) - set(gate_weights_.keys())
        if additional_keys:
            raise ValueError(
                f"Additional not supported keys in gate_weights: {additional_keys}"
            )
        gate_weights_.update(self.gate_weights)
        gate_elements = list(gate_weights_.keys())
        gate_probabilities = list(gate_weights_.values())
        picked_gates = random.choices(
            gate_elements, weights=gate_probabilities, k=self.num_gates
        )

        # Determine if parameterized gates hold a feature or a parameter
        encoding_weights_ = dict(zip(encodings.keys(), [0.0] * len(encodings.keys())))
        additional_keys = set(self.encoding_weights.keys()) - set(
            encoding_weights_.keys()
        )
        if additional_keys:
            raise ValueError(
                f"Additional not supported keys in encoding_weights: {additional_keys}"
            )
        encoding_weights_.update(self.encoding_weights)
        encoding_elements = list(encoding_weights_.keys())
        encoding_probabilities = list(encoding_weights_.values())
        num_param_gates = sum(
            [1 for gate in picked_gates if gate in parameterized_gates], 0
        )
        feature_or_param = random.choices(
            encoding_elements, weights=encoding_probabilities, k=num_param_gates
        )
        num_feature_gates = sum(
            [1 for p in feature_or_param if p in feature_encodings], 0
        )

        # In case there are not enough gates with features, add additional gates with features
        if num_feature_gates < self.num_features:
            probabilities_feature = [
                gate_weights_[gate] for gate in parameterized_gates
            ]
            extra_gates = random.choices(
                parameterized_gates,
                weights=probabilities_feature,
                k=self.num_features - num_feature_gates,
            )
            feature_encoding_probabilities = [
                encoding_weights_[gate] for gate in feature_encodings
            ]
            extra_encodings = random.choices(
                feature_encodings,
                weights=feature_encoding_probabilities,
                k=self.num_features - num_feature_gates,
            )
            if len(extra_gates) != 0:
                feature_or_param += extra_encodings
                picked_gates += extra_gates
                random.shuffle(picked_gates)
                random.shuffle(feature_or_param)

        # Increase the number of gates is larger than max_gates
        # remove first non-parameterized gates, than parameterized gates with parameters (not features)
        if len(picked_gates) > max_gates:
            while len(picked_gates) > max_gates:
                popped_gate = False
                # Try to remove a non-parameterized gate first
                for i,p in enumerate(picked_gates):
                    if p not in parameterized_gates:
                        picked_gates.pop(i)
                        popped_gate = True
                        break
                # No non-parameterized gates found -> remove a purely parameterized gate
                if not popped_gate:
                    for i in range(len(picked_gates)):
                        if feature_or_param[i] == "p":
                            picked_gates.pop(i)
                            feature_or_param.pop(i)
                            popped_gate = True
                            break
                if not popped_gate:
                    break
            # Shuffle gates again
            random.shuffle(picked_gates)

        # Random list for feature indices, keeps blocks of all features together
        self.num_gates = len(picked_gates)
        feature_indices = sum(
            [
                random.sample(range(self.num_features), k=self.num_features)
                for i in range(self.num_gates)
            ],
            [],
        )
        picked_qubits = [tuple(random.sample(range(self.num_qubits), gates[gate][1])) for gate in picked_gates]

        # Create circuit
        feature_counter = 0
        parameter_counter = 0
        parameterized_gate_counter = 0
        self._num_parameters = sum(
            [1 for p in feature_or_param if p in parameter_encodings], 0
        )
        self._x = ParameterVector("x", self.num_features)
        self._p = ParameterVector("p", self.num_parameters)
        for igate,gate in enumerate(picked_gates):
            is_parameterized = gates[gate][2]
            # sample the indices of the qubits
            qubits_arg = picked_qubits[igate]
            # switch between parameterized and non-parameterized gates (according to feature_or_param)
            if is_parameterized:
                encoding = encodings[feature_or_param[parameterized_gate_counter]][0]
                gates[gate][0](
                    encoding(
                        self._x[feature_indices[feature_counter]],
                        self._p[parameter_counter% self.num_parameters],
                    ),
                    *qubits_arg,
                )
                if feature_or_param[parameterized_gate_counter] in feature_encodings:
                    feature_counter += 1
                if feature_or_param[parameterized_gate_counter] in parameter_encodings:
                    parameter_counter = (parameter_counter + 1) % self.num_parameters
                parameterized_gate_counter += 1
            else:
                gates[gate][0](*qubits_arg)

        return qc

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the encoding circuit."""
        return self._num_parameters

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
        param["seed"] = self.seed
        param["min_gates"] = self.min_gates
        param["max_gates"] = self.max_gates
        param["encoding_weights"] = self.encoding_weights
        param["gate_weights"] = self.gate_weights
        return param

    def set_params(self, **params) -> None:
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

        self._quantum_circuit = self._gen_circle()

        return self
