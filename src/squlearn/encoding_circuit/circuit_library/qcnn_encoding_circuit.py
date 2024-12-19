import numpy as np
from typing import Union
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_gate, circuit_to_instruction

from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from squlearn.encoding_circuit.circuit_library.param_z_feature_map import ParamZFeatureMap
from squlearn.observables import CustomObservable, SummedPaulis
from squlearn.observables.observable_base import ObservableBase


class QCNNEncodingCircuit(EncodingCircuitBase):
    """
    Encoding circuit for quantum convolutional neural networks (QCNN).

    The structure is inspired by classical convolutional neural networks. The number of active
    qubits reduces with each layer. The design idea was initially proposed in reference [1].

    Args:
        num_qubits (int): Number of initial qubits of the QCNN encoding circuit.
        num_features (int): Dimension of the feature vector.
            By default this is 0, so a feature map must be provided.
            If the number of features is bigger then 0,
            then in the get_circuit function a ZFeatureMap is built to encode the features.
        default (bool): If True, the default circuit is built.

    References
    -----------
    [1]: `Cong, I., Choi, S. & Lukin, M.D. Quantum convolutional neural networks. Nat. Phys. 15,
    1273–1278 (2019). <https://doi.org/10.1038/s41567-019-0648-8>`_
    """

    def __init__(self, num_qubits: int = 0, num_features: int = 0, default: bool = False) -> None:
        super().__init__(num_qubits, num_features)
        self._num_parameters = 0
        self._left_qubits = [i for i in range(num_qubits)]
        self._operations_list = []
        self._default = default
        self._num_measurements = 0
        if default:
            if num_qubits == 0:
                raise ValueError("To generate a default circuit provide a number of qubits > 0.")
            self.default_circuit()

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters of the current encoding circuit."""
        return self._num_parameters

    @property
    def left_qubits(self) -> list:
        """Returns the qubits which one can operate on in the current circuit."""
        return self._left_qubits

    @property
    def operations_list(self) -> list:
        """Returns the list of operators currently acting on the encoding circuit."""
        return self._operations_list

    def set_params(self, **params):
        """
        Sets value of the encoding circuit hyper-parameters.

        If the number of qubits is reduced and the supplied pooling gates do not fit to this,
        this will throw a error in the troubling layer.

        Args:
            params: Hyper-parameters (num_qubits or num_features)
                and their values, e.g. ``num_qubits=2``.
        """
        super().set_params(**params)
        if "num_qubits" in params:
            self._left_qubits = [i for i in range(self.num_qubits)]
            self._num_parameters = 0
            if self._default:
                self._operations_list = []
                self.default_circuit()
            else:
                for operation in self.operations_list:
                    if operation[0] == "Conv":
                        self.__convolution(*operation[1:], _new_operation=False)
                    elif operation[0] == "Pool":
                        self.__pooling(*operation[1:], _new_operation=False)
                    elif operation[0] == "FC":
                        self.__fully_connected(*operation[1:], _new_operation=False)
                        break  # since FC should be at the end of the circuit

    def convolution(
        self,
        quantum_circuit: Union[QuantumCircuit, EncodingCircuitBase, None] = None,
        label: str = "Conv",
        alternating: bool = True,
        share_params: bool = False,
    ):
        """
        Add a convolution layer to the encoding circuit.

        Args:
            quantum_circuit Union[EncodingCircuitBase, QuantumCircuit, None]:
                The quantum circuit, which is applied in this layer.
            label (str): The name of the layer.
            alternating (bool): The gate is applied on every qubit modulo qubits of this circuit
                beginning at 0. If True it applies the gate on every qubit beginning at 1 again.
            share_params (bool): If False,
                different parameters are used for the gates in this layer.
        """
        self.__convolution(quantum_circuit, label, alternating, share_params)

    def __convolution(
        self,
        quantum_circuit: Union[QuantumCircuit, EncodingCircuitBase, None] = None,
        label: str = "Conv",
        alternating: bool = True,
        share_params: bool = False,
        _new_operation: bool = True,
    ):
        """Internal function to allow internal _new_operation argument."""

        # Define default circuit
        if not quantum_circuit:
            param = ParameterVector("a", 3)
            quantum_circuit = QuantumCircuit(2)
            quantum_circuit.rz(-np.pi / 2, 1)
            quantum_circuit.cx(1, 0)
            quantum_circuit.rz(param[0], 0)
            quantum_circuit.ry(param[1], 1)
            quantum_circuit.cx(0, 1)
            quantum_circuit.ry(param[2], 1)
            quantum_circuit.cx(1, 0)
            quantum_circuit.rz(np.pi / 2, 0)

        quantum_circuit = self.__convert_encoding_circuit(quantum_circuit)
        if self.num_qubits == 0:
            if _new_operation:
                self._operations_list.append(
                    ["Conv", quantum_circuit, label, alternating, share_params]
                )
        else:
            if quantum_circuit.num_qubits > len(self.left_qubits):
                raise ValueError(
                    "Warning on convolutional layer: The input circuit controls too many qubits: "
                    f"{quantum_circuit.num_qubits} qubits on input vs. "
                    f"{len(self.left_qubits)} qubits on the actual circuit."
                )
            # define number of gates applied
            if not share_params:
                number_of_gates_1 = int(len(self.left_qubits) / quantum_circuit.num_qubits)
                number_of_gates_2 = 0
                if alternating and len(self.left_qubits) > quantum_circuit.num_qubits:
                    number_of_gates_2 = int(len(self.left_qubits) / quantum_circuit.num_qubits)
                self._num_parameters += quantum_circuit.num_parameters * (
                    number_of_gates_1 + number_of_gates_2
                )
            else:
                self._num_parameters += quantum_circuit.num_parameters
            if _new_operation:
                self._operations_list.append(
                    ["Conv", quantum_circuit, label, alternating, share_params]
                )

    def pooling(
        self,
        quantum_circuit: Union[QuantumCircuit, EncodingCircuitBase, None] = None,
        label: str = "Pool",
        measurement: bool = False,
        input_list: list = [],
        output_list: list = [],
    ):
        """
        Add a pooling layer to the encoding circuit.

        This reduces the number of qubits to operate on from here on in this circuit
        by at least one for each circuit applied.
        Default circuit: Entangles qubit i and qubit i+1.
        Only qubit i stays in the circuit for further operations.

        Args:
            quantum_circuit Union[EncodingCircuitBase,QuantumCircuit, None]:
                The quantum circuit, which is applied in this layer.
                Must be an entangling layer, which entangles qubits.
            label (str): The name of the layer.
            measurement (bool): Sets whether the qubits,
                which are not used anymore after this layer, are measured.
                If True, quantum_circuit must consist of exactly one classical bit
                additionally to the quantum bits.
            input_list (list): Optionally one can pass the structure of the gates operating.
                The input list defines the qubits the input circuit acts on.
                The list should be structured as: [[qubit1,qubit2,..],[qubit3,qubit4,..],..].
                Every qubit can only be adressed once and the number of qubits in each list
                within the list must be equal to the number of qubits of input circuit.
                THE QUBIT NUMBERS IN THE SUBLISTS REFER TO THE INITIAL QUBIT NUMBERS!
            output_list (list):
                Exactly if an input list is entered, an output list must be entered.
                The output list defines the qubits which are left in the circuit to operate on.
                The list should be structured as: [[qubit1,qubit2,..],[qubit3,qubit4,..],..].
                It must have the same length as the input list and in each sublist
                its elements must be in the corresponding input sublist
                while beeing at least one element less.
                THE QUBIT NUMBERS IN THE SUBLISTS REFER TO THE INITIAL QUBIT NUMBERS!
        """
        self.__pooling(quantum_circuit, label, measurement, input_list, output_list)

    def __pooling(
        self,
        quantum_circuit: Union[QuantumCircuit, EncodingCircuitBase, None] = None,
        label: str = "Pool",
        measurement: bool = False,
        input_list: list = [],
        output_list: list = [],
        _new_operation: bool = True,
    ):
        """Internal function to allow internal _new_operation argument."""
        # define default circuit
        if not quantum_circuit:
            param = ParameterVector("a", 3)
            if measurement:
                quantum_circuit = QuantumCircuit(2, 1)
            else:
                quantum_circuit = QuantumCircuit(2)
            quantum_circuit.rz(-np.pi / 2, 0)
            quantum_circuit.cx(0, 1)
            quantum_circuit.ry(param[0], 0)
            if measurement:
                quantum_circuit.measure(1, 0)
                quantum_circuit.x(0).c_if(0, 1)
            else:
                quantum_circuit.rz(param[1], 1)
                quantum_circuit.cx(1, 0)
            quantum_circuit.ry(param[2], 0)

        quantum_circuit = self.__convert_encoding_circuit(quantum_circuit)

        if (measurement and quantum_circuit.num_clbits < 1) or (
            not measurement and quantum_circuit.num_clbits > 0
        ):
            raise ValueError(
                "Warning on pooling layer: Eather set measurement to True and provide a "
                "circuit with exactly one classical bit or set measurement to False."
            )

        if quantum_circuit.num_qubits > len(self.left_qubits) and self.num_qubits > 0:
            raise ValueError(
                "Warning on pooling layer: The input circuit controls too many qubits: "
                f"{quantum_circuit.num_qubits} qubits on input vs. "
                f"{len(self.left_qubits)} qubits on the actual circuit."
            )

        if len(output_list) + len(input_list) == 0:  # if no input and output lists are given
            if _new_operation:
                self._operations_list.append(
                    ["Pool", quantum_circuit, label, measurement, input_list, output_list]
                )
            if self.num_qubits > 0:
                number_of_gates = int(len(self.left_qubits) / quantum_circuit.num_qubits)
                self._num_parameters += quantum_circuit.num_parameters * number_of_gates
                if measurement:
                    self._num_measurements += quantum_circuit.num_clbits * number_of_gates
                left_qubits = [i for i in self.left_qubits]
                for j in range(number_of_gates):
                    for i in self.left_qubits[
                        j * quantum_circuit.num_qubits + 1 : (j + 1) * quantum_circuit.num_qubits
                    ]:
                        left_qubits.remove(i)
                self._left_qubits = left_qubits
        else:
            # in case a predefined order is given, test whether it is in a proper structure
            if len(input_list) != len(output_list):
                raise ValueError("The lists do not have the same length.")

            if self.num_qubits == 0:
                n_max = 0
                for i in input_list:
                    for j in i:
                        if j > n_max:
                            n_max = j
                unpooled_qubits = [i for i in range(n_max + 1)]
            else:
                unpooled_qubits = [i for i in self.left_qubits]

            for i in range(len(input_list)):
                for j in output_list[i]:
                    if j not in input_list[i]:
                        raise ValueError(
                            "The qubits adressed in the output "
                            "are not in the respective input list."
                        )

                if len(input_list[i]) <= len(output_list[i]):
                    raise ValueError(
                        "The sublists in the input list do not all have at least "
                        "one qubit more then those in the output list."
                    )

                if len(output_list[i]) == 0:
                    raise ValueError(
                        "At least one qubit must be in the sublists in the output list."
                    )

                if len(input_list[i]) != quantum_circuit.num_qubits:
                    raise ValueError(
                        "Not all sublists in the input list match the "
                        "number of qubits of the input circuit."
                    )

                for j in input_list[i]:
                    if j in unpooled_qubits:
                        unpooled_qubits.remove(j)
                    else:
                        raise ValueError(
                            "The sublists in the input list either adress the same "
                            "qubit or qubits which are not in the current circuit."
                        )

            # if the given in- and outputlists are in a proper shape
            if _new_operation:
                self._operations_list.append(
                    ["Pool", quantum_circuit, label, measurement, input_list, output_list]
                )
            if self.num_qubits > 0:
                self._num_parameters += quantum_circuit.num_parameters * len(input_list)
                if measurement:
                    self._num_measurements += quantum_circuit.num_clbits * len(input_list)
                left_qubits = [i for i in self.left_qubits]
                for i in range(len(input_list)):  # keep track of the qubits left in the circuit
                    for j in input_list[i]:
                        if j not in output_list[i]:
                            left_qubits.remove(j)
                self._left_qubits = left_qubits

    def fully_connected(
        self,
        quantum_circuit: Union[QuantumCircuit, EncodingCircuitBase, None] = None,
        label: str = "FC",
    ):
        """
        Add a fully connected layer to the encoding circuit.

        The fully connected layer should be placed at the end
        and operates on all qubits remaining in the circuit.

        Args:
            quantum_circuit Union[EncodingCircuitBase,QuantumCircuit, None]:
                The quantum circuit, which is applied in this layer.
            label: The name of the layer.
        """
        self.__fully_connected(quantum_circuit, label)

    def __fully_connected(
        self,
        quantum_circuit: Union[QuantumCircuit, EncodingCircuitBase, None] = None,
        label: str = "FC",
        _new_operation: bool = True,
    ):
        """Internal function to allow internal _new_operation argument."""
        if (
            (not quantum_circuit) and (self.num_qubits > 0) and (not _new_operation)
        ):  # overwrite with the correct gate
            self._operations_list.remove(["FC", quantum_circuit, label])
            _new_operation = True
        if self.num_qubits == 0:
            if _new_operation:
                self._operations_list.append(["FC", quantum_circuit, label])
        else:
            # define default circuit
            if not quantum_circuit:
                param = ParameterVector("a", 2 * len(self.left_qubits))
                quantum_circuit = QuantumCircuit(len(self.left_qubits))
                n_param = 0
                for i in range(len(self.left_qubits)):
                    quantum_circuit.rz(param[n_param], i)
                    n_param += 1
                for i in range(len(self.left_qubits)):
                    for j in range(len(self.left_qubits)):
                        if i != j:
                            quantum_circuit.cx(i, j)
                for i in range(len(self.left_qubits)):
                    quantum_circuit.ry(param[n_param], i)
                    n_param += 1

            quantum_circuit = self.__convert_encoding_circuit(quantum_circuit)

            if quantum_circuit.num_qubits != len(self.left_qubits):
                raise ValueError(
                    "Warning on fully connected layer: The input circuit "
                    "controls a wrong amount of qubits: "
                    f"{quantum_circuit.num_qubits} qubits on input vs. "
                    f"{len(self.left_qubits)} qubits on the actual circuit."
                )
            self._num_parameters += quantum_circuit.num_parameters
            if _new_operation:
                self._operations_list.append(["FC", quantum_circuit, label])

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the QCNN encoding circuit.

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained.
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained.

        Return:
            Returns the circuit in Qiskit's QuantumCircuit format.
        """

        if self.num_qubits == 0:
            raise ValueError(
                "Firstly, a number of qubits must be provided. "
                "Either with 'set_params', or with 'build_circuit'."
            )

        total_qc = QuantumCircuit(
            self.num_qubits, self._num_measurements
        )  # keeps track of the whole encoding circuit

        # if it is asked for a intrinsic feature map
        num_features = len(features)
        if num_features > 0:
            feature_map = ParamZFeatureMap(self.num_qubits, num_features, 1).get_circuit(
                features=features, parameters=[1] * num_features
            )
            total_qc = total_qc.compose(feature_map)
            total_qc = total_qc.compose(feature_map)

        left_qubits = [
            i for i in range(self.num_qubits)
        ]  # keeps track of the qubits which can still be adressed
        i_param = 0  # counts the number of parameters
        i_pool = 0  # counts the number of pooling layers applied
        i_clbit = 0  # counts the number of clbits used
        for gate in self.operations_list:
            quantum_circuit = gate[1]  # get the circuit which is to apply
            qc_name = gate[2] + "_" + str(i_pool)  # set name of the layer
            n_params = quantum_circuit.num_parameters
            n_qubits = quantum_circuit.num_qubits
            n_clbits = quantum_circuit.num_clbits
            if gate[0] == "Conv":
                # define number of gates applied
                number_of_gates_1 = int(len(left_qubits) / n_qubits)
                number_of_gates_2 = 0
                if gate[3] and len(left_qubits) > n_qubits:
                    number_of_gates_2 = int(len(left_qubits) / n_qubits)

                # assign parameter and add gates to circuit
                for j in range(number_of_gates_1):
                    qc_out = quantum_circuit.assign_parameters(
                        parameters[i_param : i_param + n_params]
                    )
                    qc_out.name = qc_name
                    if not gate[4]:  # if different parameters are supposed to be used
                        i_param += n_params
                    total_qc = total_qc.compose(
                        circuit_to_gate(qc_out),
                        qubits=[
                            left_qubits[i]
                            for i in range(
                                j * n_qubits,
                                (j + 1) * n_qubits,
                            )
                        ],
                    )
                for j in range(number_of_gates_2):
                    qc_out = quantum_circuit.assign_parameters(
                        parameters[i_param : i_param + n_params]
                    )
                    qc_out.name = qc_name
                    if not gate[4]:
                        i_param += n_params
                    total_qc = total_qc.compose(
                        circuit_to_gate(qc_out),
                        qubits=[
                            left_qubits[i % len(left_qubits)]
                            for i in range(
                                j * n_qubits + 1,
                                (j + 1) * n_qubits + 1,
                            )
                        ],
                    )
                if gate[4]:
                    i_param += n_params

            elif gate[0] == "Pool":
                input_list = gate[4]
                output_list = gate[5]
                i_pool += 1
                left_qubits_1 = [i for i in left_qubits]
                if len(input_list) != 0:  # if a proper in- and output list is provided
                    for j in range(len(input_list)):
                        qc_out = quantum_circuit.assign_parameters(
                            parameters[i_param : i_param + n_params]
                        )
                        qc_out.name = qc_name
                        i_param += n_params
                        if gate[3]:  # measurement
                            total_qc = total_qc.compose(
                                circuit_to_instruction(qc_out),
                                qubits=input_list[j],
                                clbits=list(range(i_clbit, i_clbit + n_clbits)),
                            )
                            i_clbit += n_clbits
                        else:
                            total_qc = total_qc.compose(
                                circuit_to_gate(qc_out), qubits=input_list[j]
                            )
                        for i in input_list[j]:
                            if i not in output_list[j]:
                                left_qubits_1.remove(i)
                else:
                    number_of_gates = int(len(left_qubits) / n_qubits)
                    # assign parameter and add gates to circuit
                    for j in range(number_of_gates):
                        qc_out = quantum_circuit.assign_parameters(
                            parameters[i_param : i_param + n_params]
                        )
                        qc_out.name = qc_name
                        i_param += n_params
                        if gate[3]:  # measurement
                            total_qc = total_qc.compose(
                                circuit_to_instruction(qc_out),
                                qubits=[
                                    left_qubits[i]
                                    for i in range(
                                        j * n_qubits,
                                        (j + 1) * n_qubits,
                                    )
                                ],
                                clbits=list(range(i_clbit, i_clbit + n_clbits)),
                            )
                            i_clbit += n_clbits
                        else:
                            total_qc = total_qc.compose(
                                circuit_to_gate(qc_out),
                                qubits=[
                                    left_qubits[i]
                                    for i in range(
                                        j * n_qubits,
                                        (j + 1) * n_qubits,
                                    )
                                ],
                            )
                        for i in left_qubits[j * n_qubits + 1 : (j + 1) * n_qubits]:
                            left_qubits_1.remove(i)
                left_qubits = left_qubits_1

            elif gate[0] == "FC":
                # assign parameter and add gates to circuit
                qc_out = quantum_circuit.assign_parameters(
                    parameters[i_param : i_param + n_params]
                )
                qc_out.name = qc_name
                i_param += n_params
                total_qc = total_qc.compose(
                    circuit_to_gate(qc_out), qubits=[i for i in left_qubits]
                )
                break  # since FC should be at the end of the circuit
        return total_qc

    def repeat_layers(self, n_times: int = 0):
        """
        Repeat the already applied gates to simply build the circuit.

        This does not work with a pooling layer with supplied in- and output lists.

        Args:
            n_times (int): The number of times the already applied gates are repeatedly applied.
            default configuration: At least once applied and until less then 4 qubits are left
                and only once if there is no pooling gate applied.
        """

        if n_times == 0 and self.num_qubits == 0:
            n_times = 1
        operations_list = [i for i in self.operations_list]
        if n_times < 0:
            raise ValueError("The argument is negative.")
        elif n_times == 0:
            while True:
                if len(self.left_qubits) <= 1:
                    raise ValueError(
                        "Warning on repeat_layers: The actual circuit has too few qubits."
                    )
                pooled = False  # so that it will not continue forever adding convolution layers
                for operation in operations_list:
                    if operation[0] == "Conv":
                        self.convolution(*operation[1:])
                    elif operation[0] == "Pool":
                        self.pooling(*operation[1:])
                        pooled = True
                    if len(self.left_qubits) <= 3 and n_times > 0:
                        break
                if len(self.left_qubits) <= 3 or not pooled:
                    break
                n_times += 1
        else:
            for n in range(n_times):
                if len(self.left_qubits) <= 1 and self.num_qubits > 0:
                    raise ValueError(
                        "Warning on repeat_layers: The actual circuit has too few qubits."
                    )
                for operation in operations_list:
                    if operation[0] == "Conv":
                        self.convolution(*operation[1:])
                    elif operation[0] == "Pool":
                        self.pooling(*operation[1:])

    def default_circuit(self):
        """A default circuit for quickly building a QCNN."""
        if len(self.left_qubits) <= 1:
            self.fully_connected()
        else:
            self.convolution()
            self.pooling()
            if len(self.left_qubits) > 1:
                self.repeat_layers()
            self.fully_connected()

    def QCNNObservable(self, obs: Union[ObservableBase, str] = "Z") -> ObservableBase:
        """
        Build a fitting observable for the current circuit.

        This function should be called after beeing finished building the circuit.
        It maps the supplied observable on the circuit, so that only the left qubits
        are measured.

        Args:
            obs Union[ObservableBase, str]: A squlearn observable can be supplied
                with n-qubit measurements, where n can not exceed the number of left qubits.
                Alternatively, a string of a pauli gate (X, Y or Z) can be supplied and
                an observable of single qubit measurements, mapped on the left qubits,
                is build.

        Return:
            Returns the fitting observable.
        """

        if isinstance(obs, str):
            if obs not in ["X", "Y", "Z"]:
                raise ValueError(
                    "For 'obs' either provide an 'ObservableBase' type or"
                    " a 'str' type Pauli gate (X, Y or Z)."
                )
            else:
                pauli = obs
            obs = SummedPaulis(len(self.left_qubits), op_str=pauli)

        obs.set_map(self.left_qubits, self.num_qubits)
        param = ParameterVector("p", obs.num_parameters)
        paulis = obs.get_pauli_mapped(param).paulis
        operator_list = []
        for i in paulis:
            operator_list.append(str(i))
        obs1 = CustomObservable(
            num_qubits=self.num_qubits,
            operator_string=operator_list,
            parameterized=(obs.num_parameters > 0),
        )
        return obs1

    def __convert_encoding_circuit(self, quantum_circuit) -> QuantumCircuit:
        """Internal function to allow also sQUlearn encoding circuits as input."""
        if not isinstance(quantum_circuit, QuantumCircuit):
            param = ParameterVector("p", quantum_circuit.num_parameters)
            if quantum_circuit.num_features > 0:
                raise ValueError(
                    "No features are allowed in the QCNN ansatz. "
                    "Please provide a circuit without features instead."
                )
            quantum_circuit = quantum_circuit.get_circuit([], param)
        return quantum_circuit

    def build_circuit(self, final_num_qubits: int = 1):
        """
        Build the circuit "backwards".

        Build the circuit by supplying the number of qubits which should be left
        after the already supplied gates. This function then generates the necessary number
        of initial qubits and applies the supplied gates.

        Args:
            final_num_qubits (int):
                The number of qubits which should be left after applying the supplied gates.
        """
        for operation in self.operations_list[::-1]:
            if operation[0] == "Pool":  # only pooling layers matter for the number of qubits
                input_list = operation[4]
                output_list = operation[5]
                quantum_circuit = operation[1]
                if len(output_list) == 0:
                    final_num_qubits *= quantum_circuit.num_qubits
                else:
                    len_input_list = 0
                    for i in input_list:
                        len_input_list += len(i)
                    len_output_list = 0
                    for i in output_list:
                        len_output_list += len(i)
                    if len_output_list > final_num_qubits:
                        raise ValueError(
                            f"The number of final qubits ({final_num_qubits}"
                            ") is to few to fit to the output of the last pooling layer "
                            f"(number of output qubits: {len(output_list)}) provided."
                        )
                    final_num_qubits = len_input_list + final_num_qubits - len_output_list
        self.set_params(num_qubits=final_num_qubits)
