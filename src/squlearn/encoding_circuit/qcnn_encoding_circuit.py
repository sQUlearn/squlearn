import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_gate, circuit_to_instruction

from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from squlearn.encoding_circuit.circuit_library.param_z_feature_map import ParamZFeatureMap
from squlearn.observables import CustomObservable


class QCNNEncodingCircuit(EncodingCircuitBase):

    """
    Encoding circuit for quantum convolutional neural networks (QCNN).

    The structure is inspired by classical neural networks.
    Therefore the number of qubits it is operated on is reduced from layer to layer.
    For more information read:
    "Quantum convolutional neural networks" by Cong, Iris and Choi, Soonwon and Lukin, Mikhail D.

    Args:
        num_qubits (int): Number of initial qubits of the QCNN encoding circuit.
        num_features (int): Dimension of the feature vector.
            By default this is 0, so a feature map must be provided.
            If the number of features is bigger then 0,
            then in the get_circuit function a ZFeatureMap is built to encode the features.
        default (bool): If True, the default circuit is built.
    """

    def __init__(self, num_qubits: int = 0, num_features: int = 0, default: bool = False) -> None:
        super().__init__(num_qubits, num_features)
        self._num_parameters = 0
        self._left_qubits = [i for i in range(num_qubits)]
        self._operations_list = []
        self._default = default
        if default:
            if num_qubits == 0:
                print("To generate a default circuit provide a number of qubits > 0.")
            else:
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

    def set_num_qubits(self, n: int = 0):
        """
        Set the number of qubits of the circuit.

        If the number of qubits is reduced and the supplied pooling gates do not fit to this,
        this will throw a error in the troubling layer.
        
        Args:
            n (int): The number of qubits in the new circuit.
        """

        if n > 0:
            if self.num_qubits == n:
                print("The number of qubits did not change.")
            else:
                self._num_qubits = n
                self._left_qubits = [i for i in range(n)]
                self._num_parameters = 0
                if self._default:
                    self._operations_list = []
                    self.default_circuit()
                else:
                    for operation in self.operations_list:
                        if operation[0] == "Conv":
                            self.convolution(*operation[1:], _new_operation=False)
                        elif operation[0] == "Pool":
                            self.pooling(*operation[1:], _new_operation=False)
                        elif operation[0] == "FC":
                            self.fully_connected(*operation[1:], _new_operation=False)
                            break  #since FC should be at the end of the circuit

    def set_num_features(self, n: int = 0):
        """
        Change the number of features in the feature map of the circuit.
        
        Args:
            n (int): The new number of features.
        """

        if n > 0:
            self._num_features = n

    def convolution(
        self,
        quantum_circuit: Union[QuantumCircuit, EncodingCircuitBase, None] = None,
        label: str = "Conv",
        alternating: bool = True,
        diff_params: bool = True,
        _new_operation: bool = True,
    ):
        """
        Add a convolution layer to the encoding circuit.

        Args:
            quantum_circuit Union[EncodingCircuitBase, QuantumCircuit, None]:
                The quantum circuit, which is applied in this layer.
            label (str): The name of the layer.
            alternating (bool): The gate is applied on every qubit modulo qubits of this circuit
                beginning at 0. If True it applies the gate on every qubit beginning at 1 again.
            diff_params (bool): If True, different parameters are used for the gates in this layer.
        """
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

        quantum_circuit = self._convert_encoding_circuit(quantum_circuit)
        if self.num_qubits == 0:
            if _new_operation:
                self._operations_list.append(
                    ["Conv", quantum_circuit, label, alternating, diff_params]
                )
        else:
            if quantum_circuit.num_qubits > len(self.left_qubits):
                print(
                    "Warning on convolutional layer: ",
                    "The input circuit controls too many qubits:",
                    quantum_circuit.num_qubits,
                    "qubits on input vs.",
                    len(self.left_qubits),
                    "qubits on the actual circuit.",
                )
            else:
                #define number of gates applied
                if diff_params:
                    number_of_gates_1 = int(len(self.left_qubits) / quantum_circuit.num_qubits)
                    number_of_gates_2 = 0
                    if alternating:
                        number_of_gates_2 = int(
                            (len(self.left_qubits) - 1) / quantum_circuit.num_qubits
                        )
                    self._num_parameters += quantum_circuit.num_parameters * (
                        number_of_gates_1 + number_of_gates_2
                    )
                else:
                    self._num_parameters += quantum_circuit.num_parameters
                if _new_operation:
                    self._operations_list.append(
                        ["Conv", quantum_circuit, label, alternating, diff_params]
                    )


    def pooling(
        self,
        quantum_circuit: Union[QuantumCircuit, EncodingCircuitBase, None] = None,
        label: str = "Pool",
        measurement: bool = False,
        input_list: list = [],
        output_list: list = [],
        _new_operation: bool = True,
    ):
        """
        Add a pooling layer to the encoding circuit.
        
        This reduces the number of qubits to operate on from here on in this circuit
        by at least one for each circuit applied.

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
        output_list (list): Exactly if an input list is entered, an output list must be entered.
            The output list defines the qubits which are left in the circuit to operate on.
            The list should be structured as: [[qubit1,qubit2,..],[qubit3,qubit4,..],..].
            It must have the same length as the input list and in each sublist
            its elements must be in the corresponding input sublist
            while beeing at least one element less.
        THE QUBIT NUMBERS IN THE LISTS REFER TO THE INITIAL QUBIT NUMBERS!
        Default circuit: Entangles qubit i and qubit i+1.
            Only qubit i stays in the circuit for further operations.
        """
        # define default circuit
        if not quantum_circuit:
            param = ParameterVector("a", 3)
            if measurement:
                quantum_circuit = QuantumCircuit(2, 1)
            else:
                quantum_circuit = QuantumCircuit(2)
            quantum_circuit.rz(-np.pi / 2, 0)
            quantum_circuit.cx(0, 1)
            quantum_circuit.rz(param[0], 1)
            quantum_circuit.ry(param[1], 0)
            if measurement:
                quantum_circuit.measure(1, 0)
                quantum_circuit.y(0).c_if(0, 1)
            else:
                quantum_circuit.cx(1, 0)
            quantum_circuit.ry(param[2], 0)

        quantum_circuit = self._convert_encoding_circuit(quantum_circuit)
        Found_error = False
        if (measurement and quantum_circuit.num_clbits != 1) or (
            not measurement and quantum_circuit.num_clbits == 1
        ):
            print(
                "Warning on pooling layer: Eather set measurement to True and provide a ",
                "circuit with exactly one classical bit or set measurement to False.",
            )
            Found_error = True

        if quantum_circuit.num_qubits > len(self.left_qubits) and self.num_qubits > 0:
            print(
                "Warning on pooling layer: The input circuit controls too many qubits:",
                quantum_circuit.num_qubits,
                "qubits on input vs.",
                len(self.left_qubits),
                "qubits on the actual circuit.",
            )
            Found_error = True

        if not Found_error:
            if len(output_list) + len(input_list) == 0: #if no input and output lists are given
                if _new_operation:
                    self._operations_list.append(
                        ["Pool", quantum_circuit, label, measurement, input_list, output_list]
                    )
                if self.num_qubits > 0:
                    number_of_gates = int(len(self.left_qubits) / quantum_circuit.num_qubits)
                    self._num_parameters += quantum_circuit.num_parameters * number_of_gates
                    left_qubits = [i for i in self.left_qubits]
                    for j in range(number_of_gates):
                        for i in self.left_qubits[
                            j * quantum_circuit.num_qubits
                            + 1 : (j + 1) * quantum_circuit.num_qubits
                        ]:
                            left_qubits.remove(i)
                    self._left_qubits = left_qubits
                Found_error = True #to skip the input list part

            #in case a predefined order is given, test whether it is in a proper structure
            if len(input_list) != len(output_list):
                print("The lists do not have the same length.")
                Found_error = True

            if self.num_qubits == 0:
                n_max = 0
                for i in input_list:
                    for j in i:
                        if j > n_max:
                            n_max = j
                unpooled_qubits = [i for i in range(n_max)]
            else:
                unpooled_qubits = [i for i in self.left_qubits]

            for i in range(len(input_list)):
                for j in output_list[i]:
                    if j not in input_list[i]:
                        print(
                            "The qubits adressed in the output ",
                            "are not in the respective input list."
                        )
                        Found_error = True
                        break
                
                if Found_error == True: #For early ending
                    break

                if len(input_list[i]) <= len(output_list[i]):
                    print(
                        "The sublists in the input list do not all have at least ",
                        "one qubit more then those in the output list."
                        )
                    Found_error = True
                    break

                if len(output_list[i]) == 0:
                    print("At least one qubit must be in the sublists in the output list.")
                    Found_error = True
                    break

                if len(input_list[i]) != quantum_circuit.num_qubits:
                    print(
                        "Not all sublists in the input list match the ",
                        "number of qubits of the input circuit."
                        )
                    Found_error = True
                    break

                for j in input_list[i]:
                    if j in unpooled_qubits:
                        unpooled_qubits.remove(j)
                    else:
                        print(
                            "The sublists in the input list either adress the same ",
                            "qubit or qubits which are not in the current circuit."
                            )
                        Found_error = True
                        break

            #if the given in- and outputlists are in a proper shape
            if not Found_error:
                if _new_operation:
                    self._operations_list.append(
                        ["Pool", quantum_circuit, label, measurement, input_list, output_list]
                    )
                if self.num_qubits > 0:
                    self._num_parameters += quantum_circuit.num_parameters * len(input_list)
                    left_qubits = [i for i in self.left_qubits]
                    for i in range(
                        len(input_list)
                    ):  #to keep track of the qubits left in the circuit
                        for j in input_list[i]:
                            if j not in output_list[i]:
                                left_qubits.remove(j)
                    self._left_qubits = left_qubits


    def fully_connected(
        self,
        quantum_circuit: Union[QuantumCircuit, EncodingCircuitBase, None] = None,
        label: str = "FC",
        _new_operation: bool = True
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
        if (
            (not quantum_circuit) and (self.num_qubits > 0) and (not _new_operation)
        ): #overwrite with the correct gate
            self._operations_list.remove(["FC", quantum_circuit, label])
            _new_operation = True
        if self.num_qubits == 0:
            if _new_operation:
                self._operations_list.append(["FC", quantum_circuit, label])
        else:
            #define default circuit
            if not quantum_circuit:
                param = ParameterVector("a", len(self.left_qubits))
                quantum_circuit = QuantumCircuit(len(self.left_qubits))
                for i in range(len(self.left_qubits)):
                    quantum_circuit.rx(param[i], i)
                for i in range(len(self.left_qubits)):
                    for j in range(i + 1, len(self.left_qubits)):
                        quantum_circuit.cx(i, j)

            quantum_circuit = self._convert_encoding_circuit(quantum_circuit)

            if quantum_circuit.num_qubits != len(self.left_qubits):
                print(
                    "Warning on fully connected layer: The input circuit ",
                    "controls a wrong amount of qubits:",
                    quantum_circuit.num_qubits,
                    "qubits on input vs.",
                    len(self.left_qubits),
                    "qubits on the actual circuit.",
                )
            else:
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
            print(
                "Firstly, a number of qubits must be provided. Either with 'set_num_qubits', or with 'build circuit'."
            )
            return QuantumCircuit(self.num_qubits, 1)

        total_qc = QuantumCircuit(self.num_qubits, 1)  # keeps track of the whole encoding circuit

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
        ]  # keeps track of the qubits which can be still adressed
        i_param = 0  # counts the number of parameters
        i_layer = 0  # counts the number of pooling layers applied
        for gate in self.operations_list:
            quantum_circuit = gate[1]  # get the circuit which is to apply
            quantum_circuit.name = gate[2] + "_" + str(i_layer)  # set name of the operation

            if gate[0] == "Conv":
                # define number of gates applied
                number_of_gates_1 = int(len(left_qubits) / quantum_circuit.num_qubits)
                number_of_gates_2 = 0
                if gate[3]:
                    number_of_gates_2 = int((len(left_qubits) - 1) / quantum_circuit.num_qubits)

                # assign parameter and add gates to circuit
                for j in range(number_of_gates_1):
                    quantum_circuit.assign_parameters(
                        parameters[i_param : i_param + quantum_circuit.num_parameters], True
                    )
                    if gate[4]:  # if different parameters are supposed to be used
                        i_param += quantum_circuit.num_parameters
                    total_qc = total_qc.compose(
                        circuit_to_gate(quantum_circuit),
                        qubits=[
                            left_qubits[i]
                            for i in range(
                                j * quantum_circuit.num_qubits,
                                (j + 1) * quantum_circuit.num_qubits,
                            )
                        ],
                    )
                for j in range(number_of_gates_2):
                    quantum_circuit.assign_parameters(
                        parameters[i_param : i_param + quantum_circuit.num_parameters], True
                    )
                    if gate[4]:
                        i_param += quantum_circuit.num_parameters
                    total_qc = total_qc.compose(
                        circuit_to_gate(quantum_circuit),
                        qubits=[
                            left_qubits[i]
                            for i in range(
                                j * quantum_circuit.num_qubits + 1,
                                (j + 1) * quantum_circuit.num_qubits + 1,
                            )
                        ],
                    )
                if not gate[4]:
                    i_param += quantum_circuit.num_parameters

            elif gate[0] == "Pool":
                ###continue
                input_list = gate[4]
                output_list = gate[5]
                i_layer += 1
                left_qubits_1 = [i for i in left_qubits]
                if len(input_list) != 0:  # if a proper in- and output list is provided
                    for j in range(len(input_list)):
                        quantum_circuit.assign_parameters(
                            parameters[i_param : i_param + quantum_circuit.num_parameters], True
                        )
                        i_param += quantum_circuit.num_parameters
                        if gate[3]:  # measurement
                            total_qc = total_qc.compose(
                                circuit_to_instruction(quantum_circuit),
                                qubits=input_list[j],
                                clbits=[0],
                            )
                        else:
                            total_qc = total_qc.compose(
                                circuit_to_gate(quantum_circuit), qubits=input_list[j]
                            )
                        for i in input_list[j]:
                            if i not in output_list[j]:
                                left_qubits_1.remove(i)
                else:
                    number_of_gates = int(len(left_qubits) / quantum_circuit.num_qubits)
                    # assign parameter and add gates to circuit
                    for j in range(number_of_gates):
                        quantum_circuit.assign_parameters(
                            parameters[i_param : i_param + quantum_circuit.num_parameters], True
                        )
                        i_param += quantum_circuit.num_parameters
                        if gate[3]:  # measurement
                            total_qc = total_qc.compose(
                                circuit_to_instruction(quantum_circuit),
                                qubits=[
                                    left_qubits[i]
                                    for i in range(
                                        j * quantum_circuit.num_qubits,
                                        (j + 1) * quantum_circuit.num_qubits,
                                    )
                                ],
                                clbits=[0],
                            )
                        else:
                            total_qc = total_qc.compose(
                                circuit_to_gate(quantum_circuit),
                                qubits=[
                                    left_qubits[i]
                                    for i in range(
                                        j * quantum_circuit.num_qubits,
                                        (j + 1) * quantum_circuit.num_qubits,
                                    )
                                ],
                            )
                        for i in left_qubits[
                            j * quantum_circuit.num_qubits
                            + 1 : (j + 1) * quantum_circuit.num_qubits
                        ]:
                            left_qubits_1.remove(i)
                left_qubits = left_qubits_1

            elif gate[0] == "FC":
                # assign parameter and add gates to circuit
                quantum_circuit.assign_parameters(
                    parameters[i_param : i_param + quantum_circuit.num_parameters], True
                )
                i_param += quantum_circuit.num_parameters
                total_qc = total_qc.compose(
                    circuit_to_gate(quantum_circuit), qubits=[i for i in left_qubits]
                )
                break  # after the "FC" gate the circuit construction ends and measurement has to follow
        return total_qc

    def repeat_layers(self, n_times: int = 0):
        """
        Repeat the already applied gates to simply build the circuit.

        This does not work with a pooling layer with supplied in- and output lists.

        Args:
        n_times (int): The number of times the already applied gates are repeatedly applied (n_times !> 0).
        default configuration: At least once applied and until less then 4 qubits are left
            and only once if there is no pooling gate applied.
        """
        if n_times == 0 and self.num_qubits == 0:
            n_times = 1
        operations_list = [i for i in self.operations_list]
        if n_times < 0:
            print("The argument is negative.")
        elif n_times == 0:
            while True:
                if len(self.left_qubits) <= 1:
                    print("The circuit has too few qubits")
                    break
                pooled = (
                    False  # so that it will not continue forever adding convolution operations
                )
                for operation in operations_list:
                    if operation[0] == "Conv":
                        self.convolution(*operation[1:])
                    elif operation[0] == "Pool":
                        self.pooling(*operation[1:])
                        pooled = True
                if len(self.left_qubits) <= 3 or not pooled:
                    break
        else:
            for n in range(n_times):
                if len(self.left_qubits) <= 1 and self.num_qubits > 0:
                    print("The circuit has too few qubits to continue")
                    break
                for operation in operations_list:
                    if operation[0] == "Conv":
                        self.convolution(*operation[1:])
                    elif operation[0] == "Pool":
                        self.pooling(*operation[1:])

    def default_circuit(self):
        """
        A default circuit for quickly building a QCNN.
        """
        if len(self.left_qubits) <= 1:
            print("The circuit has too few qubits")
        else:
            self.convolution()
            self.pooling()
            if len(self.left_qubits) > 1:
                self.repeat_layers()
            self.fully_connected()

    def QCNNObservable(self, pauli: str = "Z") -> CustomObservable:
        """
        Build a fitting observable for the current circuit.

        This function should be called after beeing finished building the circuit.

        Args:
        pauli (str): Its the used pauli gate so either X,Y or Z.
        """
        if pauli not in ["X", "Y", "Z"]:
            pauli = "Z"

        observable_list = []
        for i in self.left_qubits:
            observable = ""
            for j in range(self.num_qubits):
                if i == j:
                    observable = pauli + observable  # most right qubit in operator refers to q0
                else:
                    observable = "I" + observable
            observable_list.append(observable)
        return CustomObservable(
            num_qubits=self.num_qubits, operator_string=observable_list, parameterized=True
        )

    def _convert_encoding_circuit(self, quantum_circuit) -> QuantumCircuit:
        """Internal function to allow also sQUlearn encoding circuits as input."""
        if not isinstance(quantum_circuit, QuantumCircuit):
            param = ParameterVector("p", quantum_circuit.num_parameters)
            if quantum_circuit.num_features > 0:
                print(
                    "features parameters are set to 1 since no features are allowed in the QCNN ansatz."
                )
            quantum_circuit = quantum_circuit.get_circuit(
                [1] * quantum_circuit.num_features, param
            )
        return quantum_circuit

    def build_circuit(self, final_num_qubits: int = 1):
        """
        Build the circuit "backwards".

        Build the circuit by supplying the number of qubits which should be left after the already supplied gates.
        This function then generates the necessary number of initial qubits and applies the supplied gates.

        Args:
        final_num_qubits (int): The number of qubits which should be left after applying the supplied gates.
        """
        for operation in self.operations_list[::-1]:
            if operation[0] == "Pool":  # only pooling layers matter for the number of qubits
                output_list = operation[5]
                quantum_circuit = operation[1]
                if len(output_list) == 0:
                    final_num_qubits *= quantum_circuit.num_qubits
                else:
                    if len(output_list) > final_num_qubits:
                        print(
                            "The number of final qubits (",
                            final_num_qubits,
                            ") is to few to fit to the last pooling layer (number of final qubits: ",
                            len(output_list),
                            ") provided.",
                        )
                        break
                    final_num_qubits = quantum_circuit.num_qubits * int(
                        final_num_qubits / len(output_list)
                    ) + final_num_qubits % len(output_list)
        self.set_num_qubits(final_num_qubits)
