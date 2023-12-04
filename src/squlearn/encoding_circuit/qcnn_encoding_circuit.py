import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_gate, circuit_to_instruction

from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from squlearn.encoding_circuit.circuit_library.param_z_feature_map import ParamZFeatureMap
from squlearn.observables import CustomObservable


class QcnnEncodingCircuit(EncodingCircuitBase):

    """
    Instantiate the object for building a qcnn.
    In order to plot your circuit (called e.g. mycircuit) at any given point use: mycircuit.draw().
    One can hand over the optional arguments "mpl" to get the circuit in matplotlib drawing and "decompose = True"
    to resolve the individual components of the circuit building blocks
    (e.g. mycircuit.draw("mpl", decompose = True)).

    Args:
        num_qubits (int): Number of qubits of the qcnn encoding circuit.
        num_features (int): Dimension of the feature vector. By default this is 0,
            so an feature map must be provided (or can be changed to x by set_num_features(x)).
            If num_features is bigger then 0, then in the get_circuit function a ZFeatureMap is build.
        default (bool): If True, the default circuit is build.
    """

    def __init__(self, num_qubits : int = 8, num_features = 0, default = False) -> None:
        super().__init__(num_qubits, num_features)
        self._num_parameters = 0  # counts the number of parameters used
        self._left_qubits = [i for i in range(num_qubits)] # stores, how many qubits can be controlled yet
        self._operations_list = []    #stores the operations applied in the qcnn
        if default:
            self.default_circuit()

    @property
    def num_parameters(self) -> int:
        """ Returns the number of trainable parameters of the current encoding circuit. """
        return self._num_parameters
    
    @property
    def left_qubits(self) -> list:
        """ Returns the qubits which one can operate on in the current circuit as list. """
        return self._left_qubits
    
    @property
    def operations_list(self) -> list:
        """ Returns the list of operators currently acting on the encoding circuit. """
        return self._operations_list

    def set_num_features(self,n : int = 0):
        """ Change the number of features in the feature map of the circuit. """
        if n > 0:
            self._num_features = n

    def convolution(self, QC = "None", label : str = "C", alternating : bool = True, diff_params : bool = True):
        """
        Add a convolution layer to the encoding circuit.

        Args:
            QC Union[EncodingCircuitBase,QuantumCircuit]: The quantum circuit, which is applied on every qubit
                modulo qubits of this circuit.
            label (str): Sets the name of the operation.
            alternating (bool): It applies the QC on every qubit modulo qubits of this circuit beginning at 0
                and if True it applies the QC on every qubit beginning at 1 again.
            diff_params (bool): If True, different parameters are used for the gates build by this layer.
        """
        #Define default circuit
        if QC == "None":
            param = ParameterVector("a", 3)
            QC = QuantumCircuit(2)
            QC.rz(-np.pi / 2, 1)
            QC.cx(1, 0)
            QC.rz(param[0], 0)
            QC.ry(param[1], 1)
            QC.cx(0, 1)
            QC.ry(param[2], 1)
            QC.cx(1, 0)
            QC.rz(np.pi / 2, 0)
        
        QC = self.convert_encoding_circuit(QC) #to allow also encoding circuits as input

        if QC.num_qubits > len(self.left_qubits):
            print("Warning on convolutional layer: The quantum circuit input controls too many qubits:",
                QC.num_qubits,"qubits on input vs.",len(self.left_qubits),"qubits on the actual circuit.")
        else:
            #define number of gates applied
            if diff_params:
                number_of_gates_1 = int(len(self.left_qubits)/QC.num_qubits)
                number_of_gates_2 = 0
                if alternating: number_of_gates_2 = int((len(self.left_qubits)-1)/QC.num_qubits)
                self._num_parameters += QC.num_parameters*(number_of_gates_1+number_of_gates_2)
            else:
                self._num_parameters += QC.num_parameters
            self.operations_list.append(["C", QC, label, alternating, diff_params])
            
    def pooling(self, QC = "None", label : str = "P", measurement : bool = False,
        input_list : list = [], output_list : list = []):
        """
        Add a convolution layer to the encoding circuit. This reduces the number of qubits to operate on
        from here on in this circuit to 1 for each circuit applied plus those on which no circuit operates on.

        Args:
        QC Union[EncodingCircuitBase,QuantumCircuit]: Must be an entangling layer, which entangles qubits.
        label (str): Sets the name of the operation.
        measurement (bool): Sets whether the qubits, which are not used anymore after this layer, are measured.
            If True, QC must consist of exactly one classical bit additionally to the quantum bits.
        input_layer (list): optionally one can pass the order of the entangling gates acting
            if the structure is [[qubit1,qubit2,..],[qubit3,qubit4,..],..]. Every qubit can only be adressed once
            and the number of qubits in each list within the list must be equal to the number of qubits of QC.
        output_layer (list): optionally one can add the output list which defines the qubit
            which is kept in the circuit in the structure [[qubit1],[qubit4],..].
        Both input and output list must have the same length and in each output sublist must be one element
            of the correspoding input sublist, while the latter must have a length longer then one.
        THE QUBITS IN THE LISTS REFER TO THE QUBIT NUMBERS IN THE CIRCUIT BEFORE APPLYING ANY GATES!    
        Default circuit: Entangles qubit i and qubit i+1. Only qubit i stays in the circuit for further operations.
        """
        #define default circuit
        if QC == "None":
            param = ParameterVector("a", 3)
            if measurement:
                QC = QuantumCircuit(2, 1)
            else:
                QC = QuantumCircuit(2)
            QC.rz(-np.pi / 2, 0)
            QC.cx(0, 1)
            QC.rz(param[0], 1)
            QC.ry(param[1], 0)
            if measurement:
                QC.measure(1,0)
                QC.y(0).c_if(0,1)
            else:
                QC.cx(1, 0)
            QC.ry(param[2], 0)

        QC = self.convert_encoding_circuit(QC) #to allow also encoding circuits as input

        if (measurement and QC.num_clbits != 1) or (not measurement and QC.num_clbits == 1):
            print("Warning on pooling layer: Eather set measurement to True and provide a Circuit with exactly",
                " one classical bit or set measurement to False.")
            return

        if QC.num_qubits > len(self.left_qubits):
            print("Warning on pooling layer: The quantum circuit input controls too many qubits:",
                QC.num_qubits,"qubits on input vs.",len(self.left_qubits),"qubits on the actual circuit.")
        else:
            #in case a predefined order is given test whether it is in a proper structure
            Found_error = False
            if len(input_list) != len(output_list):
                print("Lists have not the same length")
                Found_error = True
                
            if len(input_list) == 0:
                Found_error = True

            unpooled_qubits = [i for i in self.left_qubits]
            for i in range(len(input_list)):
                if (len(input_list[i]) <= len(output_list[i]) or len(output_list[i]) != 1
                    or output_list[i][0] not in input_list[i] or len(input_list[i]) != QC.num_qubits):
                    print("the shapes of the lists are not correct.")
                    Found_error = True
                    break

                for j in input_list[i]:
                    if j in unpooled_qubits:
                        unpooled_qubits.remove(j)
                    else:
                        print("the qubits adressed in the input_list are incorrect.")
                        Found_error = True
                        break

            left_qubits = [i for i in self.left_qubits]
            if not Found_error: # if the given in- and outputlists are in a proper shape
                self._num_parameters += QC.num_parameters*len(input_list)
                self.operations_list.append(["P", QC, label, measurement, input_list, output_list])
                for i in range(len(input_list)): #to keep track of the qubits left in the circuit
                    for j in input_list[i]:
                        if j not in output_list[i]:
                            left_qubits.remove(j)
            else: # if no proper in- and outputlists are given
                number_of_gates = int(len(self.left_qubits)/QC.num_qubits)
                self._num_parameters += QC.num_parameters*number_of_gates
                self.operations_list.append(["P", QC, label, measurement, [], []])
                for j in range(number_of_gates):
                    for i in self.left_qubits[j*QC.num_qubits+1:(j+1)*QC.num_qubits]:
                        left_qubits.remove(i)
            self._left_qubits = left_qubits

                
    def fully_connected(self, QC = "None", label : str = "F"):
        """
        Final layer which is added right before measurement. It operates on all qubits remaining in the circuit.

        Args:
        QC Union[EncodingCircuitBase,QuantumCircuit]: Must be a gate, which adresses all qubits left
            and should be placed at the end of the circuit, right before measurement.
        label: Sets the name of the operation.
        """
        #define default circuit
        if QC == "None":
                param = ParameterVector("a", len(self.left_qubits))
                QC = QuantumCircuit(len(self.left_qubits))
                for i in range(len(self.left_qubits)):
                    QC.rx(param[i],i)
                for i in range(len(self.left_qubits)):
                    for j in range(i+1,len(self.left_qubits)):
                        QC.cx(i, j)

        QC = self.convert_encoding_circuit(QC) #to allow also encoding circuits as input

        if QC.num_qubits != len(self.left_qubits):
            print("Warning on fully connected layer: The quantum circuit input controls a wrong amount of qubits:",
                QC.num_qubits,"qubits on input vs.",len(self.left_qubits),"qubits on the actual circuit.")
        else:
            self._num_parameters += QC.num_parameters
            self.operations_list.append(["F", QC, label])


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
        total_QC = QuantumCircuit(self.num_qubits,1) #keeps track of the whole encoding circuit
        
        #if it is asked for a intrinsic feature map
        num_features = len(features)
        if num_features > 0:
            feature_map = ParamZFeatureMap(self.num_qubits, num_features, 1).get_circuit(
                features=features,
                parameters=[1]*num_features
                )
            total_QC = total_QC.compose(feature_map)
            total_QC = total_QC.compose(feature_map)
       
        left_qubits = [i for i in range(self.num_qubits)] #keeps track of the qubits which can be still adressed
        i_param = 0 #counts the number of parameters
        i_layer = 0 #counts the number of pooling layers applied
        for gate in self.operations_list:
            QC = gate[1] #get the circuit which is to apply
            QC.name = gate[2]+"_"+str(i_layer) #set name of the operation

            if gate[0] == "C":
                #define number of gates applied
                number_of_gates_1 = int(len(left_qubits)/QC.num_qubits)
                number_of_gates_2 = 0
                if gate[3]: number_of_gates_2 = int((len(left_qubits)-1)/QC.num_qubits)

                #assign parameter and add gates to circuit
                for j in range(number_of_gates_1):
                    QC.assign_parameters(parameters[i_param:i_param+QC.num_parameters], True)
                    if gate[4]: #if different parameters are supposed to be used
                        i_param += QC.num_parameters
                    total_QC = total_QC.compose(
                        circuit_to_gate(QC),
                        qubits = [left_qubits[i] for i in range(j*QC.num_qubits,(j+1)*QC.num_qubits)]
                        )
                for j in range(number_of_gates_2):
                    QC.assign_parameters(parameters[i_param:i_param+QC.num_parameters], True)
                    if gate[4]:
                        i_param += QC.num_parameters
                    total_QC = total_QC.compose(
                        circuit_to_gate(QC),
                        qubits = [left_qubits[i] for i in range(j*QC.num_qubits+1,(j+1)*QC.num_qubits+1)]
                        )
                if not gate[4]:
                    i_param += QC.num_parameters

            elif gate[0] == "P":
                ###continue
                input_list = gate[4]
                output_list = gate[5]
                i_layer += 1
                left_qubits_1 = [i for i in left_qubits]
                if len(input_list) != 0: #if a proper in- and output list is provided
                    for j in range(len(input_list)):
                        QC.assign_parameters(parameters[i_param:i_param+QC.num_parameters], True)
                        i_param += QC.num_parameters
                        if gate[3]:#measurement
                            total_QC = total_QC.compose(
                                circuit_to_instruction(QC),
                                qubits = input_list[j],
                                clbits = [0]
                                )
                        else:
                            total_QC = total_QC.compose(circuit_to_gate(QC), qubits = input_list[j])
                        for i in input_list[j]:
                            if i not in output_list[j]:
                                left_qubits_1.remove(i)
                else:
                    number_of_gates = int(len(left_qubits)/QC.num_qubits)
                    #assign parameter and add gates to circuit
                    for j in range(number_of_gates):
                        QC.assign_parameters(parameters[i_param:i_param+QC.num_parameters], True)
                        i_param += QC.num_parameters
                        if gate[3]:#measurement
                            total_QC = total_QC.compose(
                                circuit_to_instruction(QC),
                                qubits = [left_qubits[i] for i in range(j*QC.num_qubits,(j+1)*QC.num_qubits)],
                                clbits = [0]
                                )
                        else:
                            total_QC = total_QC.compose(
                                circuit_to_gate(QC),
                                qubits = [left_qubits[i] for i in range(j*QC.num_qubits,(j+1)*QC.num_qubits)]
                                )
                        for i in left_qubits[j*QC.num_qubits+1:(j+1)*QC.num_qubits]:
                            left_qubits_1.remove(i)
                left_qubits = left_qubits_1

            elif gate[0] == "F":
                #assign parameter and add gates to circuit
                QC.assign_parameters(parameters[i_param:i_param+QC.num_parameters], True)
                i_param += QC.num_parameters
                total_QC = total_QC.compose(circuit_to_gate(QC), qubits = [i for i in left_qubits])
                break #after the "F" gate the circuit construction ends and measurement has to follow
        return total_QC

    def repeat_layers(self, n_times : int = -1):
        """
        Repeat the already applied gates to simply build the circuit.

        Args:
        n_times (int): The number of times the already applied gates are repeatedly applied (n_times !> 0).
        default configuration: At least once applied and until less then 4 qubits are left
            and only once if there is no pooling gate applied.
        """
        operations_list = [i for i in self.operations_list]
        if n_times < 0:
            while True:
                if len(self.left_qubits) <= 1:
                    print("The circuit has too few qubits")
                    break
                pooled = False #so that it will not continue forever adding convolution operations
                for operation in operations_list:
                    if operation[0] == "C":
                        self.convolution(*operation[1:])
                    elif operation[0] == "P":
                        self.pooling(*operation[1:])
                        pooled = True
                if len(self.left_qubits) <= 3 or not pooled:
                    break
        else:
            for n in range(n_times):
                if len(self.left_qubits) <= 1:
                    print("The circuit has too few qubits to continue")
                    break
                for operation in operations_list:
                    if operation[0] == "C":
                        self.convolution(*operation[1:])
                    elif operation[0] == "P":
                        self.pooling(*operation[1:])

    def default_circuit(self):
        """
        A default circuit for quickly building a qcnn.
        """
        if len(self.left_qubits) <= 1:
                print("The circuit has too few qubits")
                return
        self.convolution()
        self.pooling()
        if len(self.left_qubits) > 1:
            self.repeat_layers()
        self.fully_connected()

    def QcnnObservable(self, pauli : str = "Z"):
        """
        This function should be called after beeing finished building the circuit,
        because it builds the fitting observable for the current circuit.

        Args:
        pauli (str): Its the used pauli gate so ether X,Y or Z.
        """
        if pauli not in ["X","Y","Z"]:
            pauli = "Z"
            
        observable_list = []
        for i in self.left_qubits:
            observable = ""
            for j in range(self.num_qubits):
                if i == j:
                    observable = pauli + observable #most right qubit in operator refers to q0 
                else:
                    observable = "I" + observable
            observable_list.append(observable)
        return CustomObservable(num_qubits=self.num_qubits, operator_string=observable_list, parameterized=True)

    def convert_encoding_circuit(self,QC):#to allow also encoding circuits as input
        if not isinstance(QC,QuantumCircuit):
            param = ParameterVector("p", QC.num_parameters)
            if QC.num_features > 0:
                print("features parameters are set to 1 since no features are allowed in the QCNN ansatz.")
            QC = QC.get_circuit([1]*QC.num_features,param)
        return QC