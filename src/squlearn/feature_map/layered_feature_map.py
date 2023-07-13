from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RXGate, RYGate, RZGate, PhaseGate, UGate
from qiskit.circuit.library import CPhaseGate, CHGate, CXGate, CYGate, CZGate
from qiskit.circuit.library import SwapGate, CRXGate, CRYGate, CRZGate, RXXGate
from qiskit.circuit.library import RYYGate, RZXGate, RZZGate, CUGate
from typing import Union, Callable

# is needed for making feature maps with numpy functions using strings:
import numpy as np

from .feature_map_base import FeatureMapBase


class VariableGroup:
    """
    class for one variable group e.g. x1, x2, p1,..., which saves the dimension of one variable
    """

    def __init__(self, variable_name: str, size=None):
        """
        Args:
            variable_name [String]: the name of the variable type, which one can see, if he draws the circuit of a feature map with this variable group
            size (int): The dimension of the variable group
            index (int): The index of the variable group (only important for creating the circuit)
            Only if size is not given:
            total_variables_used (int): counter, which saves the number of variables used (only important, if size not given, so the dimension can potentially be infinity)
        """
        self.variable_name = variable_name
        self.size = size
        self.index = 0
        if size == None:
            self.total_variables_used = 0

    def __hash__(self):
        """
        creates a hash with the name of the variable
        """
        return hash(self.variable_name)

    @property
    def num_variables(self):
        """
        returns the total number of variables that have been used;
        if number of size counts until infinity, cause size is not given, than return counter, else return size
        """
        if self.size == None:
            return self.total_variables_used
        return self.size

    def get_param_vector(self):
        """Creates a parameter vector by qiskit"""
        return ParameterVector(self.variable_name, self.num_variables)

    def increase_used_number_of_variables(self, number_of_used_variables: int):
        """Increases total number of variables , if size not given
        (this is important for initializing the parameter vectors of qiskit with get_number_of_variables)
        """
        if self.size == None:
            self.total_variables_used += number_of_used_variables
        pass

    def increase_index(self, number_of_used_variables: int):
        """Only for get_circuit: Building the circuit, it increases the index of the variable group"""
        self.index += number_of_used_variables

    def set_index_to_zero(self):
        """Sets the index to zero"""
        self.index = 0


class _operation:
    """
    parent class for a quantum operation. Each gate layer stands for one operation.
    """

    def __init__(self, num_qubits: int, variablegroup_tuple: tuple, map=None):
        """
        Attributes:
            num_qubits: The number of all qubits
            variablegroup_tuple: A tuple with every variable group used in this operation
            map: A default map, that is used, if the operation has exactly 2 variable groups and no given map (by user)
            default_map: A boolean, that checks, if the user initializes his own map
        """
        self.num_qubits = num_qubits
        self.variablegroup_tuple = variablegroup_tuple
        if map == None:
            # Default: Set map to x*y (if two arguments given, if there are more than two arguments for one operation without any given map, an error will be raised)
            self.map = lambda x, y: x * y
            self.default_map = True
        else:
            self.map = map
            self.default_map = False

    def get_circuit(self, var_param_assignment: dict):
        return None


class _H_operation(_operation):
    """class for a H operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC.h(range(self.num_qubits))
        return QC


class _X_operation(_operation):
    """class for a X operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC.x(range(self.num_qubits))
        return QC


class _Y_operation(_operation):
    """class for a Y operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC.y(range(self.num_qubits))
        return QC


class _Z_operation(_operation):
    """class for a Z operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC.z(range(self.num_qubits))
        return QC


class _Id_operation(_operation):
    """class for an identity operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC.id(range(self.num_qubits))
        return QC


class _S_operation(_operation):
    """class for a S operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC.s(range(self.num_qubits))
        return QC


class _S_conjugate_operation(_operation):
    """class for a conjugated S operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC.sdg(range(self.num_qubits))
        return QC


class _T_operation(_operation):
    """class for a conjugated S operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC.t(range(self.num_qubits))
        return QC


class _T_conjugate_operation(_operation):
    """class for a conjugated T operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC.tdg(range(self.num_qubits))
        return QC


class _rot_operation(_operation):
    def __init__(self, num_qubits: int, variablegroup_tuple: tuple, map=None):
        super().__init__(num_qubits, variablegroup_tuple, map)

    def apply_param_vectors(self, QC, r_star, var_param_assignment):
        """
        Applies the param vectors creating a quantum circuit with the given gate.
        Contains an algorithm for Rx, Ry and Rz gates, which distinguish between the following cases:
        First: The user sets no map so a default map is given and the number of variable groups used exceeds 2, than raise an error because the default map needs exactly two arguments.
        Second: The user sets no map so a default map is given and the number of variable groups used is 1, than apply the r* (stands for rx, ry or rz) gate with the given variable but without the map.
        Third: The user sets a map with some variable groups or there are exactly 2 variable groups given.
        Args:
            QC: the quantum circuit by qiskit
            r_star: Stands for r* (rx, ry or rz); a RXGate, RYGate or RZGate (also PhaseGate)
            var_param_assignment: a dictionary, that assigns hash values of variable groups with their parameter vectors (by qiskit)
        Returns:
            QuantumCircuit
        Raises:
            ValueError, if the first case occurs.

        """
        if self.default_map and (len(self.variablegroup_tuple) > 2):
            raise ValueError(
                "There are too many variable groups given without a map. There can only be one or two parameters without any given map."
            )
        elif self.default_map and (len(self.variablegroup_tuple) == 1):
            if self.variablegroup_tuple[0].size == None:
                for qubit in range(self.num_qubits):
                    QC.append(
                        r_star(
                            var_param_assignment[hash(self.variablegroup_tuple[0])][
                                self.variablegroup_tuple[0].index
                            ]
                        ),
                        [qubit],
                        [],
                    )
                    self.variablegroup_tuple[0].increase_index(1)
            else:
                for qubit in range(self.num_qubits):
                    QC.append(
                        r_star(
                            var_param_assignment[hash(self.variablegroup_tuple[0])][
                                (self.variablegroup_tuple[0].index)
                                % self.variablegroup_tuple[0].size
                            ]
                        ),
                        [qubit],
                        [],
                    )
                    self.variablegroup_tuple[0].increase_index(1)
        else:
            # So there is either two variable groups given or a given map. In the first case without given map it multiplies the variables together (using of the default map)
            for qubit in range(self.num_qubits):
                buffer_param_vectors_list = []
                for variablegroup in self.variablegroup_tuple:
                    if variablegroup.size == None:
                        buffer_param_vectors_list.append(
                            var_param_assignment[hash(variablegroup)][variablegroup.index]
                        )
                    else:
                        buffer_param_vectors_list.append(
                            var_param_assignment[hash(variablegroup)][
                                variablegroup.index % variablegroup.size
                            ]
                        )
                    variablegroup.increase_index(1)
                QC.append(r_star(self.map(*buffer_param_vectors_list)), [qubit], [])
        return QC


class _Rx_operation(_rot_operation):
    """class for a Rx operation"""

    def get_circuit(self, var_param_assignment: dict):
        """
        Args:
            var_param_assignment: a dictionary, that assigns hash values of variable groups with their parameter vectors (by qiskit)
        returns:
            QuantumCircuit (by qiskit)
        """
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, RXGate, var_param_assignment)
        return QC


class _Ry_operation(_rot_operation):
    """class for a Ry operation"""

    def get_circuit(self, var_param_assignment: dict):
        """
        Args:
            var_param_assignment: a dictionary, that assigns hash values of variable groups with their parameter vectors (by qiskit)
        returns:
            QuantumCircuit (by qiskit)
        """
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, RYGate, var_param_assignment)
        return QC


class _Rz_operation(_rot_operation):
    """class for a Rz operation"""

    def get_circuit(self, var_param_assignment: dict):
        """
        Args:
            var_param_assignment: a dictionary, that assigns hash values of variable groups with their parameter vectors (by qiskit)
        returns:
            QuantumCircuit (by qiskit)
        """
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, RZGate, var_param_assignment)
        return QC


class _P_operation(_rot_operation):
    """class for a P operation"""

    def get_circuit(self, var_param_assignment: dict):
        """
        Args:
            var_param_assignment: a dictionary, that assigns hash values of variable groups with their parameter vectors (by qiskit)
        returns:
            QuantumCircuit (by qiskit)
        """
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, PhaseGate, var_param_assignment)
        return QC


class _U_operation(_operation):
    """class for an U operation"""

    def get_circuit(self, var_param_assignment: dict):
        """
        Args:
            var_param_assignment: a dictionary, that assigns hash values of variable groups with their parameter vectors (by qiskit)
        returns:
            QuantumCircuit (by qiskit)
        """
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, UGate, var_param_assignment)
        return QC

    def apply_param_vectors(self, QC, u_gate, var_param_assignment):
        """
        Applies the param vectors creating a quantum circuit with the given gate.
        works much like the apply_param_vectors function in rot_operation but without maps
        """
        for qubit in range(self.num_qubits):
            buffer_param_vectors_list = []
            for variablegroup in self.variablegroup_tuple:
                if variablegroup.size == None:
                    buffer_param_vectors_list.append(
                        var_param_assignment[hash(variablegroup)][variablegroup.index]
                    )
                else:
                    buffer_param_vectors_list.append(
                        var_param_assignment[hash(variablegroup)][
                            variablegroup.index % variablegroup.size
                        ]
                    )
                variablegroup.increase_index(1)
            QC.append(u_gate(*buffer_param_vectors_list), [qubit], [])
        return QC


class _two_qubit_operation(_operation):
    """
    parent class for any two-qubit operation
    """

    def __init__(self, num_qubits: int, variablegroup_tuple: tuple, ent_strategy: str, map=None):
        super().__init__(num_qubits, variablegroup_tuple, map)
        self.ent_strategy = ent_strategy

    def apply_param_vectors(self, QC, gate, var_param_assignment: dict):
        """
        Applies the param vectors creating a quantum circuit with the given gate.
        We have to distinguish between the following cases:
        1st: Default map exists (or not)
        2nd: NN or AA
        3rd: Variable group is given
        4th: Dimension (size) is finite (or infinite)
        """
        if self.variablegroup_tuple == None:
            if self.ent_strategy == "AA":
                for first_qubit in range(self.num_qubits - 1):
                    for second_qubit in range(first_qubit + 1, self.num_qubits):
                        QC.append(gate(), [first_qubit, second_qubit], [])
            elif self.ent_strategy == "NN":
                for first_qubit in range(0, self.num_qubits - 1, 2):
                    QC.append(gate(), [first_qubit, first_qubit + 1], [])
                for first_qubit in range(1, self.num_qubits - 1, 2):
                    QC.append(gate(), [first_qubit, first_qubit + 1], [])
            else:
                raise ValueError("Wrong entangling strategy input.")
        elif self.default_map and len(self.variablegroup_tuple) > 2:
            raise ValueError(
                "In two qubit operation: There are too many variable groups given without a map. There can only be one or two parameters without any given map."
            )
        elif self.default_map and len(self.variablegroup_tuple) == 1:
            if self.ent_strategy == "AA":
                if self.variablegroup_tuple[0].size == None:
                    for first_qubit in range(self.num_qubits - 1):
                        for second_qubit in range(first_qubit + 1, self.num_qubits):
                            QC.append(
                                gate(
                                    var_param_assignment[hash(self.variablegroup_tuple[0])][
                                        self.variablegroup_tuple[0].index
                                    ]
                                ),
                                [first_qubit, second_qubit],
                                [],
                            )
                            self.variablegroup_tuple[0].increase_index(1)
                else:
                    for first_qubit in range(self.num_qubits - 1):
                        for second_qubit in range(first_qubit + 1, self.num_qubits):
                            QC.append(
                                gate(
                                    var_param_assignment[hash(self.variablegroup_tuple[0])][
                                        self.variablegroup_tuple[0].index
                                        % self.variablegroup_tuple[0].size
                                    ]
                                ),
                                [first_qubit, second_qubit],
                                [],
                            )
                            self.variablegroup_tuple[0].increase_index(1)
            elif self.ent_strategy == "NN":
                if self.variablegroup_tuple[0].size == None:
                    for first_qubit in range(0, self.num_qubits - 1, 2):
                        QC.append(
                            gate(
                                var_param_assignment[hash(self.variablegroup_tuple[0])][
                                    self.variablegroup_tuple[0].index
                                ]
                            ),
                            [first_qubit, first_qubit + 1],
                            [],
                        )
                        self.variablegroup_tuple[0].increase_index(1)
                    for first_qubit in range(1, self.num_qubits - 1, 2):
                        QC.append(
                            gate(
                                var_param_assignment[hash(self.variablegroup_tuple[0])][
                                    self.variablegroup_tuple[0].index
                                ]
                            ),
                            [first_qubit, first_qubit + 1],
                            [],
                        )
                        self.variablegroup_tuple[0].increase_index(1)
                else:
                    for first_qubit in range(0, self.num_qubits - 1, 2):
                        QC.append(
                            gate(
                                var_param_assignment[hash(self.variablegroup_tuple[0])][
                                    self.variablegroup_tuple[0].index
                                    % self.variablegroup_tuple[0].size
                                ]
                            ),
                            [first_qubit, first_qubit + 1],
                            [],
                        )
                        self.variablegroup_tuple[0].increase_index(1)
                    for first_qubit in range(1, self.num_qubits - 1, 2):
                        QC.append(
                            gate(
                                var_param_assignment[hash(self.variablegroup_tuple[0])][
                                    self.variablegroup_tuple[0].index
                                    % self.variablegroup_tuple[0].size
                                ]
                            ),
                            [first_qubit, first_qubit + 1],
                            [],
                        )
                        self.variablegroup_tuple[0].increase_index(1)
            else:
                raise ValueError("Wrong entangling strategy input.")
        else:
            if self.ent_strategy == "AA":
                for first_qubit in range(self.num_qubits - 1):
                    for second_qubit in range(first_qubit + 1, self.num_qubits):
                        buffer_param_vectors_list = []
                        for variablegroup in self.variablegroup_tuple:
                            if variablegroup.size == None:
                                buffer_param_vectors_list.append(
                                    var_param_assignment[hash(variablegroup)][variablegroup.index]
                                )
                            else:
                                buffer_param_vectors_list.append(
                                    var_param_assignment[hash(variablegroup)][
                                        variablegroup.index % variablegroup.size
                                    ]
                                )
                            variablegroup.increase_index(1)
                        QC.append(
                            gate(self.map(*buffer_param_vectors_list)),
                            [first_qubit, second_qubit],
                            [],
                        )
            elif self.ent_strategy == "NN":
                for first_qubit in range(0, self.num_qubits - 1, 2):
                    buffer_param_vectors_list = []
                    for variablegroup in self.variablegroup_tuple:
                        if variablegroup.size == None:
                            buffer_param_vectors_list.append(
                                var_param_assignment[hash(variablegroup)][variablegroup.index]
                            )
                        else:
                            buffer_param_vectors_list.append(
                                var_param_assignment[hash(variablegroup)][
                                    variablegroup.index % variablegroup.size
                                ]
                            )
                        variablegroup.increase_index(1)
                    QC.append(
                        gate(self.map(*buffer_param_vectors_list)),
                        [first_qubit, first_qubit + 1],
                        [],
                    )
                for first_qubit in range(1, self.num_qubits - 1, 2):
                    buffer_param_vectors_list = []
                    for variablegroup in self.variablegroup_tuple:
                        if variablegroup.size == None:
                            buffer_param_vectors_list.append(
                                var_param_assignment[hash(variablegroup)][variablegroup.index]
                            )
                        else:
                            buffer_param_vectors_list.append(
                                var_param_assignment[hash(variablegroup)][
                                    variablegroup.index % variablegroup.size
                                ]
                            )
                        variablegroup.increase_index(1)
                    QC.append(
                        gate(self.map(*buffer_param_vectors_list)),
                        [first_qubit, first_qubit + 1],
                        [],
                    )
            else:
                raise ValueError("Wrong entangling strategy input.")
        return QC


class _CH_entangle_operation(_two_qubit_operation):
    """Default class for a controlled x entangling operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, CHGate, var_param_assignment)
        return QC


class _CX_entangle_operation(_two_qubit_operation):
    """Default class for a controlled x entangling operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, CXGate, var_param_assignment)
        return QC


class _CY_entangle_operation(_two_qubit_operation):
    """Default class for a controlled y entangling operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, CYGate, var_param_assignment)
        return QC


class _CZ_entangle_operation(_two_qubit_operation):
    """Default class for a controlled z entangling operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, CZGate, var_param_assignment)
        return QC


class _SWAP_operation(_two_qubit_operation):
    """class for a SWAP operation"""

    def get_circuit(self, var_param_assignment=None):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, SwapGate, var_param_assignment)
        return QC


class _CRX_operation(_two_qubit_operation):
    """class for a controlled rx entangling operation"""

    def get_circuit(self, var_param_assignment: dict):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, CRXGate, var_param_assignment)
        return QC


class _CRY_operation(_two_qubit_operation):
    """class for a controlled ry entangling operation"""

    def get_circuit(self, var_param_assignment: dict):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, CRYGate, var_param_assignment)
        return QC


class _CRZ_operation(_two_qubit_operation):
    """class for a controlled rz entangling operation"""

    def get_circuit(self, var_param_assignment: dict):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, CRZGate, var_param_assignment)
        return QC


class _CP_operation(_two_qubit_operation):
    """class for a controlled phase entangling operation"""

    def get_circuit(self, var_param_assignment: dict):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, CPhaseGate, var_param_assignment)
        return QC


class _RXX_operation(_two_qubit_operation):
    """class for a RZX operation"""

    def get_circuit(self, var_param_assignment: dict):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, RXXGate, var_param_assignment)
        return QC


class _RYY_operation(_two_qubit_operation):
    """class for a RZX operation"""

    def get_circuit(self, var_param_assignment: dict):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, RYYGate, var_param_assignment)
        return QC


class _RZX_operation(_two_qubit_operation):
    """class for a RZX operation"""

    def get_circuit(self, var_param_assignment: dict):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, RZXGate, var_param_assignment)
        return QC


class _RZZ_operation(_two_qubit_operation):
    """class for a RZZ operation"""

    def get_circuit(self, var_param_assignment: dict):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, RZZGate, var_param_assignment)
        return QC


class _CU_operation(_two_qubit_operation):
    """class for a controlled u entangling operation"""

    def get_circuit(self, var_param_assignment: dict):
        QC = QuantumCircuit(self.num_qubits)
        QC = self.apply_param_vectors(QC, CUGate, var_param_assignment)
        return QC

    def apply_param_vectors(self, QC, cu_gate, var_param_assignment):
        """
        Applies the param vectors creating a quantum circuit with the given gate.
        Works much like the apply_param_vectors function in two_qubit_operation, but without distinguishing of maps (or variable groups because is there no variable groups given, it will raise an error).
        We have to distinguish between the following cases:
        1st: NN or AA
        2nd: Dimension (size) is finite (or infinite)
        """
        if self.ent_strategy == "AA":
            for first_qubit in range(self.num_qubits - 1):
                for second_qubit in range(first_qubit + 1, self.num_qubits):
                    buffer_param_vectors_list = []
                    for variablegroup in self.variablegroup_tuple:
                        if variablegroup.size == None:
                            buffer_param_vectors_list.append(
                                var_param_assignment[hash(variablegroup)][variablegroup.index]
                            )
                        else:
                            buffer_param_vectors_list.append(
                                var_param_assignment[hash(variablegroup)][
                                    variablegroup.index % variablegroup.size
                                ]
                            )
                        variablegroup.increase_index(1)
                    QC.append(
                        cu_gate(*buffer_param_vectors_list),
                        [first_qubit, second_qubit],
                        [],
                    )
        elif self.ent_strategy == "NN":
            for first_qubit in range(0, self.num_qubits - 1, 2):
                buffer_param_vectors_list = []
                for variablegroup in self.variablegroup_tuple:
                    if variablegroup.size == None:
                        buffer_param_vectors_list.append(
                            var_param_assignment[hash(variablegroup)][variablegroup.index]
                        )
                    else:
                        buffer_param_vectors_list.append(
                            var_param_assignment[hash(variablegroup)][
                                variablegroup.index % variablegroup.size
                            ]
                        )
                    variablegroup.increase_index(1)
                QC.append(
                    cu_gate(*buffer_param_vectors_list),
                    [first_qubit, first_qubit + 1],
                    [],
                )
            for first_qubit in range(1, self.num_qubits - 1, 2):
                buffer_param_vectors_list = []
                for variablegroup in self.variablegroup_tuple:
                    if variablegroup.size == None:
                        buffer_param_vectors_list.append(
                            var_param_assignment[hash(variablegroup)][variablegroup.index]
                        )
                    else:
                        buffer_param_vectors_list.append(
                            var_param_assignment[hash(variablegroup)][
                                variablegroup.index % variablegroup.size
                            ]
                        )
                    variablegroup.increase_index(1)
                QC.append(
                    cu_gate(*buffer_param_vectors_list),
                    [first_qubit, first_qubit + 1],
                    [],
                )
        else:
            raise ValueError("Wrong entangling strategy input.")
        return QC


class LayeredPQC:
    """The main class. Contains a list of operations. With that one can build his circuit."""

    def __init__(self, num_qubits: int, variable_groups=None):
        """
        Takes number of qubits.
        Attributes:
            num_qubits (int): Number of qubits in this feature map
            operation_list [list]: List of objects of the class operation with the tuple of the variablegroups used for each operation and the number of variables used in that operation, e.g. [[_H_operation,None], [Rx_operation, (x_var,x_var2), [5,5],...]
            variable_groups [tuple]: Tuple of all variable groups used in this feature map; ATTENTION: If there is only one variable group, be sure to type in "(x,)" and not "(x)" initializing the feature map
            Only if variable_groups is not None:
            variable_name_tuple [tuple]: Tuple of variable names of each variablegroup e.g. variablegroup x_var, x_var2, p_var; variable_name_tuple = (x,x2,p);
                this is only used to create feature maps with Strings
            variable_groups_string_tuple [tuple]: Tuple of the hash values for each variable group, with that, you can search the position of each variable_group,
                e.g. variable_groups = (x_var, x_var2,...) with type(x_var) = variable_group and variable_string_list = (hash(x_var),hash(x_var2),...)
        """
        self._num_qubits = num_qubits
        self.operation_list = []
        self.variable_groups = variable_groups
        if variable_groups != None:
            variable_groups_string_list = []
            variable_name_list = []
            for i in range(len(variable_groups)):
                variable_name_list.append(variable_groups[i].variable_name)
                variable_groups_string_list.append(hash(variable_groups[i]))
            self.variable_name_tuple = tuple(variable_name_list)
            self.variable_groups_string_tuple = tuple(variable_groups_string_list)

    @property
    def num_qubits(self):
        return self._num_qubits

    def add_operation(
        self, operation: _operation, variablegroup_tuple: tuple, variable_num_list=None
    ):
        """
        adds an operation to the operation_list
        Args:
            operation [operation]: an operation of the class operation
            variablegroup_tuple [tuple]: a tuple of variablegroups
            variable_num_list [list or None type]: gives information about how often a parameter is used in each operation;
                for example in a 5 qubit system with R_x-Layers: There are 5 (number of qubits) R_x-Gates used,
                whereas in nearest neighbour entangling there are only 4 (number of qubits - 1) variables per group used.
        """
        if variablegroup_tuple == None:
            self.operation_list.append([operation, None])
        else:
            # For the case that there are variables given but without an information about how often they are used, set variable_dif_list to [number of qubits] on default
            if variable_num_list == None:
                variable_num_list = [self.num_qubits for i in range(len(variablegroup_tuple))]
            # adds the operation with the tuple of the variable groups used for this operation and the number of variables used per group:
            self.operation_list.append([operation, variablegroup_tuple, variable_num_list])
            # counter of the variables: if size not given, that means there is no finite dimension, than increase the counter of this variablegroup by number of the qubits
            # otherwise use the size
            iteration_counter = 0
            for variablegroup in variablegroup_tuple:
                variablegroup.increase_used_number_of_variables(
                    variable_num_list[iteration_counter]
                )
                iteration_counter += 1

    def add_layer(self, layer, num_layers=1):
        """adds a layer of gates to the given feature map"""
        for i in range(num_layers):
            for operation_iter in layer.operation_list:
                if len(operation_iter) == 3:
                    self.add_operation(operation_iter[0], operation_iter[1], operation_iter[2])
                else:
                    self.add_operation(operation_iter[0], operation_iter[1], None)

    def get_number_of_variables(self, variablegroup: VariableGroup):
        """get how often the variable group was used (required for building parameter vectors by qiskit)"""
        return variablegroup.num_variables

    def get_circuit(self, *args):
        """
        returns the quantum circuit
        Args:
            *args: is a tuple with parameter vectors as its entries
        """

        # To avoid errors, we set the indices of all variable groups to zero first
        if self.variable_groups != None:
            for i in range(len(self.variable_groups)):
                self.variable_groups[i].set_index_to_zero()

        # Create assignment between variable groups and parameter vectors, e.g. {hash(x_var1):paramvec(x_var1),hash(x_var2):paramvec(x_var2),...}
        var_param_assignment = {hash(self.variable_groups[i]): args[i] for i in range(len(args))}
        QC = QuantumCircuit(self.num_qubits)
        for operation_variablegroup_iter in self.operation_list:
            operation_iter = operation_variablegroup_iter[0]
            variablegroup_tuple_iter = operation_variablegroup_iter[1]
            if variablegroup_tuple_iter == None:
                QC = QC.compose(operation_iter.get_circuit())
            else:
                QC = QC.compose(operation_iter.get_circuit(var_param_assignment))
        return QC

    def H(self):
        self.add_operation(_H_operation(self.num_qubits, None), None)

    def X(self):
        self.add_operation(_X_operation(self.num_qubits, None), None)

    def Y(self):
        self.add_operation(_Y_operation(self.num_qubits, None), None)

    def Z(self):
        self.add_operation(_Z_operation(self.num_qubits, None), None)

    def I(self):
        self.add_operation(_Id_operation(self.num_qubits, None), None)

    def S(self):
        self.add_operation(_S_operation(self.num_qubits, None), None)

    def S_conjugate(self):
        self.add_operation(_S_conjugate_operation(self.num_qubits, None), None)

    def T(self):
        self.add_operation(_T_operation(self.num_qubits, None), None)

    def T_conjugate(self):
        self.add_operation(_T_conjugate_operation(self.num_qubits, None), None)

    def Rx(self, *variablegroup_tuple, map=None):
        """
        variablegroup_tuple is a tuple of variable types (x1,x2 etc.)
        """
        if map == None:
            self.add_operation(
                _Rx_operation(self.num_qubits, variablegroup_tuple), variablegroup_tuple
            )
        else:
            self.add_operation(
                _Rx_operation(self.num_qubits, variablegroup_tuple, map),
                variablegroup_tuple,
            )

    def Ry(self, *variablegroup_tuple, map=None):
        """
        variablegroup_tuple is a tuple of variable types (x1,x2 etc.)
        """
        if map == None:
            self.add_operation(
                _Ry_operation(self.num_qubits, variablegroup_tuple), variablegroup_tuple
            )
        else:
            self.add_operation(
                _Ry_operation(self.num_qubits, variablegroup_tuple, map),
                variablegroup_tuple,
            )

    def Rz(self, *variablegroup_tuple, map=None):
        """
        variablegroup_tuple is a tuple of variable types (x1,x2 etc.)
        """
        if map == None:
            self.add_operation(
                _Rz_operation(self.num_qubits, variablegroup_tuple), variablegroup_tuple
            )
        else:
            self.add_operation(
                _Rz_operation(self.num_qubits, variablegroup_tuple, map),
                variablegroup_tuple,
            )

    def P(self, *variablegroup_tuple, map=None):
        if map == None:
            if len(variablegroup_tuple) != 1:
                raise ValueError("There must be one variable group for a P gate.")
            self.add_operation(
                _P_operation(self.num_qubits, variablegroup_tuple), variablegroup_tuple
            )
        else:
            self.add_operation(
                _P_operation(self.num_qubits, variablegroup_tuple, map),
                variablegroup_tuple,
            )

    def U(self, *variablegroup_tuple, map=None):
        if map == None:
            if len(variablegroup_tuple) != 3:
                raise ValueError("There must be three variable groups for a U gate.")
            self.add_operation(
                _U_operation(self.num_qubits, variablegroup_tuple), variablegroup_tuple
            )
        else:
            self.add_operation(
                _U_operation(self.num_qubits, variablegroup_tuple, map),
                variablegroup_tuple,
            )

    def ch_entangling(self, ent_strategy="NN"):
        """
        Adds a controlled x entangling layer.
        args:
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        self.add_operation(
            _CH_entangle_operation(self.num_qubits, None, ent_strategy, map=None), None
        )

    def cx_entangling(self, ent_strategy="NN"):
        """
        Adds a controlled x entangling layer.
        args:
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        self.add_operation(
            _CX_entangle_operation(self.num_qubits, None, ent_strategy, map=None), None
        )

    def cy_entangling(self, ent_strategy="NN"):
        """
        Adds a controlled y entangling layer.
        args:
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        self.add_operation(
            _CY_entangle_operation(self.num_qubits, None, ent_strategy, map=None), None
        )

    def cz_entangling(self, ent_strategy="NN"):
        """
        Adds a controlled z entangling layer.
        args:
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        self.add_operation(
            _CZ_entangle_operation(self.num_qubits, None, ent_strategy, map=None), None
        )

    def swap(self, ent_strategy="NN"):
        """
        Adds a swap gate layer.
        args:
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        self.add_operation(_SWAP_operation(self.num_qubits, None, ent_strategy, map=None), None)

    def cp_entangling(self, *variablegroup_tuple, ent_strategy="NN", map=None):
        """
        Adds a controlled phase entangling gate layer.
        args:
            *variablegroup_tuple: should be an empty tuple, because there are no variable groups needed
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        if ent_strategy == "NN":
            number_of_variables = self.num_qubits - 1
        else:
            number_of_variables = sum(x for x in range(1, self.num_qubits))
        variable_num_list = [number_of_variables for i in range(len(variablegroup_tuple))]
        self.add_operation(
            _CP_operation(self.num_qubits, variablegroup_tuple, ent_strategy, map),
            variablegroup_tuple,
            variable_num_list,
        )

    def crx_entangling(self, *variablegroup_tuple, ent_strategy="NN", map=None):
        """
        Adds a controlled rx gate layer.
        args:
            *variablegroup_tuple: should be an empty tuple, because there are no variable groups needed
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        if ent_strategy == "NN":
            number_of_variables = self.num_qubits - 1
        else:
            number_of_variables = sum(x for x in range(1, self.num_qubits))
        variable_num_list = [number_of_variables for i in range(len(variablegroup_tuple))]
        self.add_operation(
            _CRX_operation(self.num_qubits, variablegroup_tuple, ent_strategy, map),
            variablegroup_tuple,
            variable_num_list,
        )

    def cry_entangling(self, *variablegroup_tuple, ent_strategy="NN", map=None):
        """
        Adds a controlled ry gate layer.
        args:
            *variablegroup_tuple: should be an empty tuple, because there are no variable groups needed
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        if ent_strategy == "NN":
            number_of_variables = self.num_qubits - 1
        else:
            number_of_variables = sum(x for x in range(1, self.num_qubits))
        variable_num_list = [number_of_variables for i in range(len(variablegroup_tuple))]
        self.add_operation(
            _CRY_operation(self.num_qubits, variablegroup_tuple, ent_strategy, map),
            variablegroup_tuple,
            variable_num_list,
        )

    def crz_entangling(self, *variablegroup_tuple, ent_strategy="NN", map=None):
        """
        Adds a controlled rz gate layer.
        args:
            *variablegroup_tuple: should be an empty tuple, because there are no variable groups needed
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        if ent_strategy == "NN":
            number_of_variables = self.num_qubits - 1
        else:
            number_of_variables = sum(x for x in range(1, self.num_qubits))
        variable_num_list = [number_of_variables for i in range(len(variablegroup_tuple))]
        self.add_operation(
            _CRZ_operation(self.num_qubits, variablegroup_tuple, ent_strategy, map),
            variablegroup_tuple,
            variable_num_list,
        )

    def rxx_entangling(self, *variablegroup_tuple, ent_strategy="NN", map=None):
        """
        Adds rxx gate layer.
        args:
            *variablegroup_tuple: should be an empty tuple, because there are no variable groups needed
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        if ent_strategy == "NN":
            number_of_variables = self.num_qubits - 1
        else:
            number_of_variables = sum(x for x in range(1, self.num_qubits))
        variable_num_list = [number_of_variables for i in range(len(variablegroup_tuple))]
        self.add_operation(
            _RXX_operation(self.num_qubits, variablegroup_tuple, ent_strategy, map),
            variablegroup_tuple,
            variable_num_list,
        )

    def ryy_entangling(self, *variablegroup_tuple, ent_strategy="NN", map=None):
        """
        Adds a ryy gate layer.
        args:
            *variablegroup_tuple: should be an empty tuple, because there are no variable groups needed
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        if ent_strategy == "NN":
            number_of_variables = self.num_qubits - 1
        else:
            number_of_variables = sum(x for x in range(1, self.num_qubits))
        variable_num_list = [number_of_variables for i in range(len(variablegroup_tuple))]
        self.add_operation(
            _RYY_operation(self.num_qubits, variablegroup_tuple, ent_strategy, map),
            variablegroup_tuple,
            variable_num_list,
        )

    def rzx_entangling(self, *variablegroup_tuple, ent_strategy="NN", map=None):
        """
        Adds a rzx gate layer.
        args:
            *variablegroup_tuple: should be an empty tuple, because there are no variable groups needed
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        if ent_strategy == "NN":
            number_of_variables = self.num_qubits - 1
        else:
            number_of_variables = sum(x for x in range(1, self.num_qubits))
        variable_num_list = [number_of_variables for i in range(len(variablegroup_tuple))]
        self.add_operation(
            _RZX_operation(self.num_qubits, variablegroup_tuple, ent_strategy, map),
            variablegroup_tuple,
            variable_num_list,
        )

    def rzz_entangling(self, *variablegroup_tuple, ent_strategy="NN", map=None):
        """
        Adds a rzz gate layer.
        args:
            *variablegroup_tuple: should be an empty tuple, because there are no variable groups needed
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: a function for one or more variable groups
        """
        if ent_strategy == "NN":
            number_of_variables = self.num_qubits - 1
        else:
            number_of_variables = sum(x for x in range(1, self.num_qubits))
        variable_num_list = [number_of_variables for i in range(len(variablegroup_tuple))]
        self.add_operation(
            _RZZ_operation(self.num_qubits, variablegroup_tuple, ent_strategy, map),
            variablegroup_tuple,
            variable_num_list,
        )

    def cu_entangling(self, *variablegroup_tuple, ent_strategy="NN", map=None):
        """
        Adds a controlled unitary gate layer.
        args:
            *variablegroup_tuple: should be tuple with 4 entries
            Optional:
                ent_strategy: the entangling strategy (NN or AA)
                    Default ("NN"): Adds a controlled x nearest neighbour entangling operation
                    otherwise ("AA"): Adds a controlled x all in all entangling operation
                map: is not provided for a controlled unitary gate (raises Error if user gives a map)
        """
        if map != None:
            raise AttributeError("There must be no map for a cu entangling layer.")
        if ent_strategy == "NN":
            number_of_variables = self.num_qubits - 1
        else:
            number_of_variables = sum(x for x in range(1, self.num_qubits))
        variable_num_list = [number_of_variables for i in range(len(variablegroup_tuple))]
        self.add_operation(
            _CU_operation(self.num_qubits, variablegroup_tuple, ent_strategy, map),
            variablegroup_tuple,
            variable_num_list,
        )

    @classmethod
    def from_string(cls, num_qubits: int, gate_layers: str, variable_groups=None):
        """initializes a feature map through a given string of gates"""

        def generate_function(map_string, args):
            """Translates a string into a function"""
            function_string = """\
def math_function({var}):
    return {func}
            """.format(
                func=map_string, var=args
            )
            exec(function_string, globals())
            return math_function

        def get_closing_bracket_index(word, index):
            """gives to an open round bracket '(' the location of the closing bracket. This works especially, if there are more than one open brackets."""
            if word[index] != "(":
                raise ValueError("There must be an open bracket at index ", index)
            bracket_open_counter = 0
            for k in range(index, len(word)):
                if word[k] == ")":
                    bracket_open_counter -= 1
                elif word[k] == "(":
                    bracket_open_counter += 1

                if bracket_open_counter == 0:
                    return k
            raise ValueError("At least one closed bracket is missing.")

        def make_digit_list_to_number(digit_list):
            """Transforms a list of digit into a number"""
            number_string = "".join(digit_list)
            return int(number_string)

        featuremap = cls(num_qubits, variable_groups)
        gate_layers = gate_layers.replace(" ", "")
        string_iterator = 0
        featuremap_active = featuremap
        # Variable that detects, if all brackets "[" are closed (is needed for layers e.g. 3[H-X] etc.):
        closed_brackets = True
        while string_iterator < len(gate_layers):
            character_iter = gate_layers[string_iterator]
            if character_iter == "-":
                string_iterator += 1
            # ///////////////////////////////////////////////////////////////////////
            # To make layers of operations:
            elif character_iter.isdigit():
                # For the case that the layer number is in double digits or more, we use the following routine:
                digit_list = []
                while gate_layers[string_iterator].isdigit():
                    digit_list.append(gate_layers[string_iterator])
                    string_iterator += 1
                if gate_layers[string_iterator] != "[":
                    raise ValueError('To create different layers we need "[".')
                number_of_layers = make_digit_list_to_number(digit_list)
                featuremap_active = LayerPQC(featuremap)
            elif character_iter == "[":
                closed_brackets = False
                string_iterator += 1
            elif character_iter == "]":
                if closed_brackets == True:
                    raise ValueError("There are to many closed brackets.")
                closed_brackets = True
                featuremap.add_layer(featuremap_active, num_layers=number_of_layers)
                featuremap_active = featuremap
                string_iterator += 1
            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Operations without parameters:
            elif character_iter == "H":
                featuremap_active.H()
                string_iterator += 1
            elif character_iter == "X":
                featuremap_active.X()
                string_iterator += 1
            elif character_iter == "Y":
                featuremap_active.Y()
                string_iterator += 1
            elif character_iter == "Z":
                featuremap_active.Z()
                string_iterator += 1
            elif character_iter == "I":
                featuremap_active.I()
                string_iterator += 1
            # For the two following operations S and T there exists also the conjugated version: s_conjugated and t_conjugated ("Sc" and "Tc" not to be confused with "cs", which stands for the swap operation)
            elif character_iter == "S":
                # check first, if the entry gate_layers[string_iterator+1] exists, otherwise it will raise an error
                if string_iterator + 1 < len(gate_layers):
                    character_iter_1 = gate_layers[string_iterator + 1]
                    if character_iter_1 == "c":
                        featuremap_active.S_conjugate()
                        string_iterator += 2
                    else:
                        featuremap_active.S()
                        string_iterator += 1
                else:
                    featuremap_active.S()
                    string_iterator += 1
            elif character_iter == "T":
                # check first, if the entry gate_layers[string_iterator+1] exists, otherwise it will raise an error
                if string_iterator + 1 < len(gate_layers):
                    character_iter_1 = gate_layers[string_iterator + 1]
                    if character_iter_1 == "c":
                        featuremap_active.T_conjugate()
                        string_iterator += 2
                    else:
                        featuremap_active.T()
                        string_iterator += 1
                else:
                    featuremap_active.T()
                    string_iterator += 1
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Operations with parameters
            elif character_iter == "R":
                character_iter_1 = gate_layers[string_iterator + 1]

                # stores the index of the first open bracket in the word (e.g. Rx(...) so the index is 2)
                open_bracket_index = string_iterator + 2
                end_word = get_closing_bracket_index(gate_layers, open_bracket_index)

                # Now check, if there is a semicolon (the semicolon separates parameters from the map)
                semicolon1 = ";" in gate_layers[string_iterator:end_word]
                if semicolon1:
                    semicolon1_index = gate_layers.index(";", string_iterator)
                    # Store the parameters in a list:
                    param_vector_word_with_commas = gate_layers[
                        (string_iterator + 3) : semicolon1_index
                    ]
                    param_vector_name_list = param_vector_word_with_commas.split(",")
                    param_vector_list = []
                    # Assigning the parameter names to the right parameters:
                    for param_vector_name in param_vector_name_list:
                        param_index = featuremap.variable_name_tuple.index(param_vector_name)
                        param_vector_list.append(featuremap.variable_groups[param_index])
                    if gate_layers[semicolon1_index + 1] == "=":
                        # Evaluates all variables, that are stored in the brackets "{}", and creates a map with the given string of a function
                        map_comma_index = gate_layers.index(",", semicolon1_index)
                        map_string = gate_layers[semicolon1_index + 2 : map_comma_index]
                        bra = gate_layers.index("{", map_comma_index)
                        cket = gate_layers.index("}", map_comma_index)
                        map_args = gate_layers[bra + 1 : cket]
                        map_from_string = generate_function(map_string, map_args)
                    else:
                        raise ValueError("Wrong input 2.")

                    if character_iter_1 == "x":
                        featuremap_active.Rx(*param_vector_list, map=map_from_string)
                    elif character_iter_1 == "y":
                        featuremap_active.Ry(*param_vector_list, map=map_from_string)
                    elif character_iter_1 == "z":
                        featuremap_active.Rz(*param_vector_list, map=map_from_string)
                    else:
                        raise ValueError("Unknown rotation gate.")
                else:
                    # There is no semicolon so there is no given map. That means there must be exactly one parameter vector name, which will be assigned to its parameter vector in the following step:
                    param_vector_name = gate_layers[(string_iterator + 3) : end_word]
                    param_index = featuremap.variable_name_tuple.index(param_vector_name)
                    param_vector = featuremap.variable_groups[param_index]
                    if character_iter_1 == "x":
                        featuremap_active.Rx(param_vector)
                    elif character_iter_1 == "y":
                        featuremap_active.Ry(param_vector)
                    elif character_iter_1 == "z":
                        featuremap_active.Rz(param_vector)
                    else:
                        raise ValueError("Unknown rotation gate.")
                string_iterator = end_word + 1

            elif character_iter == "P":
                # stores the index of the first open bracket in the word (e.g. P(...) so the index is 1)
                open_bracket_index = string_iterator + 1
                end_word = get_closing_bracket_index(gate_layers, open_bracket_index)

                # Now check, if there is a semicolon (the semicolon separates parameters from the map)
                semicolon1 = ";" in gate_layers[string_iterator:end_word]
                if semicolon1:
                    semicolon1_index = gate_layers.index(";", string_iterator)
                    # Store the parameters in a list:
                    param_vector_word_with_commas = gate_layers[
                        (string_iterator + 2) : semicolon1_index
                    ]
                    param_vector_name_list = param_vector_word_with_commas.split(",")
                    param_vector_list = []
                    # Assigning the parameter names to the right parameters:
                    for param_vector_name in param_vector_name_list:
                        param_index = featuremap.variable_name_tuple.index(param_vector_name)
                        param_vector_list.append(featuremap.variable_groups[param_index])
                    if gate_layers[semicolon1_index + 1] == "=":
                        # Evaluates all variables, that are stored in the brackets "{}", and creates a map with the given string of a function
                        map_comma_index = gate_layers.index(",", semicolon1_index)
                        map_string = gate_layers[semicolon1_index + 2 : map_comma_index]
                        bra = gate_layers.index("{", map_comma_index)
                        cket = gate_layers.index("}", map_comma_index)
                        map_args = gate_layers[bra + 1 : cket]
                        map_from_string = generate_function(map_string, map_args)
                    else:
                        raise ValueError("Wrong input 2.")
                    featuremap_active.P(*param_vector_list, map=map_from_string)
                else:
                    # There is no semicolon so there is no given map. That means there must be exactly one parameter vector name, which will be assigned to its parameter vector in the following step:
                    param_vector_name = gate_layers[(string_iterator + 2) : end_word]
                    param_index = featuremap.variable_name_tuple.index(param_vector_name)
                    param_vector = featuremap.variable_groups[param_index]
                    featuremap_active.P(param_vector)
                string_iterator = end_word + 1
            elif character_iter == "U":
                open_bracket_index = string_iterator + 1
                end_word = get_closing_bracket_index(gate_layers, open_bracket_index)
                param_vector_word_with_commas = gate_layers[(string_iterator + 2) : end_word]
                param_vector_name_list = param_vector_word_with_commas.split(",")
                if len(param_vector_name_list) != 3:
                    raise ValueError("There must be exactly three parameters for an U gate.")
                param_vector_list = []
                for param_vector_name in param_vector_name_list:
                    param_index = featuremap.variable_name_tuple.index(param_vector_name)
                    param_vector_list.append(featuremap.variable_groups[param_index])
                featuremap_active.U(*param_vector_list)
                string_iterator = end_word + 1
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Beginning of the entangling layers:
            elif character_iter == "c" or character_iter == "r":
                character_iter_1 = gate_layers[string_iterator + 1]
                # //////////////////////////////////////////////////////////////////////////////////////////////
                # simple entangling layers (that don't require parameter vectors)
                # h, x, y, z, s gates work in the same way, we use a function pointer to handle all those gates at once:
                function_pointer = None
                if character_iter_1 == "h":
                    function_pointer = featuremap_active.ch_entangling
                elif character_iter_1 == "x":
                    function_pointer = featuremap_active.cx_entangling
                elif character_iter_1 == "y":
                    function_pointer = featuremap_active.cy_entangling
                elif character_iter_1 == "z":
                    function_pointer = featuremap_active.cz_entangling
                elif character_iter_1 == "s":
                    function_pointer = featuremap_active.swap

                # overwrite function pointer for rxx,ryy,... gates
                if character_iter == "r":
                    function_pointer = None

                if function_pointer != None:
                    if string_iterator + 2 < len(gate_layers):
                        character_iter_2 = gate_layers[string_iterator + 2]
                        if character_iter_2 == "(":
                            character_iter_3 = gate_layers[string_iterator + 3]
                            character_iter_4 = gate_layers[string_iterator + 4]
                            character_iter_5 = gate_layers[string_iterator + 5]
                            if character_iter_5 != ")":
                                raise ValueError("Unknown entangling strategy.")
                            if character_iter_3 == character_iter_4 == "A":
                                function_pointer(ent_strategy="AA")
                            elif character_iter_3 == character_iter_4 == "N":
                                function_pointer(ent_strategy="NN")
                            else:
                                raise ValueError("Unknown entangling strategy.")
                            string_iterator += 6
                        else:
                            function_pointer()
                            string_iterator += 2
                    else:
                        function_pointer()
                        string_iterator += 2
                # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
                # Entangling layers, that require parameter vectors
                else:
                    if character_iter_1 in ("r", "x", "y", "z"):
                        if string_iterator + 2 < len(gate_layers):
                            character_iter_2 = gate_layers[string_iterator + 2]
                        else:
                            raise ValueError("Wrong rotation entangling input.")

                        func = None
                        if character_iter_1 == "r" and character_iter_2 == "x":
                            func = featuremap_active.crx_entangling
                        elif character_iter_1 == "r" and character_iter_2 == "y":
                            func = featuremap_active.cry_entangling
                        elif character_iter_1 == "r" and character_iter_2 == "z":
                            func = featuremap_active.crz_entangling
                        elif character_iter_1 == "x" and character_iter_2 == "x":
                            func = featuremap_active.rxx_entangling
                        elif character_iter_1 == "y" and character_iter_2 == "y":
                            func = featuremap_active.ryy_entangling
                        elif character_iter_1 == "z" and character_iter_2 == "z":
                            func = featuremap_active.rzz_entangling
                        elif character_iter_1 == "z" and character_iter_2 == "x":
                            func = featuremap_active.rzx_entangling
                        else:
                            raise ValueError("Unknown rotation gate.")

                        # set entangling strategy to "NN" and there is no map given (on default)
                        ent_strategy = "NN"
                        given_map = False
                        # stores the index of the first open bracket in the word (e.g. crx(...) so the index is 3)
                        open_bracket_index = string_iterator + 3
                        end_word = get_closing_bracket_index(gate_layers, open_bracket_index)

                        # Now check, if there are semicolons (semicolons separate parameters from the map and from the entangling strategy)
                        # So that means we have to check three different cases:
                        # 1st: There is no semicolon: So there is no map and no given strategy
                        # 2nd: There is one semicolon: That can either mean, there is a given strategy or a given map
                        # 3nd: There are two semicolons: So there is a given map and a given strategy
                        semicolon1 = ";" in gate_layers[string_iterator:end_word]
                        if semicolon1:
                            semicolon1_index = gate_layers.index(";", string_iterator)
                            # Store the parameters in a list:
                            param_vector_word_with_commas = gate_layers[
                                (string_iterator + 4) : semicolon1_index
                            ]
                            param_vector_name_list = param_vector_word_with_commas.split(",")
                            param_vector_list = []
                            # Assigning the parameter names to the right parameters:
                            for param_vector_name in param_vector_name_list:
                                param_index = featuremap.variable_name_tuple.index(
                                    param_vector_name
                                )
                                param_vector_list.append(featuremap.variable_groups[param_index])

                            # if the second semicolon exists, you have to put the map (with "=") first and then the entangling strategy like this: crx(x1,p;=x*y,{x,y};AA)
                            # if the second semicolon doesn't exist, you can also write the entangling strategy without a map
                            semicolon2 = ";" in gate_layers[semicolon1_index + 1 : end_word]
                            if semicolon2:
                                semicolon2_index = gate_layers.index(";", semicolon1_index + 1)
                                if (
                                    gate_layers[semicolon2_index + 1 : semicolon2_index + 3]
                                    == "AA"
                                ):
                                    ent_strategy = "AA"
                                elif (
                                    gate_layers[semicolon2_index + 1 : semicolon2_index + 3]
                                    == "NN"
                                ):
                                    ent_strategy = "NN"
                                else:
                                    raise ValueError("Wrong input1.")
                            if gate_layers[semicolon1_index + 1] == "=":
                                # Evaluates all variables, that are stored in the brackets "{}", and creates a map with the given string of a function
                                given_map = True
                                map_comma_index = gate_layers.index(",", semicolon1_index)
                                map_string = gate_layers[semicolon1_index + 2 : map_comma_index]
                                bra = gate_layers.index("{", map_comma_index)
                                cket = gate_layers.index("}", map_comma_index)
                                map_args = gate_layers[bra + 1 : cket]
                                map_from_string = generate_function(map_string, map_args)
                            elif (
                                gate_layers[semicolon1_index + 1 : semicolon1_index + 3] == "AA"
                                and not semicolon2
                            ):
                                ent_strategy = "AA"
                            elif (
                                gate_layers[semicolon1_index + 1 : semicolon1_index + 3] == "NN"
                                and not semicolon2
                            ):
                                ent_strategy = "NN"
                            else:
                                raise ValueError("Wrong input2.")

                            if given_map:
                                func(
                                    *param_vector_list,
                                    map=map_from_string,
                                    ent_strategy=ent_strategy,
                                )
                            else:
                                func(*param_vector_list, ent_strategy=ent_strategy)

                        else:
                            # So there is no semicolon. That means there must be exactly one parameter vector and the default entangling strategy is NN:
                            param_vector_name = gate_layers[(string_iterator + 4) : end_word]
                            param_index = featuremap.variable_name_tuple.index(param_vector_name)
                            param_vector = featuremap.variable_groups[param_index]
                            func(param_vector)
                        string_iterator = end_word + 1
                    elif character_iter_1 == "p":
                        if string_iterator + 2 < len(gate_layers):
                            character_iter_2 = gate_layers[string_iterator + 2]
                        else:
                            raise ValueError("Wrong phase entangling input.")
                        # set entangling strategy to "NN" and there is no map given (on default)
                        ent_strategy = "NN"
                        given_map = False
                        # stores the index of the first open bracket in the word (e.g. cp(...) so the index is 2)
                        open_bracket_index = string_iterator + 2
                        end_word = get_closing_bracket_index(gate_layers, open_bracket_index)

                        # Now check, if there are semicolons (semicolons separate parameters from the map and from the entangling strategy)
                        # So that means we have to check three different cases:
                        # 1st: There is no semicolon: So there is no map and no given strategy
                        # 2nd: There is one semicolon: That can either mean, there is a given strategy or a given map
                        # 3nd: There are two semicolons: So there is a given map and a given strategy
                        semicolon1 = ";" in gate_layers[string_iterator:end_word]
                        if semicolon1:
                            semicolon1_index = gate_layers.index(";", string_iterator)
                            # Store the parameters in a list:
                            param_vector_word_with_commas = gate_layers[
                                (string_iterator + 3) : semicolon1_index
                            ]
                            param_vector_name_list = param_vector_word_with_commas.split(",")
                            param_vector_list = []
                            # Assigning the parameter names to the right parameters:
                            for param_vector_name in param_vector_name_list:
                                param_index = featuremap.variable_name_tuple.index(
                                    param_vector_name
                                )
                                param_vector_list.append(featuremap.variable_groups[param_index])

                            # if the second semicolon exists, you have to put the map (with "=") first and then the entangling strategy like this: cp(x1,p;=x*y,{x,y};AA)
                            # if the second semicolon doesn't exist, you can also write the entangling strategy without a map
                            semicolon2 = ";" in gate_layers[semicolon1_index + 1 : end_word]
                            if semicolon2:
                                semicolon2_index = gate_layers.index(";", semicolon1_index + 1)
                                if (
                                    gate_layers[semicolon2_index + 1 : semicolon2_index + 3]
                                    == "AA"
                                ):
                                    ent_strategy = "AA"
                                elif (
                                    gate_layers[semicolon2_index + 1 : semicolon2_index + 3]
                                    == "NN"
                                ):
                                    ent_strategy = "NN"
                                else:
                                    raise ValueError("Wrong input1.")

                            if gate_layers[semicolon1_index + 1] == "=":
                                # Evaluates all variables, that are stored in the brackets "{}", and creates a map with the given string of a function
                                given_map = True
                                map_comma_index = gate_layers.index(",", semicolon1_index)
                                map_string = gate_layers[semicolon1_index + 2 : map_comma_index]
                                bra = gate_layers.index("{", map_comma_index)
                                cket = gate_layers.index("}", map_comma_index)
                                map_args = gate_layers[bra + 1 : cket]
                                map_from_string = generate_function(map_string, map_args)
                            elif (
                                gate_layers[semicolon1_index + 1 : semicolon1_index + 3] == "AA"
                                and not semicolon2
                            ):
                                ent_strategy = "AA"
                            elif (
                                gate_layers[semicolon1_index + 1 : semicolon1_index + 3] == "NN"
                                and not semicolon2
                            ):
                                ent_strategy = "NN"
                            else:
                                raise ValueError("Wrong input2.")

                            if given_map:
                                featuremap_active.cp_entangling(
                                    *param_vector_list,
                                    map=map_from_string,
                                    ent_strategy=ent_strategy,
                                )
                            else:
                                featuremap_active.cp_entangling(
                                    *param_vector_list, ent_strategy=ent_strategy
                                )
                        else:
                            # So there is no semicolon. That means there must be exactly one parameter vector and the default entangling strategy is NN:
                            param_vector_name = gate_layers[(string_iterator + 3) : end_word]
                            param_index = featuremap.variable_name_tuple.index(param_vector_name)
                            param_vector = featuremap.variable_groups[param_index]
                            featuremap_active.cp_entangling(param_vector)
                        string_iterator = end_word + 1
                    elif character_iter_1 == "u":
                        if string_iterator + 2 < len(gate_layers):
                            character_iter_2 = gate_layers[string_iterator + 2]
                        else:
                            raise ValueError("Wrong phase entangling input.")
                        # stores the index of the first open bracket in the word (e.g. cu(...) so the index is 2)
                        open_bracket_index = string_iterator + 2
                        end_word = get_closing_bracket_index(gate_layers, open_bracket_index)
                        semicolon1 = ";" in gate_layers[string_iterator:end_word]
                        if semicolon1:
                            semicolon1_index = gate_layers.index(";", string_iterator)
                            # Store the parameters in a list:
                            param_vector_word_with_commas = gate_layers[
                                (string_iterator + 3) : semicolon1_index
                            ]
                            param_vector_name_list = param_vector_word_with_commas.split(",")
                            param_vector_list = []
                            # Assigning the parameter names to the right parameters:
                            for param_vector_name in param_vector_name_list:
                                param_index = featuremap.variable_name_tuple.index(
                                    param_vector_name
                                )
                                param_vector_list.append(featuremap.variable_groups[param_index])
                            if gate_layers[semicolon1_index + 1 : semicolon1_index + 3] == "AA":
                                featuremap_active.cu_entangling(
                                    *param_vector_list, ent_strategy="AA"
                                )
                            elif gate_layers[semicolon1_index + 1 : semicolon1_index + 3] == "NN":
                                featuremap_active.cu_entangling(
                                    *param_vector_list, ent_strategy="NN"
                                )
                            else:
                                raise ValueError("Unknown entangling strategy input.")
                        else:
                            param_vector_word_with_commas = gate_layers[
                                (string_iterator + 3) : end_word
                            ]
                            param_vector_name_list = param_vector_word_with_commas.split(",")
                            param_vector_list = []
                            # Assigning the parameter names to the right parameters:
                            for param_vector_name in param_vector_name_list:
                                param_index = featuremap.variable_name_tuple.index(
                                    param_vector_name
                                )
                                param_vector_list.append(featuremap.variable_groups[param_index])
                            featuremap_active.cu_entangling(*param_vector_list)
                        string_iterator = end_word + 1
                    else:
                        raise ValueError("Unknown entangling operation.")
            else:
                raise ValueError(
                    character_iter + " is an unknown operation input or an unknown character."
                )
        return featuremap

    def to_feature_map(
        self,
        feature_variable_group: Union[VariableGroup, list],
        parameters_variable_group: Union[VariableGroup, list],
    ):
        return ConvertedLayeredFeatureMap(self, feature_variable_group, parameters_variable_group)


class LayerPQC(LayeredPQC):
    """
    default class for a layer: the user is able to build his one list of operations and this list can be added to the main class LayeredFeatureMap
    """

    def __init__(self, featuremap: LayeredPQC):
        super().__init__(featuremap.num_qubits, featuremap.variable_groups)

    def add_operation(
        self, operation: _operation, variablegroup_tuple: tuple, variable_num_list=None
    ):
        """
        like the parent add_operation method with the exception, that we mustn't count the variable groups up, otherwise it would count once too much
        """
        if variablegroup_tuple == None:
            self.operation_list.append([operation, None])
        else:
            # For the case that there are variables given but without an information about how often they are used, set variable_dif_list to [number of qubits] on default
            if variable_num_list == None:
                variable_num_list = [self.num_qubits for i in range(len(variablegroup_tuple))]
            # adds the operation with the tuple of the variable groups used for this operation and the number of variables used per group:
            self.operation_list.append([operation, variablegroup_tuple, variable_num_list])


class ConvertedLayeredFeatureMap(FeatureMapBase):
    """
    Data structure for converting a LayeredPQC structure into sqlearn feature map structure.
    The programmer specifies, which variable groups are considered as features
    and which are parameters

    Args:
        layered_pqc [LayeredPQC]: Layered PQC that should be converted
        feature_variable_group Union[VariableGroup,list]: List of variable groups that
            are considered as the feature variable.
        parameters_variable_group Union[VariableGroup,list]: List of variable groups that
            are considered as the parameter variable.
    """

    def __init__(
        self,
        layered_pqc: LayeredPQC,
        feature_variable_group: Union[VariableGroup, list],
        parameters_variable_group: Union[VariableGroup, list],
    ) -> None:
        self._layered_pqc = layered_pqc

        if isinstance(feature_variable_group, VariableGroup):
            self._feature_variable_group = [feature_variable_group]
        else:
            self._feature_variable_group = feature_variable_group

        if isinstance(parameters_variable_group, VariableGroup):
            self._parameters_variable_group = [parameters_variable_group]
        else:
            self._parameters_variable_group = parameters_variable_group

    @property
    def num_qubits(self) -> int:
        """Returns number of qubits of the Layered Feature Map"""
        return self._layered_pqc.num_qubits

    @property
    def num_features(self) -> int:
        """Returns number of features of the Layered Feature Map"""
        num_features = 0
        for vg in self._feature_variable_group:
            num_features += self._layered_pqc.get_number_of_variables(vg)
        return num_features

    @property
    def num_parameters(self) -> int:
        """Returns number of parameters of the Layered Feature Map"""
        num_parameters = 0
        for vg in self._parameters_variable_group:
            num_parameters += self._layered_pqc.get_number_of_variables(vg)
        return num_parameters

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the Layered Feature Map

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the circuit in qiskit QuantumCircuit format
        """
        split = []
        feature_name = []
        ioff = 0
        for vg in self._feature_variable_group:
            ioff += self._layered_pqc.get_number_of_variables(vg)
            split.append(ioff)
            feature_name.append(vg.variable_name)
        split_features = np.split(features, split)

        split = []
        parameter_name = []
        ioff = 0
        for vg in self._parameters_variable_group:
            ioff += self._layered_pqc.get_number_of_variables(vg)
            split.append(ioff)
            parameter_name.append(vg.variable_name)
        split_parameters = np.split(parameters, split)

        vg_dict = dict(zip(feature_name, split_features))
        vg_dict.update(dict(zip(parameter_name, split_parameters)))

        input_list = []
        for name in self._layered_pqc.variable_name_tuple:
            input_list.append(vg_dict[name])

        return self._layered_pqc.get_circuit(*input_list)


class LayeredFeatureMap(FeatureMapBase):
    r"""
    A class for a simple creation of layered feature maps.

    Gates are added to all qubits by calling the associated function similar to Qiskit's circuits.
    Single qubit gates are added to all qubits, while two qubits gates can be added with different
    entanglement patterns. The implemented one and two qubit gates are listed below.

    Some gates have a input variable, as for example rotation gates, that can be set by supplying
    the string ``"x"`` for feature or ``"p"`` for parameter. Non-linear mapping can
    be added by setting the map variable ``map=``. Two qubit gates can be placed either
    in a nearest-neighbour ``NN`` or a all to all entangling pattern ``AA``.

    **Simple Layered Feature Map**

    .. code-block:: python

       from squlearn.feature_map import LayeredFeatureMap
       feature_map = LayeredFeatureMap(num_qubits=4,num_features=2)
       feature_map.H()
       feature_map.Rz("x")
       feature_map.Ry("p")
       feature_map.cx_entangling("NN")
       feature_map.draw()

    .. plot::

       from squlearn.feature_map import LayeredFeatureMap
       feature_map = LayeredFeatureMap(num_qubits=4,num_features=2)
       feature_map.H()
       feature_map.Rz("x")
       feature_map.Ry("p")
       feature_map.cx_entangling("NN")
       plt = feature_map.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt


    **Create a layered feature map with non-linear input encoding**

    It is also possible to define a non-linear function for encoding variables in gates by
    supplying a function for the encoding as the second argument

    .. code-block:: python

       import numpy as np
       from squlearn.feature_map import LayeredFeatureMap

       def func(a,b):
           return a*np.arccos(b)

       feature_map = LayeredFeatureMap(num_qubits=4,num_features=2)
       feature_map.H()
       feature_map.Rz("p","x",encoding=func)
       feature_map.cx_entangling("NN")
       feature_map.draw()

    .. plot::

       import numpy as np
       from squlearn.feature_map import LayeredFeatureMap

       def func(a,b):
           return a*np.arccos(b)

       feature_map = LayeredFeatureMap(num_qubits=4,num_features=2)
       feature_map.H()
       feature_map.Rz("p","x",encoding=func)
       feature_map.cx_entangling("NN")
       plt = feature_map.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt


    **Create a layered feature map with layers**

    Furthermore, it is possible to define layers and repeat them.

    .. code-block:: python

       from squlearn.feature_map import LayeredFeatureMap
       from squlearn.feature_map.layered_feature_map import Layer
       feature_map = LayeredFeatureMap(num_qubits=4,num_features=2)
       feature_map.H()
       layer = Layer(feature_map)
       layer.Rz("x")
       layer.Ry("p")
       layer.cx_entangling("NN")
       feature_map.add_layer(layer,num_layers=3)
       feature_map.draw()

    .. plot::

       from squlearn.feature_map import LayeredFeatureMap
       from squlearn.feature_map.layered_feature_map import Layer
       feature_map = LayeredFeatureMap(num_qubits=4,num_features=2)
       feature_map.H()
       layer = Layer(feature_map)
       layer.Rz("x")
       layer.Ry("p")
       layer.cx_entangling("NN")
       feature_map.add_layer(layer,num_layers=3)
       plt = feature_map.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt

    **Create a layered feature map from string**

    Another very useful feature is the creation from feature maps from strings.
    This can be achieved by the function ``LayeredFeatureMap.from_string()``.

    Gates are separated by ``-``, layers can be specified by ``N[...]`` where ``N`` is the
    number of repetitions. The entangling strategy can be set by adding ``NN`` or ``AA``.
    Adding a encoding function is possible by adding a ``=`` and the function definition as a
    string. The variables used in the function are given within curly brackets,
    e.g. ``crz(p;=a*np.arccos(b),{y,x};NN)``.

    The following strings are used for the gates:

    .. list-table:: Single qubit gates and their string representation
       :widths: 15 25 15 25 15 25
       :header-rows: 1

       * - String
         - Function
         - String
         - Function
         - String
         - Function
       * - ``"H"``
         - :meth:`H`
         - ``"I"``
         - :meth:`I`
         - ``"P"``
         - :meth:`P`
       * - ``"Rx"``
         - :meth:`Rx`
         - ``"Ry"``
         - :meth:`Ry`
         - ``"Rz"``
         - :meth:`Rz`
       * - ``"S"``
         - :meth:`S`
         - ``"Sc"``
         - :meth:`S_conjugate`
         - ``"T"``
         - :meth:`T`
       * - ``"Tc"``
         - :meth:`T_conjugate`
         - ``"U"``
         - :meth:`U`
         - ``"X"``
         - :meth:`X`
       * - ``"Y"``
         - :meth:`Y`
         - ``"Z"``
         - :meth:`Z`
         -
         -

    .. list-table:: Two qubit gates and their string representation
       :widths: 25 25 25 25 25 25
       :header-rows: 1

       * - String
         - Function
         - String
         - Function
         - String
         - Function
       * - ``"ch"``
         - :meth:`ch_entangling`
         - ``"cx"``
         - :meth:`cx_entangling`
         - ``"cy"``
         - :meth:`cy_entangling`
       * - ``"cz"``
         - :meth:`cz_entangling`
         - ``"s"``
         - :meth:`swap`
         - ``"cp"``
         - :meth:`cp_entangling`
       * - ``"crx"``
         - :meth:`crx_entangling`
         - ``"cry"``
         - :meth:`cry_entangling`
         - ``"crz"``
         - :meth:`crz_entangling`
       * - ``"rxx"``
         - :meth:`rxx_entangling`
         - ``"ryy"``
         - :meth:`ryy_entangling`
         - ``"rzz"``
         - :meth:`rzz_entangling`
       * - ``"rzx"``
         - :meth:`rzx_entangling`
         - ``"cu"``
         - :meth:`cu_entangling`
         -
         -

    .. code-block:: python

       from squlearn.feature_map import LayeredFeatureMap
       feature_map = LayeredFeatureMap.from_string(
           "Ry(p)-3[Rx(p,x;=y*np.arccos(x),{y,x})-crz(p)]-Ry(p)", num_qubits=4, num_features=1
       )
       feature_map.draw()

    .. plot::

       from squlearn.feature_map import LayeredFeatureMap
       feature_map = LayeredFeatureMap.from_string(
           "Ry(p)-3[Rx(p,x;=y*np.arccos(x),{y,x})-crz(p)]-Ry(p)", num_qubits=4, num_features=1
       )
       plt = feature_map.draw(style={'fontsize':15,'subfontsize': 10})
       plt.tight_layout()
       plt

    Args:
        num_qubits (int): Number of qubits of the feature map
        num_features (int): Dimension of the feature vector
        feature_str (str): Label for identifying the feature variable group (default: ``"x"``).
        parameter_str (str): Label for identifying the parameter variable group (default: ``"p"``).
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        feature_str: str = "x",
        parameter_str: str = "p",
    ) -> None:
        super().__init__(num_qubits, num_features)
        self._feature_str = feature_str
        self._parameter_str = parameter_str
        self._x = VariableGroup(self._feature_str, size=num_features)
        self._p = VariableGroup(self._parameter_str)
        self._layered_pqc = LayeredPQC(num_qubits=num_qubits, variable_groups=(self._x, self._p))

    @property
    def num_parameters(self) -> int:
        """Returns number of parameters of the Layered Feature Map"""
        return self._layered_pqc.get_number_of_variables(self._p)

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the Layered Feature Map

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the circuit in qiskit QuantumCircuit format
        """
        return self._layered_pqc.get_circuit(features, parameters)

    @classmethod
    def from_string(
        cls,
        feature_map_str: str,
        num_qubits: int,
        num_features: int,
        feature_str: str = "x",
        parameter_str: str = "p",
    ):
        """
        Constructs a Layered Feature Map through a given string of gates.

        Args:
            feature_map_str (str): String that specifies the feature map
            num_qubits (int): Number of qubits in the feature map
            num_features (int): Dimension of the feature vector.
            feature_str (str): String that used in feature_map_str to label features (default: 'x')
            parameter_str (str): String that used in feature_map_str to label parameters (default: 'p')

        Returns:
            Returns a LayeredFeatureMap object that contains the specified feature map.

        """

        layered_feature_map = cls(num_qubits, num_features, feature_str, parameter_str)
        layered_feature_map._layered_pqc = LayeredPQC.from_string(
            num_qubits,
            feature_map_str,
            (layered_feature_map._x, layered_feature_map._p),
        )
        return layered_feature_map

    def add_layer(self, layer, num_layers=1) -> None:
        """
        Add a layer num_layers times.

        Args:
            layer: Layer structure
            num_layers (int): Number of times that the layer is repeated

        """
        self._layered_pqc.add_layer(layer.layered_pqc, num_layers)

    def _str_to_variable_group(self, input_string: str) -> VariableGroup:
        """
        Internal function to convert a string to the
        feature or parameter variable group

        Args:
            input_string (str): String that is either feature_str or parameter_str

        Returns:
            Associated variable group
        """
        if input_string == self._feature_str:
            return self._x
        elif input_string == self._parameter_str:
            return self._p
        else:
            raise ValueError("Unknown variable type!")

    def _param_gate(self, *variable, function, encoding: Union[Callable, None] = None):
        """
        Internal conversion routine for one qubit gates that calls the LayeredPQC routines with the correct
        variable group data
        """
        vg_list = [self._str_to_variable_group(str) for str in variable]
        return function(*vg_list, map=encoding)

    def _two_param_gate(
        self, *variable, function, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """
        Internal conversion routine for two qubit gates that calls the LayeredPQC routines with the correct
        variable group data
        """
        vg_list = [self._str_to_variable_group(str) for str in variable]
        return function(*vg_list, ent_strategy=ent_strategy, map=encoding)

    def H(self):
        """Adds a layer of H gates to the Layered Feature Map"""
        self._layered_pqc.H()

    def X(self):
        """Adds a layer of X gates to the Layered Feature Map"""
        self._layered_pqc.X()

    def Y(self):
        """Adds a layer of Y gates to the Layered Feature Map"""
        self._layered_pqc.Y()

    def Z(self):
        """Adds a layer of Z gates to the Layered Feature Map"""
        self._layered_pqc.Z()

    def I(self):
        """Adds a layer of I gates to the Layered Feature Map"""
        self._layered_pqc.I()

    def S(self):
        """Adds a layer of S gates to the Layered Feature Map"""
        self._layered_pqc.S()

    def S_conjugate(self):
        """Adds a layer of conjugated S gates to the Layered Feature Map"""
        self._layered_pqc.S_conjugate()

    def T(self):
        """Adds a layer of T gates to the Layered Feature Map"""
        self._layered_pqc.T()

    def T_conjugate(self):
        """Adds a layer of conjugated T gates to the Layered Feature Map"""
        self._layered_pqc.T_conjugate()

    def Rx(self, *variable_str: str, encoding: Union[Callable, None] = None):
        """Adds a layer of Rx gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._param_gate(*variable_str, function=self._layered_pqc.Rx, encoding=encoding)

    def Ry(self, *variable_str, encoding: Union[Callable, None] = None):
        """Adds a layer of Ry gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._param_gate(*variable_str, function=self._layered_pqc.Ry, encoding=encoding)

    def Rz(self, *variable_str, encoding: Union[Callable, None] = None):
        """Adds a layer of Rz gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._param_gate(*variable_str, function=self._layered_pqc.Rz, encoding=encoding)

    def P(self, *variable_str, encoding: Union[Callable, None] = None):
        """Adds a layer of P gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._param_gate(*variable_str, function=self._layered_pqc.P, encoding=encoding)

    def U(self, *variable_str, encoding: Union[Callable, None] = None):
        """Adds a layer of U gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._param_gate(*variable_str, function=self._layered_pqc.U, encoding=encoding)

    def ch_entangling(self, ent_strategy="NN"):
        """Adds a layer of controlled H gates to the Layered Feature Map

        Args:
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
        """
        self._layered_pqc.ch_entangling(ent_strategy)

    def cx_entangling(self, ent_strategy="NN"):
        """Adds a layer of controlled X gates to the Layered Feature Map

        Args:
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
        """
        self._layered_pqc.cx_entangling(ent_strategy)

    def cy_entangling(self, ent_strategy="NN"):
        """Adds a layer of controlled Y gates to the Layered Feature Map

        Args:
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
        """
        self._layered_pqc.cy_entangling(ent_strategy)

    def cz_entangling(self, ent_strategy="NN"):
        """Adds a layer of controlled Z gates to the Layered Feature Map

        Args:
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
        """
        self._layered_pqc.cz_entangling(ent_strategy)

    def swap(self, ent_strategy="NN"):
        """Adds a layer of swap gates to the Layered Feature Map

        Args:
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
        """
        self._layered_pqc.swap(ent_strategy)

    def cp_entangling(
        self, *variable_str, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """Adds a layer of controlled P gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """

        self._two_param_gate(
            *variable_str,
            function=self._layered_pqc.cp_entangling,
            ent_strategy=ent_strategy,
            encoding=encoding,
        )

    def crx_entangling(
        self, *variable_str, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """Adds a layer of controlled Rx gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._two_param_gate(
            *variable_str,
            function=self._layered_pqc.crx_entangling,
            ent_strategy=ent_strategy,
            encoding=encoding,
        )

    def cry_entangling(
        self, *variable_str, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """Adds a layer of controlled Ry gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._two_param_gate(
            *variable_str,
            function=self._layered_pqc.cry_entangling,
            ent_strategy=ent_strategy,
            encoding=encoding,
        )

    def crz_entangling(
        self, *variable_str, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """Adds a layer of controlled Rz gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._two_param_gate(
            *variable_str,
            function=self._layered_pqc.crz_entangling,
            ent_strategy=ent_strategy,
            encoding=encoding,
        )

    def rxx_entangling(
        self, *variable_str, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """Adds a layer of Rxx gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._two_param_gate(
            *variable_str,
            function=self._layered_pqc.rxx_entangling,
            ent_strategy=ent_strategy,
            encoding=encoding,
        )

    def ryy_entangling(
        self, *variable_str, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """Adds a layer of Ryy gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._two_param_gate(
            *variable_str,
            function=self._layered_pqc.ryy_entangling,
            ent_strategy=ent_strategy,
            encoding=encoding,
        )

    def rzx_entangling(
        self, *variable_str, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """Adds a layer of Rzx gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._two_param_gate(
            *variable_str,
            function=self._layered_pqc.rzx_entangling,
            ent_strategy=ent_strategy,
            encoding=encoding,
        )

    def rzz_entangling(
        self, *variable_str, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """Adds a layer of Rzz gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._two_param_gate(
            *variable_str,
            function=self._layered_pqc.rzz_entangling,
            ent_strategy=ent_strategy,
            encoding=encoding,
        )

    def cu_entangling(
        self, *variable_str, ent_strategy="NN", encoding: Union[Callable, None] = None
    ):
        """Adds a layer of controlled U gates to the Layered Feature Map

        Args:
            variable_str (str): Labels of variables that are used in the gate
            ent_strategy (str): Entanglement strategy that is used to determine the entanglement,
                                either ``"NN"`` or ``"AA"``.
            encoding (Callable): Encoding function that is applied to the variables, input in the
                                 same order as the given labels in variable_str
        """
        self._two_param_gate(
            *variable_str,
            function=self._layered_pqc.cu_entangling,
            ent_strategy=ent_strategy,
            encoding=encoding,
        )


class Layer(LayeredFeatureMap):
    """Class for defining a Layer of the Layered Feature Map"""

    def __init__(self, feature_map: LayeredFeatureMap):
        super().__init__(
            feature_map.num_qubits,
            feature_map.num_features,
            feature_map._feature_str,
            feature_map._parameter_str,
        )
        # Copy relevant data from input feature_map
        self._x = feature_map._x
        self._p = feature_map._p
        self._layered_pqc = LayerPQC(feature_map._layered_pqc)

    @property
    def layered_pqc(self):
        """Returns the LayerPQC object of the Layered Feature Map"""
        return self._layered_pqc
