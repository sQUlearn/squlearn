import numpy as np
from typing import Union
from abc import ABC, abstractmethod

from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli
from qiskit.opflow import ListOp, PauliOp, PauliSumOp, TensoredOp
from qiskit.opflow import SummedOp, Zero, One
from qiskit.opflow import OperatorBase, StateFn


class ExpectationOperatorBase(ABC):
    """Base class for expectation operators.

    Args:
        num_qubits (int): Number of qubits.

    Attributes:
        num_parameters (int): Number of trainable parameters in the expectation operator.
        num_qubits (int): Number of qubits in the expectation operator.

    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._num_all_qubits = num_qubits
        self._qubit_map = np.linspace(0, num_qubits - 1, num_qubits, dtype=int)

    def set_map(self, qubit_map: Union[list, dict], num_all_qubits: int):
        """
        Function for setting a qubit mapping from physical qubits to the ones of the operator.

        This function is necessary whenever the number of physical qubits are different from the
        operator definition, as for example when running on a real backend.
        The number of qubits in the system has to be larger than the number
        of qubits in the expectation operator.

        Args:
            qubit_map (Union[list, dict]): A list or dictionary specifying which of the input
                                           qubits are mapped to the output qubits.
            num_all_qubits (int): The total number of qubits in the system.
        """
        self._qubit_map = qubit_map
        self._num_all_qubits = num_all_qubits
        if self._num_all_qubits < self._num_qubits:
            raise ValueError(
                """Number of qubits in the system is smaller than the number
                                of qubits in the expectation operator."""
            )

    @property
    def num_parameters(self):
        """Number of free parameters in the expectation operator."""
        return 0

    @property
    def num_qubits(self):
        """Number of qubits in the expectation operator."""
        return self._num_qubits

    def get_operator(self, parameters: Union[ParameterVector, np.ndarray]):
        """Returns Operator in as a opflow measurement operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used in
                                                             the operator
        Return:
            StateFn expression of the expectation operator.
        """

        return StateFn(self.get_pauli_mapped(parameters), is_measurement=True)

    @abstractmethod
    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
        """
        Returns the PauliOp expression of the expectation operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used
                                                             in the operator

        Returns:
            Expectation operator in qiskit's PauliOp class
        """
        raise NotImplementedError

    def get_pauli_mapped(self, parameters: Union[ParameterVector, np.ndarray]):
        """
        Returns the mapped PauliOp expression of the expectation operator.
        The previously the qubit map has to be set via :meth:`set_map`!

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used in
                                                             the operator

        Return:
            Expectation operator in qiskit's PauliOp class with qubits mapped to
            physical ones
        """

        def map_expectation_op(operator: OperatorBase) -> OperatorBase:
            """Recrusive method that replaces that resets the qubits to the mapped ones."""

            # We reached the Composed object or the wavefunction
            if isinstance(operator, PauliOp):
                blank = Pauli("I" * self._num_all_qubits)
                for i, p in enumerate(operator.primitive):
                    blank[self._qubit_map[i]] = p
                return PauliOp(blank, operator.coeff)
            elif isinstance(operator, PauliSumOp):
                PauliList = []
                for i, p in enumerate(operator.primitive.paulis):
                    blank = Pauli("I" * self._num_all_qubits)
                    for j, op in enumerate(p):
                        blank[self._qubit_map[j]] = op
                    PauliList.append(PauliOp(blank, coeff=operator.coeffs[i]))
                return_op = sum(PauliList)
                return_op._coeff = operator.coeff
                return return_op
            elif isinstance(operator, ListOp):
                # List object reached, recursive call of the function
                op_list = [map_expectation_op(op) for op in operator.oplist]

                # Sort out the Zero @ One terms
                if operator.combo_fn == ListOp.default_combo_fn:  # If using default
                    return ListOp(
                        oplist=[op for op in op_list if op != ~Zero @ One],
                        coeff=operator.coeff,
                    )
                elif isinstance(operator, SummedOp):
                    return SummedOp(
                        oplist=[op for op in op_list if op != ~Zero @ One],
                        coeff=operator.coeff,
                    )
                elif isinstance(operator, TensoredOp):
                    return TensoredOp(
                        oplist=[op for op in op_list if op != ~Zero @ One],
                        coeff=operator.coeff,
                    )
                else:
                    raise ValueError("Unknown Type in ListOp")
            else:
                raise ValueError("Wrong Type in operator: ", type(operator))

        return map_expectation_op(self.get_pauli(parameters))

    def __str__(self):
        """Return a string representation of the ExpectationOperatorBase."""
        p = ParameterVector("p", self.num_parameters)
        return str(self.get_pauli_mapped(p))

    def __repr__(self):
        """Return a string representation of the ExpectationOperatorBase."""
        p = ParameterVector("p", self.num_parameters)
        return repr(self.get_pauli_mapped(p))

    def __add__(self, x):
        """Addition of two expectation operators."""
        if not isinstance(x, ExpectationOperatorBase):
            raise ValueError("Only the addition with other expectation operator is allowed!")

        class AddedExpectationOperator(ExpectationOperatorBase):
            """Internal class for a sum of expectation operators.

            Args:
                op1 (ExpectationOperatorBase): Left expectation operator
                op2 (ExpectationOperatorBase): Right expectation operator
            """

            def __init__(self, op1: ExpectationOperatorBase, op2: ExpectationOperatorBase):
                if op1.num_qubits != op2.num_qubits:
                    raise ValueError("Number of qubits is not equal in both expectation operator.")

                super().__init__(op1.num_qubits)

                self._op1 = op1
                self._op2 = op2

            @property
            def num_parameters(self) -> int:
                """The number of trainable parameters of added expectation operator.

                Is equal to the sum of both trainable parameters.
                """
                if self._op1 == self._op2:
                    return self._op1.num_parameters
                else:
                    return self._op1.num_parameters + self._op2.num_parameters

            def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
                """Returns the PauliOp expression of the added expectation operator.

                Args:
                    parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used
                                                                     in the operator

                Return:
                    PauliOp: Expectation operator in qiskit's PauliOp class
                """
                if self._op1 == self._op2:
                    paulis_op = self._op1.get_pauli(parameters)
                    return (paulis_op + paulis_op).reduce()
                else:
                    paulis_op1 = self._op1.get_pauli(parameters[: self._op1.num_parameters])
                    paulis_op2 = self._op2.get_pauli(parameters[self._op1.num_parameters :])
                    return (paulis_op1 + paulis_op2).reduce()

        return AddedExpectationOperator(self, x)

    def __mul__(self, x):
        """Multiplication of two expectation operators."""
        if not isinstance(x, ExpectationOperatorBase):
            raise ValueError("Only the multiplication with other expectation operator is allowed!")

        class MultipliedExpectationOperator(ExpectationOperatorBase):
            """Internal class for a multiplication of expectation operators.

            Args:
                op1 (ExpectationOperatorBase): Left expectation operator
                op2 (ExpectationOperatorBase): Right expectation operator
            """

            def __init__(self, op1: ExpectationOperatorBase, op2: ExpectationOperatorBase):
                if op1.num_qubits != op2.num_qubits:
                    raise ValueError("Number of qubits is not equal in both expectation operator.")

                super().__init__(op1.num_qubits)

                self._op1 = op1
                self._op2 = op2

            @property
            def num_parameters(self) -> int:
                """The number of trainable parameters of multiplied expectation operator.

                Is equal to the sum of both trainable parameters.
                """
                if self._op1 == self._op2:
                    return self._op1.num_parameters
                else:
                    return self._op1.num_parameters + self._op2.num_parameters

            def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
                """Returns the PauliOp expression of the multiplied expectation operator.

                Args:
                    parameters (Union[ParameterVector, np.ndarray]): Vector of parameters used
                                                                     in the operator

                Return:
                    PauliOp: Expectation operator in qiskit's PauliOp class
                """
                if self._op1 == self._op2:
                    paulis_op = self._op1.get_pauli(parameters)
                    return (paulis_op @ paulis_op).reduce().reduce()
                else:
                    paulis_op1 = self._op1.get_pauli(parameters[: self._op1.num_parameters])
                    paulis_op2 = self._op2.get_pauli(parameters[self._op1.num_parameters :])
                    return (paulis_op1 @ paulis_op2).reduce().reduce()

        return MultipliedExpectationOperator(self, x)
