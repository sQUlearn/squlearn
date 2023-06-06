import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SinglePauli(ExpectationOperatorBase):
    r"""
    Expectation operator for evaluating the expectation value of a single Pauli operator.

    Args:
        num_qubits (int): Number of qubits.
        qubit (int): Qubit for which the Pauli expectation value is obtained.
        op_str (str): Pauli operator to measure. Must be one of "I", "X", "Y", "Z".
        parameterized (bool): If True, the operator is parameterized.

    """

    def __init__(
        self, num_qubits: int,
        qubit: int,
        op_str: str = "Z",
        parameterized: bool = False
    ) -> None:

        super().__init__(num_qubits)

        self.qubit = qubit
        self.op_str = op_str
        self.parameterized = parameterized

        if self.op_str not in ["I", "X", "Y", "Z"]:
            raise ValueError("Specified operator not supported")

    @property
    def num_parameters(self):
        """Returns the number of free parameters in the single pauli operator"""

        if self.parameterized:
            return 1
        else:
            return 0

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
        """
        Function for generating the PauliOp expression of the single pauli operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): parameters of the single
                pauli operator.

        Returns:
            PauliOp expression of the specified single pauli operator.
        """

        i = self.qubit
        if 0 > i or self.num_qubits <= i:
            raise ValueError("Specified qubit out of range")

        H = "I" * self.num_qubits
        H = PauliOp(Pauli(H[(i + 1) :] + self.op_str + H[:i]))

        if self.parameterized:
            H = H * parameters[0]

        return H.reduce()
