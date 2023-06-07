import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SingleProbability(ExpectationOperatorBase):
    r"""
    Expectation operator for measuring the probability of being in state 0 or 1 of
    a specified qubit.

    Implemented by :math:`\ket{0}\bra{0} = 0.5(\hat{I}+\hat{Z})` and :math:`\ket{1}\bra{1} = 0.5(\hat{I}-\hat{Z})`

    Args:
        num_qubits (int): Number of qubits.
        qubit (int): Qubit to measure.
        one_state (bool): If True, measure the probability of being in state 1, otherwise state 0.
        parameterized (bool): If True, the operator is parameterized.
    """

    def __init__(
        self,
        num_qubits: int,
        qubit: int = 0,
        one_state: bool = False,
        parameterized: bool = False,
    ) -> None:
        super().__init__(num_qubits)

        self.qubit = qubit
        self.one_state = one_state
        self.parameterized = parameterized

    @property
    def num_parameters(self):
        """Returns the number of free parameters in the single probability operator"""

        if self.parameterized:
            return 1
        else:
            return 0

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray] = None):
        """
        Function for generating the PauliOp expression of the single probability operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Parameters of the single
                probability operator.

        Returns:
            PauliOp expression of the specified single probability operator.
        """

        i = self.qubit
        if 0 > i or self.num_qubits <= i:
            raise ValueError("Specified qubit out of range")
        I = "I" * self.num_qubits
        Z = I[(i + 1) :] + "Z" + I[:i]

        if self.one_state:
            H = 0.5 * PauliOp(Pauli(I)) - 0.5 * PauliOp(Pauli(Z))
        else:
            H = 0.5 * PauliOp(Pauli(I)) + 0.5 * PauliOp(Pauli(Z))

        if self.parameterized:
            H = H * parameters[0]

        return H.reduce()
