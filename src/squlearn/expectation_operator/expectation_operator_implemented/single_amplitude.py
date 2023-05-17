import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SingleAmplitude(ExpectationOperatorBase):
    """Initialization

    Operator for measuring the probability of being in state 0 or 1 of a qubit.
    Implemented by |0><0| = 0.5*(I+Z) and |1><1| = 0.5*(I-Z)

    Args:
        number_of_qubits: Number of qubits.
        qubits: index of the qubit for which the probability is measured
        one_state: if false the |0><0| state is measured, if true the |1><1| state is measured
    """

    def __init__(self, num_qubits: int, qubit=0, one_state=False) -> None:
        super().__init__(num_qubits)
        self.qubit = qubit
        self.one_state = one_state

    @property
    def num_parameters(self):
        return 0

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray] = None):
        """
        Returns:
            Return PauliOp expression of the specified Expectation operator.
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

        return H.reduce()
