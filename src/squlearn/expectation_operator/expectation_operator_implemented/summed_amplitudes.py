import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SummedAmplitudes(ExpectationOperatorBase):
    """
    Operator for a + sum_i b_i |0_i><0_i|
    Implemented by |0><0| = 0.5*(I+Z) and |1><1| = 0.5*(I-Z)

    Args:
        num_qubits: Number of qubits.
        one_state: if false the |0><0| state is measured, if true the |1><1| state is measured

    """

    def __init__(self, num_qubits: int, one_state=False, full_sum: bool = True) -> None:
        super().__init__(num_qubits)
        self.one_state = one_state
        self.full_sum = full_sum

    @property
    def num_parameters(self):
        if self.full_sum:
            return 1 + self.num_qubits
        else:
            return 2

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray] = None):
        """
        Returns:
            Return PauliOp expression of the specified Expectation operator.
        """

        nparam = len(parameters)
        H = PauliOp(Pauli("I" * self.num_qubits)) * parameters[0 % nparam]

        ioff = 1
        for i in range(self.num_qubits):
            I = "I" * self.num_qubits
            Z = I[(i + 1) :] + "Z" + I[:i]

            if self.one_state:
                H = (
                    H
                    + parameters[ioff % nparam] * 0.5 * PauliOp(Pauli(I))
                    - parameters[ioff % nparam] * 0.5 * PauliOp(Pauli(Z))
                )
            else:
                H = (
                    H
                    + parameters[ioff % nparam] * 0.5 * PauliOp(Pauli(I))
                    + parameters[ioff % nparam] * 0.5 * PauliOp(Pauli(Z))
                )
            if self.full_sum:
                ioff += 1

        return H.reduce()
