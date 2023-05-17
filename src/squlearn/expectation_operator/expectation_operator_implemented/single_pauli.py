import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SinglePauli(ExpectationOperatorBase):
    """Expectation operator: H = c*I + sum_i Z_i*h_i"""

    def __init__(self, num_qubits: int, qubits, op_str="Z", parameterized=False) -> None:
        super().__init__(num_qubits)
        self.qubits = qubits
        self.op_str = op_str
        self.parameterized = parameterized
        if not isinstance(qubits, list):
            self.qubits = [qubits]

    @property
    def num_parameters(self):
        """Returns the number of free parameters in the Expectation operator"""
        if self.parameterized:
            return len(self.qubits)
        else:
            return 0

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
        """
        Returns:
            Return PauliOp expression of the specified Expectation operator.
        """

        nparam = len(parameters)

        i = self.qubits[0]
        if 0 > i or self.num_qubits <= i:
            raise ValueError("Specified qubit out of range")
        H = "I" * self.num_qubits
        H = H[i + 1 :] + self.op_str + H[:i]
        if self.parameterized:
            ioff = 0
            HH = PauliOp(Pauli(H)) * parameters[ioff % nparam]
            ioff = ioff + 1
        else:
            HH = PauliOp(Pauli(H))

        for j in range(1, len(self.qubits)):
            i = self.qubits[j]
            if 0 > i or self.num_qubits <= i:
                raise ValueError("Specified qubit out of range")
            H = "I" * self.num_qubits
            H = H[i + 1 :] + self.op_str + H[:i]
            if self.parameterized:
                HH += PauliOp(Pauli(H)) * parameters[ioff % nparam]
                ioff = ioff + 1
            else:
                HH += PauliOp(Pauli(H))
        return HH.reduce()
