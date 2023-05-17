import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SummedPaulis(ExpectationOperatorBase):
    def __init__(self, num_qubits: int, op_str: str = "Z", full_sum: bool = True) -> None:
        super().__init__(num_qubits)
        self.op_str = op_str
        self.full_sum = full_sum

    @property
    def num_parameters(self):
        """Returns the number of free parameters in the Expectation operator"""
        if self.full_sum:
            return 1 + len(self.op_str) * self.num_qubits
        else:
            return 1 + len(self.op_str)

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
        """
        Returns:
            Return PauliOp expression of the specified Expectation operator.
        """

        def gen_string(i, op_str):
            H = "I" * self.num_qubits
            H = H[i + 1 :] + op_str + H[:i]
            return H

        nparam = len(parameters)

        H = PauliOp(Pauli("I" * self.num_qubits)) * parameters[0 % nparam]

        ioff = 1
        for s in self.op_str:
            for i in range(self.num_qubits):
                H = H + PauliOp(Pauli(gen_string(i, s))) * parameters[ioff % nparam]
                if self.full_sum:
                    ioff += 1
            if not self.full_sum:
                ioff += 1

        return H.reduce()
