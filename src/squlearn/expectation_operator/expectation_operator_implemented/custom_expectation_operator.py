import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class CustomExpectationOperator(ExpectationOperatorBase):
    def __init__(
        self, num_qubits: int, operator_string: Union[str, list], parameterized=False
    ) -> None:
        super().__init__(num_qubits)
        self.operator_string = operator_string
        if isinstance(self.operator_string, str):
            self.operator_string = [self.operator_string]
        self.parameterized = parameterized

        for s in self.operator_string:
            if len(s) != self.num_qubits:
                raise ValueError(
                    "Supplied string has not the same size as the number of qubits, "
                    + "please add missing identities as 'I'"
                )

    @property
    def num_parameters(self):
        """Returns the number of free parameters in the Expectation operator"""
        if self.parameterized:
            return len(self.operator_string)
        else:
            return 0

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray] = None):
        """
        Returns:
            Return PauliOp expression of the specified Expectation operator.
        """

        if self.parameterized:
            nparam = len(parameters)
            H = PauliOp(Pauli(self.operator_string[0])) * parameters[0 % nparam]
            ioff = 1
            for j in range(1, len(self.operator_string)):
                H = H + PauliOp(Pauli(self.operator_string[j])) * parameters[ioff % nparam]
                ioff = ioff + 1
            return H.reduce()

        else:
            H = PauliOp(Pauli(self.operator_string[0]))
            for j in range(1, len(self.operator_string)):
                H = H + PauliOp(Pauli(self.operator_string[j]))
            return H.reduce()
