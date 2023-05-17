import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class IsingHamiltonian(ExpectationOperatorBase):
    def __init__(
        self, num_qubits: int, I: str = "S", Z: str = "F", X: str = "N", ZZ: str = "F"
    ) -> None:
        super().__init__(num_qubits)

        self.I = I
        self.Z = Z
        self.X = X
        self.ZZ = ZZ

        if self.I not in ["S", "N"]:
            raise ValueError(
                "Only the characters 'S' and 'N' are" + "supported as characters for I"
            )

        if (
            self.Z not in ["F", "S", "N"]
            or self.ZZ not in ["F", "S", "N"]
            or self.ZZ not in ["F", "S", "N"]
        ):
            raise ValueError(
                "Only the characters 'F','S','N' are" + "supported as characters for Z, ZZ, and X"
            )

    @property
    def num_parameters(self):
        """Returns the number of free parameters in the Expectation operator"""

        num_parameters = 0
        if self.I == "S":
            num_parameters += 1

        if self.Z == "S":
            num_parameters += 1
        elif self.Z == "F":
            num_parameters += self.num_qubits

        if self.X == "S":
            num_parameters += 1
        elif self.X == "F":
            num_parameters += self.num_qubits

        if self.ZZ == "S":
            num_parameters += 1
        elif self.ZZ == "F":
            num_parameters += (self.num_qubits * (self.num_qubits - 1)) // 2

        return num_parameters

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
        """
        Returns:
            Return PauliOp expression of the specified Expectation operator.
        """

        def gen_double_ising_string(i, j):
            H = "I" * self.num_qubits
            H = H[i + 1 :] + "Z" + H[:i]
            if i != j:
                H = H[: self.num_qubits - j - 1] + "Z" + H[self.num_qubits - j :]
            return H

        def gen_single_ising_string(i, str):
            H = "I" * self.num_qubits
            H = H[i + 1 :] + str + H[:i]
            return H

        nparam = len(parameters)

        H = PauliOp(Pauli("I" * self.num_qubits))  # is removed later
        ioff = 0

        if self.I == "S":
            H += PauliOp(Pauli("I" * self.num_qubits)) * parameters[ioff % nparam]
            ioff += 1

        if self.Z == "S" or self.Z == "F":
            for i in range(self.num_qubits):
                H += PauliOp(Pauli(gen_single_ising_string(i, "Z"))) * parameters[ioff % nparam]
                if self.Z == "F":
                    ioff += 1
            if self.Z == "S":
                ioff += 1

        if self.X == "S" or self.X == "F":
            for i in range(self.num_qubits):
                H += PauliOp(Pauli(gen_single_ising_string(i, "X"))) * parameters[ioff % nparam]
                if self.X == "F":
                    ioff += 1
            if self.X == "S":
                ioff += 1

        if self.ZZ == "S" or self.ZZ == "F":
            for i in range(self.num_qubits):
                for j in range(i):
                    H += PauliOp(Pauli(gen_double_ising_string(i, j))) * parameters[ioff % nparam]
                    if self.ZZ == "F":
                        ioff += 1
            if self.ZZ == "S":
                ioff += 1

        H = H - PauliOp(Pauli("I" * self.num_qubits))

        return H.reduce()
