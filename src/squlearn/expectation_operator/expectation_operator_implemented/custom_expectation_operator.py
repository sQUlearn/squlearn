import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class CustomExpectationOperator(ExpectationOperatorBase):
    r"""
    Class for evaluating the expectation value of a custom operator.

    The operator is supplied as a string of Pauli operators, e.g. ``operator_string='ZI'`` for
    a two qubit operator with a Z operator on the second qubit.
    Note that the index of the qubits is reversed, i.e. the first qubit is the last character
    in the string, similar to the Qiskit computational state numbering.

    Multiple operators that are summed can be specified by a list of strings, e.g.
    ``operator_string=['ZZ', 'XX']``.

    Args:
        num_qubits (int): Number of qubits.
        operator_string (Union[str, list[str], tuple[str]]): String of operator to measure.
            Also list or tuples of strings are allowed for multiple operators.
        parameterized (bool): If True, the operator is parameterized.

    Attributes:
        num_qubits (int): Number of qubits.
        num_parameters (int): Number of trainable parameters in the custom operator.
        operator_string (Union[str, list[str], tuple[str]]): String of operator to measure.
        parameterized (bool): If True, the operator is parameterized.

    """

    def __init__(
        self,
        num_qubits: int,
        operator_string: Union[str, list[str], tuple[str]],
        parameterized: bool = False,
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
            for s_ in s:
                if s_ not in ["I", "X", "Y", "Z"]:
                    raise ValueError("Only Pauli operators I, X, Y, Z are allowed.")

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters in the custom operator"""
        if self.parameterized:
            return len(self.operator_string)
        else:
            return 0

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray] = None):
        """
        Function for generating the PauliOp expression of the custom operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Parameters of the custom operator.

        Returns:
            PauliOp expression of the specified custom operator.
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
