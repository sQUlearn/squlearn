import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SinglePauli(ExpectationOperatorBase):
    r"""
    Single Pauli operator of a single Qubit.

    **Equation for Z Pauli operator:**

    .. math::

       \hat{H} = \hat{Z_i} \qquad \text{or} \qquad \hat{H} = \theta\hat{Z_i}~~~~
       \text{  (parameterized)}

    Can be parameterized or not, the four Pauli operators :math:`\hat{X},\hat{Y},\hat{Z}`
    and :math:`\hat{I}` are supported.

    Args:
        num_qubits (int): Number of qubits.
        qubit (int): Qubit on which the Pauli operator acts.
        op_str (str): Pauli operator to measure. Must be one of ``'I'``, ``'X'``, ``'Y'``, ``'Z'``
                      (default: ``'Z'``).
        parameterized (bool): If True, the operator is parameterized (default: False).

    Attributes:
        num_qubits (int): Number of qubits.
        num_parameters (int): Number of trainable parameters in the single pauli operator.
        qubit (int): Qubit on which the Pauli operator acts.
        op_str (str): Pauli operator to measure.
        parameterized (bool): If True, the operator is parameterized.

    """

    def __init__(
        self, num_qubits: int, qubit: int, op_str: str = "Z", parameterized: bool = False
    ) -> None:
        super().__init__(num_qubits)

        self.qubit = qubit
        self.op_str = op_str
        self.parameterized = parameterized

        if self.op_str not in ["I", "X", "Y", "Z"]:
            raise ValueError("Specified operator not supported")

    @property
    def num_parameters(self):
        """The number of trainable parameters in the single pauli operator"""

        if self.parameterized:
            return 1
        else:
            return 0

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
        """
        Function for generating the PauliOp expression of the single pauli operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Parameters of the single
                                                             pauli operator.

        Return:
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
