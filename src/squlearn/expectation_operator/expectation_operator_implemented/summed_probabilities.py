import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SummedProbabilities(ExpectationOperatorBase):
    r"""
    Operator for summing single Qubit probabilities of binary states.

    **Equation for a sum of 0-states:**

    .. math::
        \hat{H} = a\hat{I} + \sum_i b_i (\ket{0}\bra{0})_i

    States are implemented by :math:`\ket{0}\bra{0} = 0.5(\hat{I}+\hat{Z})`
    and :math:`\ket{1}\bra{1} = 0.5(\hat{I}-\hat{Z})`.

    The parameter can be optionally equal for all states.

    Args:
        num_qubits (int): Number of qubits.
        one_state (bool): If false the :math:`\ket{0}\bra{0}` state is measured,
                   if true the :math:`\ket{1}\bra{1}` state is measured (default: False).
        full_sum (bool): If False, the parameter is the same for all states,
                         i.e. the sum is :math:`b\sum_i (\ket{0}\bra{0})_i`
                         instead of :math:`\sum_i b_i (\ket{0}\bra{0})_i`
                         (default: True).
        include_identity (bool): If True, the identity operator is included in the sum.
                                 (default: True)

    Attributes:
        num_qubits (int): Number of qubits.
        num_parameters (int): Number of trainable parameters in the summed probabilities operator.
        one_state (bool): If false the :math:`\ket{0}\bra{0}` state is measured,
                   if true the :math:`\ket{1}\bra{1}` state is measured.
        full_sum (bool): If False, the parameter is the same for all states.
        include_identity (bool): If True, the identity operator is included in the sum.

    """

    def __init__(
        self,
        num_qubits: int,
        one_state=False,
        full_sum: bool = True,
        include_identity: bool = True,
    ) -> None:
        super().__init__(num_qubits)
        self.one_state = one_state
        self.full_sum = full_sum
        self.include_identity = include_identity

    @property
    def num_parameters(self):
        """The number of trainable parameters in the summed probabilities operator"""
        num_param = 0
        if self.include_identity:
            num_param += 1
        if self.full_sum:
            num_param += self.num_qubits
        else:
            num_param += 1
        return num_param

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray] = None):
        """
        Function for generating the PauliOp expression of the summed probabilities operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Parameters of the summed
                probabilities operator.

        Returns:
            PauliOp expression of the specified summed probabilities operator.
        """

        nparam = len(parameters)
        if self.include_identity:
            H = PauliOp(Pauli("I" * self.num_qubits)) * parameters[0 % nparam]
        else:
            H = PauliOp(Pauli("I" * self.num_qubits)) * 0.0

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
