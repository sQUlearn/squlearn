import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SummedProbabilities(ExpectationOperatorBase):
    r"""
    Operator for summed probabilities of being in state 0 or 1.

    Operator reads:

    .. math::

        \hat{H} = a\hat{I} + \sum_i b_i (\ket{0}\bra{0})_i

    Implemented by :math:`\ket{0}\bra{0} = 0.5(\hat{I}+\hat{Z})` and :math:`\ket{1}\bra{1} = 0.5(\hat{I}-\hat{Z})`

    Args:
        num_qubits (int): Number of qubits.
        one_state: If false the :math:`\ket{0}\bra{0}` state is measured,
            if true the :math:`\ket{1}\bra{1}` state is measured
        full_sum (bool): If False, the parameters are excluded from the sum.
            (i.e. the sum is :math:`b\sum_i (\ket{0}\bra{0})_i`
            instead of :math:`\sum_i b_i (\ket{0}\bra{0})_i)`

    """

    def __init__(self, num_qubits: int, one_state=False, full_sum: bool = True) -> None:

        super().__init__(num_qubits)
        self.one_state = one_state
        self.full_sum = full_sum

    @property
    def num_parameters(self):
        """Returns the number of free parameters in the summed probabilities operator"""
        if self.full_sum:
            return 1 + self.num_qubits
        else:
            return 2

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
