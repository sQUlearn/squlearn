import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli

from ..expectation_operator_base import ExpectationOperatorBase


class SummedPaulis(ExpectationOperatorBase):
    r"""

    Operator for summed Pauli expectation values.

    For example, if Z operator is specified, the operator is given by

    .. math::
        \hat{H} = a\hat{I} + \sum_i b_i \hat{Z}_i

    Multiple Pauli operators can be specified by list or tuple of strings, e.g. ``op_str=['X','Z']``.

    .. math::
        \hat{H} = a\hat{I} + \sum_i b_i \hat{X}_i + \sum_i c_i \hat{Z}_i

    Args:
        num_qubits (int): Number of qubits.
        op_str (Union[list[str],str,tuple[str]]): Pauli operator string to be measured.
            Also list or tuples of strings are allowed for multiple Pauli operators.
        full_sum (bool): If False, the parameters are excluded from the sum;
            i.e. the sum is :math:`b\sum_i \hat{Z}_i` instead of :math:`\sum_i b_i \hat{Z}_i`

    """

    def __init__(
        self,
        num_qubits: int,
        op_str: Union[list[str], str, tuple[str]] = "Z",
        full_sum: bool = True,
    ) -> None:
        super().__init__(num_qubits)
        self.op_str = op_str
        self.full_sum = full_sum

        for s in self.op_str:
            if s not in ["I", "X", "Y", "Z"]:
                raise ValueError("Only Pauli operators I, X, Y, Z are allowed.")

    @property
    def num_parameters(self):
        """Number of free parameters in the summed pauli operator"""
        if self.full_sum:
            return 1 + len(self.op_str) * self.num_qubits
        else:
            return 1 + len(self.op_str)

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]):
        """
        Function for generating the PauliOp expression of the summed Paulis operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Parameters of the summed
                Paulis operator.

        Returns:
            PauliOp expression of the specified summed Paulis operator.
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
