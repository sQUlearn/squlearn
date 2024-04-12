import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from ..observable_base import ObservableBase


class SummedPaulis(ObservableBase):
    r"""
    Observable for summation of single Pauli operators.

    **Equation for Z Pauli operator:**

    .. math::
        \hat{H} = a\hat{I} + \sum_i b_i \hat{Z}_i

    Multiple Pauli operators can be specified by a tuple of strings, e.g. ``op_str=("X","Z")``:

    .. math::
        \hat{H} = a\hat{I} + \sum_i b_i \hat{X}_i + \sum_i c_i \hat{Z}_i

    The parameter can optionally be equal for the same kind of Pauli operators.

    Args:
        num_qubits (int): Number of qubits.
        op_str (Union[str,tuple[str]]): String of the Pauli operator that is  measured.
                                        Possible values are ``"I"``, ``"X"``, ``"Y"``, ``"Z"``.
                                        Also a tuples of strings are allowed for multiple Pauli
                                        operators (default: ``"Z"``).
        full_sum (bool): If False, only one parameter is used for each Pauli operator,
                         i.e. the sum is :math:`b\sum_i \hat{Z}_i`
                         instead of :math:`\sum_i b_i \hat{Z}_i` (default: True).
        include_identity (bool): If True, the identity operator is included in the sum.
                                 (default: True)

    Attributes:
    -----------

    Attributes:
        num_qubits (int): Number of qubits.
        num_parameters (int): Number of trainable parameters in the summed Paulis operator.
        op_str (Union[str,tuple[str]]): String of the Pauli operator that is  measured.
        full_sum (bool): If False, only one parameter is used for each Pauli operator.
        include_identity (bool): If True, the identity operator is included in the sum.

    Methods:
    --------
    """

    def __init__(
        self,
        num_qubits: int,
        op_str: Union[str, tuple[str]] = "Z",
        full_sum: bool = True,
        include_identity: bool = True,
    ) -> None:
        super().__init__(num_qubits)
        self.op_str = op_str
        self.full_sum = full_sum
        self.include_identity = include_identity

        for s in self.op_str:
            if s not in ["I", "X", "Y", "Z"]:
                raise ValueError("Only Pauli operators I, X, Y, Z are allowed.")

    @property
    def num_parameters(self):
        """Number of trainable parameters in the summed Pauli operator"""
        num_param = 0
        if self.include_identity:
            num_param += 1
        if self.full_sum:
            return num_param + len(self.op_str) * self.num_qubits
        else:
            return num_param + len(self.op_str)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the single summed Pauli operator.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["op_str"] = self.op_str
        params["full_sum"] = self.full_sum
        params["include_identity"] = self.include_identity
        return params

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]) -> SparsePauliOp:
        """
        Function for generating the PauliOp expression of the summed Paulis operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Parameters of the summed
                                                             Paulis operator.

        Return:
            PauliOp expression of the specified summed Paulis operator.
        """

        def gen_string(i, op_str):
            H = "I" * self.num_qubits
            H = H[i + 1 :] + op_str + H[:i]
            return H

        nparam = len(parameters)
        ioff = 0

        op_list = []
        param_list = []

        if self.include_identity:
            op_list.append("I" * self.num_qubits)
            param_list.append(parameters[ioff % nparam])
            ioff += 1

        for op_str in self.op_str:
            for i in range(self.num_qubits):
                op_list.append(gen_string(i, op_str))
                param_list.append(parameters[ioff % nparam])
                if self.full_sum:
                    ioff += 1
            if not self.full_sum:
                ioff += 1

        return SparsePauliOp(op_list, np.array(param_list))
