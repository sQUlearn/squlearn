import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from ..observable_base import ObservableBase


class SinglePauli(ObservableBase):
    r"""
    Observable constructed from a single Pauli operator of a single Qubit.

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
    -----------

    Attributes:
        num_qubits (int): Number of qubits.
        num_parameters (int): Number of trainable parameters in the single Pauli operator.
        qubit (int): Qubit on which the Pauli operator acts.
        op_str (str): Pauli operator to measure.
        parameterized (bool): If True, the operator is parameterized.

    Methods:
    --------
    """

    def __init__(
        self, num_qubits: int, qubit: int = 0, op_str: str = "Z", parameterized: bool = False
    ) -> None:
        super().__init__(num_qubits)

        self.qubit = qubit
        self.op_str = op_str
        self.parameterized = parameterized

        if self.op_str not in ["I", "X", "Y", "Z"]:
            raise ValueError("Specified operator not supported")

    @property
    def num_parameters(self):
        """The number of trainable parameters in the single Pauli operator"""

        if self.parameterized:
            return 1
        else:
            return 0

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Single Pauli operator.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["qubit"] = self.qubit
        params["op_str"] = self.op_str
        params["parameterized"] = self.parameterized
        return params

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray]) -> SparsePauliOp:
        """
        Function for generating the SparsePauliOp expression of the single Pauli operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Parameters of the single
                                                             Pauli operator.

        Return:
            SparsePauliOp expression of the specified single Pauli operator.
        """

        i = self.qubit
        if 0 > i or self.num_qubits <= i:
            raise ValueError("Specified qubit out of range")

        H = "I" * self.num_qubits
        if self.parameterized:
            return SparsePauliOp([H[(i + 1) :] + self.op_str + H[:i]], [parameters[0]])

        return SparsePauliOp([H[(i + 1) :] + self.op_str + H[:i]])
