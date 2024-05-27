import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from ..observable_base import ObservableBase


class SingleProbability(ObservableBase):
    r"""
    Observable for measuring the probability of being in state 0 or 1 of a specified qubit.

    **Equation as the operator is implemented:**

    .. math::

       \hat{H} = 0.5(\hat{I}_i+\hat{Z}_i) (= \ket{0}\bra{0}_i) \qquad \text{or} \qquad
       \hat{H} = 0.5(\hat{I}_i-\hat{Z}_i) (= \ket{1}\bra{1}_i)

    Operator can be optionally parameterized.

    Args:
        num_qubits (int): Number of qubits.
        qubit (int): Qubit to measure the probability of.
        one_state (bool): If True, measure the probability of being in state 1, otherwise state 0
                          (default: False).
        parameterized (bool): If True, the operator is parameterized (default: false).

    Attributes:
    -----------

    Attributes:
        num_qubits (int): Number of qubits.
        num_parameters (int): Number of trainable parameters in the single Pauli operator.
        qubit (int): Qubit to measure the probability of.
        one_state (bool): If True, measure the probability of being in state 1, otherwise state 0.
        parameterized (bool): If True, the operator is parameterized.

    Methods:
    --------
    """

    def __init__(
        self,
        num_qubits: int,
        qubit: int = 0,
        one_state: bool = False,
        parameterized: bool = False,
    ) -> None:
        super().__init__(num_qubits)

        self.qubit = qubit
        self.one_state = one_state
        self.parameterized = parameterized

    @property
    def num_parameters(self):
        """Number of trainable parameters in the single probability operator."""

        if self.parameterized:
            return 1
        else:
            return 0

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the single probability operator.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["qubit"] = self.qubit
        params["one_state"] = self.one_state
        params["parameterized"] = self.parameterized
        return params

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray] = None) -> SparsePauliOp:
        """
        Function for generating the SparsePauliOp expression of the single probability operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Parameters of the single
                                                             probability operator.

        Return:
            SparsePauliOp expression of the specified single probability operator.
        """

        i = self.qubit
        if 0 > i or self.num_qubits <= i:
            raise ValueError("Specified qubit out of range")
        I = "I" * self.num_qubits
        Z = I[(i + 1) :] + "Z" + I[:i]

        if self.parameterized:
            coeff = 0.5 * parameters[0]
        else:
            coeff = 0.5

        if self.one_state:
            return SparsePauliOp([I, Z], [coeff, -coeff])

        else:
            return SparsePauliOp([I, Z], [coeff, coeff])
