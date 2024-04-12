import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from ..observable_base import ObservableBase


class CustomObservable(ObservableBase):
    r"""
    Class for defining a custom observable.

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
    -----------

    Attributes:
        num_qubits (int): Number of qubits.
        num_parameters (int): Number of trainable parameters in the custom operator.
        operator_string (Union[str, list[str], tuple[str]]): String of operator to measure.
        parameterized (bool): If True, the operator is parameterized.

    Methods:
    --------
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

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the custom operator.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["operator_string"] = self.operator_string
        params["parameterized"] = self.parameterized
        return params

    def get_pauli(self, parameters: Union[ParameterVector, np.ndarray] = None) -> SparsePauliOp:
        """
        Function for generating the SparsePauliOp expression of the custom operator.

        Args:
            parameters (Union[ParameterVector, np.ndarray]): Parameters of the custom operator.

        Returns:
            SparsePauliOp expression of the specified custom operator.
        """

        op_list = []
        param_list = []

        if self.parameterized:
            nparam = len(parameters)
            op_list.append(self.operator_string[0])
            param_list.append(parameters[0 % nparam])

            ioff = 1
            for j in range(1, len(self.operator_string)):
                op_list.append(self.operator_string[j])
                param_list.append(parameters[ioff % nparam])
                ioff = ioff + 1
            return SparsePauliOp(op_list, param_list)

        else:
            op_list.append(self.operator_string[0])
            for j in range(1, len(self.operator_string)):
                op_list.append(self.operator_string[j])
            return SparsePauliOp(op_list)
