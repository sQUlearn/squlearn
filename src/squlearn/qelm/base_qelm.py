from abc import abstractmethod, ABC
from typing import Callable, Union
import numpy as np
from sklearn.base import BaseEstimator

from qiskit.quantum_info import random_pauli_list, SparsePauliOp
from qiskit.quantum_info.operators.random import random_unitary



from ..observables.observable_base import ObservableBase
from ..observables import CustomObservable
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ..util import Executor
from ..qnn.lowlevel_qnn import LowLevelQNN

class BaseQELM(BaseEstimator, ABC):

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        executor: Executor,
        ml_model: str = 'linear', # or 'mlp'
        num_operators: int = 100,
        operator_seed: int = 0,
        operators: Union[ObservableBase, list[ObservableBase]] = None,
        param_ini: Union[np.ndarray, None] = None,
        param_op_ini: Union[np.ndarray, None] = None,
        parameter_seed: Union[int, None] = 0,
        caching: bool = True,
        ) -> None:

        super().__init__()

        self.encoding_circuit = encoding_circuit
        self.executor = executor
        self.ml_model = ml_model
        self.num_operators = num_operators
        self.operator_seed = operator_seed
        self.operators = operators
        self.param_ini = param_ini
        self.param_op_ini = param_op_ini
        self.parameter_seed = parameter_seed
        self.caching = caching

        if self.operators is None:
            # Generate random operators
            paulis = random_pauli_list(self.encoding_circuit.num_qubits,self.num_operators,seed=self.operator_seed,phase=False)
            self.operators = [CustomObservable(self.encoding_circuit.num_qubits,str(p)) for p in paulis]
        else:
            if isinstance(self.operators, ObservableBase):
                self.operators = [self.operators]
            self.num_operators = len(self.operators)

        self._initialize_lowlevel_qnn()

    def _initialize_lowlevel_qnn(self):
        self._qnn = LowLevelQNN(
            self.encoding_circuit, self.operators, self.executor, result_caching=self.caching
        )

        if self.param_ini is not None:
            if len(self.param_ini) != self.encoding_circuit.num_parameters:
                self.param_ini = self.encoding_circuit.generate_initial_parameters(
                    seed=self.parameter_seed
                )
        else:
            self.param_ini = self.encoding_circuit.generate_initial_parameters(
                seed=self.parameter_seed
            )

        num_op_parameters = sum(operator.num_parameters for operator in self.operators)
        if self.param_op_ini is not None:
            if len(self.param_ini) != self.encoding_circuit.num_parameters:
                if num_op_parameters != len(self.param_op_ini):
                    self.param_op_ini = np.concatenate(
                        [
                            operator.generate_initial_parameters(seed=self.parameter_seed)
                            for operator in self.operators
                        ]
                    )
        else:
            self.param_op_ini = np.concatenate(
                [
                    operator.generate_initial_parameters(seed=self.parameter_seed)
                    for operator in self.operators
                ]
            )



    def get_params(self, deep: bool = True) -> dict:
        """
        Returns a dictionary of parameters for the current object.

        Parameters:
            deep: If True, includes the parameters from the base class.

        Returns:
            dict: A dictionary of parameters for the current object.
        """
        # Create a dictionary of all public parameters
        params = super().get_params(deep=False)

        if deep:
            params.update(self._qnn.get_params(deep=True))
        return params