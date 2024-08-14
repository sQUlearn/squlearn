from typing import Callable, Union
import numpy as np

from sklearn.base import ClassifierMixin


from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase

from .base_qelm import BaseQELM
from ..util import Executor


class QELMClassifier(BaseQELM, ClassifierMixin):
    """
    Quantum Extreme Learning Machine for classification.

    Parameters:
        encoding_circuit: EncodingCircuitBase
            The encoding circuit to use for encoding the data.
        executor: Executor
    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        executor: Executor,
        ml_model: str = "mlp",  # or 'linear'
        num_operators: int = 100,
        operator_seed: int = 0,
        operators: Union[ObservableBase, list[ObservableBase], str] = "random_paulis",
        param_ini: Union[np.ndarray, None] = None,
        param_op_ini: Union[np.ndarray, None] = None,
        parameter_seed: Union[int, None] = 0,
        caching: bool = True,
    ) -> None:

        super().__init__(
            encoding_circuit,
            executor,
            ml_model,
            num_operators,
            operator_seed,
            operators,
            param_ini,
            param_op_ini,
            parameter_seed,
            caching,
        )

        if self.ml_model == "mlp":
            self.ml_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=300, random_state=42)
        elif self.ml_model == "linear":
            self._ml_model = LogisticRegression()
        else:
            raise ValueError("Invalid ml_model. Please choose 'mlp' or 'linear'.")