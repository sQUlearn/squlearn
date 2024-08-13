
from typing import Callable, Union
import numpy as np

from sklearn.base import RegressorMixin


from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

from ..observables.observable_base import ObservableBase
from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase

from .base_qelm import BaseQELM
from ..util import Executor


class QELMRegressor(BaseQELM, RegressorMixin):
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
        ml_model: str = 'linear', # or 'linear'
        num_operators: int = 200,
        operator_seed: int = 0,
        operators: Union[ObservableBase, list[ObservableBase]] = None,
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

        if self.ml_model == 'mlp':
            self._ml_model = MLPRegressor(hidden_layer_sizes=(1,1,), max_iter=300, random_state=42)
        elif self.ml_model == 'linear':
            self._ml_model = LinearRegression()
        else:
            raise ValueError("Invalid ml_model. Please choose 'mlp' or 'linear'.")

    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters:
            X: np.ndarray
                The input data.
            y: np.ndarray
                The target data.
        """

        X_qnn = self._qnn.evaluate(X, self.param_ini, self.param_op_ini, "f")["f"]
        self._ml_model.fit(X_qnn, y)

    def predict(self, X):
        """
        Predict the target data.

        Parameters:
            X: np.ndarray
                The input data.

        Returns:
            np.ndarray: The predicted target data.
        """

        X_qnn = self._qnn.evaluate(X, self.param_ini, self.param_op_ini, "f")["f"]
        return self._ml_model.predict(X_qnn)