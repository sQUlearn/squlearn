
from typing import Callable, Union
import numpy as np

from sklearn.base import RegressorMixin


from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

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
        ml_model: str = 'linear',
        ml_model_options: dict = None,
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
            ml_model_options,
            num_operators,
            operator_seed,
            operators,
            param_ini,
            param_op_ini,
            parameter_seed,
            caching,
        )

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

    def _initialize_ml_model(self):
        if self.ml_model == 'mlp':
            if self.ml_model_options is None:
                self.ml_model_options = {}
            self._ml_model = MLPRegressor(**self.ml_model_options)
        elif self.ml_model == 'linear':
            if self.ml_model_options is None:
                self.ml_model_options = {}
            self._ml_model = LinearRegression(**self.ml_model_options)
        elif self.ml_model == 'kernel':
            if self.ml_model_options is None:
                self.ml_model_options = {}
            self._ml_model = KernelRidge(**self.ml_model_options)
        else:
            raise ValueError("Invalid ml_model. Please choose 'mlp' or 'linear'.")