"""QNN Base Implemenation"""
from typing import Callable, Union
from warnings import warn

from abc import abstractmethod, ABC
import numpy as np
from sklearn.base import BaseEstimator

from ..expectation_operator.expectation_operator_base import ExpectationOperatorBase
from ..feature_map.feature_map_base import FeatureMapBase
from ..optimizers.optimizer_base import OptimizerBase, SGDMixin
from ..util import Executor

from .loss import LossBase
from .qnn import QNN
from .training import shot_adjusting_options


class BaseQNN(BaseEstimator, ABC):
    """Base Class for Quantum Neural Networks.

    Args:
        pqc : Parameterized quantum circuit in feature map format
        operator : Operator that are used in the expectation value of the QNN. Can be a list for
            multiple outputs.
        executor : Executor instance
        optimizer : Optimizer instance
        param_ini : Initialization values of the parameters of the PQC
        param_op_ini : Initialization values of the cost operator
        batch_size : Number of datapoints in each batch, for SGDMixin optimizers
        epochs : Number of epochs of SGD to perform, for SGDMixin optimizers
        shuffle : If True, datapoints get shuffled before each epoch (default: False),
            for SGDMixin optimizers
        opt_param_op : If True, operators parameters get optimized
        variance : Variance factor
        parameter_seed : Seed for the random number generator for the parameter initialization
    """

    def __init__(
        self,
        pqc: FeatureMapBase,
        operator: Union[ExpectationOperatorBase, list[ExpectationOperatorBase]],
        executor: Executor,
        loss: LossBase,
        optimizer: OptimizerBase,
        param_ini: Union[np.ndarray, None] = None,
        param_op_ini: Union[np.ndarray, None] = None,
        batch_size: int = None,
        epochs: int = None,
        shuffle: bool = None,
        opt_param_op: bool = True,
        variance: Union[float, Callable] = None,
        shot_adjusting: shot_adjusting_options = None,
        parameter_seed: Union[int, None] = 0,
    ) -> None:
        super().__init__()
        self.pqc = pqc
        self.operator = operator
        self.loss = loss
        self.optimizer = optimizer
        self.variance = variance
        self.parameter_seed = parameter_seed

        if param_ini is None:
            self.param_ini = pqc.generate_initial_parameters(seed=parameter_seed)
        else:
            self.param_ini = param_ini
        self.param = self.param_ini.copy()

        if param_op_ini is None:
            if isinstance(operator, list):
                self.param_op_ini = np.concatenate(
                    [op.generate_initial_parameters(seed=parameter_seed) for op in operator]
                )
            else:
                self.param_op_ini = operator.generate_initial_parameters(seed=parameter_seed)
        else:
            self.param_op_ini = param_op_ini
        self.param_op = self.param_op_ini.copy()

        if not isinstance(optimizer, SGDMixin) and any(
            param is not None for param in [batch_size, epochs, shuffle]
        ):
            warn(
                f"{optimizer.__class__.__name__} is not of type SGDMixin, thus batch_size, epochs"
                " and shuffle will be ignored."
            )
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle

        self.opt_param_op = opt_param_op

        self.shot_adjusting = shot_adjusting

        self.executor = executor
        self._qnn = QNN(self.pqc, self.operator, executor)

        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Fit a new model to data.

        This method will reinitialize the models parameters and fit it to the provided data.

        Args:
            X: Input data
            y: Labels
            weights: Weights for each datapoint
        """
        self.param = self.param_ini.copy()
        self.param_op = self.param_op_ini.copy()
        self._is_fitted = False
        self._fit(X, y, weights)

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Internal fit function."""
        raise NotImplementedError()
