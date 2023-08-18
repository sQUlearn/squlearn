"""QNN Base Implemenation"""
from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Callable, Union
from warnings import warn

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
        feature_map : Parameterized quantum circuit in feature map format
        operator : Operator that are used in the expectation value of the QNN. Can be a list for
            multiple outputs.
        executor : Executor instance
        optimizer : Optimizer instance
        param_ini : Initialization values of the parameters of the PQC
        param_op_ini : Initialization values of the cost operator
        batch_size : Number of data points in each batch, for SGDMixin optimizers
        epochs : Number of epochs of SGD to perform, for SGDMixin optimizers
        shuffle : If True, data points get shuffled before each epoch (default: False),
            for SGDMixin optimizers
        opt_param_op : If True, operators parameters get optimized
        variance : Variance factor
        parameter_seed : Seed for the random number generator for the parameter initialization
    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.feature_map = feature_map
        self.operator = operator
        self.loss = loss
        self.optimizer = optimizer
        self.variance = variance
        self.parameter_seed = parameter_seed

        if param_ini is None:
            self.param_ini = feature_map.generate_initial_parameters(seed=parameter_seed)
        else:
            self.param_ini = param_ini
        self._param = self.param_ini.copy()

        if param_op_ini is None:
            if isinstance(operator, list):
                self.param_op_ini = np.concatenate(
                    [
                        operator.generate_initial_parameters(seed=parameter_seed)
                        for operator in operator
                    ]
                )
            else:
                self.param_op_ini = operator.generate_initial_parameters(seed=parameter_seed)
        else:
            self.param_op_ini = param_op_ini
        self._param_op = self.param_op_ini.copy()

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
        self._qnn = QNN(self.feature_map, self.operator, executor)

        update_params = self.get_params().keys() & kwargs.keys()
        if update_params:
            self.set_params(**{key: kwargs[key] for key in update_params})

        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Fit a new model to data.

        This method will reinitialize the models parameters and fit it to the provided data.

        Args:
            X: Input data
            y: Labels
            weights: Weights for each data point
        """
        self._param = self.param_ini.copy()
        self._param_op = self.param_op_ini.copy()
        self._is_fitted = False
        self._fit(X, y, weights)

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

    def set_params(self: BaseQNN, **params) -> BaseQNN:
        """
        Sets the hyper-parameters of the BaseQNN.

        Args:
            params: Hyper-parameters of the BaseQNN.

        Returns:
            updated BaseQNN
        """
        # Create dictionary of valid parameters
        valid_params = self.get_params().keys()
        for key in params.keys():
            # Check if parameter is valid
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        # Set parameters
        self_params = params.keys() & self.get_params(deep=False).keys()
        for key in self_params:
            setattr(self, key, params[key])

        # Set parameters of the QNN
        qnn_params = params.keys() & self._qnn.get_params(deep=True).keys()
        if qnn_params:
            self._qnn.set_params(
                **{key: value for key, value in params.items() if key in qnn_params}
            )

            # If the number of parameters has changed, reinitialize the parameters
            if self.feature_map.num_parameters != len(self.param_ini):
                self.param_ini = self.feature_map.generate_initial_parameters(
                    seed=self.parameter_seed
                )
            if isinstance(self.operator, list):
                num_op_parameters = sum(operator.num_parameters for operator in self.operator)
                if num_op_parameters != len(self.param_op_ini):
                    self.param_op_ini = np.concatenate(
                        [
                            operator.generate_initial_parameters(seed=self.parameter_seed)
                            for operator in self.operator
                        ]
                    )
            elif self.operator.num_parameters != len(self.param_op_ini):
                self.param_op_ini = self.operator.generate_initial_parameters(
                    seed=self.parameter_seed
                )
            if isinstance(self.optimizer, SGDMixin):
                self.optimizer.reset()

        self._is_fitted = False

        return self

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> None:
        """Internal fit function."""
        raise NotImplementedError()
