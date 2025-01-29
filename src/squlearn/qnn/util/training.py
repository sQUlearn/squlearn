"""QNN Training Utilities."""

from typing import Union

import numpy as np
from sklearn.utils import gen_batches

from ..loss.qnn_loss_base import QNNLossBase
from ..lowlevel_qnn.lowlevel_qnn_base import LowLevelQNNBase
from ...optimizers.optimizer_base import OptimizerBase, SGDMixin, IterativeMixin
from ...util import Executor


def get_variance_fac(v: float, a: float, b: float, offset: int = 0):
    r"""
    Function for adjusting the variance regularization along the iterations.

    Based on the sigmoid function, see Ref. [1] for details:

    .. math::
        \alpha_{a,b,v}(i) = (1-v)\frac{\exp(a(b-i))}{\exp(a(b-i))+\frac{1}{b}}+v

    Args:
        v (float): Minimal variance factor value
        a (float): Decay of the variance factor
        b (float): Length of the plateau in the beginning
        offset (int): Offset for the number of iterations (e.g. for restart) (default:0).

    Returns:
        Returns function with iteration as input for adjusting the variance factor

    References:
        [1] D. A. Kreplin and M. Roth "Reduction of finite sampling noise in quantum neural networks".
        `arXiv:2306.01639 <https://arxiv.org/abs/2306.01639>`_ (2023).
    """

    def get_variance_fac_func(iteration: int):
        """Function that return the variance parameter for a given iteration."""
        return (1 - v) * (np.exp(a * (b - offset - iteration))) / (
            np.exp(a * (b - offset - iteration)) + 1 / b
        ) + v

    return get_variance_fac_func


def get_lr_decay(lr_start: float, lr_end: float, iter_decay: float, iter_plateau: int = 0):
    """
    Function for running an Adam optimization with a decay in the learning rate.

    Can be inputted to the learning rate of the Adam optimization.

    Args:
        lr_start (float): start value of the learning rate
        lr_end (float): final value of the learning rate
        iter_decay (float): decay of the learning rate
        iter_plateau (int): length of the plateau of the start value (default: 0)

    Returns:
        Returns function with iteration as input for adjusting the learning rate
    """
    a = (np.log(lr_end) - np.log(lr_start)) / (iter_decay)

    def lr_decay(iteration: int):
        """Function that return the learning rate for a given iteration."""
        val = lr_start * np.exp(a * float(iteration - iter_plateau))
        val = np.clip(val, lr_end, lr_start)
        return val

    return lr_decay


class ShotControlBase:
    """Base Class for shot control"""

    def __init__(self) -> None:
        self._executor = None
        self._initial_shots = None

    def set_executor(self, executor: Executor) -> None:
        """Function for setting the executor that is used for the shot control.

        Args:
            executor (Executor): Executor instance
        """
        self._executor = executor
        self._initial_shots = self._executor.shots

    @property
    def executor(self) -> Executor:
        """Executor of that is used for shot control"""
        return self._executor

    @property
    def shots(self) -> int:
        """Current number of shots"""
        if self._executor is None:
            raise ValueError("Executor not set, call set_executor() first")
        return self._executor.shots

    def reset_shots(self) -> None:
        """Reset the shots to the initial value."""
        if self._executor is None:
            raise ValueError("Executor not set, call set_executor() first")
        if self._initial_shots is not None:
            self._executor.set_shots(self._initial_shots)

    def set_shots_for_loss(self, **kwargs):
        """Function for setting the shots for the loss function evaluation.

        Default function resets the shots to the initial value.

        Args:
            kwargs: Keyword arguments for the loss function evaluation
        """
        self.reset_shots()

    def set_shots_for_grad(self, **kwargs):
        """Function for setting the shots for the gradient evaluation.

        Default function resets the shots to the initial value.

        Args:
            kwargs: Keyword arguments for the loss function evaluation
        """
        if self._executor is None:
            raise ValueError("Executor not set, call set_executor() first")
        self.reset_shots()


class ShotsFromRSTD(ShotControlBase):
    r"""Shot control for setting the shots of the gradient evaluation after the RSTD of the loss.

    The number of shots in the gradient evaluation is set to:

    .. math::
        N_\text{shots} = \frac{\sigma_L^2}{L^2 \beta^2},

    where :math:`\sigma_L` is the standard deviation of the loss, :math:`L` is the loss value and
    :math:`\text{\beta}` is the bound for the relative standard deviation (RSTD) of the loss.
    See Ref. [1] for details:

    Args:
        rstd_bound (float): Bound for the RSTD of the loss (default: 0.1)
        min_shots (int): Minimal number of shots (default: 100)
        max_shots (int): Maximal number of shots, is also used for function evaluation
                         (default: 5000)

    References:
        [1] D. A. Kreplin and M. Roth "Reduction of finite sampling noise in quantum neural networks".
        `arXiv:2306.01639 <https://arxiv.org/abs/2306.01639>`_ (2023).
    """

    def __init__(
        self, rstd_bound: float = 0.1, min_shots: int = 100, max_shots: int = 5000
    ) -> None:
        super().__init__()
        self.rstd_bound = rstd_bound
        self.min_shots = min_shots
        self.max_shots = max_shots
        self._initial_shots = max_shots

    def set_executor(self, executor: Executor) -> None:
        """Function for setting the executor that is used for the shot control.

        Args:
            executor (Executor): Executor instance
        """
        self._executor = executor

    def set_shots_for_loss(self, **kwargs):
        """Function for setting the shots for the loss function evaluation.

        Sets the shots to the maximal value.

        Args:
            kwargs: Keyword arguments for the loss function evaluation
        """
        self._executor.set_shots(self.max_shots)

    def set_shots_for_grad(self, **kwargs):
        """Function for setting the shots for the gradient evaluation.

        Sets the shots to the value that is determined by the RSTD of the loss.

        Args:
            kwargs: Keyword arguments for the loss function evaluation

        """
        if "value" not in kwargs:
            raise AttributeError("Value requires ground_truth.")

        if "variance" not in kwargs:
            raise AttributeError("Variance requires variance.")

        if self._executor is None:
            raise ValueError("Executor not set, call set_executor() first")

        value = kwargs["value"]
        variance = kwargs["variance"]

        shots = int(np.divide(variance, np.square(value) * np.square(self.rstd_bound)))
        num_shots = min(max(shots, self.min_shots), self.max_shots)
        self._executor.set_shots(num_shots)
        # print(
        #     "Set shots for gradient evaluation to: ",
        #     num_shots,
        #     " ( RSTD: ",
        #     "%0.3f" % np.divide(np.sqrt(variance / num_shots), value),
        #     ")",
        # )


def train(
    qnn: LowLevelQNNBase,
    input_values: Union[list, np.ndarray],
    ground_truth: Union[list, np.ndarray],
    param_ini: Union[list, np.ndarray],
    param_op_ini: Union[list, np.ndarray],
    loss: QNNLossBase,
    optimizer: OptimizerBase,
    shot_control: ShotControlBase = None,
    weights: Union[list, np.ndarray] = None,
    opt_param_op: bool = True,
):
    """
    Function for training a given QNN.

    Args:
        QNN (LowLevelQNNBase): QNN instance that is trained
        input_values (Union[list,np.ndarray]): List of input values, i.e. training data
        ground_truth (Union[list,np.ndarray]): List of ground truth values,
                                               e.g. labels of the training data
        param_ini (Union[list,np.ndarray]): Initial parameters of the encoding circuit
        param_op_ini (Union[list,np.ndarray]): Initial parameters of the observable
        loss (QNNLossBase): Loss instance that is minimized
        optimizer (OptimizerBase): Optimizer instance that is used for the minimization
        shot_control (ShotControlBase): Shot control instance that is used for setting the shots
                                        for each optimization step (default: None)
        weights (Union[list,np.ndarray]): Weighting of the reference values. Has to be the same
                                          size as input and ground_truth (default : None)
        opt_param_op (bool): If True, observable parameters are optimized as well (default: True)

    Returns:
        Optimized parameters of the PQC, and, if opt_param_op=True,
        the optimized parameters of the observable
    """
    if isinstance(weights, np.ndarray):
        weights_values = weights
    elif weights is None:
        weights_values = np.ones(np.shape(ground_truth))
    else:
        raise TypeError(f"Unknown weight format: {type(weights)}")

    # Tell the loss function if the cost operator parameters are optimized
    loss.set_opt_param_op(opt_param_op)

    if weights_values.shape != np.shape(ground_truth):
        raise ValueError(
            f"Shape {weights_values.shape} of weight values doesn't match shape"
            f" {np.shape(ground_truth)} of reference values"
        )

    # Preprocessing of the input values in case of lists
    if not isinstance(param_ini, np.ndarray):
        param = np.array([param_ini])
    else:
        param = param_ini
    if not isinstance(param_op_ini, np.ndarray):
        param_op = np.array([param_op_ini])
    else:
        param_op = param_op_ini

    # Merge initialization values for minimize
    val_ini = param
    if opt_param_op:
        val_ini = np.concatenate((val_ini, param_op), axis=None)

    iteration = 0

    def _fun(theta):
        nonlocal iteration
        nonlocal optimizer
        nonlocal param_op
        if isinstance(optimizer, IterativeMixin):
            iteration = optimizer.iteration
        else:
            iteration = None

        # Splitting theta in the arrays
        if opt_param_op:
            param_ = theta[: len(param_ini)]
            param_op_ = theta[len(param_ini) :]
        else:
            param_ = theta
            param_op_ = param_op

        # Shot controlling
        if shot_control is not None:
            if isinstance(shot_control, ShotsFromRSTD):
                shot_control.set_shots_for_loss()

        loss_values = qnn.evaluate(input_values, param_, param_op_, *loss.loss_args_tuple)

        loss_value = loss.value(
            loss_values,
            ground_truth=ground_truth,
            weights=weights_values,
            iteration=iteration,
        )
        return loss_value

    def _grad(theta):
        nonlocal iteration
        nonlocal optimizer
        nonlocal param_op
        if isinstance(optimizer, IterativeMixin):
            iteration = optimizer.iteration
        else:
            iteration = None

        # Splitting theta in the arrays
        if opt_param_op:
            param_ = theta[: len(param_ini)]
            param_op_ = theta[len(param_ini) :]
        else:
            param_ = theta
            param_op_ = param_op

        # Shot controlling
        if shot_control is not None:
            if isinstance(shot_control, ShotsFromRSTD):
                if loss.loss_variance_available:
                    loss_variance = loss.variance(
                        qnn.evaluate(input_values, param_, param_op_, *loss.variance_args_tuple),
                        ground_truth=ground_truth,
                        weights=weights_values,
                        iteration=iteration,
                    )
                    loss_values = loss.value(
                        qnn.evaluate(input_values, param_, param_op_, *loss.loss_args_tuple),
                        ground_truth=ground_truth,
                        weights=weights_values,
                        iteration=iteration,
                    )
                    shot_control.set_shots_for_grad(value=loss_values, variance=loss_variance)
                else:
                    raise ValueError("Loss variance necessary for ShotsFromRSTD shot control")

        grad_values = qnn.evaluate(input_values, param_, param_op_, *loss.gradient_args_tuple)
        grad = np.concatenate(
            loss.gradient(
                grad_values,
                ground_truth=ground_truth,
                weights=weights_values,
                iteration=iteration,
                multiple_output=qnn.multiple_output,
                opt_param_op=opt_param_op,
            ),
            axis=None,
        )
        return grad

    if len(val_ini) == 0:
        if opt_param_op:
            return np.array([]), np.array([])
        else:
            return np.array([])

    result = optimizer.minimize(_fun, val_ini, _grad, bounds=None)

    if hasattr(result, "x"):
        result = result.x

    if opt_param_op:
        param = result[: len(param_ini)]
        param_op = result[len(param_ini) :]
        return param, param_op

    param = result
    return param


def train_mini_batch(
    qnn: LowLevelQNNBase,
    input_values: Union[list, np.ndarray],
    ground_truth: Union[list, np.ndarray],
    param_ini: Union[list, np.ndarray],
    param_op_ini: Union[list, np.ndarray],
    loss: QNNLossBase,
    optimizer: OptimizerBase,
    shot_control: ShotControlBase = None,
    weights: Union[list, np.ndarray] = None,
    opt_param_op: bool = True,
    epochs: int = 10,
    batch_size: int = None,
    shuffle=False,
):
    """Minimize a loss function using mini-batch gradient descent.

    Args:
        QNN (LowLevelQNNBase): QNN instance that is trained
        input_values (Union[list,np.ndarray]): List of input values, i.e. training data
        ground_truth (Union[list,np.ndarray]): List of ground truth values,
                                               e.g. labels of the training data
        param_ini (Union[list,np.ndarray]): Initial parameters of the encoding circuit
        param_op_ini (Union[list,np.ndarray]): Initial parameters of the observable
        loss (QNNLossBase): Loss instance that is minimized
        optimizer (OptimizerBase): Optimizer instance that is used for the minimization
        shot_control (ShotControlBase): Shot control instance that is used for setting the shots
                                        for each optimization step (default: None)
        weights (Union[list,np.ndarray]): Weighting of the reference values. Has to be the same
                                          size as input and ground_truth (default : None)
        opt_param_op (bool): If True, observable parameters are optimized as well (default: True)        epochs : Number of epochs of SGD to perform
        batch_size : Number of data points in each batch
        shuffle : If True, data points get shuffled before each epoch (default: False)

    Returns:
        Optimized parameters of the PQC, and, if opt_param_op=True,
        the optimized parameters of the observable
    """
    if not isinstance(optimizer, SGDMixin):
        raise TypeError(
            f"Optimizer {optimizer.__class__.__name__} is not supported for mini-batch gradient "
            "descent."
        )

    if isinstance(weights, np.ndarray):
        weights_values = weights
    elif weights is None:
        weights_values = np.ones(ground_truth.shape)
    else:
        raise TypeError(f"Unknown weight format: {type(weights)}")

    # Tell the loss function if the cost operator parameters are optimized
    loss.set_opt_param_op(opt_param_op)

    if weights_values.shape != ground_truth.shape:
        raise ValueError(
            f"Shape {weights_values.shape} of weight values doesn't match shape"
            f" {ground_truth.shape} of reference values"
        )

    n_samples = len(input_values)

    idcs = np.arange(n_samples)

    if epochs is None:
        epochs = 10

    if batch_size is None:
        batch_size = min(100, n_samples)

    if not isinstance(param_ini, np.ndarray):
        param = np.array([param_ini])
    else:
        param = param_ini

    if not isinstance(param_op_ini, np.ndarray):
        param_op = np.array([param_op_ini])
    else:
        param_op = param_op_ini

    if len(param_ini) == 0:
        if opt_param_op:
            if len(param_op_ini) == 0:
                return np.array([]), np.array([])
        else:
            return np.array([])

    for epoch in range(epochs):
        accumulated_loss = 0.0
        if shuffle:
            idcs = np.random.permutation(idcs)
        for batch_slice in gen_batches(n_samples, batch_size):
            # Shot controlling
            if shot_control is not None:
                if isinstance(shot_control, ShotsFromRSTD):
                    shot_control.set_shots_for_loss()

            loss_values = qnn.evaluate(
                input_values[idcs[batch_slice]], param, param_op, *loss.loss_args_tuple
            )

            batch_loss = loss.value(
                loss_values,
                ground_truth=ground_truth[idcs[batch_slice]],
                weights=weights_values[idcs[batch_slice]],
                iteration=epoch,
            )

            accumulated_loss += batch_loss

            # Shot controlling
            if shot_control is not None:
                if isinstance(shot_control, ShotsFromRSTD):
                    if loss.loss_variance_available:
                        batch_loss_variance = loss.variance(
                            qnn.evaluate(
                                input_values[idcs[batch_slice]],
                                param,
                                param_op,
                                *loss.variance_args_tuple,
                            ),
                            ground_truth=ground_truth[idcs[batch_slice]],
                            weights=weights_values[idcs[batch_slice]],
                            iteration=epoch,
                        )

                        shot_control.set_shots_for_grad(
                            value=batch_loss, variance=batch_loss_variance
                        )
                    else:
                        raise ValueError("Loss variance necessary for ShotsFromRSTD shot control")

            diff_values = qnn.evaluate(
                input_values[idcs[batch_slice]], param, param_op, *loss.gradient_args_tuple
            )

            grad = loss.gradient(
                diff_values,
                ground_truth=ground_truth[idcs[batch_slice]],
                weights=weights_values[idcs[batch_slice]],
                iteration=epoch,
                multiple_output=qnn.multiple_output,
                opt_param_op=opt_param_op,
            )

            if opt_param_op:
                updated_params = optimizer.step(
                    x=np.concatenate((param, param_op), axis=None),
                    grad=np.concatenate(grad, axis=None),
                )
                param = updated_params[: len(param_ini)]
                param_op = updated_params[len(param_ini) :]
            else:
                param = optimizer.step(
                    x=param,
                    grad=grad,
                )
        if optimizer.callback:
            if opt_param_op:
                optimizer.callback(
                    epoch,
                    np.concatenate((param, param_op), axis=None),
                    np.concatenate(grad, axis=None),
                    accumulated_loss / n_samples,
                )

    if opt_param_op:
        return param, param_op

    return param
