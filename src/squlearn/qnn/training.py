import numpy as np

from abc import ABCMeta
from sklearn.utils import gen_batches
from typing import Union

from .loss import LossBase
from .qnn import QNN
from ..optimizers.optimizer_base import OptimizerBase, SGDMixin


def get_variance_fac(v, a, b, offset=0):
    """
    Function for adjusting the variance regularization along the iterations.
    Based on the sigmoid function:

    var_fac(iter,v,a,b,offset) = (1-v)*(exp(a*(b-offset-iter)))/(exp(a*(b-offset-iter))+1/b)+v

    Args:
        v : minimal variance factor value
        a : decay of the variance factor
        b : length of the plateau in the beginning
        offset : offset for the number of iterations (e.g. for restart) (default:0).

    Returns:
        Returns function with iteration as input for adjusting the variance factor
    """

    def get_variance_fac_func(x):
        return (1 - v) * (np.exp(a * (b - offset - x))) / (
            np.exp(a * (b - offset - x)) + 1 / b
        ) + v

    return get_variance_fac_func


def get_annealed_lr(lr_start, lr_end, iter_plateau, iter_decay):
    """
    Function for annealed ADAM optimization.

    Args:
        lr_start : start value of the learning rate
        lr_end : final value of the learning rate
        iter_plateau : length of the plateau of the start value
        iter_decay : decay of the learning rate

    Returns:
        Return function with iteration as input for adjusting the learning rate
    """
    a = (np.log(lr_end) - np.log(lr_start)) / (iter_decay)

    def get_annealed_lr_(x):
        val = lr_start * np.exp(a * (x - iter_plateau))
        val = np.clip(val, lr_end, lr_start)
        return val

    return get_annealed_lr_


class shot_adjusting_options:
    # Move to loss function class?
    """
    Class for automatic adjustment of the shots in an optimization.
    Shots are adjusted such that the relative standard deviation is lower than the given parameter

    Args:
        parameter [float]: Parameter value between 1 and 0 for adjusting the shots
        min_shots [int]: Minimal number of shots
        max_shots [int]: Maximum number of allows shots
    """

    def __init__(
        self, parameter: float = 0.1, min_shots: int = 100, max_shots: int = 5000
    ) -> None:
        self.parameter = parameter
        self.min_shots = min_shots
        self.max_shots = max_shots
        self.func = lambda *args: None
        self.grad = lambda *args: None

    def set_l2_rstd(self, qnn: QNN, Y: np.ndarray) -> None:
        """
        Initialization function for adjusting the number of shots for a L2 Loss.
        Number of shots are adjusted with respect to the relative standard deviation of the L2 loss.

        Args:
            qnn [QNN]: Instance of the QNN that is optimized
            Y [np.ndarray]: Output data that is used in the L2 loss.

        """

        self.qnn = qnn

        def func(x, param, param_op):
            self.qnn.reset_shots()

        self.func = func

        def grad(x, param, param_op):
            self.qnn.reset_shots()
            value_dict = self.qnn.evaluate(("f", "var"), x, param, param_op)
            diff_square = np.square(value_dict["f"] - Y)
            var = np.sum(4 * np.multiply(diff_square, value_dict["var"]))
            exp = np.sum(diff_square)
            shots = int(np.divide(var, np.square(exp) * np.square(self.parameter)))
            num_shots = min(max(shots, self.min_shots), self.max_shots)
            print(
                "Set shots for gradient evaluation to: ",
                num_shots,
                " ( RSTD: ",
                "%0.3f" % np.divide(np.sqrt(var / num_shots), exp),
                ")",
            )
            self.qnn.set_shots(num_shots)

        self.grad = grad


def solve_param(
    qnn: QNN,
    x_space,
    param_ini,
    param_op_ini,
    loss_function,
    input_loss: tuple,
    loss_function_gradient,
    input_grad: tuple,
    minimize: OptimizerBase,
    opt_param_op=True,
    bounds=None,
    shot_adjusting: shot_adjusting_options = None,
):
    """Wrapper function for optimizing only the parameters

    Args:
        x_space : Supporting vectors in x (constant in opt.)
        param_ini : Initial values for the PQC parameters
        param_op_ini : Initial values of the cost operator parameters
        loss_function : Loss function for the optimization (gets a dictionary as input)
        input_loss : Derivatives which has to be accessible in the loss_function dictionary
        loss_function_gradient : Gradient of the loss function (gets a dictionary as input)
        input_grad : Derivatives which has to be accessible in the loss_function_gradient dictionary
        opt_param_op : Option for optimizing the parameter of the cost operator
        bounds : Boundaries for the x variable

    Returns:
        Optimized parameters of the PQC, and, if opt_param_op=True, the parameters
        of the cost operator
    """

    return solve_all(
        qnn,
        x_space,
        param_ini,
        param_op_ini,
        loss_function,
        input_loss,
        loss_function_gradient,
        input_grad,
        minimize,
        (False, True, opt_param_op),
        bounds=bounds,
        shot_adjusting=shot_adjusting,
    )


def solve_x(
    qnn,
    x_ini,
    param,
    param_op,
    loss_function,
    input_loss: tuple,
    loss_function_gradient,
    input_grad: tuple,
    minimize: OptimizerBase,
    bounds=None,
    shot_adjusting: shot_adjusting_options = None,
):
    """Wrapper function for optimizing only the x values

    Args:
        x_space : Initial values of x
        param_ini : Values for the PQC parameters
        param_op_ini : Values of the cost operator parameters
        loss_function : Loss function for the optimization (gets a dictionary as input)
        input_loss : Derivatives which has to be accessible in the loss_function dictionary
        loss_function_gradient : Gradient of the loss function (gets a dictionary as input)
        input_grad : Derivatives which has to be accessible in the loss_function_gradient dictionary
        bounds : Boundaries for the x variable

    Returns:
        Optimized parameters of the PQC, and, if opt_param_op=True, the parameters
        of the cost operator
    """

    return solve_all(
        qnn,
        x_ini,
        param,
        param_op,
        loss_function,
        input_loss,
        loss_function_gradient,
        input_grad,
        minimize,
        (True, False, False),
        bounds=bounds,
        shot_adjusting=shot_adjusting,
    )


def solve_all(
    qnn: QNN,
    x_ini,
    param_ini,
    param_op_ini,
    loss_function,
    input_loss: tuple,
    loss_function_gradient,
    input_grad: tuple,
    minimize: OptimizerBase,
    opt_tuple: tuple = (False, True, True),
    bounds=None,
    shot_adjusting: shot_adjusting_options = None,
):
    """General function for minimizing a given loss function

    Args:
        x_ini : Initial values for x
        param_ini : Initial values for the PQC parameters
        param_op_ini : Initial values of the cost operator parameters
        loss_function : Loss function for the optimization (gets a dictionary as input)
        input_loss : Derivatives which has to be accessible in the loss_function dictionary
        loss_function_gradient : Gradient of the loss function (gets a dictionary as input)
        input_grad : Derivatives which has to be accessible in the loss_function_gradient dictionary
        opt_tuple : Three entry boolean tuple which specifies which variables are optimized
                    Structure: (opt x, opt param, opt param_op)
        bounds : Boundaries for the optimization variable

    Returns:
        Returns a tuple of array containing the optimized values (specified by opt_tuple)

    """
    # # Container class for storing already calculated data to avoid
    # # recomputation of already known data points
    # cont = self.FContainer(self)

    # Size of the parameters
    nx = qnn.num_features
    nparam = qnn.num_parameters
    nparam_op = qnn.num_parameters_operator

    global iter_counter
    iter_counter = 0

    def opt_func(theta):
        """Optimization function for minimize"""
        global iter_counter

        # Splitting theta in the arrays
        ioff = 0
        if opt_tuple[0]:
            x = theta[:nx]
            ioff += nx
        else:
            x = x_ini
        if opt_tuple[1]:
            param = theta[ioff : (ioff + nparam)]
            ioff += nparam
        else:
            param = param_ini
        if opt_tuple[2]:
            param_op = theta[ioff : (ioff + nparam_op)]
        else:
            param_op = param_op_ini

        # adjust shots
        if shot_adjusting is not None:
            shot_adjusting.func(x, param, param_op)

        # Evaluate the necessary derivatives and call the loss function
        return_value = loss_function(qnn.evaluate(input_loss, x, param, param_op), iter_counter)

        if shot_adjusting is not None:
            qnn.reset_shots()

        return return_value

    def opt_func_grad(theta):
        """Optimization function gradient for minimize"""
        global iter_counter
        iter_counter = iter_counter + 1

        # Splitting theta in the arrays
        ioff = 0
        if opt_tuple[0]:
            x = theta[:nx]
            ioff += nx
        else:
            x = x_ini
        if opt_tuple[1]:
            param = theta[ioff : (ioff + nparam)]
            ioff += nparam
        else:
            param = param_ini
        if opt_tuple[2]:
            param_op = theta[ioff : (ioff + nparam_op)]
        else:
            param_op = param_op_ini

        # adjust shots
        if shot_adjusting is not None:
            shot_adjusting.grad(x, param, param_op)

        # Evaluate the necessary derivatives and call the loss function gradient
        return_value = np.concatenate(
            loss_function_gradient(qnn.evaluate(input_grad, x, param, param_op), iter_counter),
            axis=None,
        )

        if shot_adjusting is not None:
            qnn.reset_shots()

        return return_value

    # Merge initialization values for minimize
    val_ini = np.array([])
    if opt_tuple[0]:
        val_ini = np.concatenate((val_ini, x_ini), axis=None)
    if opt_tuple[1]:
        val_ini = np.concatenate((val_ini, param_ini), axis=None)
    if opt_tuple[2]:
        val_ini = np.concatenate((val_ini, param_op_ini), axis=None)

    # Call optimization function
    result = minimize(opt_func, val_ini, opt_func_grad, bounds=bounds)

    if hasattr(result, "x"):
        result = result.x

    # Split up final result into its x, param, param_op pieces
    return_list = []
    ioff = 0
    if opt_tuple[0]:
        return_list.append(result[:nx])
        ioff += nx
    if opt_tuple[1]:
        return_list.append(result[ioff : (ioff + nparam)])
        ioff += nparam
    if opt_tuple[2]:
        return_list.append(result[ioff : (ioff + nparam_op)])
    if len(return_list) == 1:
        return return_list[0]
    else:
        return tuple(return_list)


def solve_mini_batch(
    qnn: QNN,
    input_values,
    ground_truth,
    param_ini,
    param_op_ini,
    loss: LossBase,
    optimizer: OptimizerBase,
    weights=None,
    opt_param_op: bool = True,
    epochs: int = 10,
    batch_size: int = None,
    shuffle=False,
):
    """Minimize a loss function using mini-batch gradient descent.

    Args:
        qnn : QNN instance
        input_values : Array of input values
        ground_truth : Array of ground truth values for the inputs
        param_ini : Initialization values of the parameters of the PQC
        param_op_ini : Initialization values of the cost operator
        loss: Loss instance
        optimizer : Optimizer instance, needs to be of SGDMixin type
        weigths : Weighting of the reference values. Has to be the same size as input and
                  ground_truth (default : None)
        opt_param_op : If True, cost operator parameters are optimized as well (default: True)
        epochs : Number of epochs of SGD to perform
        batch_size : Number of data points in each batch
        shuffle : If True, data points get shuffled before each epoch (default: False)

    Returns:
        optimized parameters of the PQC, and, if opt_param_op=True,
        the optimized parameters of the cost operator
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

    for epoch in range(epochs):
        accumulated_loss = 0.0
        if shuffle:
            idcs = np.random.permutation(idcs)
        for batch_slice in gen_batches(n_samples, batch_size):
            loss_values = qnn.evaluate(
                loss.loss_args_tuple, input_values[idcs[batch_slice]], param, param_op
            )

            accumulated_loss += loss.value(
                loss_values,
                ground_truth=ground_truth[idcs[batch_slice]],
                weights=weights_values[idcs[batch_slice]],
                iteration=epoch,
            )

            diff_values = qnn.evaluate(
                loss.gradient_args_tuple, input_values[idcs[batch_slice]], param, param_op
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
        print(
            f"Epoch {epoch+1: >5}/{epochs: >5}:\taverage_loss = {accumulated_loss/n_samples:.5f}"
        )

    if opt_param_op:
        return param, param_op

    return param


def regression(
    qnn,
    x_space,
    ref_values,
    param_ini,
    param_op_ini,
    loss: LossBase,
    minimize: OptimizerBase,
    weights=None,
    opt_param_op=True,
    shot_adjusting: shot_adjusting_options = None,
):
    """Function for a simple regression of the given reference values

    Args:
        x_space : array of the supporting vectors
        ref_values : array if the reference values at the supporting vectors
        param_ini : Initialization values of the parameters of the PQC
        param_op_ini : Initialization values of the cost operator
        loss: Loss instance
        weigths : Weighting of the reference values. Has to be the same size as x_space and ref_values (default : None)
        opt_param_op : If true, cost operator parameters are optimized as well (default: true)

    Returns:
        optimized parameters of the PQC, and, if opt_param_op=True,
        the optimized parameters of the cost operator
    """
    if isinstance(weights, np.ndarray):
        weights_values = weights
    elif weights is None:
        weights_values = np.ones(ref_values.shape)
    else:
        raise TypeError("Unknown weight format")

    # Tell the loss function if the cost operator parameters are optimized
    loss.set_opt_param_op(opt_param_op)

    if weights_values.shape != ref_values.shape:
        raise ValueError(
            f"Shape {weights_values.shape} of weight values doesn't match shape"
            f" {ref_values.shape} of reference values"
        )

    # Loss function of the regression problem
    def loss_function(value_dict, iteration):
        return loss.value(
            value_dict,
            ground_truth=ref_values,
            weights=weights_values,
            iteration=iteration,
        )

    # Gradient of the loss function
    def loss_function_gradient(value_dict, iteration):
        return loss.gradient(
            value_dict,
            ground_truth=ref_values,
            weights=weights_values,
            iteration=iteration,
            multiple_output=qnn.multiple_output,
        )

    # Call optimization function
    return solve_all(
        qnn,
        x_space,
        param_ini,
        param_op_ini,
        loss_function,
        loss.loss_args_tuple,
        loss_function_gradient,
        loss.gradient_args_tuple,
        minimize,
        (False, True, opt_param_op),
        shot_adjusting=shot_adjusting,
    )
