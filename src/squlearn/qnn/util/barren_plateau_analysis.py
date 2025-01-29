import numpy as np
from sklearn.linear_model import LinearRegression

from ..lowlevel_qnn.lowlevel_qnn_base import LowLevelQNNBase

from ...util.data_preprocessing import adjust_parameters

# TODO: This file needs a lot of revision. Not properly working with the current codebase.


def calc_var_dg(
    qnn: LowLevelQNNBase,
    x,
    param_op,
    n_sample=100,
    p_lim=None,
    p_val=None,
    p_index=None,
    verbose=1,
    seed=0,
):
    """
    Calculates the variance and the mean of the gradient of the given qnn.

    Args:
        qnn (LowLevelQNNBase): QNN object from which the variance of the gradient is calculated
        x : Single value or array of the x values of the QNN
        param_op : Values of the cost-operator
        n_sample = 100 : Number of samples considered for the variance computation of the gradient
        p_lim = None : Limits of the unified sampling of parameters for the variance calculation
        p_val = None : Array containing the parameters for sampling.
                       If equal None, the parameters are chosen randomly (default).
        p_index = None : Array of indices of the parameters chosen for the derivative.
                         If equal None (default), the parameters are chosen randomly
        verbose = 1 : Verbosity of the method.
        seed = 0: Random seed the random operations.

    Returns:
        Variance of the gradient values
        Mean value of the absolute values of the gradient
    """

    seed_backup = np.random.get_state()
    np.random.seed(seed)

    # Set-up default limits
    if p_lim is None:
        p_lim = (-np.pi, np.pi)

    # Process p-values
    if p_val is not None:
        p_val, multi = adjust_parameters(p_val, qnn.num_parameters)
        if not isinstance(p_val, np.ndarray):
            p = np.array(p_val)
        else:
            p = p_val
        if p.shape[1] != qnn.num_parameters:
            raise ValueError("Wrong size of the input p_val")
    else:
        p = np.random.uniform(low=p_lim[0], high=p_lim[1], size=(n_sample, qnn.num_parameters))

    if p_index is None:
        p_index = np.random.randint(0, qnn.num_parameters, size=n_sample)

    np.random.set_state(seed_backup)

    # Radom sampling of the gradient derivatives
    grad_val = []
    for i in range(p.shape[0]):
        grad_val.append(qnn.evaluate_diff_tuple(x, p[i], param_op, qnn.parameters[p_index[i]]))

    # Returns variance and absolute mean value of all sampled gradient entries
    np_array = np.array(grad_val).flatten()
    return np.var(np_array), np.mean(np.abs(np_array))


def get_barren_slope(pqc_func, cost_op_func, x, QI, num_qubits, layer_fac=5, n_sample=100):
    """
    Calculates the variance of the gradient for different number of qubits.
    Returns a linear regression model to the data

    Args:
        pqc_func : function which returns the initialized pqc class for a given number of qubits and layers
                   pqc_func(number_of_qubits, number_of_layers)
        cost_op_func : function which returns the initialized cost_op class for a given number of qubits
                       cost_op_func(number_of_qubits)
        x : single value or array of the x values
        QI : Quantum Instance for evaluating the quantum circuits
        num_qubits : Either a list containing the considered qubits or
                     an integer value that triggers a list of [2,...,num_qubits] qubits
        layer_fac = 5 : (optional) Number of layer is  given by number of qubits times layer_fac
        n_sample = 100 : (optional) Number of samples considered for the variance computation of the gradient

    Returns:
        Model containing the linear regression (based on :class:`sklearn.linear_model.LinearRegression`)
        Numpy array with the variance values
    """

    if isinstance(num_qubits, int):
        num_qubits_ = np.arange(2, num_qubits + 1)
    else:
        num_qubits_ = num_qubits

    var = []
    mean = []

    for iqubits in num_qubits_:
        number_of_layers = layer_fac * iqubits
        pqc_ = pqc_func(iqubits, number_of_layers)
        cost_op_ = cost_op_func(iqubits)
        qnn = QNN(pqc_, cost_op_, QI)
        mean_val, var_val = calc_var_dg(
            qnn,
            x=x,
            param_op=np.ones(cost_op_.get_number_of_parameters()),
            n_sample=n_sample,
        )
        var.append(var_val)
        mean.append(mean_val)

    model = LinearRegression()
    model.fit(np.array(num_qubits_).reshape((-1, 1)), np.log(np.array(var)))
    model.x_val = np.array(num_qubits_).reshape((-1, 1))
    model.y_val = np.log(np.array(var))
    return model, np.array(var)


def get_barren_slop_from_model(model):
    """
    Returns the slop of the linear regression of the barren plateau fit

    Args:
        model : linear regression model created by get_barren_slope

    Returns:
        slop of the linear regression
    """
    return model.coef_[0]


def get_barren_plot_from_model(model, plt):
    """
    Plots the regression output and the measured numbers into a semilogarithmic plot-

    Args:
        model : linear regression model created by get_barren_slope
        plt : matplotplib.pyplot object

    Returns:
        matplotlib handles to the two plots
    """
    y = np.exp(model.predict(model.x_val))
    handle = []
    handle.append(plt.semilogy(model.x_val, y))
    handle.append(plt.semilogy(model.x_val, np.exp(model.y_val)))
    return handle


def get_barren_layer(pqc_func, cost_op_func, x, QI, num_qubits, num_layers, n_sample=100):
    """
    Calculate the variance of the gradient for different number of qubits and layers.
    Can be used to create a plot for visualizing the plateauing for specific number of layers.

    Args:
        pqc_func : function which returns the initialized pqc class for a given number of qubits and layers
                   pqc_func(number_of_qubits, number_of_layers)
        cost_op_func : function which returns the initialized cost_op class for a given number of qubits
                       cost_op_func(number_of_qubits)
        x : single value or array of the x values
        QI : Quantum Instance for evaluating the quantum circuits
        num_qubits : Either a list containing the considered qubits or
                     an integer value that triggers a list of [2,...,num_qubits] qubits
        num_layers : Either a list containing the considered qubits or
                     an integer value that triggers a list of [2,...,num_qubits] qubits
        n_sample = 100 : (optional) Number of samples considered for the variance computation of the gradient

    Returns:
        Returns two dictionaries with the variance and the mean absolute value for the given layers
    """

    if isinstance(num_qubits, int):
        num_qubits_ = np.arange(2, num_qubits + 1)
    else:
        num_qubits_ = num_qubits

    if isinstance(num_layers, int):
        num_layers_ = np.arange(1, num_layers + 1, 5)
    else:
        num_layers_ = num_layers

    var = {}
    mean = {}

    for iqubits in num_qubits_:
        qubit_var = []
        qubit_mean = []
        for ilayers in num_layers_:
            number_of_layers = ilayers
            pqc_ = pqc_func(iqubits, number_of_layers)
            cost_op_ = cost_op_func(iqubits)
            qnn = QNN(pqc_, cost_op_, QI)
            mean_val, var_val = calc_var_dg(
                qnn,
                x=x,
                param_op=np.ones(cost_op_.get_number_of_parameters()),
                n_sample=n_sample,
            )
            qubit_var.append(var_val)
            qubit_mean.append(mean_val)
        var[iqubits] = qubit_var
        mean[iqubits] = qubit_mean

    return var, mean


def get_barren_layer_plot(var, num_layers, plt):
    """
    Creates a barren plateau visualization with the
    number of layers on the X-axis. Returns a semilogarithmic
    plot of the variance of the gradient (see get_barren_layer).

    Args:
        var : dictionary containing the variance connecting qubits
            and the variance for the layers
        num_layers : Either a list containing the considered
            number of layers or an integer value that triggers a list of
            [1,5,num_layers] qubits
        plt : matplotplib.pyplot object

    Returns:
        Handles of the different curves
    """

    if isinstance(num_layers, int):
        num_layers_ = np.arange(1, num_layers + 1, 5)
    else:
        num_layers_ = num_layers

    handles = {}
    for i in var.keys():
        handles[i] = plt.semilogy(num_layers_, var[i])
    return handles
