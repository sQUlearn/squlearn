import numpy as np
from qiskit.circuit import ParameterVector
from qiskit_algorithms.gradients import LinCombQGT, QFI

from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from .executor import Executor
from .data_preprocessing import adjust_features, adjust_parameters


def get_quantum_fisher(
    encoding_circuit: EncodingCircuitBase,
    x: np.ndarray,
    p: np.ndarray,
    executor: Executor,
    mode: str = "p",
):
    """
    Function for evaluating the Quantum Fisher Information Matrix of a encoding circuit.

    The Quantum Fisher Information Matrix (QFIM) is evaluated the supplied numerical
    features and parameter value.

    Mode enables the user to choose between different modes of evaluation:
    * ``"p"`` : QFIM for parameters only
    * ``"x"`` : QFIM for features only
    * ``"px"`` : QFIM for parameters and features (order parameters first)
    * ``"xp"`` : QFIM for features and parameters (order features first)

    In case of multiple inputs for ``x`` and ``p``, the QFIM is evaluated for each input separately and
    returned as a numpy matrix.

    Args:
        encoding_circuit (EncodingCircuitBase): Encoding circuit for which the QFIM is evaluated
        x (np.ndarray): Input data values for replacing the features in the encoding circuit
        p (np.ndarray): Parameter values for replacing the parameters in the encoding circuit
        executor (Executor): Executor for evaluating the QFIM (utilizes estimator)
        mode (str): Mode for evaluating the QFIM, possibilities: ``"p"``, ``"x"``,
                    ``"px"``, ``"xp"`` (default: ``"p"``)

    Return:
        Numpy matrix with the QFIM, in case of multiple inputs, the array is nested.
    """
    # Get Qiskit QFI primitive
    qfi = QFI(LinCombQGT(executor.get_estimator()))

    p_ = ParameterVector("p", encoding_circuit.num_parameters)
    x_ = ParameterVector("x", encoding_circuit.num_features)
    circuit = encoding_circuit.get_circuit(x_, p_)

    # Adjust input
    x_list, multi_x = adjust_features(x, encoding_circuit.num_features)
    p_list, multi_p = adjust_parameters(p, encoding_circuit.num_parameters)

    circ_list = []
    param_values_list = []
    param_list = []
    if mode == "p":
        for xval in x_list:
            circ_temp = circuit.assign_parameters(dict(zip(x_, xval)))
            for pval in p_list:
                circ_list.append(circ_temp)
                param_values_list.append(pval)
                param_list.append(p_)
    elif mode == "x":
        for xval in x_list:
            for pval in p_list:
                circ_list.append(circuit.assign_parameters(dict(zip(p_, pval))))
                param_values_list.append(xval)
                param_list.append(x_)
    elif mode == "xp":
        for xval in x_list:
            for pval in p_list:
                circ_list.append(circuit)
                param_values_list.append(np.concatenate((xval, pval)))
                param_list.append(list(x_) + list(p_))
    elif mode == "px":
        for xval in x_list:
            for pval in p_list:
                circ_list.append(circuit)
                param_values_list.append(np.concatenate((pval, xval)))
                param_list.append(list(p_) + list(x_))

    # Evaluate QFIM with Qiskit Primitive
    qfis = np.array(qfi.run(circ_list, param_values_list, param_list).result().qfis)

    # Reformating in case of multiple inputs
    reshape_list = []
    if multi_x:
        reshape_list.append(len(x_list))
    if multi_p:
        reshape_list.append(len(p_list))

    if len(reshape_list) > 0:
        qfis = qfis.reshape(reshape_list + list(qfis[0].shape))
    else:
        qfis = qfis[0]

    executor.clear_estimator_cache()
    return qfis


# WORK IN PROGRESS:


# def regression_fisher(qnn : qfit, x, y, param, param_op):
#     # empirical log-likelihood Fischer log p(y|f) = log exp(-0.5(y-f)^2)
#     # see for example doi:10.5555/3454287.3454661
#     # F(p) = sum_n (f(x_n)-y_n)^2 grad f(x_n) grad f(x_n)'

#     x_,multi_x = _adjust_input_(x,qnn.get_num_x())
#     param_,multi_param = _adjust_input_(param,qnn.num_parameters)
#     param_op_,multi_param_op = _adjust_input_(param_op,qnn.get_num_param_op())

#     print("x_",x_)
#     print("param_",param_)
#     print("param_op_",param_op_)

#     f = qnn.eval_f(x_,param_,param_op_)
#     dfdp = qnn.eval_dfdp(x_,param_,param_op_)
#     dfdop = qnn.eval_dfdop(x_,param_,param_op_)

#     print("f",f)
#     print("dfdp",dfdp)
#     print("dfdop",dfdop)

#     if multi_x:
#         prefac = []
#         for i in range(len(y)):
#             prefac.append(np.square((f[i]-y[i])))
#     else:
#         prefac = []
#         prefac.append(np.square(f[0]-y))
#     prefac = np.array(prefac)

#     print("prefac",prefac)
#     ax = 2
#     if multi_param_op:
#         ax = 3
#     dp_dop = np.concatenate((dfdp,dfdop),axis=ax)

#     print("dp_dop",dp_dop)

#     dp_dop_multi = dp_dop.copy()

#     for i in range(len(x_)):
#         for j in range(len(param_)):
#             for k in range(len(param_op_)):
#                 dp_dop_multi[i,j,k] = np.multiply(dp_dop_multi[i,j,k],prefac[i,j,k])

#     print("dp_dop_multi",dp_dop_multi)

#     fischers = np.einsum('ijm,ijn->ijmn',dp_dop,dp_dop_multi)

#     print("fischers",fischers)

#     return np.sum(fischers,axis=0)


# def calculate_fhat(qnn : qfit, x,
#                             param_min,param_max,num_param,
#                             param_op_min,param_op_max,num_param_op,
#                             param_op=True, seed = 0):

#     # averaging over x!!!

#     # is all wrong

#     # Create Random variables
#     state = np.random.get_state()
#     np.random.seed(seed)
#     param_rand = np.random.uniform(param_min, param_max, size=(num_param, qnn.num_parameters))
#     param_op_rand = np.random.uniform(param_op_min, param_op_max, size=(num_param_op, qnn.get_num_param_op()))
#     np.random.set_state(state)

#     volume_param = (param_max - param_min) * qnn.num_parameters + (param_op_max - param_op_min)*qnn.get_num_param_op()

#     fischers = qnn.eval_fischer(x,param_rand,param_op_rand)
#     fischer_trace = np.trace(np.average(fischers,axis=(0,1,2)))
#     fischers_avrg = np.average(np.reshape(fischers, (len(x), num_param*num_param_op,
#     qnn.num_parameters+qnn.get_num_param_op(),
#     qnn.num_parameters+qnn.get_num_param_op())),axis=0)

#     f_hat = fischers_avrg / fischer_trace * volume_param
#     return f_hat,fischer_trace,volume_param

# def calculate_effective_dimension(f_hat,n,gamma,volume_param):
#     #n?

#     if gamma <= 0.0 or gamma > 1.0:
#         raise ValueError("Gamma out of range (0,1]: gamma = ",gamma)

#     for f in f_hat:
#         w,v = np.linalg.eig(f)
#         print("eigenvalues",w)


#     factor = gamma*float(n)/(2.0*np.pi*np.log(float(n)))
#     print("factor",factor)
#     print("det(f_hat)",np.linalg.det(f_hat))
#     print("det(f_hat*factor)",np.linalg.det(f_hat*factor))
#     # rewriting of the original formular, since this is more robust
#     # following the code of Abbas et al:
#     # log(1/V sum sqrt(det(A)) ) = log(sum exp(log(det(A))/2)-log(V)
#     # special more robust function for log(det(A)) -> slogdet
#     IplusF = np.eye(f_hat.shape[1]) + f_hat*factor
#     print("det(IplusF)",np.linalg.det(IplusF))
#     print("np.linalg.slogdet(IplusF)[0]",np.linalg.slogdet(IplusF)[0])
#     print("np.linalg.slogdet(IplusF)[1]",np.linalg.slogdet(IplusF)[1])
#     logdet = logsumexp(np.linalg.slogdet(IplusF)[1]/2)
#     print("logdet",logdet)
#     print("np.log(volume_param)",np.log(volume_param))
#     print("np.log(factor)",np.log(factor))
#     value = 2 * (logdet - np.log(volume_param))/np.log(factor)
#     print("eff dim",value)
#     return value
