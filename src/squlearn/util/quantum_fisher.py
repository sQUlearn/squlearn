import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.utils import QuantumInstance
from qiskit.opflow.gradients import QFI
from qiskit.opflow import CircuitStateFn
from qiskit.opflow import CircuitSampler

from ..feature_map import FeatureMapBase
from ..util.data_preprocessing import adjust_input, assign_all_parameters


def get_quantum_fisher(pqc: FeatureMapBase, x, param, QI: QuantumInstance):
    """
    Function for evaluating the Quantum Fisher Information Matrix for an
    inputted parameterized quantum circuit.

    Args:
        pqc : Parameterized quantum circuit that will be pruned. Has to be in the pqc format of quantum_fit!
        x : Input data values for replacing the features in the pqc (can also be an array for multiple Fishers)
        param : Parameter values for replacing the parameters in the pqc (can also be an array for multiple Fishers)
        QI : Quantum Instance for evaluating the Quantum Fisher Information Matrix

    Returns: Numpy matrix with the quantum Fisher matrix
             (or matrices for multi dimensional x and param)

    """

    x_ = ParameterVector("x", pqc.num_features)
    p_ = ParameterVector("p", pqc.num_parameters)
    opflow_circ = CircuitStateFn(primitive=pqc.get_circuit(x_, p_), coeff=1.0)
    qfi = QFI(qfi_method="lin_comb_full").convert(operator=opflow_circ, params=p_)
    qfi_with_param = assign_all_parameters(qfi, x=x_, param=p_, x_values=x, param_values=param)
    sampler = CircuitSampler(QI)
    eval_conv = sampler.convert(qfi_with_param)
    return np.real(eval_conv.eval())


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
