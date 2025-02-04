import cProfile
import pstats
import io

import numpy as np
import matplotlib.pyplot as plt

from squlearn import Executor
from squlearn.encoding_circuit import ChebyshevRx
from squlearn.observables import IsingHamiltonian
from squlearn.qnn import QNNRegressor, SquaredLoss
from squlearn.optimizers import SLSQP

executor = Executor("pennylane", shots=None)

nqubits = 12
number_of_layers = 2

pqc = ChebyshevRx(nqubits, 1, num_layers=number_of_layers)

ising_op = IsingHamiltonian(nqubits, I="S", Z="S", ZZ="S")

np.random.seed(13)
param_ini = np.random.rand(pqc.num_parameters)
param_op_ini = np.random.rand(ising_op.num_parameters)

qnn = QNNRegressor(pqc, ising_op, executor, SquaredLoss(), SLSQP(), param_ini, param_op_ini)


x_space = np.arange(0.1, 0.9, 0.1).reshape(-1, 1)
ref_values = np.log(x_space).ravel()

#pr = cProfile.Profile()
#pr.enable()

qnn.fit(x_space, ref_values)

#pr.disable()
#s = io.StringIO()
#ps = pstats.Stats(pr, stream=s).sort_stats('time')
#ps.print_stats()
#print(s.getvalue())  # Print profiling results