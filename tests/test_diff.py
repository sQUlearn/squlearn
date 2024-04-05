from squlearn import Executor
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel
from squlearn.encoding_circuit import ChebyshevTower, ChebyshevPQC, HubregtsenEncodingCircuit


import pennylane as qml
import pennylane.numpy as pnp

import numpy as np

from functools import partial

dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_qasm_simulator', ibmqx_token="1e1fcef3940bc7181262dbc135a052e83b17a669f187b1d6d7bde96ef3167078b55eb494f44f3b3de8ea934023da7a5eef3df4c6d27bf8ce7d48ca174c654cb7")

@qml.qnode(dev, diff_method="best")
def circuit(param,inp):

    # repeatedly apply each layer in the circuit
    qml.RX(param[0], wires=0)
    qml.RX(inp[0], wires=1)
    qml.CNOT(wires=[0, 1])

    qml.RY(param[1], wires=0)
    qml.RY(inp[1], wires=1)
    qml.CNOT(wires=[0, 1])

    qml.RX(param[2], wires=0)
    qml.RX(inp[2], wires=1)
    qml.CNOT(wires=[0, 1])

    # returns the expectation of the input matrix A on the first qubit
    return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))


deriv = qml.jacobian(circuit, argnum=[0])

p = [[0.1,0.2,0.8],[0.2,0.3,0.1]]
x = [[0.1,0.2,0.8],[0.2,0.3,0.1]]

pp = pnp.array(p[0],requires_grad=True)
xx = pnp.array(x[0],requires_grad=False)

print(deriv(pp,xx))