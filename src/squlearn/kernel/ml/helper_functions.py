import numpy as np


# TODO change to preprocessing
# helper function to stack input vector if num_features and num_qubits are not equal
def stack_input(x_vec, num_features):
    x_stack = np.array([x_vec.ravel() for n in range(num_features)]).T
    return x_stack
