import numpy as np
import pytest
from squlearn import Executor
from squlearn.encoding_circuit.circuit_library.chebyshev_pqc import ChebyshevPQC
from squlearn.encoding_circuit.circuit_library.yz_cx_encoding_circuit import YZ_CX_EncodingCircuit
from squlearn.kernel.matrix import FidelityKernel
from squlearn.kernel.matrix.kernel_optimizer import KernelOptimizer
from squlearn.kernel.ml import QGPR
from squlearn.kernel.ml.qgpc import QGPC
from squlearn.kernel.ml.qkrr import QKRR
from squlearn.kernel.ml.qsvc import QSVC
from squlearn.kernel.ml.qsvr import QSVR
from squlearn.kernel.optimization.negative_log_likelihood import NLL
from squlearn.optimizers.adam import Adam


def get_regression_data():

    def func(x):
        return np.sin(3 * x) * x * x

    noise_std = 0.1
    X_train = np.linspace(-1, 1, 15)
    Y_train = func(X_train) + np.random.normal(scale=noise_std, size=len(X_train))
    return X_train, Y_train


def get_classification_data(num_samples=100, num_features=2, noise_std=0.1):
    X = np.random.rand(num_samples, num_features)  # Random feature values between 0 and 1
    coefficients = np.random.randn(num_features)  # Random coefficients for linear separation
    noise = np.random.normal(scale=noise_std, size=num_samples)  # Gaussian noise
    y = (X.dot(coefficients) + noise > 0).astype(
        int
    )  # Labels based on linear separation with noise

    return X, y


@pytest.fixture(scope="module")
def setup_kernel_optimizer_for_regressor():
    enc_circ = ChebyshevPQC(num_qubits=4, num_features=1, num_layers=2)
    q_kernel = FidelityKernel(
        encoding_circuit=enc_circ, executor=Executor("pennylane"), parameter_seed=0, regularization="tikhonov"
    )
    nll_loss = NLL(quantum_kernel=q_kernel)
    optimzer = Adam(options={"maxiter": 20, "lr": 0.1})
    kernel_optimizer = KernelOptimizer(quantum_kernel=q_kernel, loss=nll_loss, optimizer=optimzer)
    regression_data = get_regression_data()

    return kernel_optimizer, regression_data


@pytest.fixture(scope="module")
def setup_kernel_optimizer_for_classifier():
    enc_circ = YZ_CX_EncodingCircuit(2, num_features=2, num_layers=2, c=1.0)
    q_kernel = FidelityKernel(
        encoding_circuit=enc_circ, executor=Executor("pennylane"), parameter_seed=0, regularization="tikhonov"
    )
    nll_loss = NLL(quantum_kernel=q_kernel)
    optimzer = Adam(options={"maxiter": 20, "lr": 0.1})
    kernel_optimizer = KernelOptimizer(quantum_kernel=q_kernel, loss=nll_loss, optimizer=optimzer)
    classification_data = get_classification_data()
    return kernel_optimizer, classification_data


@pytest.mark.parametrize("high_level_class", [QGPR, QKRR, QSVR])
def test_kernel_optimizer_for_regressor_classes(
    setup_kernel_optimizer_for_regressor,
    high_level_class,
):
    kernel_optimizer, regression_data = setup_kernel_optimizer_for_regressor

    model = high_level_class(quantum_kernel=kernel_optimizer)
    model.fit(regression_data[0].reshape(-1, 1), regression_data[1])
    assert kernel_optimizer.get_optimal_parameters() is not None
    assert kernel_optimizer._optimizer is not None


# TODO: Fix the issue causing the "LinAlgError: 17-th leading minor of the array is not positive definite" error.
# This error occurs during the Cholesky decomposition in the NLL loss function's compute method.
# The matrix is not positive definite, possibly due to numerical instability or insufficient regularization.
# Investigate the root cause and implement a solution to ensure the matrix is always positive definite.

@pytest.mark.parametrize("high_level_class", [QGPC, QSVC])
def test_kernel_opimizer_for_classification_classes(
    setup_kernel_optimizer_for_classifier,
    high_level_class,
):
    kernel_optimizer, classification_data = setup_kernel_optimizer_for_classifier

    model = high_level_class(quantum_kernel=kernel_optimizer)
    model.fit(classification_data[0], classification_data[1])
    assert kernel_optimizer.get_optimal_parameters() is not None
    assert kernel_optimizer._optimizer is not None
