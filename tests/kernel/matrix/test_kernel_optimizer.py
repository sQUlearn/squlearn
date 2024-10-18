import numpy as np
import pytest

from sklearn.datasets import make_regression, make_classification

from squlearn import Executor
from squlearn.encoding_circuit.circuit_library.yz_cx_encoding_circuit import YZ_CX_EncodingCircuit
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel, KernelOptimizer
from squlearn.kernel import KernelOptimizer
from squlearn.kernel import QGPR, QGPC, QKRR, QSVC, QSVR
from squlearn.kernel.loss import NLL, TargetAlignment

from squlearn.optimizers.adam import Adam


@pytest.fixture(scope="module")
def setup_kernel_optimizer_for_regressor():
    """Setup kernel optimizer for regression tasks."""
    enc_circ = YZ_CX_EncodingCircuit(num_qubits=4, num_features=4, num_layers=2)
    q_kernel = FidelityKernel(
        encoding_circuit=enc_circ,
        executor=Executor("pennylane"),
        parameter_seed=0,
        regularization="tikhonov",
    )
    nll_loss = NLL()
    optimzer = Adam(options={"maxiter": 3, "lr": 0.1})
    kernel_optimizer = KernelOptimizer(quantum_kernel=q_kernel, loss=nll_loss, optimizer=optimzer)
    regression_data = make_regression(n_samples=10, n_features=4, random_state=0)
    return kernel_optimizer, regression_data


@pytest.fixture(scope="module")
def setup_kernel_optimizer_for_classifier():
    """Setup kernel optimizer for classification tasks."""
    enc_circ = YZ_CX_EncodingCircuit(4, num_features=4, num_layers=2, c=1.0)
    q_kernel = ProjectedQuantumKernel(
        encoding_circuit=enc_circ,
        executor=Executor("pennylane"),
        parameter_seed=0,
        regularization="tikhonov",
    )
    nll_loss = TargetAlignment()
    optimzer = Adam(options={"maxiter": 3, "lr": 0.1})
    kernel_optimizer = KernelOptimizer(quantum_kernel=q_kernel, loss=nll_loss, optimizer=optimzer)
    classification_data = make_classification(n_samples=10, n_features=4, random_state=0)
    return kernel_optimizer, classification_data


@pytest.mark.parametrize("high_level_class", [QGPR, QKRR, QSVR])
def test_kernel_optimizer_for_regressor_classes(
    setup_kernel_optimizer_for_regressor,
    high_level_class,
):
    """Test kernel optimizer for high-level regression methods.

    Args:
        setup_kernel_optimizer_for_regressor: A fixture that returns a kernel optimizer and
                                                regression data.
        high_level_class: A high-level regression class
    """
    kernel_optimizer, regression_data = setup_kernel_optimizer_for_regressor
    model = high_level_class(quantum_kernel=kernel_optimizer)
    model.fit(regression_data[0], regression_data[1])
    assert kernel_optimizer.get_optimal_parameters() is not None
    assert kernel_optimizer._optimizer is not None
    assert kernel_optimizer.is_fitted


@pytest.mark.parametrize("high_level_class", [QGPC, QSVC])
def test_kernel_opimizer_for_classification_classes(
    setup_kernel_optimizer_for_classifier,
    high_level_class,
):
    """Test kernel optimizer for high-level classification methods.

    Args:
        setup_kernel_optimizer_for_classifier: A fixture that returns a kernel optimizer and
                                                classification data.
        high_level_class: A high-level classification class
    """
    kernel_optimizer, classification_data = setup_kernel_optimizer_for_classifier
    model = high_level_class(quantum_kernel=kernel_optimizer)
    model.fit(classification_data[0], classification_data[1])
    assert kernel_optimizer.get_optimal_parameters() is not None
    assert kernel_optimizer._optimizer is not None
    assert kernel_optimizer.is_fitted
