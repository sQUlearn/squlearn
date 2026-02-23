import pytest
from squlearn import Executor
from squlearn.encoding_circuit.circuit_library.yz_cx_encoding_circuit import YZ_CX_EncodingCircuit
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel, KernelOptimizer
from squlearn.kernel import QGPR, QGPC, QKRR, QSVC, QSVR
from squlearn.kernel.loss import NLL, TargetAlignment

from squlearn.optimizers.adam import Adam


def regression_data():
    x = [[0.1], [0.2], [0.3], [0.4], [0.5]]
    y = [0.1, 0.2, 0.3, 0.4, 0.5]
    return x, y


def classification_data():
    x = [[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]]
    y = [0, 0, 1, 1]
    return x, y


@pytest.fixture(scope="module")
def setup_kernel_optimizer_for_regressor():
    """Setup kernel optimizer for regression tasks."""
    enc_circ = YZ_CX_EncodingCircuit(num_qubits=4, num_layers=2)
    q_kernel = FidelityKernel(
        encoding_circuit=enc_circ,
        executor=Executor("pennylane"),
        parameter_seed=0,
        regularization="tikhonov",
    )
    nll_loss = NLL()
    optimzer = Adam(options={"maxiter": 3, "lr": 0.1})
    kernel_optimizer = KernelOptimizer(quantum_kernel=q_kernel, loss=nll_loss, optimizer=optimzer)
    return kernel_optimizer


@pytest.fixture(scope="module")
def setup_kernel_optimizer_for_classifier():
    """Setup kernel optimizer for classification tasks."""
    enc_circ = YZ_CX_EncodingCircuit(4, num_layers=2, c=1.0)
    q_kernel = ProjectedQuantumKernel(
        encoding_circuit=enc_circ,
        executor=Executor("pennylane"),
        parameter_seed=0,
        regularization="tikhonov",
    )
    nll_loss = TargetAlignment()
    optimzer = Adam(options={"maxiter": 3, "lr": 0.1})
    kernel_optimizer = KernelOptimizer(quantum_kernel=q_kernel, loss=nll_loss, optimizer=optimzer)
    return kernel_optimizer


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
    kernel_optimizer = setup_kernel_optimizer_for_regressor
    model = high_level_class(quantum_kernel=kernel_optimizer)
    data = regression_data()
    model.fit(data[0], data[1])
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
    kernel_optimizer = setup_kernel_optimizer_for_classifier
    model = high_level_class(quantum_kernel=kernel_optimizer)
    data = classification_data()
    model.fit(data[0], data[1])
    assert kernel_optimizer.get_optimal_parameters() is not None
    assert kernel_optimizer._optimizer is not None
    assert kernel_optimizer.is_fitted
