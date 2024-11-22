import pytest
import numpy as np

from squlearn import Executor
from squlearn.encoding_circuit.circuit_library.chebyshev_pqc import ChebyshevPQC
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel


@pytest.mark.parametrize("kernel_class", ["FidelityKernel", "ProjectedQuantumKernel"])
@pytest.mark.parametrize(
    "input_features, expected_features, warn_expected",
    [
        (np.array([[1, 2], [3, 4]]), 2, False),  # Matching features, no warning
        (np.array([[1, 2, 3], [4, 5, 6]]), 3, True),  # Mismatched features, warning
        (np.array([1, 2, 3]), 1, True),  # Single feature array, mismatched, warning
        (np.array([[1]]), 1, False),  # Single feature array, matching, no warning
    ],
)
def test_check_feature_consistency(kernel_class, input_features, expected_features, warn_expected):

    encoding_circuit = ChebyshevPQC(num_qubits=2, num_features=2, num_layers=1)
    executor = Executor("pennylane")

    # Initialize kernel
    if kernel_class == "FidelityKernel":
        kernel = FidelityKernel(encoding_circuit, executor)
    elif kernel_class == "ProjectedQuantumKernel":
        kernel = ProjectedQuantumKernel(encoding_circuit, executor)

    # Validate features and check for warnings
    if warn_expected:
        with pytest.warns(UserWarning, match="Number of features in the input data"):
            kernel._check_feature_consistency(input_features)
    else:
        kernel._check_feature_consistency(input_features)

    # Assert num_features was updated correctly
    assert kernel.num_features == expected_features
