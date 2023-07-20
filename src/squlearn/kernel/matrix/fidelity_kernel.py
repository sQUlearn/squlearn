""" Fidelity Quatnum Kernel class"""
import numpy as np
from typing import Union
from .kernel_matrix_base import KernelMatrixBase
from ...feature_map.feature_map_base import FeatureMapBase
from ...util.executor import Executor

from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.fidelity_quantum_kernel import (
    FidelityQuantumKernel,
)
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
from qiskit.utils import QuantumInstance
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit import ParameterVector
from qiskit.primitives import BaseSampler, BaseEstimator
from qiskit_ibm_runtime import IBMBackend


class FidelityKernel(KernelMatrixBase):
    """
    Fidelity Quantum Kernel.

    The Fidelity Quantum Kernel is a based on the overlap of the quantum states. These quantum states
    can be defined by a parameterized quantum circuit. The Fidelity Quantum Kernel is defined as:

    .. math::

        K(x,y) = |\\langle \\phi(x) | \\phi(y) \\rangle|^2

    This class wraps to the respective Quantum Kernel implemenations from *`Qiskit Machine Learning
    <https://qiskit.org/ecosystem/machine-learning/apidocs/qiskit_machine_learning.kernels.html>`_.
    Depending on the choice of the Qiskit Primitive or Quantum Instance, and dependent on the choice of trainable parameters, the
    appropraite Quantum Kernel implementation is chosen.

    Args:
        feature_map (FeatureMapBase): PQC feature map.
        executor (Executor): Executor object.
        initial_parameters (Union[np.ndarray, None], default=None):
            Initial parameters for the feature map.
        evaluate_duplicates (str), default='off_diagonal':
            Option for evaluating duplicates ('all', 'off_diagonal', 'none').
        mit_depol_noise (Union[str, None]), default=None:
            Option for mitigating depolarizing noise ('msplit' or 'mmean') after
            Ref. [4]. Only meaningful for
            FQKs computed on a real backend.

    References:
        [1]: `Havlicek et al., Supervised learning with quantum-enhanced feature spaces, Nature 567, 209-212 (2019). <https://www.nature.com/articles/s41586-019-0980-2>`_

        [2]: `Schuld et al., Quantum Machine Learning in Feature Hilbert Spaces, Phys. Rev. Lett. 122, 040504 (2019). <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040504>`_

        [3]: `Schuld et al., Quantum Machine Learning Models are Kernel Methods: Noise-Enhanced Quantum Embeddings, arXiv:2105.02276 (2021). <https://arxiv.org/abs/2105.02276>`_

        [4]: `T. Hubregtsen et al., "Training Quantum Embedding Kernels on Near-Term Quantum Computers", arXiv:2105.02276v1 (2021) <https://arxiv.org/pdf/2105.02276.pdf>`_


    """

    def __init__(
        self,
        feature_map: FeatureMapBase,
        executor: Executor,
        initial_parameters=None,
        evaluate_duplicates: str = "off_diagonal",
        mit_depol_noise: Union[str, None] = None,
    ) -> None:
        super().__init__(feature_map, executor, initial_parameters)

        self._quantum_kernel = None
        self._evaluate_duplicates = evaluate_duplicates.lower()
        self._mit_depol_noise = mit_depol_noise

        self._feature_vector = ParameterVector("x", self.num_features)
        if self.num_parameters > 0:
            self._parameter_vector = ParameterVector("θ", self.num_parameters)
        else:
            self._parameter_vector = None

        self._fmap_circuit = self._feature_map.get_circuit(
            self._feature_vector, self._parameter_vector
        )

        if self._executor.execution == "Sampler" or isinstance(self._executor.backend, IBMBackend):
            fidelity = ComputeUncompute(sampler=self._executor.get_sampler())
            if self._parameter_vector is None:
                # Fidelity Quantum Kernel without any parameters
                self._quantum_kernel = FidelityQuantumKernel(
                    feature_map=self._fmap_circuit,
                    fidelity=fidelity,
                    evaluate_duplicates=self._evaluate_duplicates,
                )
            else:
                # Fidelity Quantum Kernel with any parameters -> TrainableFidelityQuantumKernel
                self._quantum_kernel = TrainableFidelityQuantumKernel(
                    feature_map=self._fmap_circuit,
                    fidelity=fidelity,
                    training_parameters=self._parameter_vector,
                    evaluate_duplicates=self._evaluate_duplicates,
                )
        else:
            self._quantum_kernel = QuantumKernel(
                feature_map=self._fmap_circuit,
                quantum_instance=self._executor.backend,
                training_parameters=self._parameter_vector,
                evaluate_duplicates=self._evaluate_duplicates,
            )

    def evaluate(self, x: np.ndarray, y: Union[np.ndarray, None] = None) -> np.ndarray:
        if y is None:
            y = x
        kernel_matrix = np.zeros((x.shape[0], y.shape[0]))
        if self._parameter_vector is not None:
            if self._parameters is None:
                raise ValueError(
                    "Parameters have to been set with assign_parameters or as initial parameters!"
                )

            self._quantum_kernel.assign_training_parameters(self._parameters)

        kernel_matrix = self._quantum_kernel.evaluate(x, y)
        if self._mit_depol_noise is not None:
            print("WARNING: Adavnced option. Do not use it within an squlearn.kernel.ml workflow")
            if not np.array_equal(x, y):
                raise ValueError(
                    "Mitigating depolarizing noise works only for square matrices computed on real backend"
                )
            else:
                if self._mit_depol_noise == "msplit":
                    kernel_matrix = self._get_msplit_kernel(kernel_matrix)
                elif self._mit_depol_noise == "mmean":
                    kernel_matrix = self._get_mmean_kernel(kernel_matrix)

        return kernel_matrix
        # return self._quantum_kernel.evaluate(x, y)

    ###########
    ## Mitigating depolarizing noise after http://arxiv.org/pdf/2105.02276v1
    ###########
    def _get_msplit_kernel(self, kernel: np.ndarray) -> np.ndarray:
        msplit_kernel_matrix = np.zeros((kernel.shape[0], kernel.shape[1]))
        survival_prob = self._survival_probability(kernel)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                msplit_kernel_matrix[i, j] = (
                    kernel[i, j]
                    - 2 ** (-1.0 * self._num_qubits) * (1 - survival_prob[i] * survival_prob[j])
                ) / (survival_prob[i] * survival_prob[j])
        return msplit_kernel_matrix

    def _get_mmean_kernel(self, kernel: np.ndarray) -> np.ndarray:
        mmean_kernel_matrix = np.zeros((kernel.shape[0], kernel.shape[1]))
        survival_prob_mean = self._survival_probability_mean(kernel)
        mmean_kernel_matrix = (
            kernel - 2 ** (-1.0 * self._num_qubits) * (1 - survival_prob_mean**2)
        ) / survival_prob_mean**2
        return mmean_kernel_matrix

    def _survival_probability(self, kernel: np.ndarray) -> np.ndarray:
        kernel_diagonal = np.diag(kernel)
        surv_prob = np.sqrt(
            (kernel_diagonal - 2 ** (-1.0 * self._num_qubits))
            / (1 - 2 ** (-1.0 * self._num_qubits))
        )
        return surv_prob

    def _survival_probability_mean(self, kernel: np.ndarray) -> float:
        surv_prob = self._survival_probability(kernel)
        return np.mean(surv_prob)
