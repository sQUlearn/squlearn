""" Fidelity Quantum Kernel class"""
from typing import Union
import numpy as np

from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel,
    FidelityStatevectorKernel,
    TrainableFidelityQuantumKernel,
    TrainableFidelityStatevectorKernel,
)
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit import ParameterVector

from .kernel_matrix_base import KernelMatrixBase
from ...encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from ...util.executor import Executor


class FidelityKernel(KernelMatrixBase):
    """
    Fidelity Quantum Kernel.

    The Fidelity Quantum Kernel is a based on the overlap of the quantum states.
    These quantum states
    can be defined by a parameterized quantum circuit. The Fidelity Quantum Kernel is defined as:

    .. math::

        K(x,y) = |\\langle \\phi(x) | \\phi(y) \\rangle|^2

    This class wraps to the respective Quantum Kernel implementations from `Qiskit Machine Learning
    <https://qiskit.org/ecosystem/machine-learning/apidocs/qiskit_machine_learning.kernels.html>`_.
    Depending on the choice of the backend and the choice of trainable parameters, the appropriate
    Quantum Kernel implementation is chosen.

    Args:
        encoding_circuit (EncodingCircuitBase): PQC encoding circuit.
        executor (Executor): Executor object.
        evaluate_duplicates (str), default='off_diagonal':
            Option for evaluating duplicates ('all', 'off_diagonal', 'none').
        mit_depol_noise (Union[str, None]), default=None:
            Option for mitigating depolarizing noise (``"msplit"`` or ``"mmean"``) after
            Ref. [4]. Only meaningful for
            FQKs computed on a real backend.
        initial_parameters (Union[np.ndarray, None], default=None):
            Initial parameters for the encoding circuit.
        parameter_seed (Union[int, None], default=0):
            Seed for the random number generator for the parameter initialization, if
            initial_parameters is None.
        regularization  (Union[str, None], default=None):
            Option for choosing different regularization techniques (``"thresholding"`` or
            ``"tikhonov"``) after Ref. [4] for the training kernel matrix, prior to  solving the
            linear system in the ``fit()``-procedure.

    References:
        [1]: `Havlicek et al., Supervised learning with quantum-enhanced feature spaces,
        Nature 567, 209-212 (2019).
        <https://www.nature.com/articles/s41586-019-0980-2>`_

        [2]: `Schuld et al., Quantum Machine Learning in Feature Hilbert Spaces,
        Phys. Rev. Lett. 122, 040504 (2019).
        <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040504>`_

        [3]: `Schuld et al., Quantum Machine Learning Models are Kernel Methods:
        Noise-Enhanced Quantum Embeddings, arXiv:2105.02276 (2021).
        <https://arxiv.org/abs/2105.02276>`_

        [4]: `T. Hubregtsen et al.,
        "Training Quantum Embedding Kernels on Near-Term Quantum Computers",
        arXiv:2105.02276v1 (2021)
        <https://arxiv.org/abs/2105.02276>`_


    """

    def __init__(
        self,
        encoding_circuit: EncodingCircuitBase,
        executor: Executor,
        evaluate_duplicates: str = "off_diagonal",
        mit_depol_noise: Union[str, None] = None,
        initial_parameters: Union[np.ndarray, None] = None,
        parameter_seed: Union[int, None] = 0,
        regularization: Union[str, None] = None,
    ) -> None:
        super().__init__(
            encoding_circuit, executor, initial_parameters, parameter_seed, regularization
        )

        self._quantum_kernel = None
        self._evaluate_duplicates = evaluate_duplicates
        self._mit_depol_noise = mit_depol_noise

        self._feature_vector = ParameterVector("x", self.num_features)
        if self.num_parameters > 0:
            self._parameter_vector = ParameterVector("θ", self.num_parameters)
        else:
            self._parameter_vector = None

        self._enc_circ = self._encoding_circuit.get_circuit(
            self._feature_vector, self._parameter_vector
        )

        if "statevector_simulator" in str(self._executor._backend):
            if self._parameter_vector is None:
                self._quantum_kernel = FidelityStatevectorKernel(feature_map=self._enc_circ)
            else:
                self._quantum_kernel = TrainableFidelityStatevectorKernel(
                    feature_map=self._enc_circ,
                    training_parameters=self._parameter_vector,
                )
        else:
            fidelity = ComputeUncompute(sampler=self._executor.get_sampler())
            if self._parameter_vector is None:
                self._quantum_kernel = FidelityQuantumKernel(
                    feature_map=self._enc_circ,
                    fidelity=fidelity,
                    evaluate_duplicates=self._evaluate_duplicates,
                )
            else:
                self._quantum_kernel = TrainableFidelityQuantumKernel(
                    feature_map=self._enc_circ,
                    fidelity=fidelity,
                    training_parameters=self._parameter_vector,
                    evaluate_duplicates=self._evaluate_duplicates,
                )

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the fidelity kernel.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params(deep=False)
        params["evaluate_duplicates"] = self._evaluate_duplicates
        params["mit_depol_noise"] = self._mit_depol_noise
        params["regularization"] = self._regularization
        params["encoding_circuit"] = self._encoding_circuit
        if deep:
            params.update(self._encoding_circuit.get_params())
        return params

    def set_params(self, **params):
        """
        Sets value of the fidelity kernel hyper-parameters.

        Args:
            params: Hyper-parameters and their values, e.g. ``num_qubits=2``
        """
        num_parameters_backup = self.num_parameters
        parameters_backup = self._parameters

        # Check if all parameters are valid
        valid_params = self.get_params()
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        if "encoding_circuit" in params:
            self._encoding_circuit = params["encoding_circuit"]
            params.pop("encoding_circuit")

        dict_encoding_circuit = {}
        for key in params.keys():
            if key in self._encoding_circuit.get_params().keys():
                dict_encoding_circuit[key] = params[key]
        for key in dict_encoding_circuit.keys():
            params.pop(key)

        self._encoding_circuit.set_params(**dict_encoding_circuit)

        if "evaluate_duplicates" in params.keys():
            self._evaluate_duplicates = params["evaluate_duplicates"].lower()
            params.pop("evaluate_duplicates")
        if "mit_depol_noise" in params.keys():
            self._mit_depol_noise = params["mit_depol_noise"]
            params.pop("mit_depol_noise")
        if "regularization" in params.keys():
            self._regularization = params["regularization"]
            params.pop("regularization")

        self.__init__(
            self._encoding_circuit,
            self._executor,
            self._evaluate_duplicates,
            self._mit_depol_noise,
            None,
            self._parameter_seed,
            self._regularization,
        )

        if self.num_parameters == num_parameters_backup:
            self._parameters = parameters_backup

        if len(params) > 0:
            raise ValueError("The following parameters could not be assigned:", params)

    def evaluate(self, x: np.ndarray, y: Union[np.ndarray, None] = None) -> np.ndarray:
        """
        Evaluates the fidelity kernel matrix.

        Args:
            x (np.ndarray) :
                Vector of training or test data for which the kernel matrix is evaluated
            y (np.ndarray, default=None) :
                Vector of training or test data for which the kernel matrix is evaluated
        Returns:
            Returns the quantum kernel matrix as 2D numpy array.
        """

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
            print("WARNING: Advanced option. Do not use it within an squlearn.kernel.ml workflow")
            if not np.array_equal(x, y):
                raise ValueError(
                    "Mitigating depolarizing noise works only for square matrices computed on real"
                    " backend"
                )
            else:
                if self._mit_depol_noise == "msplit":
                    kernel_matrix = self._get_msplit_kernel(kernel_matrix)
                elif self._mit_depol_noise == "mmean":
                    kernel_matrix = self._get_mmean_kernel(kernel_matrix)

        if (self._regularization is not None) and (
            kernel_matrix.shape[0] == kernel_matrix.shape[1]
        ):
            kernel_matrix = self._regularize_matrix(kernel_matrix)
        return kernel_matrix

    ###########
    ## Mitigating depolarizing noise after http://arxiv.org/abs/2105.02276v1
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
