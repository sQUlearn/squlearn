from qiskit.primitives import BaseEstimator
from qiskit.providers import Options
from qiskit_ibm_runtime.options import Options as qiskit_ibm_runtime_Options
import copy
from typing import Union, Callable, Optional, List, Tuple
from dataclasses import asdict
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives.base import EstimatorResult
from qiskit.circuit import QuantumCircuit
from qiskit.providers import JobV1 as Job
from qiskit import Aer, transpile
from qiskit.primitives import Estimator as qiskit_primitives_Estimator
from qiskit.primitives import BackendEstimator as qiskit_primitives_BackendEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Estimator as qiskit_ibm_runtime_Estimator
from qiskit.primitives.utils import _circuit_key


def custom_result_method(self):
    return self._result


class ParallelEstimator(BaseEstimator):
    """
    Special Estimator Primitive that parallelize circuits to be passed to the estimator.

    Args:
        estimator (BaseEstimator): The estimator instance to use
        num_parallel (int, optional): The number of times the circuit is duplicated. Defaults to None, which means automatic determination.
        transpiler (callable, optional): A function for transpiling quantum circuits. Defaults to a standard transpile function if not provided.
        options (Options or qiskit_ibm_runtime_Options, optional): Configuration settings for the instance.

    """

    def __init__(
        self, estimator: BaseEstimator, num_parallel: Optional[int] = None, transpiler: Optional[Callable] = None, options=None
    ) -> None:
        if isinstance(options, Options) or isinstance(options, qiskit_ibm_runtime_Options):
            try:
                options_ini = copy.deepcopy(options).__dict__
            except Exception:
                options_ini = asdict(copy.deepcopy(options))
        else:
            options_ini = options

        super().__init__(options=options_ini)
        self._estimator = estimator
        self._num_parallel = num_parallel
        if transpiler is None:
            self._transpiler = transpile
        else:
            self._transpiler = transpiler
        self.check_estimator()
        self._cache = {}

    def check_estimator(self) -> None:
        """Configures the backend and shot settings based on the properties of the provided estimator object."""
        self.shots = None
        if hasattr(self._estimator.options, "execution"):
            self.shots = self._estimator.options.get("execution").get("shots", None)
        if isinstance(self._estimator, qiskit_primitives_Estimator):
            # this is only a hack, there is no real backend in the Primitive Estimator class
            self._backend = Aer.get_backend("statevector_simulator")
        elif isinstance(self._estimator, qiskit_primitives_BackendEstimator):
            self._backend = self._estimator._backend
            shots_estimator = self._estimator.options.get("shots", 0)
            if shots_estimator == 0:
                if self.shots is None:
                    self.shots = 1024
                self._estimator.set_options(shots=self.shots)
            else:
                self.shots = shots_estimator
        # Real Backend
        elif hasattr(self._estimator, "session"):
            self._backend = self._estimator.session.service.get_backend(
                self._estimator.session.backend()
            )
            self._session_active = True
        else:
            raise RuntimeError("No backend found in the given Estimator Primitive!")

    def set_shots(self, num_shots: Union[int, None]) -> None:
        """Sets the number shots for the next evaluations.

        Args:
            num_shots (int or None): Number of shots that are set
        """

        self._shots = num_shots

        if num_shots is None:
            self._logger.info("Set shots to {}".format(num_shots))
            num_shots = 0

        # Update shots in backend
        if self._backend is not None:
            if "statevector_simulator" not in str(self._backend):
                self._backend.options.shots = num_shots

        # Update shots in estimator primitive
        if self._estimator is not None:
            if isinstance(self._estimator, qiskit_primitives_Estimator):
                if num_shots == 0:
                    self._estimator.set_options(shots=None)
                else:
                    self._estimator.set_options(shots=num_shots)
                try:
                    self._options_estimator["shots"] = num_shots
                except Exception:
                    pass  # no option available
            elif isinstance(self._estimator, qiskit_primitives_BackendEstimator):
                self._estimator.set_options(shots=num_shots)
                try:
                    self._options_estimator["shots"] = num_shots
                except Exception:
                    pass  # no option available
            elif isinstance(self._estimator, qiskit_ibm_runtime_Estimator):
                execution = self._estimator.options.get("execution")
                execution["shots"] = num_shots
                self._estimator.set_options(execution=execution)
                try:
                    self._options_estimator["execution"]["shots"] = num_shots
                except Exception:
                    pass  # no options_estimator or no execution in options_estimator
            else:
                raise RuntimeError("Unknown estimator type!")

    def _call(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        observables: Union[
            List[BaseOperator], List[PauliSumOp], List[str], BaseOperator, PauliSumOp, str
        ],
        parameter_values: Union[List[float], List[List[float]]] = None,
        **run_options,
    ) -> EstimatorResult:
        """Has to be passed through, otherwise python will complain about the abstract method.
        Input arguments are the same as in Qiskit's estimator.call()
        """
        return self._estimator._call(circuits, observables, parameter_values, **run_options)

    def _run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        observables: Union[
            List[BaseOperator], List[PauliSumOp], List[str], BaseOperator, PauliSumOp, str
        ],
        parameter_values: Union[List[float], List[List[float]]] = None,
        **run_options,
    ) -> Job:
        """Has to be passed through, otherwise python will complain about the abstract method.
        Input arguments are the same as in Qiskit's estimator.run().
        """
        return self._estimator._run(
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            **run_options,
        )

    def run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        observables: Union[
            List[BaseOperator], List[PauliSumOp], List[str], BaseOperator, PauliSumOp, str
        ],
        parameter_values: Union[List[float], List[List[float]]] = None,
        **run_options,
    ) -> Job:
        """
        Overwrites the sampler primitive run method, to evaluate expectation values.

        Input arguments are the same as in Qiskit's estimator.run()

        """
        dupl_circuits = []
        dupl_observables = []
        if not isinstance(circuits, list):
            circuits = [circuits]
        if not isinstance(observables, list):
            observables = [observables]

        if "shots" in run_options:
            self.shots = run_options["shots"]
            run_options.pop("shots")

        for circ, obs in zip(circuits, observables):
            duplicated_circ, duplicated_obs = self.create_mapped_circuit(
                circ, observable=obs, num_parallel=self._num_parallel
            )
            dupl_circuits.append(duplicated_circ)
            dupl_observables.append(duplicated_obs)

        result_job = self._estimator.run(
            circuits=dupl_circuits,
            observables=dupl_observables,
            parameter_values=parameter_values,
            **run_options,
        )
        result = result_job.result()
        for meta in result.metadata:
            meta["shots"] = self.shots
        result_job._result = result
        result_job.result = custom_result_method.__get__(result_job, type(result_job))
        return result_job

    @property
    def circuits(self):
        """Quantum circuits that represents quantum states.

        Returns:
            The quantum circuits.
        """
        return tuple(self._estimator.circuits)

    @property
    def observables(self):
        """Observables to be estimated.

        Returns:
            The observables.
        """
        return tuple(self._estimator.observables)

    @property
    def parameters(self):
        """Parameters of the quantum circuits.

        Returns:
            Parameters, where ``parameters[i][j]`` is the j- :spelling:word:`th` parameter of the
            i-th circuit.
        """
        return tuple(self._estimator.parameters)

    @property
    def options(self) -> Options:
        """Return options values for the estimator.

        Returns:
            options
        """
        return self._estimator.options

    # def clear_cache(self):
    #     self._estimator.clear_cache()

    def set_options(self, **fields) -> None:
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self._estimator.set_options(**fields)
        self._estimator._options_estimator = self._estimator.options

    def create_mapped_circuit(
        self,
        circuit: QuantumCircuit,
        observable: Optional[Union[BaseOperator, PauliSumOp]] = None,
        num_parallel: Optional[int] = None,
        return_duplications: Optional[bool] = False,
        max_qubits: Optional[int] = None,
    ) -> Union[
        QuantumCircuit,
        Tuple[QuantumCircuit, Optional[Union[BaseOperator, PauliSumOp]], Optional[int]],
    ]:
        """
        Maps a given quantum circuit, optionally duplicating it to fill the backend capacity.

        This method maps the provided quantum circuit, potentially duplicating it to utilize as much of the backend's capacity as possible.
        The duplication is controlled by the 'n_duplications' or 'max_qubits' Args.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be mapped.
            observable (Optional[BaseOperator, PauliSumOp], optional): The observable to be estimated. Defaults to None.
            n_duplications (Optional[int], optional): Specifies the number of times the circuit should be duplicated. Defaults to None.
            return_duplications (Optional[bool], optional): If True, returns a tuple of the mapped circuit and the number of duplications. Defaults to False.
            max_qubits (Optional[int], optional): The maximum number of qubits to use from the backend. Defaults to the number of qubits in the backend if None.

        Returns:
        Union[QuantumCircuit, Tuple[QuantumCircuit, Optional[Union[BaseOperator, PauliSumOp, str]], Optional[int]]]: The mapped quantum circuit. Depending on the parameters, the return can be:
            - A single QuantumCircuit.
            - A tuple of QuantumCircuit and the number of duplications.
            - A tuple of QuantumCircuit, the duplicated observable, and the number of duplications.
            - A tuple of QuantumCircuit and the duplicated observable.

        Raises:
            Warning: If no number of qubits is found in the given Estimator Primitive.
            ValueError: If the total number of qubits required for duplications exceeds the backend's capacity.
        """

        if max_qubits is None:
            max_qubits = self._backend.configuration().n_qubits
            if max_qubits is None:
                raise Warning("No number of qubits found in the given Estimator Primitive!")

        # check that n_duplication is None, i.e. not provided.
        if num_parallel is None:
            num_parallel = int(max_qubits // circuit.num_qubits)

        if num_parallel * circuit.num_qubits > max_qubits:
            raise ValueError(
                f"The number of qubits in the circuit ({circuit.num_qubits}) * n_duplications ({num_parallel}) "
                f"is greater than the total number of qubits in the backend ({max_qubits})"
            )

        # create the circuit
        mapped_circuit = circuit.copy()

        # duplicate the circuit
        for _ in range(num_parallel - 1):
            mapped_circuit.tensor(circuit, inplace=True)

        shots = self.shots
        if shots is None:
            shots = 0
        print(
            f"Circuit with {circuit.num_qubits} qubits has been duplicated {num_parallel} times."
            f"\nReducing shots from {shots} to {int(shots/num_parallel)}"
            f"\nBackend has {max_qubits} qubits.\n"
        )

        # Set the shots=shots/n_duplications
        if self.shots is not None:
            self.set_shots(int(self.shots / num_parallel))

        # if observable is provided, duplicate it and return it as well
        if observable is not None:
            mapped_obs = self.duplicate_observable(observable, num_parallel)
            if return_duplications:
                return mapped_circuit, mapped_obs, num_parallel
            else:
                return mapped_circuit, mapped_obs

        if return_duplications:
            return mapped_circuit, num_parallel
        else:
            return mapped_circuit

    def duplicate_observable(
        self, observable: Union[BaseOperator, PauliSumOp], n_duplications: int
    ) -> SparsePauliOp:
        """
        Duplicates a given quantum observable multiple times and combines them into a single SparsePauliOp.

        This function takes a quantum observable and creates 'n_duplications' copies of it, with each duplicate padded with identity operators ('I') to align with the qubits' positions.
        The duplicates are then combined into one SparsePauliOp. The coefficients of the combined observable are adjusted to account for the number of duplications.

        Args:
            observable (Union[BaseOperator, PauliSumOp]): The quantum observable to be duplicated.
            n_duplications (int): The number of times the observable should be duplicated.

        Returns:
            SparsePauliOp: The combined observable after duplicating and merging the original observable 'n_duplications' times.

        The method adjusts the observable's coefficients to ensure the combined observable accurately represents the sum of its parts.
        """

        # Get number of qubits in the observable
        n_qubits = observable.num_qubits
        # Total number of qubits in the final combined observable
        total_qubits = n_qubits * n_duplications

        # Initialize a list to hold the duplicated observables
        duplicated_observables = []

        # Duplicate the observable with appropriate padding
        for i in range(n_duplications):
            # Padding on the left and right for each duplication
            padding_left = "I" * (n_qubits * i)
            padding_right = "I" * (total_qubits - n_qubits * (i + 1))
            padded_observable = SparsePauliOp.from_list(
                [
                    (padding_left + pauli_str + padding_right, coeff)
                    for pauli_str, coeff in zip(observable.paulis.to_labels(), observable.coeffs)
                ]
            )
            duplicated_observables.append(padded_observable)

        # Combine all the observables into one SparsePauliOp
        combined_observable = sum(duplicated_observables)

        # Adjust the coefficients by dividing by the number of duplications
        combined_observable = SparsePauliOp(
            combined_observable.paulis, combined_observable.coeffs / n_duplications
        ).simplify()

        return combined_observable

    def transpile(self, circuit: QuantumCircuit, **options) -> QuantumCircuit:
        """
        Transpiles a given quantum circuit, using cached results if available.

        This method checks if the provided circuit has already been transpiled by looking it up in a cache.
        If it's in the cache, the cached transpiled circuit is returned.
        Otherwise, the circuit is transpiled using the provided transpiler function, and the result is cached for future use.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be transpiled.
            **options: Additional keyword arguments for the transpiler. These options allow for customization of the transpilation process.

        Returns:
            The transpiled quantum circuit.
        """
        key = _circuit_key(circuit)
        if key in self._cache:
            transpiled_circuit = self._cache[key]
        else:
            transpiled_circuit = self._transpiler(circuit, **options)
            self._cache[key] = transpiled_circuit
        return transpiled_circuit
