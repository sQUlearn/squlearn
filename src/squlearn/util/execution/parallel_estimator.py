import copy
from dataclasses import asdict
from typing import Callable, Optional, Tuple, Union
from collections.abc import Iterable

from packaging import version
from qiskit.circuit import QuantumCircuit
from qiskit import __version__ as qiskit_version
from qiskit.compiler import transpile
from qiskit.primitives import Estimator as PrimitiveEstimatorV1
from qiskit.primitives.base import EstimatorResult
from qiskit.primitives.utils import _circuit_key
from qiskit.providers import JobV1
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_aer import Aer
from qiskit_ibm_runtime import __version__ as ibm_runtime_version

QISKIT_SMALLER_1_0 = version.parse(qiskit_version) < version.parse("1.0.0")
QISKIT_SMALLER_1_2 = version.parse(qiskit_version) < version.parse("1.2.0")

if QISKIT_SMALLER_1_0:
    # pylint: disable=ungrouped-imports
    from qiskit.primitives import (
        BackendEstimator as BackendEstimatorV1,
        BaseEstimator as BaseEstimatorV1,
    )

    class BaseEstimatorV2:
        """Dummy BaseEstimatorV2"""

    class StatevectorEstimator:
        """Dummy StatevectorEstimator"""

    class EstimatorPubLike:
        """Dummy EstimatorPubLike"""

    class EstimatorPub:
        """Dummy EstimatorPub"""

    class BasePrimitiveJob:
        """Dummy BasePrimitiveJob"""

    class ObservablesArray:
        """Dummy ObservablesArray"""

else:
    from qiskit.primitives import (
        BackendEstimator as BackendEstimatorV1,
        BaseEstimatorV1,
        BaseEstimatorV2,
        BasePrimitiveJob,
        StatevectorEstimator,
    )
    from qiskit.primitives.containers import EstimatorPubLike
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from qiskit.primitives.containers.observables_array import ObservablesArray

if QISKIT_SMALLER_1_2:

    class BackendEstimatorV2:
        """Dummy BackendEstimatorV2"""

else:
    # pylint: disable=ungrouped-imports
    from qiskit.primitives import (
        BackendEstimatorV2,
    )


QISKIT_RUNTIME_SMALLER_0_21 = version.parse(ibm_runtime_version) < version.parse("0.21.0")
QISKIT_RUNTIME_SMALLER_0_28 = version.parse(ibm_runtime_version) < version.parse("0.28.0")

if QISKIT_RUNTIME_SMALLER_0_21:
    # pylint: disable=ungrouped-imports
    from qiskit_ibm_runtime import (
        Estimator as RuntimeEstimatorV1,
    )

    # pylint: disable=ungrouped-imports
    from qiskit_ibm_runtime.options import Options as RuntimeOptions

    class RuntimeEstimatorV2:
        """Dummy RuntimeEstimatorV2"""

elif QISKIT_RUNTIME_SMALLER_0_28:
    from qiskit_ibm_runtime import (
        EstimatorV1 as RuntimeEstimatorV1,
        EstimatorV2 as RuntimeEstimatorV2,
    )

    class RuntimeOptions:
        """Dummy RuntimeOptions"""

else:
    from qiskit_ibm_runtime import (
        Estimator as RuntimeEstimatorV2,
    )

    class RuntimeEstimatorV1:
        """Dummy RuntimeEstimatorV1"""

    class RuntimeOptions:
        """Dummy RuntimeOptions"""


import squlearn.util.executor


def _custom_result_method(self):
    return self._result


class ParallelEstimatorV1(BaseEstimatorV1):
    """
    Special Estimator Primitive that parallelize circuits to be passed to the estimator.

    Args:
        estimator (BaseEstimator): The estimator instance to use
        num_parallel (int, optional): The number of times the circuit is duplicated.
                                      Defaults to None, which means automatic determination.
        transpiler (callable, optional): A function for transpiling quantum circuits.
                                         Defaults to a standard transpile function if not provided.
        options (Options or RuntimeOptions, optional): Configuration settings for
                                                                   the instance.
    """

    def __init__(
        self,
        estimator: BaseEstimatorV1,
        num_parallel: Optional[int] = None,
        transpiler: Optional[Callable] = None,
        options: Union[Options, RuntimeOptions, None] = None,
    ) -> None:
        if isinstance(options, (Options, RuntimeOptions)):
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

        self.shots = None
        self._backend = None
        self._check_estimator()
        self._cache = {}

    def _check_estimator(self) -> None:
        """Configures the backend and shot settings based on the provided estimator object."""
        if hasattr(self._estimator.options, "execution"):
            self.shots = self._estimator.options.get("execution").get("shots", None)

        if isinstance(self._estimator, PrimitiveEstimatorV1):
            # this is only a hack, there is no real backend in the Primitive Estimator class
            self._backend = Aer.get_backend("statevector_simulator")
            self.shots = self._estimator.options.get("shots", 0)
            if self.shots == 0:
                self.shots = None
        elif isinstance(self._estimator, BackendEstimatorV1):
            self._backend = self._estimator._backend
            shots_estimator = self._estimator.options.get("shots", 0)
            if shots_estimator == 0:
                if self.shots is None:
                    self.shots = 1024
                self._estimator.set_options(shots=self.shots)
            else:
                self.shots = shots_estimator
        # Real Backend
        elif isinstance(self._estimator, RuntimeEstimatorV1):
            self._session = self._estimator._session
            self._service = self._estimator._service
            self._backend = self._estimator._backend
        elif isinstance(self._estimator, squlearn.util.executor.ExecutorEstimatorV1):
            self._backend = self._estimator._executor.backend
            self.shots = self._estimator._executor.get_shots()
        else:
            raise RuntimeError("No backend found in the given Estimator Primitive!")

    def _set_shots(self, num_shots: Union[int, None]) -> None:
        """Sets the number shots for the next evaluations.

        Args:
            num_shots (int or None): Number of shots that are set
        """
        if num_shots is None:
            num_shots = 0

        # Update shots in backend
        if self._backend is not None:
            if "statevector_simulator" not in str(self._backend):
                self._backend.options.shots = num_shots

        # Update shots in estimator primitive
        if self._estimator is not None:
            if isinstance(self._estimator, PrimitiveEstimatorV1):
                if num_shots == 0:
                    self._estimator.set_options(shots=None)
                else:
                    self._estimator.set_options(shots=num_shots)
            elif isinstance(self._estimator, BackendEstimatorV1):
                self._estimator.set_options(shots=num_shots)
            elif isinstance(self._estimator, RuntimeEstimatorV1):
                execution = self._estimator.options.get("execution")
                execution["shots"] = num_shots
                self._estimator.set_options(execution=execution)
            elif isinstance(self._estimator, squlearn.util.executor.ExecutorEstimatorV1):
                self._estimator._executor.set_shots(num_shots)
            else:
                raise RuntimeError("Unknown estimator type!")

    def _call(
        self,
        circuits,
        observables,
        parameter_values=None,
        **run_options,
    ) -> EstimatorResult:
        """Calls the estimator primitive call method and returns an EstimatorResult.

        Args:
            circuits: Quantum circuits to execute.
            observables: Observable to measure.
            parameter_values: Values for the parameters in circuits.
            run_options: Additional arguments that are passed to the estimator.

        Returns:
            An EstimatorResult object containing the expectation values.
        """
        return self._estimator._call(circuits, observables, parameter_values, **run_options)

    def _run(
        self,
        circuits,
        observables,
        parameter_values=None,
        **run_options,
    ) -> JobV1:
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
        circuits,
        observables,
        parameter_values=None,
        **run_options,
    ) -> JobV1:
        """
        Overwrites the executor primitive run method, to evaluate expectation values.

        Input arguments are the same as in Qiskit's estimator.run()

        Args:
            circuits: The quantum circuits to be executed.
            observables: The observables to be measured.
            parameter_values: The parameter values to be used for each circuit.
            **run_options: Additional keyword arguments for the Estimator run call.
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
            duplicated_circ, duplicated_obs = self._create_mapped_circuit(
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
        result_job.result = _custom_result_method.__get__(result_job, type(result_job))
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

    def set_options(self, **fields) -> None:
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        # TODO: Maybe shots??
        self._estimator.set_options(**fields)

    def _create_mapped_circuit(
        self,
        circuit: QuantumCircuit,
        observable: Optional[Union[BaseOperator]] = None,
        num_parallel: Optional[int] = None,
        return_duplications: Optional[bool] = False,
        max_qubits: Optional[int] = None,
    ) -> Union[
        QuantumCircuit,
        Tuple[QuantumCircuit, Optional[Union[BaseOperator]], Optional[int]],
    ]:
        """
        Maps a given quantum circuit, optionally duplicating it to fill the backend capacity.

        This method maps the provided quantum circuit, potentially duplicating it to utilize as much of the backend's capacity as possible.
        The duplication is controlled by the 'num_parallel' or 'max_qubits' Args.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be mapped.
            observable (Optional[BaseOperator], optional): The observable to be estimated. Defaults to None.
            num_parallel (Optional[int], optional): Specifies the number of times the circuit should be duplicated. Defaults to None.
            return_duplications (Optional[bool], optional): If True, returns a tuple of the mapped circuit and the number of duplications. Defaults to False.
            max_qubits (Optional[int], optional): The maximum number of qubits to use from the backend. Defaults to the number of qubits in the backend if None.

        Returns:
        Union[QuantumCircuit, Tuple[QuantumCircuit, Optional[Union[BaseOperator, str]], Optional[int]]]: The mapped quantum circuit. Depending on the parameters, the return can be:
            - A single QuantumCircuit.
            - A tuple of QuantumCircuit and the number of duplications.
            - A tuple of QuantumCircuit, the duplicated observable, and the number of duplications.
            - A tuple of QuantumCircuit and the duplicated observable.

        Raises:
            Warning: If no number of qubits is found in the given Estimator Primitive.
            ValueError: If the total number of qubits required for duplications exceeds the backend's capacity.
        """

        if max_qubits is None:
            try:
                max_qubits = self._backend.configuration().n_qubits
            except AttributeError:
                max_qubits = self._backend.num_qubits

            if max_qubits is None:
                raise Warning("No number of qubits found in the given Estimator Primitive!")

        # create the circuit
        mapped_circuit = circuit.copy()

        # check that n_duplication is None, i.e. not provided.
        if num_parallel is None:
            num_parallel = int(max_qubits // mapped_circuit.num_qubits)

        if num_parallel * mapped_circuit.num_qubits > max_qubits:
            raise ValueError(
                f"The number of qubits in the circuit ({mapped_circuit.num_qubits}) * num_parallel ({num_parallel}) "
                f"is greater than the total number of qubits in the backend ({max_qubits})"
            )

        # duplicate the circuit
        for _ in range(num_parallel - 1):
            mapped_circuit.tensor(circuit, inplace=True)

        shots = self.shots
        if shots is None:
            shots = 0

        # Set the shots=shots/num_parallel
        if self.shots is not None:
            self._set_shots(int(self.shots / num_parallel))

        # if observable is provided, duplicate it and return it as well
        if observable is not None:
            mapped_obs = self._duplicate_observable(observable, num_parallel)
            if return_duplications:
                return mapped_circuit, mapped_obs, num_parallel
            else:
                return mapped_circuit, mapped_obs

        if return_duplications:
            return mapped_circuit, num_parallel
        else:
            return mapped_circuit

    def _remove_unused_qubits(self, circuit: QuantumCircuit, observable) -> QuantumCircuit:
        """
        Removes unused qubits from a given quantum circuit.

        This method removes all unused qubits from a given quantum circuit, as well as any gates that act on those qubits.
        The resulting circuit is equivalent to the original circuit, but with fewer qubits.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be simplified.
            observable: The observable to be simplified

        Returns:
            QuantumCircuit: The simplified quantum circuit.
        """

        gate_count = {qubit: 0 for qubit in circuit.qubits}
        for gate in circuit.data:
            for qubit in gate.qubits:
                gate_count[qubit] += 1
        for qubit, count in gate_count.items():
            if count == 0:
                circuit.qubits.remove(qubit)

        return circuit, observable

    def _duplicate_observable(
        self, observable: Union[BaseOperator], n_duplications: int
    ) -> SparsePauliOp:
        """
        Duplicates a given quantum observable multiple times and combines them into a single SparsePauliOp.

        This function takes a quantum observable and creates 'n_duplications' copies of it, with each duplicate padded with identity operators ('I') to align with the qubits' positions.
        The duplicates are then combined into one SparsePauliOp. The coefficients of the combined observable are adjusted to account for the number of duplications.

        Args:
            observable (Union[BaseOperator]): The quantum observable to be duplicated.
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

    def _transpile(self, circuit: QuantumCircuit, **options) -> QuantumCircuit:
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


class ParallelEstimatorV2(BaseEstimatorV2):
    """
    Special Estimator Primitive that parallelize circuits to be passed to the estimator.

    Args:
        estimator (BaseEstimator): The estimator instance to use
        num_parallel (int, optional): The number of times the circuit is duplicated.
                                      Defaults to None, which means automatic determination.
        transpiler (callable, optional): A function for transpiling quantum circuits.
                                         Defaults to a standard transpile function if not provided.
    """

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        num_parallel: Optional[int] = None,
        transpiler: Optional[Callable] = None,
    ) -> None:
        self._estimator = estimator
        self._num_parallel = num_parallel
        if transpiler is None:
            self._transpiler = transpile
        else:
            self._transpiler = transpiler

        self.shots = None
        self._backend = None
        self._check_estimator()
        self._cache = {}

    def _check_estimator(self) -> None:
        """Configures the backend and shot settings based on the provided estimator object."""

        if isinstance(self._estimator, StatevectorEstimator):
            # this is only a hack, there is no real backend in the StatevectorEstimator class
            self._backend = Aer.get_backend("statevector_simulator")
            if self._estimator.default_precision:
                self.shots = int(1.0 / self._estimator.default_precision**2)
        elif isinstance(self._estimator, BackendEstimatorV2):
            self._backend = self._estimator.backend
            if not self._estimator.options.default_precision:
                raise ValueError("Precision of the estimator must be greater than 0!")
            self.shots = int((1.0 / self._estimator.options.default_precision) ** 2)
        # Real Backend
        elif isinstance(self._estimator, RuntimeEstimatorV2):
            self._session = self._estimator._mode
            self._service = self._estimator._service
            self._backend = self._estimator._backend
            if self._estimator.options.default_shots:
                self.shots = self._estimator.options.default_shots
            elif self._estimator.options.default_precision:
                self.shots = int((1.0 / self._estimator.options.default_precision) ** 2)
            else:
                raise ValueError("Either default_shots or default_precision must be set!")
        elif isinstance(self._estimator, squlearn.util.executor.ExecutorEstimatorV2):
            self._backend = self._estimator._executor.backend
            self.shots = self._estimator._executor.get_shots()
        else:
            raise RuntimeError("No backend found in the given Estimator Primitive!")

    def _set_shots(self, num_shots: Union[int, None]) -> None:
        """Sets the number shots for the next evaluations.

        Args:
            num_shots (int or None): Number of shots that are set
        """
        if num_shots is None:
            num_shots = 0

        # Update shots in backend
        if self._backend is not None:
            if "statevector_simulator" not in str(self._backend):
                self._backend.options.shots = num_shots

        # Update shots in estimator primitive
        if self._estimator is not None:
            if isinstance(self._estimator, StatevectorEstimator):
                self._estimator._default_precision = 1.0 / num_shots**0.5
            elif isinstance(self._estimator, BackendEstimatorV2):
                self._estimator._options.default_precision = 1.0 / num_shots**0.5
            elif isinstance(self._estimator, RuntimeEstimatorV2):
                self._estimator._options.update(**{"default_shots": num_shots})
            elif isinstance(self._estimator, squlearn.util.executor.ExecutorEstimatorV2):
                self._estimator._executor.set_shots(num_shots)
            else:
                raise RuntimeError("Unknown estimator type!")

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: Union[float, None] = None
    ) -> BasePrimitiveJob:
        """
        Overwrites the executor primitive run method, to evaluate expectation values.

        Input arguments are the same as in Qiskit's estimator.run()

        Args:
            pubs: An iterable of pub-like objects, such as tuples ``(circuit, observables)``
                or ``(circuit, observables, parameter_values)``.
            precision: The target precision for expectation value estimates of each
                run Estimator Pub that does not specify its own precision. If None
                the estimator's default precision value will be used.
        """
        if precision is None:
            if self.shots:
                precision = 1 / self.shots**0.5
            else:
                precision = 0.0
        duplicated_pubs = []

        for pub in pubs:
            coerced_pub = EstimatorPub.coerce(pub, precision=precision)
            coerced_pub._circuit, coerced_pub._observables, num_parallel = (
                self._create_mapped_circuit(
                    coerced_pub.circuit,
                    coerced_pub.observables,
                    num_parallel=self._num_parallel,
                    return_duplications=True,
                )
            )
            coerced_pub._precision *= num_parallel**0.5
            duplicated_pubs.append(coerced_pub)

        result_job = self._estimator.run(pubs=duplicated_pubs, precision=precision)

        results = result_job.result()
        for result in results:
            if "shots" in result.metadata:
                result.metadata["shots"] *= num_parallel
            result.metadata["target_precision"] /= num_parallel**0.5
        result_job._pub_results = results
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

    def set_options(self, **fields) -> None:
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        # TODO: Maybe shots??
        self._estimator.set_options(**fields)

    def _create_mapped_circuit(
        self,
        circuit: QuantumCircuit,
        observable: Optional[Union[BaseOperator]] = None,
        num_parallel: Optional[int] = None,
        return_duplications: Optional[bool] = False,
        max_qubits: Optional[int] = None,
    ) -> Union[
        QuantumCircuit,
        Tuple[QuantumCircuit, Optional[Union[BaseOperator]], Optional[int]],
    ]:
        """
        Maps a given quantum circuit, optionally duplicating it to fill the backend capacity.

        This method maps the provided quantum circuit, potentially duplicating it to utilize as much of the backend's capacity as possible.
        The duplication is controlled by the 'num_parallel' or 'max_qubits' Args.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be mapped.
            observable (Optional[BaseOperator], optional): The observable to be estimated. Defaults to None.
            num_parallel (Optional[int], optional): Specifies the number of times the circuit should be duplicated. Defaults to None.
            return_duplications (Optional[bool], optional): If True, returns a tuple of the mapped circuit and the number of duplications. Defaults to False.
            max_qubits (Optional[int], optional): The maximum number of qubits to use from the backend. Defaults to the number of qubits in the backend if None.

        Returns:
        Union[QuantumCircuit, Tuple[QuantumCircuit, Optional[Union[BaseOperator, str]], Optional[int]]]: The mapped quantum circuit. Depending on the parameters, the return can be:
            - A single QuantumCircuit.
            - A tuple of QuantumCircuit and the number of duplications.
            - A tuple of QuantumCircuit, the duplicated observable, and the number of duplications.
            - A tuple of QuantumCircuit and the duplicated observable.

        Raises:
            Warning: If no number of qubits is found in the given Estimator Primitive.
            ValueError: If the total number of qubits required for duplications exceeds the backend's capacity.
        """

        if max_qubits is None:
            try:
                max_qubits = self._backend.configuration().n_qubits
            except AttributeError:
                max_qubits = self._backend.num_qubits

            if max_qubits is None:
                raise Warning("No number of qubits found in the given Estimator Primitive!")

        # create the circuit
        mapped_circuit = circuit.copy()

        # check that n_duplication is None, i.e. not provided.
        if num_parallel is None:
            num_parallel = int(max_qubits // mapped_circuit.num_qubits)

        if num_parallel * mapped_circuit.num_qubits > max_qubits:
            raise ValueError(
                f"The number of qubits in the circuit ({mapped_circuit.num_qubits}) * num_parallel ({num_parallel}) "
                f"is greater than the total number of qubits in the backend ({max_qubits})"
            )

        # duplicate the circuit
        for _ in range(num_parallel - 1):
            mapped_circuit.tensor(circuit, inplace=True)

        shots = self.shots
        if shots is None:
            shots = 0

        # Set the shots=shots/num_parallel
        if self.shots is not None:
            self._set_shots(int(self.shots / num_parallel))

        # if observable is provided, duplicate it and return it as well
        if observable is not None:
            mapped_obs = self._duplicate_observable(observable, num_parallel)
            if return_duplications:
                return mapped_circuit, mapped_obs, num_parallel
            else:
                return mapped_circuit, mapped_obs

        if return_duplications:
            return mapped_circuit, num_parallel
        else:
            return mapped_circuit

    def _remove_unused_qubits(self, circuit: QuantumCircuit, observable) -> QuantumCircuit:
        """
        Removes unused qubits from a given quantum circuit.

        This method removes all unused qubits from a given quantum circuit, as well as any gates that act on those qubits.
        The resulting circuit is equivalent to the original circuit, but with fewer qubits.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be simplified.
            observable: The observable to be simplified

        Returns:
            QuantumCircuit: The simplified quantum circuit.
        """

        gate_count = {qubit: 0 for qubit in circuit.qubits}
        for gate in circuit.data:
            for qubit in gate.qubits:
                gate_count[qubit] += 1
        for qubit, count in gate_count.items():
            if count == 0:
                circuit.qubits.remove(qubit)

        return circuit, observable

    def _duplicate_observable(
        self, observable: Union[BaseOperator], n_duplications: int
    ) -> SparsePauliOp:
        """
        Duplicates a given quantum observable multiple times and combines them into a single SparsePauliOp.

        This function takes a quantum observable and creates 'n_duplications' copies of it, with each duplicate padded with identity operators ('I') to align with the qubits' positions.
        The duplicates are then combined into one SparsePauliOp. The coefficients of the combined observable are adjusted to account for the number of duplications.

        Args:
            observable (Union[BaseOperator]): The quantum observable to be duplicated.
            n_duplications (int): The number of times the observable should be duplicated.

        Returns:
            SparsePauliOp: The combined observable after duplicating and merging the original observable 'n_duplications' times.

        The method adjusts the observable's coefficients to ensure the combined observable accurately represents the sum of its parts.
        """
        # Get number of qubits in the observable
        n_qubits = len(next(iter(observable.reshape(-1)[0])))
        # Total number of qubits in the final combined observable
        total_qubits = n_qubits * n_duplications

        # Initialize a list to hold the duplicated observables
        duplicated_observables = []

        for obs in observable.reshape(-1):
            for key, value in obs.items():
                dop = {}
                for i in range(n_duplications):
                    dop["I" * (n_qubits * i) + key + "I" * (total_qubits - n_qubits * (i + 1))] = (
                        value / n_duplications
                    )
                duplicated_observables.append(dop)

        return ObservablesArray.coerce(duplicated_observables)

    def _transpile(self, circuit: QuantumCircuit, **options) -> QuantumCircuit:
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


class ParallelEstimator:
    """
    Special Estimator Primitive that parallelize circuits to be passed to the estimator.

    Args:
        estimator (BaseEstimator): The estimator instance to use
        num_parallel (int, optional): The number of times the circuit is duplicated.
            Defaults to None, which means automatic determination.
        transpiler (callable, optional): A function for transpiling quantum circuits.
            Defaults to a standard transpile function if not provided.
        options (Options or RuntimeOptions, optional): Configuration settings for the instance.
            Only supported for ParallelEstimatorV1.
    """

    def __new__(
        cls,
        estimator: Union[BaseEstimatorV1, BaseEstimatorV2],
        num_parallel: Optional[int] = None,
        transpiler: Optional[Callable] = None,
        options: Union[Options, RuntimeOptions, None] = None,
    ) -> Union[ParallelEstimatorV1, ParallelEstimatorV2]:
        if isinstance(estimator, BaseEstimatorV1):
            return ParallelEstimatorV1(estimator, num_parallel, transpiler, options)
        elif isinstance(estimator, BaseEstimatorV2):
            if options:
                raise TypeError("Options are not supported for ParallelEstimatorV2")
            return ParallelEstimatorV2(estimator, num_parallel, transpiler)
        else:
            raise TypeError("Estimator must be either BaseEstimatorV1 or BaseEstimatorV2")
