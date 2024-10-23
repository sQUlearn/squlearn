from typing import Optional, Callable, Union, List, Tuple
import copy
import json
import os
from logging import Logger

import networkx as nx
import numpy as np

from packaging import version
from qiskit import qasm3, transpile, QuantumCircuit
from qiskit import __version__ as qiskit_version
from qiskit.providers import Backend, BackendV2
from qiskit.providers.fake_provider.utils.json_decoder import (
    decode_backend_properties,
)
from qiskit.providers.models import BackendProperties
from qiskit_ibm_runtime import QiskitRuntimeService
import mapomatic as mm

# HQAA additions
from .hqaa import parse_openqasm, heuristic

QISKIT_SMALLER_1_0 = version.parse(qiskit_version) < version.parse("1.0.0")


def get_num_qubits(backend):
    """Gets the number of qubits of a backend.

    Args:
        backend (Backend or BackendV1 or BackendV2): The backend.

    Returns:
        int: The number of qubits of the backend.
    """
    try:
        return backend.configuration().n_qubits
    except AttributeError:
        return backend.num_qubits


def get_backend_name(backend):
    """Gets the name of a backend.

    Args:
        backend (Backend or BackendV1 or BackendV2): The backend.

    Returns:
        str: The name of the backend.
    """
    if callable(getattr(backend, "name", None)):
        return backend.name()
    else:
        return backend.name


class NoSuitableBackendError(Exception):
    """Raised when no suitable backend is found."""


class AutomaticBackendSelection:
    """Class for automatically selecting an IBM backend and layout from the available backends, given a circuit."""

    def __init__(
        self,
        service: Optional[QiskitRuntimeService] = None,
        min_num_qubits: Optional[int] = None,
        max_num_qubits: Optional[int] = None,
        backends_to_use: Optional[Union[Backend, List[Backend]]] = None,
        cost_function: Optional[Callable] = None,
        optimization_level: int = 3,
        n_trials_transpile=1,
        call_limit: Optional[int] = int(30000000),
        verbose: bool = False,
        logger: Logger = None,
    ):
        """Initialize AutoSelectionBackend with service.

        Args:
            service (QiskitRuntimeService, optional): Service object. Default to None.
            min_num_qubits (int, optional): Minimum number of qubits that a backend can have. Defaults to None.
            max_num_qubits (int, optional): Maximum number of qubits that a backend can have. Defaults to None.
            backends_to_use (Union[Backend, List[Backend]], optional) : List of backends to use. Defaults to None.
            cost_function (callable, optional): Custom cost function, if not specified use Mapomatic default.
            optimization_level (int, optional): Optimization level for the transpilation. Defaults to 3.
            n_trials_transpile (int, optional): Number of times to try to transpile the circuit. Defaults to 1.
            call_limit(int, optional): Maximum number of calls to VF2 mapper (Mapomatic). Defaults to 30000000.
            verbose (bool, optional): Whether to print information. Defaults to False.
            logger (logger, optional): Logger object. Defaults to None.
        """

        self.service = service

        self.min_num_qubits = min_num_qubits if min_num_qubits is not None else 0
        self.max_num_qubits = max_num_qubits
        self.cost_function = cost_function

        if isinstance(backends_to_use, Backend):
            backends_to_use = [backends_to_use]
        self.backends_to_use = backends_to_use
        self.optimization_level = optimization_level
        self.n_trials_transpile = n_trials_transpile
        self.call_limit = call_limit
        self.verbose = verbose
        self.use_hqaa = False
        self.logger = logger

        if self.service is None:
            self._obtain_backends_from_service = False
            if self.backends_to_use is not None:
                try:
                    self.service = self.backends_to_use[0].service  # get service from a backend
                except AttributeError:

                    class dummy_service:
                        def __init__(self, backend) -> None:
                            self.backend = backend

                        def least_busy(self, *args, **kwargs):
                            return self.backend

                    self.service = dummy_service(self.backends_to_use[0])
            else:
                raise ValueError("Error: Either provide service or backends_to_use.")
        else:
            self._obtain_backends_from_service = True

        self.backends = self._get_backend_list()

        if QISKIT_SMALLER_1_0:

            class BackendPropertiesWrapper:
                def __init__(self, backend: BackendV2) -> None:
                    self._backend = backend

                def properties(self) -> BackendProperties:
                    self._backend._set_props_dict_from_json()
                    return BackendProperties.from_dict(self._backend._props_dict)

                @property
                def configuration(self):
                    config_dict = copy.copy(self._backend._conf_dict)
                    config_dict["num_qubits"] = config_dict["n_qubits"]
                    config = type("config", (object,), config_dict)
                    return config

            for backend in self.backends:
                if isinstance(backend, BackendV2):
                    backend.properties = BackendPropertiesWrapper(backend).properties
                    backend.configuration = BackendPropertiesWrapper(backend).configuration

        self._print("Automatic backend selection started")
        if self.backends:
            self._print(f"Number of backends available with given parameters:{len(self.backends)}")
        else:
            raise NoSuitableBackendError("No suitable backend found with given parameters")

    def _print(self, message: str):
        """Print message if verbose is True, if logger is set, log message."""
        if self.verbose:
            print(message)
        if self.logger is not None:
            self.logger.info(message)

    def _filters(self, backend: Backend) -> bool:
        """Filter for selecting backends, returns True if backend is suitable.

        Args:
            backend (Backend): Backend object.
        """

        # Check minimum number of qubits
        min_qubits_condition = True
        if self.min_num_qubits:
            min_qubits_condition = get_num_qubits(backend) >= self.min_num_qubits

        # Check maximum number of qubits
        max_qubits_condition = True
        if self.max_num_qubits:
            max_qubits_condition = get_num_qubits(backend) <= self.max_num_qubits

        # Check backend name if self.backend_to_use exists
        backend_condition = True
        if self.backends_to_use:
            backend_condition = backend in self.backends_to_use

        # Filter out simulators (if present)
        if "simulator" in get_backend_name(backend):
            return False

        # Return True only if both conditions are met
        return min_qubits_condition and max_qubits_condition and backend_condition

    def _get_backend_list(self) -> List[Backend]:
        """Return a list of backends matching the given parameters.

        Returns:
            list: A list of backends.
        """
        if self._obtain_backends_from_service:
            return self.service.backends(
                min_num_qubits=self.min_num_qubits,
                simulator=False,
                operational=True,
                filters=self._filters,
            )
        else:
            return [backend for backend in self.backends_to_use if self._filters(backend)]

    def _get_specific_backend(self, backend_name: str) -> Optional[Backend]:
        """Return a specific backend by name.

        Args:
            backend_name (str): Name of the backend.

        Returns
            Backend: The specific backend object.
        """
        for backend in self.backends:
            if get_backend_name(backend) == backend_name:
                return backend
        return None

    def _find_least_busy_backend(self, circuit: QuantumCircuit) -> Backend:
        """Find a least busy backend for a given circuit size or min qubits specified by the user.

        Args:
            circuit (QuantumCircuit): Circuit object.

        Returns:
            Backend: The least busy backend object.

        """
        least_busy_backend = self.service.least_busy(
            min_num_qubits=max([self.min_num_qubits, circuit.num_qubits]),
            simulator=False,
            operational=True,
            filters=self._filters,
        )
        self._print(
            f"Least busy backend: {get_backend_name(least_busy_backend)} with {get_num_qubits(least_busy_backend)} qubits"
        )
        return least_busy_backend

    def _find_compatible_backends(self, circuit: QuantumCircuit) -> dict:
        """Return a list of compatible backends for a given circuit size (num_qubits).
        We want one per IBM processor family, to find which processor family offers the best transpilation (i.e. lowest two-qubit gates count).

        Args:
            circuit : Circuit object.

        Returns:
            object: A compatible backend.
        """
        backend_by_family = {}

        for backend in self.backends:
            backend_qubits = get_num_qubits(backend)

            if hasattr(backend, "processor_type"):
                if hasattr(backend.processor_type, "family"):
                    backend_processor_family = backend.processor_type.get("family", "unknown")
                else:
                    backend_processor_family = "unknown"
            else:
                backend_processor_family = "unknown"
            if backend_qubits >= circuit.num_qubits:
                # Add one backend per family of processor.
                backend_by_family[backend_processor_family] = backend

        if len(backend_by_family) == 0:
            raise NoSuitableBackendError("No suitable backend found with given parameters")
        else:
            return backend_by_family

    def _tensor_circuits(self, circuit_list: List[QuantumCircuit]) -> QuantumCircuit:
        """Tensor all circuits in the list to create a combined circuit.

        Args:
            circuit_list (list[QuantumCircuit]): List of QuantumCircuit objects to be combined..

        Returns:
            QuantumCircuit:  A new QuantumCircuit that is the tensor product of all circuits in the provided list.
        """
        # Initialize an empty Quantum Circuit
        combined_qc = QuantumCircuit(0)

        # Tensor all circuits in the list
        for circuit in circuit_list:
            combined_qc = combined_qc.tensor(circuit)

        return combined_qc

    def _evaluate_quality_mode(
        self, circuit: QuantumCircuit, compatible_backends: dict
    ) -> Tuple[Tuple, QuantumCircuit, Backend]:
        """Evaluate the best layout for a given circuit across all compatible backends.
        Because the SWAP mappers in Qiskit are stochastic, the number of inserted SWAP gates can vary with each run.
        The spread in this number can be quite large, and can impact the performance of your circuit.
        It is thus beneficial to transpile many instances of a circuit and take the best one. This can be set at init via 'n_trials_transpile'.

        Args:
            circuit (QuantumCircuit): Circuit to evaluate.
            compatible_backends (dict): Dictionary of compatible backends.
        Returns
            Tuple[Tuple, QuantumCircuit, Backend]: A tuple containing the best layout, the transpiled circuit, and the best compatible backend.
        """

        # transpile the circuit to the compatible backend.
        self._print(
            f"Transpiling circuit {self.n_trials_transpile} times for {len(compatible_backends)} compatible backends..."
        )

        trans_qc_list = []
        for backend in compatible_backends.values():
            trans_qc = transpile(
                [circuit] * self.n_trials_transpile,
                backend,
                optimization_level=self.optimization_level,
            )
            trans_qc_list.extend(trans_qc)

        best_two_q_gates_count = [
            (
                circ.count_ops().get("cx", 0)
                if "cx" in circ.count_ops()
                else circ.count_ops().get("ecr", 0)
            )
            for circ in trans_qc_list
        ]

        self._print(f"Two-qubit gate count: {best_two_q_gates_count}")
        best_idx = np.argmin(best_two_q_gates_count)
        best_qc = trans_qc_list[best_idx]
        self._print(
            f"Best Two-qubit gate count (ID:{best_idx}): {best_two_q_gates_count[best_idx]} "
        )
        # deflate the transpiled circuit,i.e. remove unused qubits
        small_qc = mm.deflate_circuit(best_qc)
        self._print(f"Transpiled circuit needs {small_qc.num_qubits} qubits")

        # Filter out self.backend based on the circuit number of qubits
        possible_backends = []

        for backend in self.backends:
            if get_num_qubits(backend) >= small_qc.num_qubits:
                possible_backends.append(backend)

        if self.use_hqaa:
            best_qc, best_backend, score, layout = self._evaluate_via_hqaa(
                possible_backends, small_qc
            )
            self._print(f"Best sub-layout: {layout}. Error_rate: {score}")

        else:
            # find the best layout for the circuit
            self._print(f"Searching best sub-layout on {len(possible_backends)} backends...")
            best_layout = mm.best_overall_layout(
                small_qc,
                possible_backends,
                call_limit=self.call_limit,
                cost_function=self.cost_function,
            )
            # retrieve the backend from result of best_overall_layout
            best_backend = self._get_specific_backend(best_layout[1])
            self._print(
                f"Best sub-layout: {best_layout[0]} on backend: {get_backend_name(best_backend)}. Error_rate: {best_layout[2]}"
            )
            # retranspile the circuit to the best backend using best_layout
            best_qc = transpile(
                small_qc,
                best_backend,
                initial_layout=best_layout[0],
                optimization_level=self.optimization_level,
            )
            score = best_layout[2]

        # check if transpilation changed layout.
        if best_qc.layout.final_layout:
            self._print(
                "Last transpilation: sub-layout has been changed"
            )  # from {best_qc.layout.initial_layout} to {best_qc.layout.final_layout}")
            return (best_qc.layout.final_layout, score), best_qc, best_backend
        else:
            return (best_qc.layout.initial_layout, score), best_qc, best_backend

    def _evaluate_speed_mode(
        self, circuit: QuantumCircuit, least_busy_backend: Backend
    ) -> Tuple[Tuple, QuantumCircuit, Backend]:
        """Evaluate the layout on the least busy backend for a given circuit.
         Args:
            circuit (QuantumCircuit): Circuit to evaluate.
            least_busy_backend (Backend): Least busy backend.
        Returns:
            Tuple[Tuple, QuantumCircuit, Backend]: A tuple containing the best layout, the transpiled circuit, and the least busybackend.

        """
        # find the best layouts for the given backend (least busy)
        trans_qc = transpile(
            circuit, least_busy_backend, optimization_level=self.optimization_level
        )
        small_qc = mm.deflate_circuit(trans_qc)
        self._print(
            f"Transpiled circuit needs {small_qc.num_qubits} qubits on {get_backend_name(least_busy_backend)}"
        )
        if self.use_hqaa:
            final_circuit, least_busy_backend, score, layout = self._evaluate_via_hqaa(
                least_busy_backend, small_qc
            )
            self._print(f"Best sub-layout: {layout}. Error_rate: {score}")
        else:
            layouts = mm.matching_layouts(small_qc, least_busy_backend, call_limit=self.call_limit)

            # if layouts: #if there is a sub-layout
            scores = mm.evaluate_layouts(
                small_qc, layouts, least_busy_backend, cost_function=self.cost_function
            )
            # run on the best sub-layout
            score = scores[0][1]
            layout = scores[0][0]
            self._print(f"Best sub-layout: {layout}. Error_rate: {score}")
            # transpile the circuit to the least busy backend, using best sub-layout
            final_circuit = transpile(
                small_qc,
                least_busy_backend,
                initial_layout=layout,
                optimization_level=self.optimization_level,
            )

        if final_circuit.layout.final_layout:
            self._print("Last transpilation: sub-layout has been changed")
            return (final_circuit.layout.final_layout, score), final_circuit, least_busy_backend
        else:
            return (final_circuit.layout.initial_layout, score), final_circuit, least_busy_backend

    def _evaluate_via_hqaa(
        self, backends: list[Backend], small_qc: QuantumCircuit
    ) -> Tuple[QuantumCircuit, Backend, float, List[int]]:
        """
        Evaluate the best sub-layout for a given circuit using the HQAA algorithm.

        Args:
            backends (list): List of backends.
            small_qc (QuantumCircuit): Circuit to evaluate.

        Returns:
            Tuple[Tuple, QuantumCircuit, Backend]: A tuple containing the best layout,
                the transpiled circuit, and the best backend.
        """
        if not isinstance(backends, list):
            backends = [backends]
        # HQAA requires the qasm3 format of the circuit.
        qasm_string = qasm3.dumps(small_qc, experimental=qasm3.ExperimentalFeatures.SWITCH_CASE_V1)
        circuit_parsed = parse_openqasm(qasm_string, small_qc.num_qubits)
        nx_graph = nx.DiGraph()
        best_score = float("inf")
        best_qc: QuantumCircuit = None
        best_backend: Backend = None
        best_layout: List[int] = None
        for backend in backends:
            edges = backend.coupling_map.get_edges()
            qubits = backend.coupling_map.physical_qubits
            props = backend.properties()
            basis_gates = backend.configuration().basis_gates
            # Check for two-qubit gates
            two_qubit_gate = (
                "cx" if "cx" in basis_gates else "ecr" if "ecr" in basis_gates else None
            )
            if two_qubit_gate is None:
                self.logger.warning(
                    f"Backend {get_backend_name(backend)} does not support two-qubit gates."
                    "Skipping this backend."
                )
                continue

            for qubit in qubits:
                nx_graph.add_node(
                    qubit,
                    weight=props.gate_error("sx", qubit),
                    read_out=props.readout_error(qubit),
                )

            for src, dst in edges:
                avg_gate_error = props.gate_error(two_qubit_gate, [src, dst])
                nx_graph.add_edge(src, dst, weight=avg_gate_error)

            final_mapping = heuristic(nx_graph, circuit_parsed, print_function=self._print)
            this_layout = list(final_mapping.values())
            final_circuit = transpile(
                small_qc,
                backend,
                initial_layout=this_layout,
                optimization_level=self.optimization_level,
            )
            this_score = mm.evaluate_layouts(
                final_circuit,
                range(get_num_qubits(backend)),
                backend,
                cost_function=self.cost_function,
            )
            self._print(f"{this_score=}")
            if this_score[0][1] < best_score:
                best_qc = final_circuit
                best_backend = backend
                best_score = this_score[0][1]
                best_layout = this_layout
        if any(best is None for best in [best_qc, best_backend, best_score, best_layout]):
            raise NoSuitableBackendError("No suitable backend found for this circuit using HQAA.")
        return best_qc, best_backend, best_score, best_layout

    def evaluate(
        self,
        circuit: QuantumCircuit,
        mode: Optional[str] = "quality",
        use_hqaa: Optional[bool] = False,
    ) -> Tuple[Tuple, QuantumCircuit, Backend]:
        """Evaluate the best backend for a given circuit and mode.
        Modes can be 'quality' or'speed'. Speed mode is available only if the service is set.

        Args:
            circuit (QuantumCircuit): Circuit object.
            mode (str, optional): Evaluation mode. Defaults to 'quality'.
            use_hqaa (bool, optional): Use HQAA or Mapomatic. Defaults to Mapomatic/False.
        Returns:
            tuple: Evaluation results.
        """
        self.use_hqaa = use_hqaa
        self._print(f"Mode: {mode}" + (" using HQAA" if self.use_hqaa else " using Mapomatic"))

        # If the input is a list of circuits, tensor them together
        if isinstance(circuit, list):
            self._print(f"Combining {len(circuit)} circuits into one")
            circuit = self._tensor_circuits(circuit)

        self._print(f"Input circuit needs {circuit.num_qubits} qubits")
        if mode == "quality":
            compatible_backends = self._find_compatible_backends(circuit)
            return self._evaluate_quality_mode(circuit, compatible_backends)
        if mode == "speed":
            if self.service is None:
                raise ValueError("Service not set. Only 'quality' mode available")
            least_busy_backend = self._find_least_busy_backend(circuit)
            return self._evaluate_speed_mode(circuit, least_busy_backend)

        raise ValueError(f"Invalid mode: {mode}. Expected 'quality' or 'speed'.")
