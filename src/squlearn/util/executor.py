"""Executor module."""

import copy
import logging
import os
import time
import traceback
from dataclasses import asdict
from hashlib import blake2b
from logging import handlers
from pathlib import Path
from typing import Any, List, Union
from types import MethodType
from collections.abc import Iterable
import dill as pickle
import numpy as np
from packaging import version

import pennylane as qml
from pennylane.devices import Device as PennylaneDevice
from qiskit.circuit import QuantumCircuit
from qiskit import __version__ as qiskit_version
from qiskit.circuit import ParameterVector
from qiskit.exceptions import QiskitError
from qiskit.primitives import (
    Estimator as PrimitiveEstimatorV1,
    Sampler as PrimitiveSamplerV1,
)
from qiskit.primitives.base import EstimatorResult, SamplerResult
from qiskit.providers import JobV1
from qiskit.providers import Options
from qiskit.providers.backend import Backend
from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import __version__ as ibm_runtime_version
from qiskit_ibm_runtime.exceptions import IBMRuntimeError, RuntimeJobFailureError

if version.parse(qml.__version__) < version.parse("0.39.0"):
    from pennylane import QubitDevice
else:
    from pennylane.devices import QubitDevice

if version.parse(qiskit_version) <= version.parse("0.45.0"):
    from qiskit.utils import algorithm_globals
from qiskit_algorithms.utils import algorithm_globals as qiskit_algorithm_globals

QISKIT_SMALLER_1_0 = version.parse(qiskit_version) < version.parse("1.0.0")
QISKIT_SMALLER_1_2 = version.parse(qiskit_version) < version.parse("1.2.0")

if QISKIT_SMALLER_1_0:
    # pylint: disable=ungrouped-imports
    from qiskit.primitives import (
        BackendEstimator as BackendEstimatorV1,
        BackendSampler as BackendSamplerV1,
        BaseEstimator as BaseEstimatorV1,
        BaseSampler as BaseSamplerV1,
    )

    class BaseEstimatorV2:
        """Dummy BaseEstimatorV2"""

    class BaseSamplerV2:
        """Dummy BaseSamplerV2"""

    class StatevectorEstimator:
        """Dummy StatevectorEstimator"""

    class StatevectorSampler:
        """Dummy StatevectorSampler"""

    class EstimatorPubLike(object):
        """Dummy EstimatorPubLike"""

    class EstimatorPub(object):
        """Dummy EstimatorPub"""

    class SamplerPubLike(object):
        """Dummy EstimatorPubLike"""

    class SamplerPub(object):
        """Dummy EstimatorPub"""

else:
    from qiskit.primitives import (
        BackendEstimator as BackendEstimatorV1,
        BackendSampler as BackendSamplerV1,
        BaseEstimatorV1,
        BaseEstimatorV2,
        BaseSamplerV1,
        BaseSamplerV2,
        StatevectorEstimator,
        StatevectorSampler,
    )

    from qiskit.primitives.containers import EstimatorPubLike, SamplerPubLike
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from qiskit.primitives.containers.sampler_pub import SamplerPub


if QISKIT_SMALLER_1_2:

    class BackendEstimatorV2:
        """Dummy BackendEstimatorV2"""

    class BackendSamplerV2:
        """Dummy BackendSamplerV2"""

else:
    # pylint: disable=ungrouped-imports
    from qiskit.primitives import (
        BackendEstimatorV2,
        BackendSamplerV2,
    )


QISKIT_RUNTIME_SMALLER_0_21 = version.parse(ibm_runtime_version) < version.parse("0.21.0")
QISKIT_RUNTIME_SMALLER_0_23 = version.parse(ibm_runtime_version) < version.parse("0.23.0")
QISKIT_RUNTIME_SMALLER_0_28 = version.parse(ibm_runtime_version) < version.parse("0.28.0")

if QISKIT_RUNTIME_SMALLER_0_21:
    # pylint: disable=ungrouped-imports
    from qiskit_ibm_runtime import (
        Estimator as RuntimeEstimatorV1,
        Sampler as RuntimeSamplerV1,
    )

    # pylint: disable=ungrouped-imports
    from qiskit_ibm_runtime.options import Options as RuntimeOptionsV1

    class RuntimeEstimatorV2:
        """Dummy RuntimeEstimatorV2"""

    class RuntimeSamplerV2:
        """Dummy RuntimeSamplerV2"""

    class RuntimeOptionsV2:
        """Dummy RuntimeOptionsV2"""

elif QISKIT_RUNTIME_SMALLER_0_28:
    from qiskit_ibm_runtime import (
        EstimatorV1 as RuntimeEstimatorV1,
        EstimatorV2 as RuntimeEstimatorV2,
        SamplerV1 as RuntimeSamplerV1,
        SamplerV2 as RuntimeSamplerV2,
    )

    # pylint: disable=ungrouped-imports
    from qiskit_ibm_runtime.options import Options as RuntimeOptionsV1
    from qiskit_ibm_runtime.options import OptionsV2 as RuntimeOptionsV2

else:
    from qiskit_ibm_runtime import (
        Estimator as RuntimeEstimatorV2,
        Sampler as RuntimeSamplerV2,
    )

    from qiskit_ibm_runtime.options import OptionsV2 as RuntimeOptionsV2

    class RuntimeEstimatorV1:
        """Dummy RuntimeEstimatorV1"""

    class RuntimeSamplerV1:
        """Dummy RuntimeSamplerV1"""

    class RuntimeOptionsV1:
        """Dummy RuntimeOptionsV1"""


# pylint: disable=wrong-import-position
from .execution import AutomaticBackendSelection, ParallelEstimator, ParallelSampler
from .execution.parallel_estimator import ParallelEstimatorV1, ParallelEstimatorV2
from .execution.parallel_sampler import ParallelSamplerV1, ParallelSamplerV2


class Executor:
    r"""
    A class for executing quantum jobs on IBM Quantum systems or simulators.

    The Executor class is the central component of sQUlearn, responsible for running quantum jobs.
    Both high- and low-level methods utilize the Executor class to execute jobs seamlessly.
    It for example automatically creates the necessary Qiskit primitives when they are
    required in the sQUlearn sub-program or takes care of the execution of PennyLane circuits.
    The Executor takes care about Qiskit Runtime session handling, result caching, and automatic
    restarts of failed jobs.

    The Estimator can be initialized with various objects that specify the execution environment,
    as for example a Qiskit backend or a PennyLane device.

    A detailed introduction to the Executor can be found in the
    :doc:`User Guide: The Executor Class </user_guide/executor>`

    The version of Qiskit Primitives used by the Executor depends on the installed Qiskit version:
    - For Qiskit versions 1.2 and above, the Executor uses Qiskit Primitives V2.
    - For versions below 1.2, it defaults to Primitives V1.

    Note: The Sampler in Primitives V2 uses shots, even with statevector simulators, whereas
    Primitives V1 provides exact probabilities.

    **Important**: When using the Executor to run jobs on IBM Quantum systems, sessions are
    created automatically. If you are working in a Jupyter notebook, ensure you close the session
    once calculations are complete to avoid unnecessary open sessions (:meth:`close_session`),
    to avoide being charged for the opened but unused session.

    Args:
        execution (Union[str, Backend, List[Backend], QiskitRuntimeService, Session,BaseEstimatorV1, BaseSamplerV1, BaseEstimatorV2, BaseSamplerV2, PennylaneDevice]):
            The execution environment, possible inputs are:

                * A string, that specifics the simulator backend. For Qiskit this can be
                  ``"qiskit"``,``"statevector_simulator"`` or ``"qasm_simulator"``.
                  For PennyLane this can be ``"pennylane"``, ``"default.qubit"``.
                * A PennyLane device, to run the jobs with PennyLane (e.g. AWS Braket plugin
                  for PennyLane)
                * A Qiskit backend, to run the jobs on IBM Quantum systems or simulators
                * A list of Qiskit backends for automatic backend selection later on
                * A QiskitRuntimeService, to run the jobs on the Qiskit Runtime service.
                  In this case the backend has to be provided separately via ``backend=``
                * A Session, to run the jobs on the Qiskit Runtime service
                * A Estimator primitive (either simulator or Qiskit Runtime primitive - V1 or V2)
                * A Sampler primitive (either simulator or Qiskit Runtime primitive - V1 or V2)

            Default is the initialization with PennyLane's
            :class:`DefaultQubit <pennylane.devices.default_qubit.DefaultQubit>` simulator.
        backend (Union[Backend, str, None]): The backend that is used for the execution.
            Only mandatory if a service is provided.
        options_estimator (Union[Any]): The options for the created estimator primitives.
        options_sampler (Union[Any]): The options for the created sampler primitives.
        log_file (str): The name of the log file, if empty, no log file is created.
        caching (Union[bool, None]): Whether to cache the results of the jobs.
        cache_dir (str): The directory where to cache the results of the jobs.
        max_session_time (str): The maximum time for a session, similar input as in Qiskit.
        max_jobs_retries (int): The maximum number of retries for a job
            until the execution is aborted.
        wait_restart (int): The time to wait before restarting a job in seconds.
        shots (Union[int, None]): The number of initial shots that is used for the execution.
        seed (Union[int, None]): The seed that is used for finite samples in the execution.
        qpu_parallelization (Union[int, str, None]): The number of parallel executions on the QPU.
            If set to ``"auto"``, the number of parallel executions is automatically determined.
            If set to ``None``, no parallelization is used. Default is ``None``.
        auto_backend_mode (str): The mode for automatic backend selection. Possible values are:

            * ``"quality"``: Automatically selects the best backend for the provided circuit using
              the mapomatic tool. This is the default value.
            * ``"quality_hqaa"``: Same as ``"quality"``, but uses the HQAA algorithm.
            * ``"speed"``: Automatically selects the backend with the smallest queue using the
              mapomatic tool.
            * ``"speed_hqaa"``: Same as ``"speed"``, but uses the HQAA algorithm.

    Attributes:
    -----------

    Attributes:
        execution (str): String of the execution environment.
        backend (Backend): The backend that is used in the Executor.
        backend_list (List[Backend]): The list of backends used for the automatic backend
            selection.
        backend_chosen (Bool): True, if the backend was chosen automatically.
        backend_name (str): The name of the backend that is used in the Executor.
        is_statevector (Bool): Returns true if the backend is a statevector simulator.
        qpu_parallelization (Bool): Returns true if QPU parallelization is used.
        session (Session): The session that is used in the Executor.
        service (QiskitRuntimeService): The service that is used in the Executor.
        quantum_framework (str): The framework used in the Executor (``"qiskit"`` or
            ``"pennylane"``).
        IBMQuantum (bool): Whether the backend is an IBM Quantum backend.
        estimator (BaseEstimatorV1, BaseEstimatorV2): The Qiskit estimator primitive that is used
                                   in the Executor.
                                   Different to :meth:`get_estimator`,
                                   which creates a new estimator object with overwritten methods
                                   that runs everything through the Executor with
                                   :meth:`estimator_run`.
        sampler (BaseSamplerV1, BaseEstimatorV2): The Qiskit sampler primitive that is used in the
                               Executor. Different to :meth:`get_sampler`,
                               which creates a new sampler object with overwritten methods
                               that runs everything through the Executor with
                               :meth:`estimator_run`.
        shots (int): The number of shots that is used in the Executor.
        estimator_options: Options of the Runtime Estiamtor V2
        sampler_options: Options of the Runtime Sampler V2

    See Also:
       * :doc:`User Guide: The Executor Class </user_guide/executor>`
       * `Qiskit Runtime <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime>`_
       * `Qsikit Primitives <https://docs.quantum.ibm.com/api/qiskit/primitives>`_
       * `PennyLane Devices <https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html>`_

    **Example: Different PennyLane based initializations of the Executor**

    .. code-block:: python

        from squlearn import Executor
        import pennylane as qml

        # Executor with a PennyLane device (statevector)
        executor = Executor(qml.device("default.qubit"))

        # Executor with a PennyLane device (shot-based)
        executor = Executor(qml.device("default.qubit", shots=1000))

        # Executor with a PennyLane lightining device
        executor = Executor(qml.device("lightning.qubit"))

        # Executor with a AWS Braket device with 4 qubits
        # (requires a valid AWS credential to be set)
        dev = qml.device(
            "braket.aws.qubit",
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            wires=4
        )
        executor = Executor(dev)

    **Example: Different Qiskit based initializations of the Executor**

    .. code-block:: python

       from squlearn import Executor
       from qiskit_ibm_runtime import QiskitRuntimeService

       # Executor with a ideal simulator backend
       exec = Executor("statevector_simulator")

       # Executor with a shot-based simulator backend and 1000 shots
       exec = Executor("qasm_simulator")
       exec.set_shots(1000)

       # Executor with a IBM Quantum backend
       service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
       executor = Executor(service.get_backend('ibm_brisbane'))

       # Executor with a IBM Quantum backend and caching and logging
       service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
       executor = Executor(service.get_backend('ibm_brisbane'), caching=True,
                            cache_dir='cache', log_file="log.log")

    **Example: Get the Executor based Qiskit primitives**

    .. jupyter-execute::

       from squlearn import Executor

       # Initialize the Executor
       executor = Executor("statevector_simulator")

       # Get the Executor based Estimator with all execusions routed through the Executor
       estimator = executor.get_estimator()

       # Get the Executor based Sampler with all execusions routed through the Executor
       sampler = executor.get_sampler()

       # Run a circuit with the Executor based Sampler
       from qiskit.circuit.random import random_circuit
       circuit = random_circuit(2, 2, seed=1, measure=True).decompose(reps=1)
       job = sampler.run([(circuit,)])
       result = job.result()

    **Example: Automatic backend selection**

    .. code-block:: python

       import numpy as np
       from squlearn import Executor
       from qiskit_ibm_runtime import QiskitRuntimeService
       from squlearn.encoding_circuit import ChebyshevRx
       from squlearn.kernel import FidelityKernel, QKRR

       # Executor is initialized with a service, and considers all available backends
       # (except simulators)
       service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
       executor = Executor(service, auto_backend_mode="quality")

       # Create a QKRR model with a FidelityKernel and the ChebyshevRx encoding circuit
       qkrr = QKRR(FidelityKernel(ChebyshevRx(4,1),executor))

       # Backend is automatically selected based on the encoding circuit
       # All the following functions will be executed on the selected backend
       X_train, y_train = np.array([[0.1],[0.2]]), np.array([0.1,0.2])
       qkrr.fit(X_train, y_train)

       # Close the session to avoid being charged for the opened but unused session
       executor.close_session()

    **Example: QPU parallelization**

    .. jupyter-execute::

       from squlearn import Executor

       # All circuit executions are copied four times and are executed in parallel
       executor = Executor("statevector_simulator", qpu_parallelization=4)

       # The level of parallelization is determined automatically to reach a maximum
       # parallelization level of number of qubits of the backend divided by the number of qubits
       # of the circuit
       executor = Executor("statevector_simulator", qpu_parallelization="auto")


    Methods:
    --------
    """

    def __init__(
        self,
        execution: Union[
            str,
            Backend,
            List[Backend],
            QiskitRuntimeService,
            Session,
            BaseEstimatorV1,
            BaseSamplerV1,
            BaseEstimatorV2,
            BaseSamplerV2,
            PennylaneDevice,
        ] = "pennylane",
        backend: Union[Backend, str, None] = None,
        options_estimator: Union[Any, None] = None,
        options_sampler: Union[Any, None] = None,
        log_file: str = "",
        caching: Union[bool, None] = None,
        cache_dir: str = "_cache",
        max_session_time: str = "8h",
        max_jobs_retries: int = 10,
        wait_restart: int = 1,
        shots: Union[int, None] = None,
        seed: Union[int, None] = None,
        qpu_parallelization: Union[int, str, None] = None,
        auto_backend_mode: str = "quality",
    ) -> None:
        # Default values for internal variables
        self._backend = None
        self._session = None
        self._service = None
        self._estimator = None
        self._sampler = None
        self._execution_origin = ""

        # Copy estimator options and make a dict
        if options_estimator is not None:
            self._options_estimator = _convert_options_to_dict(options_estimator)
        else:
            self._options_estimator = None

        # Copy sampler options and make a dict
        if options_sampler is not None:
            self._options_sampler = _convert_options_to_dict(options_sampler)
        else:
            self._options_sampler = None

        if seed is not None:
            # Hack that seed is not equal to 0 since this gets fake backends confuced
            if seed >= 0:
                seed += 1
            if version.parse(qiskit_version) <= version.parse("0.45.0"):
                algorithm_globals.random_seed = seed
            qiskit_algorithm_globals.random_seed = seed
        self._set_seed_for_primitive = seed
        self._pennylane_seed = seed

        # Copy Executor options
        self._log_file = log_file
        self._caching = caching
        self._max_session_time = max_session_time
        self._max_jobs_retries = max_jobs_retries
        self._wait_restart = wait_restart
        self._qpu_parallelization = qpu_parallelization
        if auto_backend_mode in ["quality_hqaa", "speed_hqaa"]:
            self._auto_backend_options = {
                "mode": auto_backend_mode.split("_")[0],
                "use_hqaa": True,
            }
        elif auto_backend_mode in ["quality", "speed"]:
            self._auto_backend_options = {
                "mode": auto_backend_mode,
                "use_hqaa": False,
            }
        else:
            raise ValueError(
                "auto_backend_mode must be one of 'quality_hqaa', 'speed_hqaa', 'quality' or"
                " 'speed'"
            )
        self._ibm_quantum_backend = False

        self._backend_list = None

        if self._log_file != "":
            fh = handlers.RotatingFileHandler(
                self._log_file, maxBytes=(1048576 * 5), backupCount=100
            )
            log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(log_format)
            self._logger = logging.getLogger("executor")
            self._logger.addHandler(fh)
            self._logger.setLevel(logging.INFO)
        else:
            self._logger = logging.getLogger("executor")
            self._logger.setLevel(logging.INFO)

        if execution is None and backend is not None:
            # Only backend is given
            execution = backend

        self._quantum_framework = "qiskit"
        self._pennylane_device = None

        if isinstance(execution, str):
            # Execution is a string -> get backend
            if execution in ["qiskit", "statevector_simulator", "aer_simulator_statevector"]:
                execution = "aer_simulator_statevector"
                self._backend = Aer.get_backend(execution)
                if shots is None:
                    self._backend.options.shots = None
            elif execution in ["qasm_simulator", "aer_simulator"]:
                execution = "aer_simulator"
                self._backend = Aer.get_backend(execution)
                shots_backend = self._backend.options.shots
                if shots is None:
                    shots = shots_backend
            elif "ibm" in execution:
                raise ValueError(
                    "IBM backend are not supported by string input, since credentials are missing "
                    + execution
                )
            elif execution in ["pennylane", "default.qubit"]:
                self._quantum_framework = "pennylane"
                self._pennylane_device = qml.device("default.qubit")
                if shots is None:
                    shots = self._pennylane_device.shots.total_shots
            else:
                raise ValueError("Unknown backend string: " + execution)
            self._execution_origin = "Simulator"

        elif isinstance(execution, QubitDevice) or isinstance(execution, PennylaneDevice):
            self._quantum_framework = "pennylane"
            self._pennylane_device = execution

            if self._pennylane_seed is not None:
                if hasattr(self._pennylane_device, "_rng"):
                    self._pennylane_device._rng = np.random.default_rng(self._pennylane_seed)
                if hasattr(self._pennylane_device, "_prng_key"):
                    self._pennylane_device._prng_key = None

            if isinstance(self._pennylane_device.shots, qml.measurements.Shots):
                if len(self._pennylane_device.shots.shot_vector) > 2:
                    raise ValueError("Shot vector in PennyLane device is not supported yet!")
                else:
                    if shots is None:
                        shots = self._pennylane_device.shots.total_shots
            elif isinstance(self._pennylane_device.shots, int):
                if shots is None:
                    shots = self._pennylane_device.shots

        elif isinstance(execution, Backend):
            # Execution is a backend class
            if hasattr(execution, "service"):
                self._service = execution.service
            self._backend = execution
            self._execution_origin = "Backend"
            if shots is None:
                if hasattr(self._backend.options, "shots"):
                    shots = self._backend.options.shots
                    if self.is_statevector:
                        shots = None
        elif isinstance(execution, list):
            # Execution is a list of backends -> backands will be automatically selected
            if all(isinstance(exec, Backend) for exec in execution):
                self._backend = None
                self._backend_list = execution
                self._execution_origin = "BackendList"
                # Execution is a backend class
                if hasattr(execution[0], "service"):
                    self._service = execution[0].service
            else:
                raise ValueError("Only list of backends are supported!")
        elif isinstance(execution, QiskitRuntimeService):
            self._service = execution
            if isinstance(backend, str):
                self._backend = self._service.get_backend(backend)
            elif isinstance(backend, Backend):
                self._backend = backend
            elif isinstance(backend, list):
                self._backend_list = backend
                self._backend = None
            elif backend is None:
                self._backend = None
                self._backend_list = self._service.backends()
            else:
                raise ValueError("Unknown backend type: " + backend)
            if shots is None and self._backend is not None:
                shots = self._backend.options.shots
                if self.is_statevector:
                    shots = None
            self._execution_origin = "QiskitRuntimeService"
        elif isinstance(execution, Session):
            # Execution is a active? session
            self._session = execution
            self._service = self._session.service
            self._backend = self._session.service.get_backend(self._session.backend())
            self._execution_origin = "Session"
            if shots is None:
                shots = self._backend.options.shots
                if self.is_statevector:
                    shots = None
        elif isinstance(execution, BaseEstimatorV1):
            self._estimator = execution
            if isinstance(self._estimator, PrimitiveEstimatorV1):
                # this is only a hack, there is no real backend in the Primitive Estimator class
                self._backend = Aer.get_backend("aer_simulator_statevector")
            elif isinstance(self._estimator, BackendEstimatorV1):
                self._backend = self._estimator._backend
                # TODO: check if this is duplicate
                if not shots:
                    shots_estimator = self._estimator.options.get("shots", 0)
                    if not shots_estimator:
                        shots = 1024
                        self._estimator.set_options(shots=shots)
                    else:
                        shots = shots_estimator
            # Real Backend
            elif isinstance(self._estimator, RuntimeEstimatorV1):
                self._session = self._estimator._session
                self._service = self._estimator._service
                self._backend = self._estimator._backend
                # TODO: check if this is duplicate
                if not shots:
                    shots = self._estimator.options["execution"]["shots"]
            else:
                raise ValueError("Unknown estimator type: " + str(execution))

            # Set options for the estimator
            if self._options_estimator is not None:
                self._estimator.set_options(**self._options_estimator)
            self._execution_origin = "Estimator"
        elif isinstance(execution, BaseSamplerV1):
            self._sampler = execution

            if isinstance(self._sampler, PrimitiveSamplerV1):
                # this is only a hack, there is no real backend in the Primitive Sampler class
                self._backend = Aer.get_backend("aer_simulator_statevector")
            elif isinstance(self._sampler, BackendSamplerV1):
                self._backend = self._sampler._backend
                shots_sampler = self._sampler.options.get("shots", 0)
                # TODO: check if this is duplicate
                if not shots:
                    if not shots_sampler:
                        shots = 1024
                        self._sampler.set_options(shots=shots)
                    else:
                        shots = shots_sampler
            elif isinstance(self._sampler, RuntimeSamplerV1):
                self._session = self._sampler._session
                self._service = self._sampler._service
                self._backend = self._sampler._backend
                # TODO: check if this is duplicate
                if not shots:
                    shots = self._sampler.options["execution"]["shots"]
            else:
                raise ValueError("Unknown sampler type: " + str(execution))

            # Set options for the sampler
            if self._options_sampler is not None:
                self._sampler.set_options(**self._options_sampler)

            self._execution_origin = "Sampler"
        elif isinstance(execution, BaseEstimatorV2):
            self._estimator = execution
            if isinstance(self._estimator, StatevectorEstimator):
                self._backend = Aer.get_backend("aer_simulator_statevector")
                if shots is None and self._estimator.default_precision:
                    shots = int((1.0 / self._estimator.default_precision) ** 2)
                self._estimator._seed = self._set_seed_for_primitive
            elif isinstance(self._estimator, BackendEstimatorV2):
                self._backend = self._estimator.backend
                if shots is None:
                    if self._estimator.options.default_precision <= 0.0:
                        shots = 1024
                        self._estimator._options.default_precision = 1.0 / shots**0.5
                    else:
                        shots = int((1.0 / self._estimator.options.default_precision) ** 2)
                self._estimator._options.seed_simulator = self._set_seed_for_primitive
            elif isinstance(self._estimator, RuntimeEstimatorV2):
                if hasattr(self._estimator, "_session"):
                    self._session = self._estimator._session
                elif hasattr(self._estimator, "_mode"):
                    self._session = self._estimator._mode
                self._service = self._estimator._service
                self._backend = self._estimator._backend
                if shots is None:
                    if self._estimator.options.default_shots:
                        shots = self._estimator.options.default_shots
                    elif self._estimator.options.default_precision:
                        shots = int((1.0 / self._estimator.options.default_precision) ** 2)
                    else:
                        shots = 1024
                        self._estimator.options.default_shots = 1024
                if self._set_seed_for_primitive:
                    self._estimator.options.update(
                        simulator={"seed_simulator": self._set_seed_for_primitive}
                    )
            else:
                raise ValueError("Unknown execution type: " + str(type(execution)))
        elif isinstance(execution, BaseSamplerV2):
            self._sampler = execution
            if isinstance(self._sampler, StatevectorSampler):
                self._backend = Aer.get_backend("aer_simulator_statevector")
                if shots is None:
                    shots = self._sampler.default_shots
                self._sampler._seed = self._set_seed_for_primitive
            elif isinstance(self._sampler, BackendSamplerV2):
                self._backend = self._sampler.backend
                if shots is None:
                    shots = self._sampler.options.default_shots
                self._sampler._options.seed_simulator = self._set_seed_for_primitive
            elif isinstance(self._sampler, RuntimeSamplerV2):
                if hasattr(self._sampler, "_session"):
                    self._session = self._sampler._session
                elif hasattr(self._sampler, "_mode"):
                    self._session = self._sampler._mode
                self._service = self._sampler._service
                self._backend = self._sampler._backend
                if shots is None:
                    if self._sampler.options.default_shots:
                        shots = self._sampler.options.default_shots
                    else:
                        shots = 1024
                        self._sampler.options.default_shots = 1024
                if self._set_seed_for_primitive:
                    self._sampler.options.update(
                        simulator={"seed_simulator": self._set_seed_for_primitive}
                    )
            else:
                raise ValueError("Unknown execution type: " + str(type(execution)))
        else:
            raise ValueError("Unknown execution type: " + str(type(execution)))

        # Check if execution is on a remote backend
        if self.quantum_framework == "qiskit":
            if "ibm" in str(self._backend).lower() or "ibm" in str(self._backend_list).lower():
                # Sort out fake backends
                isfake = (
                    "fake" in str(self._backend).lower()
                    or "fake" in str(self._backend_list).lower()
                )
                self._remote_backend = not isfake
                self._ibm_quantum_backend = not isfake
            else:
                self._ibm_quantum_backend = False
                # Check if backend is a simulator
                self._remote_backend = not any(
                    str(substring) in str(self._backend) for substring in Aer.backends()
                )

            if self._backend_list is None:
                self._backend_list = [self._backend]
            else:
                if not self._ibm_quantum_backend:
                    # If fake backends are given
                    # automatic backend selection is supported
                    if (
                        "fake" not in str(self._backend).lower()
                        and "fake" not in str(self._backend_list).lower()
                    ):
                        raise ValueError(
                            "Automatic backend selection is only supported"
                            + " for IBM Quantum backends or IBM Fake backends!"
                        )

        elif self.quantum_framework == "pennylane":
            if self._backend_list is not None:
                raise ValueError(
                    "Automatic backend selection is only supported for IBM Quantum backends!"
                )
            if self.qpu_parallelization:
                raise ValueError("QPU parallelization is not supported for PennyLane devices!")

            self._remote_backend = not any(
                substring in str(self._pennylane_device)
                for substring in [
                    "default.qubit",
                    "default.mixed",
                    "default.clifford",
                    "Lightning Qubit",
                ]
            )
        else:
            raise RuntimeError("Unknown quantum framework!")

        # set initial shots
        self.set_shots(shots)
        self._inital_num_shots = self.get_shots()

        if self._estimator is not None and self._options_estimator is not None:
            self.set_options_estimator(**self._options_estimator)
        if self._sampler is not None and self._options_sampler is not None:
            self.set_options_sampler(**self._options_sampler)

        if self._caching is None:
            self._caching = self.remote

        if self._caching:
            self._cache = ExecutorCache(self._logger, cache_dir)

        self._logger.info(f"Executor initialized with {{}}".format(self.quantum_framework))
        if self._backend is not None:
            self._logger.info(f"Executor initialized with backend: {{}}".format(self._backend))
        if self._backend_list is not None:
            if len(self._backend_list) > 1:
                self._logger.info(
                    f"Executor initialized with backend list: {{}}".format(self._backend_list)
                )
        if self._service is not None:
            self._logger.info(f"Executor initialized with service: {{}}".format(self._service))
        if self._session is not None:
            self._logger.info(
                f"Executor initialized with session: {{}}".format(self._session.session_id)
            )
        if self._estimator is not None:
            self._logger.info(f"Executor initialized with estimator: {{}}".format(self._estimator))
        if self._sampler is not None:
            self._logger.info(f"Executor initialized with sampler: {{}}".format(self._sampler))
        self._logger.info(f"Executor intial shots: {{}}".format(self._inital_num_shots))

    @property
    def quantum_framework(self) -> str:
        """Return the quantum framework that is used in the executor."""
        return self._quantum_framework

    def pennylane_execute(self, pennylane_circuit: callable, *args, **kwargs):
        """
        Function for executing of PennyLane circuits with the Executor with caching and restarts

        Args:
            pennylane_circuit (callable): The PennyLane circuit function
            args: Arguments for the circuit
            kwargs: Keyword arguments for the circuit

        Returns:
            The result of the circuit
        """
        # Get hash value of the circuit
        if hasattr(pennylane_circuit, "hash"):
            hash_value = [pennylane_circuit.hash, args]
        else:
            hash_value = [hash(pennylane_circuit), args]

        # Helper function for execution
        def execute_circuit():
            return pennylane_circuit(*args, **kwargs)

        # Call function for cached execution
        return self._pennylane_execute_cached(execute_circuit, hash_value)

    def pennylane_execute_batched(
        self, pennylane_circuit: callable, arg_tuples: Union[list, tuple], **kwargs
    ) -> Union[np.array, list]:
        """
        Function for batched execution of PennyLane circuits.

        Args:
            pennylane_circuit (callable): The PennyLane circuit function
            arg_tuples (Union[list,tuple]): List of tuples with arguments for the circuit

        Returns
            Union[np.array,list]: List of results of the circuits
        """
        input_list = True
        if not isinstance(pennylane_circuit, list):
            pennylane_circuit = [pennylane_circuit]
            input_list = False

        if not isinstance(arg_tuples, list):
            arg_tuples = [arg_tuples]
            input_list = False

        if len(pennylane_circuit) != len(arg_tuples):
            raise ValueError("Length of pennylane_circuit and arg_tuples does not match")

        # Build tapes for batched execution and get the hash value of the circuits
        hash_value = ""
        batched_tapes = []
        for i, arg_tuple in enumerate(arg_tuples):
            pennylane_circuit[i].pennylane_circuit.construct(arg_tuple, kwargs)

            if hasattr(pennylane_circuit[i].pennylane_circuit, "hash"):
                hash_value += str(pennylane_circuit[i].pennylane_circuit.hash)
            else:
                hash_value += str(hash(pennylane_circuit[i].pennylane_circuit))

            batched_tapes.append(pennylane_circuit[i].pennylane_circuit.tape)

        hash_value = [hash_value, arg_tuples]

        # Helper function for execution
        def execute_tapes():
            return qml.execute(batched_tapes, self.backend)

        # Call function for cached execution
        if input_list:
            return self._pennylane_execute_cached(execute_tapes, hash_value)
        else:
            return self._pennylane_execute_cached(execute_tapes, hash_value)[0]

    def _pennylane_execute_cached(self, function: callable, hash_value: Union[str, int]):
        """
        Function for cached execution of PennyLane circuits with the Executor

        Args:
            function (callable): The function that is executed
            hash_value (Union[str,int]): Hash value for the caching

        Returns:
            The result of the circuit
        """
        success = False
        critical_error = False
        critical_error_message = None
        for repeat in range(self._max_jobs_retries):

            try:
                result = None
                cached = False
                if self._caching:

                    # Generate hash value for caching
                    hash_value_adjusted = self._cache.hash_variable(
                        [
                            "pennylane_execute",
                            hash_value,
                            self._pennylane_device.name,
                            self.shots,
                        ]
                    )

                    result = self._cache.get_file(hash_value_adjusted)
                    cached = True
                else:
                    hash_value_adjusted = None

                if result is None:
                    cached = False
                    if self._caching:
                        self._logger.info(
                            f"Execution of pennylane circuit function with hash value: {{}}".format(
                                hash_value_adjusted
                            )
                        )
                    else:
                        self._logger.info(f"Execution of pennylane circuit function")
                    # Execution of pennylane circuit function
                    result = function()
                    self._logger.info(f"Execution of pennylane circuit successful")
                elif self._caching:
                    self._logger.info(
                        f"Cached result found with hash value: {{}}".format(hash_value_adjusted)
                    )

                success = True

            except (
                NotImplementedError,
                RuntimeError,
                ValueError,
                NotImplementedError,
                TypeError,
                qml.numpy.NonDifferentiableError,
            ) as e:
                critical_error = True
                critical_error_message = e

            except Exception as e:
                if repeat == self._max_jobs_retries - 1:
                    critical_error = True
                    critical_error_message = e
                else:
                    self._logger.info(
                        f"Executor failed to run pennylane_execute because of unknown error!"
                    )
                    self._logger.info("Error message: {}".format(str(e)))
                    self._logger.info("Traceback: {}".format(str(traceback.format_exc())))
                    print("Executor failed to run pennylane_execute because of unknown error!")
                    print("Error message: {}".format(str(e)))
                    print("Traceback: {}".format(str(traceback.format_exc())))
                    print("Execution will be restarted")
                    success = False

            if success:
                break
            elif not critical_error:
                self._logger.info(f"Restarting PennyLane execution")
                success = False

            if critical_error:
                self._logger.info(f"Critical error detected; abort execution")
                raise critical_error_message

        if success is not True:
            raise RuntimeError(
                f"Could not run job successfully after {{}} retries".format(self._max_jobs_retries)
            )

        if self._caching and not cached:
            self._cache.store_file(hash_value_adjusted, copy.copy(result))

        return result

    @property
    def execution(self) -> str:
        """Returns a string of the execution that is used to initialize the executor class."""
        return self._execution_origin

    @property
    def backend(self) -> Union[Backend, None, PennylaneDevice]:
        """Returns the backend that is used in the executor."""

        if self.quantum_framework == "qiskit":
            return self._backend
        elif self.quantum_framework == "pennylane":
            return self._pennylane_device
        else:
            raise RuntimeError("Unknown quantum framework!")

    @property
    def remote(self) -> bool:
        """Returns a boolean if the execution is on a remote backend."""
        return self._remote_backend

    @property
    def IBMQuantum(self) -> bool:
        """Returns a boolean if the execution is on a IBM Quantum backend."""
        return self._ibm_quantum_backend

    @property
    def backend_list(self) -> List[Backend]:
        """Returns the backend list that is used in the executor."""
        return self._backend_list

    @property
    def backend_chosen(self) -> bool:
        """Returns true if the backend has been chosen."""
        if self.backend is None:
            return False
        else:
            return True

    @property
    def qpu_parallelization(self) -> bool:
        """Returns true if QPU parallelization is used."""
        return self._qpu_parallelization is not None

    @property
    def session(self) -> Session:
        """Returns the session that is used in the executor."""
        return self._session

    @property
    def service(self) -> QiskitRuntimeService:
        """Returns the service that is used in the executor."""
        return self._service

    def _estimator_v1(self) -> BaseEstimatorV1:
        """Returns the Estimator V1 primitive that is used for the execution.

        This function created automatically estimators and checks for an expired session and
        creates a new one if necessary.
        Note that the run function is the same as in the Qiskit primitives, and
        does not support caching and restarts
        For this use :meth:`estimator_v1_run` or :meth:`get_estimator`.

        The estimator that is created depends on the backend that is used for the execution.
        """

        if self._estimator is not None:
            if self.IBMQuantum and self._session is not None and not self._session._active:
                # Session is expired, create a new session and a new estimator
                self.create_session()
                self._estimator = RuntimeEstimatorV1(
                    session=self._session, options=self._options_estimator
                )
            estimator = self._estimator
            initialize_parallel_estimator = not isinstance(estimator, ParallelEstimatorV1)
        else:
            # Create a new Estimator
            shots = self.get_shots()
            initialize_parallel_estimator = True
            if self.IBMQuantum:
                if self._session is not None:
                    if not self._session._active:
                        self.create_session()
                    self._estimator = RuntimeEstimatorV1(
                        session=self._session, options=self._options_estimator
                    )
                elif self._service is not None:
                    # No session but service -> create a new session
                    self.create_session()
                    self._estimator = RuntimeEstimatorV1(
                        session=self._session, options=self._options_estimator
                    )
                else:
                    raise RuntimeError(
                        "Missing Qiskit Runtime service for Estimator initialization!"
                    )
            else:
                if self.is_statevector:
                    # No session, no service, but state_vector simulator -> Estimator
                    self._estimator = PrimitiveEstimatorV1(options=self._options_estimator)
                    self._estimator.set_options(shots=self._shots)
                elif self._backend is None:
                    raise RuntimeError("Backend missing for Estimator initialization!")
                else:
                    # No session, no service and no state_vector simulator -> BackendEstimator
                    self._estimator = BackendEstimatorV1(
                        backend=self._backend, options=self._options_estimator
                    )
                    if shots is None:
                        shots = 1024

            if not self._options_estimator:
                self.set_shots(shots)

        # Generate a in-QPU parallelized estimator
        if self._qpu_parallelization is not None:
            if initialize_parallel_estimator:
                if isinstance(self._qpu_parallelization, str):
                    if self._qpu_parallelization == "auto":
                        self._estimator = ParallelEstimator(self._estimator, num_parallel=None)
                    else:
                        raise ValueError(
                            "Unknown qpu_parallelization value: " + self._qpu_parallelization
                        )
                elif isinstance(self._qpu_parallelization, int):
                    self._estimator = ParallelEstimator(
                        self._estimator, num_parallel=self._qpu_parallelization
                    )
                else:
                    raise TypeError(
                        "Unknown qpu_parallelization type: " + type(self._qpu_parallelization)
                    )

        estimator = self._estimator

        return estimator

    def _estimator_v2(self) -> BaseEstimatorV2:
        """Returns the Estimator V2 primitive that is used for the execution.

        This function created automatically estimators and checks for an expired session and
        creates a new one if necessary.
        Note that the run function is the same as in the Qiskit primitives, and
        does not support caching and restarts
        For this use :meth:`estimator_v2_run` or :meth:`get_estimator`.

        The estimator that is created depends on the backend that is used for the execution.
        """

        if self._estimator is not None:
            # Store already exisiting options
            if isinstance(self._estimator, RuntimeEstimatorV2):
                self._options_estimator = _convert_options_to_dict(self._estimator.options)
            if self.IBMQuantum and self._session is not None and not self._session._active:
                # Session is expired, create a new session and a new estimator
                self.create_session()
                if QISKIT_RUNTIME_SMALLER_0_28:
                    self._estimator = RuntimeEstimatorV2(
                        session=self._session, options=self._options_estimator
                    )
                else:
                    self._estimator = RuntimeEstimatorV2(
                        mode=self._session, options=self._options_estimator
                    )
            estimator = self._estimator
            initialize_parallel_estimator = not isinstance(estimator, ParallelEstimatorV2)
        else:
            # Create a new Estimator
            shots = self.get_shots()
            initialize_parallel_estimator = True
            if self.IBMQuantum:
                if shots:
                    if not self._options_estimator:
                        self._options_estimator = {"default_shots": shots}
                    else:
                        self._options_estimator["default_shots"] = shots
                if self._session is not None:
                    if not self._session._active:
                        self.create_session()
                    if QISKIT_RUNTIME_SMALLER_0_23:
                        self._estimator = RuntimeEstimatorV2(
                            session=self._session, options=self._options_estimator
                        )
                    else:
                        self._estimator = RuntimeEstimatorV2(
                            mode=self._session, options=self._options_estimator
                        )
                elif self._service is not None:
                    # No session but service -> create a new session
                    self.create_session()
                    if QISKIT_RUNTIME_SMALLER_0_23:
                        self._estimator = RuntimeEstimatorV2(
                            session=self._session, options=self._options_estimator
                        )
                    else:
                        self._estimator = RuntimeEstimatorV2(
                            mode=self._session, options=self._options_estimator
                        )
                else:
                    raise RuntimeError(
                        "Missing Qiskit Runtime service for Estimator initialization!"
                    )
            else:
                if "fake" in str(self._backend):
                    if shots:
                        if not self._options_estimator:
                            self._options_estimator = {"default_shots": shots}
                        else:
                            self._options_estimator["default_shots"] = shots
                    if QISKIT_RUNTIME_SMALLER_0_23:
                        self._estimator = RuntimeEstimatorV2(
                            backend=self._backend, options=self._options_estimator
                        )
                    else:
                        self._estimator = RuntimeEstimatorV2(
                            mode=self._backend, options=self._options_estimator
                        )
                elif self.is_statevector:
                    # No session, no service, but state_vector simulator -> Estimator
                    self._estimator = StatevectorEstimator(
                        default_precision=1 / shots**0.5 if shots else 0.0
                    )
                elif self._backend is None:
                    raise RuntimeError("Backend missing for Estimator initialization!")
                else:
                    if shots:
                        if not self._options_estimator:
                            self._options_estimator = {"default_precision": 1 / shots**0.5}
                        else:
                            self._options_estimator["default_precision"] = 1 / shots**0.5
                    # No session, no service and no state_vector simulator -> BackendEstimator
                    self._estimator = BackendEstimatorV2(
                        backend=self._backend, options=self._options_estimator
                    )
                    if shots is None:
                        shots = 1024

            if not self._options_estimator:
                self.set_shots(shots)

        # Generate a in-QPU parallelized estimator
        if self._qpu_parallelization and initialize_parallel_estimator:
            if isinstance(self._qpu_parallelization, str):
                if self._qpu_parallelization == "auto":
                    self._estimator = ParallelEstimator(self._estimator, num_parallel=None)
                else:
                    raise ValueError(
                        "Unknown qpu_parallelization value: " + self._qpu_parallelization
                    )
            elif isinstance(self._qpu_parallelization, int):
                self._estimator = ParallelEstimator(
                    self._estimator, num_parallel=self._qpu_parallelization
                )
            else:
                raise TypeError(
                    "Unknown qpu_parallelization type: " + type(self._qpu_parallelization)
                )

        estimator = self._estimator

        return estimator

    @property
    def estimator(self) -> Union[BaseEstimatorV1, BaseEstimatorV2]:
        """Returns the estimator primitive that is used for the execution.

        For Qiskit >= 1.2 the Estimator V2 is used, for Qiskit < 1.2 the Estimator V1 is returned.
        """

        if self.quantum_framework != "qiskit":
            raise RuntimeError("Estimator is only available for Qiskit backends")

        if self._estimator is not None:
            if isinstance(self._estimator, BaseEstimatorV1):
                return self._estimator_v1()
            return self._estimator_v2()

        if QISKIT_SMALLER_1_2 or "Braket" in str(self._backend):
            return self._estimator_v1()

        return self._estimator_v2()

    def clear_estimator_cache(self) -> None:
        """Function for clearing the cache of the EstimatorV1 primitive to avoid memory overflow."""
        if self._estimator is not None and (
            isinstance(self._estimator, PrimitiveEstimatorV1)
            or isinstance(self._estimator, BackendEstimatorV1)
        ):
            self._estimator._circuits = []
            self._estimator._observables = []
            self._estimator._parameters = []
            self._estimator._circuit_ids = {}
            self._estimator._observable_ids = {}

    def _sampler_v1(self) -> BaseSamplerV1:
        """Returns the Sampler V1 primitive that is used for the execution.

        This function created automatically estimators and checks for an expired session and
        creates a new one if necessary.

        Note that the run function is the same as in the Qiskit primitives, and
        does not support caching, session handing, etc.
        For this use :meth:`sampler_run_v1` or :meth:`get_sampler`.

        The sampler that is created depends on the backend that is used for the execution.
        """

        if self._sampler is not None:
            if self.IBMQuantum and self._session is not None and not self._session._active:
                # Session is expired, create a new one and a new estimator
                self.create_session()
                self._sampler = RuntimeSamplerV1(
                    session=self._session, options=self._options_sampler
                )
            sampler = self._sampler
            initialize_parallel_sampler = not isinstance(sampler, ParallelSamplerV1)
        else:
            # Create a new Sampler
            shots = self.get_shots()
            initialize_parallel_sampler = True

            if self.IBMQuantum:
                if self._session is not None:
                    if not self._session._active:
                        self.create_session()
                    self._sampler = RuntimeSamplerV1(
                        session=self._session, options=self._options_sampler
                    )

                elif self._service is not None:
                    # No session but service -> create a new session
                    self.create_session()
                    self._sampler = RuntimeSamplerV1(
                        session=self._session,
                        options=self._options_sampler,
                    )
                else:
                    raise RuntimeError(
                        "Missing Qiskit Runtime service for Sampler initialization!"
                    )
            else:
                if self.is_statevector:
                    # No session, no service, but state_vector simulator -> Sampler
                    self._sampler = PrimitiveSamplerV1(options=self._options_sampler)
                    self._sampler.set_options(shots=self._shots)
                elif self._backend is None:
                    raise RuntimeError("Backend missing for Sampler initialization!")
                else:
                    # No session, no service and no state_vector simulator -> BackendSampler
                    self._sampler = BackendSamplerV1(
                        backend=self._backend, options=self._options_sampler
                    )
                    if shots is None:
                        shots = 1024

            if not self._options_sampler:
                self.set_shots(shots)

        # Generate a in-QPU parallelized sampler
        if self._qpu_parallelization is not None:
            if initialize_parallel_sampler:
                if isinstance(self._qpu_parallelization, str):
                    if self._qpu_parallelization == "auto":
                        self._sampler = ParallelSampler(self._sampler, num_parallel=None)
                    else:
                        raise ValueError(
                            "Unknown qpu_parallelization value: " + self._qpu_parallelization
                        )
                elif isinstance(self._qpu_parallelization, int):
                    self._sampler = ParallelSampler(
                        self._sampler, num_parallel=self._qpu_parallelization
                    )
                else:
                    raise TypeError(
                        "Unknown qpu_parallelization type: " + type(self._qpu_parallelization)
                    )

        sampler = self._sampler

        return sampler

    def _sampler_v2(self) -> BaseSamplerV2:
        """Returns the Sampler V2 primitive that is used for the execution.

        This function created automatically estimators and checks for an expired session and
        creates a new one if necessary.

        Note that the run function is the same as in the Qiskit primitives, and
        does not support caching, session handing, etc.
        For this use :meth:`sampler_run_v2` or :meth:`get_sampler`.

        The sampler that is created depends on the backend that is used for the execution.
        """

        if self._sampler is not None:
            # Store already exisiting options
            if isinstance(self._sampler, RuntimeSamplerV2):
                self._options_sampler = _convert_options_to_dict(self._sampler.options)
            if self.IBMQuantum and self._session is not None and not self._session._active:
                # Session is expired, create a new one and a new estimator
                self.create_session()
                self._sampler = RuntimeSamplerV2(mode=self._session, options=self._options_sampler)
            sampler = self._sampler
            initialize_parallel_sampler = not isinstance(sampler, ParallelSamplerV2)
        else:
            # Create a new Sampler
            shots = self.get_shots()
            if shots:
                if not self._options_sampler:
                    self._options_sampler = {"default_shots": shots}
                else:
                    self._options_sampler["default_shots"] = shots
            initialize_parallel_sampler = True

            if self.IBMQuantum:
                if self._session is not None:
                    if not self._session._active:
                        self.create_session()
                    if QISKIT_RUNTIME_SMALLER_0_23:
                        self._sampler = RuntimeSamplerV2(
                            session=self._session, options=self._options_sampler
                        )
                    else:
                        self._sampler = RuntimeSamplerV2(
                            mode=self._session, options=self._options_sampler
                        )

                elif self._service is not None:
                    # No session but service -> create a new session
                    self.create_session()
                    if QISKIT_RUNTIME_SMALLER_0_23:
                        self._sampler = RuntimeSamplerV2(
                            session=self._session,
                            options=self._options_sampler,
                        )
                    else:
                        self._sampler = RuntimeSamplerV2(
                            mode=self._session,
                            options=self._options_sampler,
                        )

                else:
                    raise RuntimeError(
                        "Missing Qiskit Runtime service for Sampler initialization!"
                    )
            else:
                if "fake" in str(self._backend).lower():
                    if QISKIT_RUNTIME_SMALLER_0_23:
                        self._sampler = RuntimeSamplerV2(
                            backend=self._backend, options=self._options_sampler
                        )
                    else:
                        self._sampler = RuntimeSamplerV2(
                            mode=self._backend, options=self._options_sampler
                        )
                elif self.is_statevector:
                    # No session, no service, but state_vector simulator -> Sampler
                    if shots:
                        self._sampler = StatevectorSampler(default_shots=shots)
                    else:
                        self._sampler = StatevectorSampler()
                        shots = self._sampler.default_shots
                elif self._backend is None:
                    raise RuntimeError("Backend missing for Sampler initialization!")
                else:
                    # No session, no service and no state_vector simulator -> BackendSampler
                    self._sampler = BackendSamplerV2(
                        backend=self._backend, options=self._options_sampler
                    )
                    if shots is None:
                        shots = 1024

            if not self._options_sampler:
                self.set_shots(shots)

        # Generate a in-QPU parallelized sampler
        if self._qpu_parallelization is not None:
            if initialize_parallel_sampler:
                if isinstance(self._qpu_parallelization, str):
                    if self._qpu_parallelization == "auto":
                        self._sampler = ParallelSampler(self._sampler, num_parallel=None)
                    else:
                        raise ValueError(
                            "Unknown qpu_parallelization value: " + self._qpu_parallelization
                        )
                elif isinstance(self._qpu_parallelization, int):
                    self._sampler = ParallelSampler(
                        self._sampler, num_parallel=self._qpu_parallelization
                    )
                else:
                    raise TypeError(
                        "Unknown qpu_parallelization type: " + type(self._qpu_parallelization)
                    )

        sampler = self._sampler

        return sampler

    @property
    def sampler(self) -> Union[BaseSamplerV1, BaseSamplerV2]:
        """Returns the sampler primitive that is used for the execution.

        For Qiskit >= 1.2 the Sampler V2 is used, for Qiskit < 1.2 the Sampler V1 is returned.
        """

        if self.quantum_framework != "qiskit":
            raise RuntimeError("Estimator is only available for Qiskit backends")

        if self._sampler is not None:
            if isinstance(self._sampler, BaseSamplerV1):
                return self._sampler_v1()
            return self._sampler_v2()

        if QISKIT_SMALLER_1_2 or "Braket" in str(self._backend):
            return self._sampler_v1()

        return self._sampler_v2()

    def clear_sampler_cache(self) -> None:
        """Function for clearing the cache of the SamplerV1 primitive to avoid memory overflow."""
        if self._sampler is not None and (
            isinstance(self._sampler, PrimitiveSamplerV1)
            or isinstance(self._sampler, BackendSamplerV1)
        ):
            self._sampler._circuits = []
            self._sampler._parameters = []
            self._sampler._circuit_ids = {}
            self._sampler._qargs_list = []

    def _primitive_run(
        self, run: callable, label: str, hash_value: Union[str, None] = None
    ) -> JobV1:
        """Run function that allow restarting, session handling and caching.

        Parent implementation that is used for both, Estimator and Sampler.

        Args:
            run (callable): Run function of the primitive
            label (str): Label that is used for logging.
            hash_value (str,None): Hash value that is used for caching.

        Returns:
            A qiskit job containing the results of the run.
        """
        success = False
        critical_error = False
        critical_error_message = None

        if "v2" in label and self.IBMQuantum:
            final_states = ("DONE", "CANCELLED", "ERROR")
        else:
            final_states = JOB_FINAL_STATES

        for repeat in range(self._max_jobs_retries):
            try:
                job = None
                cached = False
                if hash_value is not None and self._caching:
                    # TODO: except cache errors
                    job = self._cache.get_file(hash_value)

                if job is None:
                    # TODO: try and except errors
                    job = run()
                    self._logger.info(
                        f"Executor runs " + label + f" with job: {{}}".format(job.job_id())
                    )
                else:
                    self._logger.info(f"Cached job found with hash value: {{}}".format(hash_value))
                    cached = True

            except IBMRuntimeError as e:
                if '"code":1217' in e.message:
                    self._logger.info(
                        f"Executor failed to run "
                        + label
                        + f" because the session has been closed!"
                    )
                    continue

            except NotImplementedError as e:
                critical_error = True
                critical_error_message = e

            except QiskitError as e:
                critical_error = True
                critical_error_message = e

            except Exception as e:
                critical_error = True
                critical_error_message = e
                self._logger.info(
                    f"Executor failed to run " + label + f" because of unknown error!"
                )
                self._logger.info(f"Error message: {{}}".format(e))
                self._logger.info(f"Traceback: {{}}".format(traceback.format_exc()))

            # Wait for the job to complete
            if job is None:
                if "simulator" in self.backend_name:
                    critical_error = True
                    critical_error_message = RuntimeError("Failed to execute job on simulator!")
            else:
                if not cached:
                    status = JobStatus.QUEUED
                    last_status = None
                else:
                    status = JobStatus.DONE
                    if "v2" in label and self.IBMQuantum:
                        last_status = "DONE"
                while status not in final_states:
                    try:
                        status = job.status()
                        if status != last_status:
                            self._logger.info(f"Job status: {{}}".format(status))
                        last_status = status
                    except Exception as e:
                        self._logger.info(
                            f"Executor failed to get job status because of unknown error!"
                        )
                        self._logger.info(f"Error message: {{}}".format(e))
                        self._logger.info(f"Traceback: {{}}".format(traceback.format_exc()))
                        break

                    if self.remote:
                        time.sleep(1)
                    else:
                        time.sleep(0.01)

                # Job is completed, check if it was successful
                if status == JobStatus.ERROR or status == "ERROR":
                    self._logger.info(f"Failed executation of the job!")
                    if hasattr(job, "error_message"):
                        self._logger.info(f"Error message: {{}}".format(job.error_message()))
                    try:
                        job.result()
                    except Exception as e2:
                        critical_error = True
                        critical_error_message = e2
                elif status == JobStatus.CANCELLED:
                    self._logger.info(f"Job has been manually cancelled, and is resubmitted!")
                    self._logger.info(
                        f"To stop resubmitting the job, cancel the execution script first."
                    )
                else:
                    success = True
                    result_success = False
                    for retry_result in range(3):
                        # check if result is available
                        try:
                            result = job.result()
                            result_success = True
                        except RuntimeJobFailureError as e:
                            self._logger.info(f"Executor unable to retriev job result!")
                            self._logger.info(f"Error message: {{}}".format(e))
                        except Exception as e:
                            self._logger.info(
                                f"Executor failed to get job result because of unknown error!"
                            )
                            self._logger.info(f"Error message: {{}}".format(e))
                            self._logger.info(f"Traceback: {{}}".format(traceback.format_exc()))
                        if result_success:
                            break
                        else:
                            self._logger.info(f"Retrying to get job result")
                            time.sleep(self._wait_restart)

            if success and result_success:
                break
            elif not critical_error:
                self._logger.info(f"Restarting " + label + f" run")
                success = False
                result_success = False

            if critical_error:
                self._logger.info(f"Critical error detected; abort execution")
                self._logger.info(f"Error message: {{}}".format(critical_error_message))
                raise critical_error_message

        if success is not True:
            raise RuntimeError(
                f"Could not run job successfully after {{}} retries".format(self._max_jobs_retries)
            )

        if self._caching and not cached:
            job_pickle = copy.copy(job)
            # remove _future and _function from job since this creates massive file sizes
            # and the information is not really needed.
            job_pickle._future = None
            job_pickle._function = None
            job_pickle._api_client = None
            job_pickle._service = None
            job_pickle._ws_client_future = None
            job_pickle._ws_client = None
            try:
                job_pickle._backend = str(job.backend())
            except (QiskitError, AttributeError):
                job_pickle._backend = str(self.backend)

            if "v2" in label:
                # Modify the result function for V2 primitives
                # to be able to pickle the result
                job_pickle.pubs_data = [r.data.__dict__ for r in result]
                job_pickle.pubs_metadata = [r.metadata for r in result]
                job_pickle.primitive_result_metadata = result.metadata
                from qiskit.primitives.containers import (
                    DataBin,
                    PrimitiveResult,
                    SamplerPubResult,
                    PubResult,
                )

                result_type = None
                if "sampler" in label:
                    result_type = SamplerPubResult
                elif "estimator" in label:
                    result_type = PubResult
                else:
                    raise RuntimeError("Unknown result type: " + label)

                def result_(self):
                    return PrimitiveResult(
                        [
                            result_type(DataBin(**data), metadata)
                            for data, metadata in zip(self.pubs_data, self.pubs_metadata)
                        ],
                        self.primitive_result_metadata,
                    )

                job_pickle.result = MethodType(result_, job_pickle)

            else:
                # overwrite result function with the obtained result
                def result_():
                    return result

                job_pickle.result = result_

            self._cache.store_file(hash_value, job_pickle)
            self._logger.info(f"Stored job in cache with hash value: {{}}".format(hash_value))

        return job

    def estimator_run_v1(
        self, circuits, observables, parameter_values=None, **kwargs: Any
    ) -> JobV1:
        """
        Function similar to the Qiskit Estimator V1 run function, but this one includes caching,
        automatic session handling, and restarts of failed jobs.

        Args:
            circuits: Quantum circuits to execute.
            observables: Observable to measure.
            parameter_values: Values for the parameters in circuits.
            kwargs (Any): Additional arguments that are passed to the estimator.

        Returns:
            A qiskit job containing the results of the run.
        """

        # Checks and handles in-circuit measurements in the circuit
        containes_incircuit_measurement = False
        if isinstance(circuits, QuantumCircuit):
            containes_incircuit_measurement = check_for_incircuit_measurements(
                circuits, mode="clbits"
            )
        else:
            for circuit in circuits:
                containes_incircuit_measurement = (
                    containes_incircuit_measurement
                    or check_for_incircuit_measurements(circuit, mode="clbits")
                )

        if containes_incircuit_measurement:
            if self.shots is None:
                raise ValueError(
                    "In-circuit measurements with the Estimator are only possible with shots."
                )
            else:
                if self.is_statevector:
                    self._switch_to_backend_primitive("estimator_v1")

        # Set seed for the primitive
        instance_estimator = self.estimator
        if isinstance(instance_estimator, BaseEstimatorV2):
            raise RuntimeError("Estimator is a BaseEstimatorV2, please use estimator_run_v2.")

        if isinstance(instance_estimator, ParallelEstimatorV1):
            instance_estimator = instance_estimator._estimator
        if isinstance(instance_estimator, BackendEstimatorV1):
            if self._set_seed_for_primitive is not None:
                kwargs["seed_simulator"] = self._set_seed_for_primitive
                self._set_seed_for_primitive += 1
        elif isinstance(instance_estimator, PrimitiveEstimatorV1):
            if self._set_seed_for_primitive is not None:
                self._estimator.set_options(seed=self._set_seed_for_primitive)
                self._set_seed_for_primitive += 1

        def run():
            return self.estimator.run(circuits, observables, parameter_values, **kwargs)

        if self._caching:
            # Generate hash value for caching
            hash_value = self._cache.hash_variable(
                [
                    "estimator",
                    circuits,
                    observables,
                    parameter_values,
                    kwargs,
                    self._options_estimator,
                    self._backend,
                    self.get_shots(),
                ]
            )
        else:
            hash_value = None

        return self._primitive_run(run, "estimator", hash_value)

    def _switch_to_backend_primitive(self, primitive: str):
        """Helperfunction for swapping to the BackendPrimitive for the Executor.

        Args:
            primitive (str): The primitive to swap to. Either "estimator" or "sampler"
        """
        if self.is_statevector and self._shots is not None:

            if primitive == "estimator_v1":
                self._estimator = BackendEstimatorV1(
                    backend=self._backend, options=self._options_estimator
                )
                self._logger.info(
                    "Changed from the EstimatorV1() to the BackendEstimatorV1() primitive"
                )

            elif primitive == "sampler_v1":
                self._sampler = BackendSamplerV1(
                    backend=self._backend, options=self._options_sampler
                )
                self._logger.info(
                    "Changed from the SamplerV1() to the BackendSamplerV1() " "primitive"
                )

            elif primitive == "estimator_v2":
                self._estimator = BackendEstimatorV2(
                    backend=self._backend, options=self._options_estimator
                )
                self._logger.info(
                    "Changed from the StatevectorEstimator() to the BackendEstimatorV2() primitive"
                )

            elif primitive == "sampler_v2":
                self._sampler = BackendSamplerV2(
                    backend=self._backend, options=self._options_sampler
                )
                self._logger.info(
                    "Changed from the StatevectorSampler() to the BackendSamplerV2()" " primitive"
                )
            else:
                raise ValueError("Unknown primitive type: " + primitive)

            self.set_shots(self._shots)

        else:
            raise ValueError(
                "Swapping to BackendPrimitive is only possible for "
                + "statevector simulator with shots"
            )

    def estimator_run_v2(
        self, pubs: Iterable[EstimatorPubLike], precision: Union[float, None] = None
    ):
        """
        Function similar to the Qiskit Estimator V2 run function, but this one includes caching,
        automatic session handling, and restarts of failed jobs.

        Args:
            pubs (Iterable[EstimatorPubLike]): An iterable of pub-like objects, such as
                tuples ``(circuit, observables)`` or ``(circuit, observables, parameter_values)``.
            precision (Union[float, None]): The target precision for expectation value estimates
                of each run Estimator Pub that does not specify its own precision. If None
                the the precision is set by the executor number of shots.

        Returns:
            A qiskit job containing the results of the run.
        """

        pubs = [EstimatorPub.coerce(pub, precision=precision) for pub in pubs]

        # Checks and handles in-circuit measurements in the circuit
        containes_incircuit_measurement = False
        for pub in pubs:
            containes_incircuit_measurement = (
                containes_incircuit_measurement
                or check_for_incircuit_measurements(pub.circuit, mode="clbits")
            )

        if containes_incircuit_measurement:
            if self.shots is None:
                raise ValueError(
                    "In-circuit measurements with the Estimator are only possible with shots."
                )
            else:
                if self.is_statevector:
                    self._switch_to_backend_primitive("estimator_v2")

        # Set seed for the primitive
        instance_estimator = self.estimator
        if isinstance(instance_estimator, BaseEstimatorV1):
            raise RuntimeError("Estimator is a BaseEstimatorV1, please use estimator_run_v1.")

        if isinstance(instance_estimator, ParallelEstimatorV2):
            instance_estimator = instance_estimator._estimator
        if self._set_seed_for_primitive is not None:
            if isinstance(instance_estimator, StatevectorEstimator):
                instance_estimator._seed = self._set_seed_for_primitive
                self._set_seed_for_primitive += 1
            elif isinstance(instance_estimator, BackendEstimatorV2):
                instance_estimator._options.seed_simulator = self._set_seed_for_primitive
                self._set_seed_for_primitive += 1
            elif isinstance(instance_estimator, RuntimeEstimatorV2):
                instance_estimator.options.update(
                    simulator={"seed_simulator": self._set_seed_for_primitive}
                )
                self._set_seed_for_primitive += 1

        if precision is None:
            if self._shots is None or self._shots == 0:
                precision = 0.0
            else:
                precision = 1.0 / self._shots**0.5

        if self._caching:
            # Generate hash value for caching
            hash_value = self._cache.hash_variable(
                ["estimator_v2", pubs, self._options_estimator, self._backend, self.get_shots()]
            )
        else:
            hash_value = None

        def run():
            return self.estimator.run(pubs=pubs, precision=precision)

        return self._primitive_run(run, "estimator_v2", hash_value)

    def sampler_run_v1(self, circuits, parameter_values=None, **kwargs: Any) -> JobV1:
        """
        Function similar to the Qiskit Sampler V1 run function, but this one includes caching,
        automatic session handling, and restarts of failed jobs.

        Args:
            circuits: Quantum circuits to execute.
            parameter_values: Values for the parameters in circuits.
            kwargs (Any): Additional arguments that are passed to the estimator.

        Returns:
            A qiskit job containing the results of the run.
        """

        # Check and handle conditions in the circuit
        circuits_contains_conditions = False
        if isinstance(circuits, QuantumCircuit):
            circuits_contains_conditions = check_for_incircuit_measurements(
                circuits, mode="condition"
            )
        else:
            for circuit in circuits:
                circuits_contains_conditions = (
                    circuits_contains_conditions
                    or check_for_incircuit_measurements(circuit, mode="condition")
                )
        if circuits_contains_conditions:
            if self.shots is None:
                raise ValueError("Conditioned gates on the Sampler are only possible with shots!")
            else:
                if self.is_statevector:
                    self._switch_to_backend_primitive("sampler_v1")

        # Set seed for the primitive
        instance_sampler = self.sampler
        if isinstance(instance_sampler, BaseSamplerV2):
            raise RuntimeError("Sampler is a BaseSamplerV2, please use sampler_run_v2.")

        if isinstance(instance_sampler, ParallelSamplerV1):
            instance_sampler = instance_sampler._sampler
        if isinstance(instance_sampler, BackendSamplerV1):
            if self._set_seed_for_primitive is not None:
                kwargs["seed_simulator"] = self._set_seed_for_primitive
                self._set_seed_for_primitive += 1
        elif isinstance(instance_sampler, PrimitiveSamplerV1):
            if self._set_seed_for_primitive is not None:
                self.sampler.set_options(seed=self._set_seed_for_primitive)
                self._set_seed_for_primitive += 1

        def run():
            return self.sampler.run(circuits, parameter_values, **kwargs)

        if self._caching:
            # Generate hash value for caching
            hash_value = self._cache.hash_variable(
                [
                    "sampler",
                    circuits,
                    parameter_values,
                    kwargs,
                    self._options_sampler,
                    self._backend,
                    self.get_shots(),
                ]
            )
        else:
            hash_value = None

        return self._primitive_run(run, "sampler", hash_value)

    def sampler_run_v2(self, pubs: Iterable[SamplerPubLike], *, shots: Union[int, None] = None):
        """
        Function similar to the Qiskit Sampler V2 run function, but this one includes caching,
        automatic session handling, and restarts of failed jobs.

        Args:
            pubs (Iterable[EstimatorPubLike]): An iterable of pub-like objects, such as
                tuples ``(circuit,)`` or ``(circuit, parameter_values)``.
            shots (Union[int, None]): The number of shots used for the sampling. If None
                the Executors numer of shot will be used.

        Returns:
            A qiskit job containing the results of the run.
        """

        # Check and handle conditions in the circuit

        pubs = [SamplerPub.coerce(pub, shots=shots) for pub in pubs]

        # Checks and handles in-circuit measurements in the circuit
        circuits_contains_conditions = False
        for pub in pubs:
            circuits_contains_conditions = (
                circuits_contains_conditions
                or check_for_incircuit_measurements(pub.circuit, mode="condition")
            )

        if circuits_contains_conditions:
            if self.shots is None and shots is None:
                raise ValueError("Conditioned gates on the Sampler are only possible with shots!")
            if self.is_statevector:
                self._switch_to_backend_primitive("sampler_v2")

        # Set seed for the primitive
        instance_sampler = self.sampler
        if isinstance(instance_sampler, BaseSamplerV1):
            raise RuntimeError("Sampler is a BaseSamplerV1, please use sampler_run_v1.")

        if isinstance(instance_sampler, ParallelSamplerV2):
            instance_sampler = instance_sampler._sampler
        if self._set_seed_for_primitive is not None:
            if isinstance(instance_sampler, StatevectorSampler):
                instance_sampler._seed = self._set_seed_for_primitive
                self._set_seed_for_primitive += 1
            elif isinstance(instance_sampler, BackendSamplerV2):
                instance_sampler._options.seed_simulator = self._set_seed_for_primitive
                self._set_seed_for_primitive += 1
            elif isinstance(instance_sampler, RuntimeSamplerV2):
                instance_sampler._options.update(
                    simulator={"seed_simulator": self._set_seed_for_primitive}
                )
                self._set_seed_for_primitive += 1

        if shots is None:
            shots = self._shots

        if self._caching:
            # Generate hash value for caching
            hash_value = self._cache.hash_variable(
                ["sampler_v2", pubs, self._options_sampler, self._backend, self.get_shots()]
            )
        else:
            hash_value = None

        def run():
            return self.sampler.run(pubs=pubs, shots=shots)

        return self._primitive_run(run, "sampler_v2", hash_value)

    def get_estimator(self):
        """
        Returns a Estimator primitive that overwrites the Qiskit Estimator primitive.

        This Estimator runs all executions through the Executor and
        includes result caching, automatic session handling, and restarts of failed jobs.

        For Qiskit >= 1.2 the Estimator V2 is used, for Qiskit < 1.2 the Estimator V1 is returned.
        """

        if self._estimator is not None:
            if isinstance(self._estimator, BaseEstimatorV1):
                return ExecutorEstimatorV1(executor=self, options=self._options_estimator)
            return ExecutorEstimatorV2(executor=self)

        if QISKIT_SMALLER_1_2 or "Braket" in str(self._backend):
            return ExecutorEstimatorV1(executor=self, options=self._options_estimator)

        return ExecutorEstimatorV2(executor=self)

    def get_sampler(self):
        """
        Returns a Sampler primitive that overwrites the Qiskit Sampler primitive.

        This Sampler runs all executions through the Executor and
        includes result caching, automatic session handling, and restarts of failed jobs.

        For Qiskit >= 1.2 the Sampler V2 is used, for Qiskit < 1.2 the Sampler V1 is returned.
        """

        if self._sampler is not None:
            if isinstance(self._sampler, BaseSamplerV1):
                return ExecutorSamplerV1(executor=self, options=self._options_estimator)
            return ExecutorSamplerV2(executor=self)

        if QISKIT_SMALLER_1_2 or "Braket" in str(self._backend):
            return ExecutorSamplerV1(executor=self, options=self._options_sampler)

        return ExecutorSamplerV2(executor=self)

    @property
    def optree_executor(self) -> str:
        """A string that indicates which executor is used for OpTree execution."""
        if self._estimator is not None:
            return "estimator"
        if self._sampler is not None:
            return "sampler"
        return "estimator"

    def qiskit_execute(self, run_input, **options):
        """Routine that runs the given circuits on the backend.

        Args:
            run_input: An object to run on the backend (typically a circuit).
            options: Additional arguments that are passed to the backend.

        Return:
            The Qiskit job object from the run.
        """
        return self.backend.run(run_input, **options)

    def set_shots(self, num_shots: Union[int, None]) -> None:
        """Sets the number shots for the next evaluations.

        Args:
            num_shots (int or None): Number of shots that are set
        """
        self._shots = num_shots

        self._logger.info("Set shots to %s", num_shots)

        # Update shots in backend
        if num_shots is None:
            num_shots = 0

        if self.quantum_framework == "pennylane":

            if self._pennylane_device is not None:
                if isinstance(self._pennylane_device.shots, qml.measurements.Shots):
                    if num_shots == 0:
                        self._pennylane_device._shots = qml.measurements.Shots(None)
                    else:
                        self._pennylane_device._shots = qml.measurements.Shots(num_shots)
                elif (
                    isinstance(self._pennylane_device.shots, int)
                    or self._pennylane_device.shots is None
                ):
                    if num_shots == 0:
                        self._pennylane_device._shots = None
                    else:
                        self._pennylane_device._shots = num_shots

        elif self.quantum_framework == "qiskit":

            # Update shots in backend
            if self._backend is not None and self.is_statevector:
                self._backend.options.shots = num_shots

            # Update shots in estimator primitive
            if self._estimator is not None:
                if isinstance(self._estimator, PrimitiveEstimatorV1):
                    if num_shots == 0:
                        self._estimator.set_options(shots=None)
                    else:
                        self._estimator.set_options(shots=num_shots)
                    if self._options_estimator:
                        self._options_estimator["shots"] = num_shots
                elif isinstance(self._estimator, BackendEstimatorV1):
                    self._estimator.set_options(shots=num_shots)
                    if self._options_estimator:
                        self._options_estimator["shots"] = num_shots
                elif isinstance(self._estimator, RuntimeEstimatorV1):
                    execution = self._estimator.options.get("execution")
                    execution["shots"] = num_shots
                    self._estimator.set_options(execution=execution)
                    try:
                        self._options_estimator["execution"]["shots"] = num_shots
                    except (TypeError, KeyError):
                        pass  # no options_estimator or no execution in options_estimator
                elif isinstance(self._estimator, (ParallelEstimatorV1, ParallelEstimatorV2)):
                    self._estimator.shots = num_shots
                elif isinstance(self._estimator, BaseEstimatorV2):
                    self._shots = num_shots
                else:
                    raise RuntimeError("Unknown estimator type!")

            # Update shots in sampler primitive
            if self._sampler is not None:
                if isinstance(self._sampler, PrimitiveSamplerV1):
                    if num_shots == 0:
                        self._sampler.set_options(shots=None)
                    else:
                        self._sampler.set_options(shots=num_shots)
                    try:
                        self._options_sampler["shots"] = num_shots
                    except:
                        pass  # no option available
                elif isinstance(self._sampler, BackendSamplerV1):
                    self._sampler.set_options(shots=num_shots)
                    try:
                        self._options_sampler["shots"] = num_shots
                    except:
                        pass  # no option available
                elif isinstance(self._sampler, RuntimeSamplerV1):
                    execution = self._sampler.options.get("execution")
                    execution["shots"] = num_shots
                    self._sampler.set_options(execution=execution)
                    try:
                        self._options_sampler["execution"]["shots"] = num_shots
                    except:
                        pass  # no options_sampler or no execution in options_sampler
                elif isinstance(self._sampler, (ParallelSamplerV1, ParallelSamplerV2)):
                    self._sampler.shots = num_shots
                elif isinstance(self._sampler, BaseSamplerV2):
                    self._shots = num_shots
                else:
                    raise RuntimeError("Unknown sampler type!")
        else:
            raise RuntimeError("Unknown quantum framework!")

    def get_shots(self) -> int:
        """Getter for the number of shots.

        Returns:
            Returns the number of shots that are used for the current evaluation.
        """
        shots = self._shots

        if self.quantum_framework == "pennylane":

            if self._pennylane_device is not None:
                if isinstance(self._pennylane_device.shots, qml.measurements.Shots):
                    shots = self._pennylane_device.shots.total_shots
                elif (
                    isinstance(self._pennylane_device.shots, int)
                    or self._pennylane_device.shots is None
                ):
                    shots = self._pennylane_device.shots
            else:
                return None  # No shots available

        elif self.quantum_framework == "qiskit":

            if self._estimator is not None or self._sampler is not None:
                shots_estimator = 0
                shots_sampler = 0
                if self._estimator is not None:
                    if isinstance(self._estimator, PrimitiveEstimatorV1):
                        shots_estimator = self._estimator.options.get("shots", 0)
                    elif isinstance(self._estimator, BackendEstimatorV1):
                        shots_estimator = self._estimator.options.get("shots", 0)
                    elif isinstance(self._estimator, RuntimeEstimatorV1):
                        execution = self._estimator.options.get("execution")
                        shots_estimator = execution["shots"]
                    elif isinstance(self._estimator, (ParallelEstimatorV1, ParallelEstimatorV2)):
                        shots_estimator = self._estimator.shots
                    elif isinstance(self._estimator, BaseEstimatorV2):
                        shots_estimator = self._shots
                    else:
                        raise RuntimeError("Unknown estimator type!")

                if self._sampler is not None:
                    if isinstance(self._sampler, PrimitiveSamplerV1):
                        shots_sampler = self._sampler.options.get("shots", 0)
                    elif isinstance(self._sampler, BackendSamplerV1):
                        shots_sampler = self._sampler.options.get("shots", 0)
                    elif isinstance(self._sampler, RuntimeSamplerV1):
                        execution = self._sampler.options.get("execution")
                        shots_sampler = execution["shots"]
                    elif isinstance(self._sampler, (ParallelSamplerV1, ParallelSamplerV2)):
                        shots_sampler = self._sampler.shots
                    elif isinstance(self._sampler, BaseSamplerV2):
                        shots_sampler = self._shots
                    else:
                        raise RuntimeError("Unknown sampler type!")

                if self._estimator is not None and self._sampler is not None:
                    if shots_estimator != shots_sampler:
                        raise ValueError(
                            "The number of shots of the given \
                                        Estimator and Sampler is not equal!"
                        )
                if shots_estimator is None:
                    shots_estimator = 0
                if shots_sampler is None:
                    shots_sampler = 0

                shots = max(shots_estimator, shots_sampler)
            elif self._backend is not None:
                if self.is_statevector:
                    shots = self._backend.options.shots
            else:
                return None  # No shots available
        else:
            raise RuntimeError("Unknown quantum framework!")

        if shots == 0:
            shots = None

        self._shots = shots
        return shots

    def reset_shots(self) -> None:
        """Resets the shots to the initial values when the executor was created."""
        self.set_shots(self._inital_num_shots)

    @property
    def shots(self) -> int:
        """Number of shots in the execution."""
        return self.get_shots()

    def create_session(self):
        """Creates a new session, is called automatically."""

        if self.quantum_framework != "qiskit":
            raise RuntimeError("Session can only be created for Qiskit framework!")

        if not self.IBMQuantum:
            raise RuntimeError("Sessions can only be created for IBM Quantum devices!")

        if self._service is not None:
            if self._backend is not None:
                self._session = Session(
                    self._service, backend=self._backend, max_time=self._max_session_time
                )
            else:
                raise RuntimeError("Session can not started because of missing backend!")
            self._logger.info("Executor created a new session.")
        else:
            raise RuntimeError("Session can not started because of missing service!")

    def close_session(self):
        """Closes the current session, is called automatically."""

        if self.quantum_framework != "qiskit":
            raise RuntimeError("Session can only be closed for Qiskit framework!")

        if self._session is not None:
            self._logger.info("Executor closed session: %s", self._session.session_id)
            self._session.close()
            self._session = None
        else:
            raise RuntimeError("No session found!")

    def __del__(self):
        """Terminate the session in case the executor is deleted"""
        if self._session is not None:
            try:
                self.close_session()
            except:
                pass

    @property
    def estimator_options(self):
        """Returns the options of the Estimator V2 primitive."""
        if not isinstance(self.estimator, RuntimeEstimatorV2):
            raise RuntimeError("Options are only available for Qiskit Runtime V2 primitives!")
        return self.estimator.options

    @property
    def sampler_options(self):
        """Returns the options of the Sampler V2 primitive."""
        if not isinstance(self.sampler, RuntimeSamplerV2):
            raise RuntimeError("Options are only available for Qiskit Runtime V2 primitives!")
        return self.sampler.options

    def set_options_estimator(self, **fields):
        """Set options values for the estimator (V1 and V2).

        Args:
            **fields: The fields to update the options
        """

        if isinstance(self.estimator, BaseEstimatorV1):
            self.estimator.set_options(**fields)
            self._options_estimator = _convert_options_to_dict(self.estimator.options)
        elif isinstance(self.estimator, BaseEstimatorV2):
            if isinstance(self.sampler, StatevectorEstimator) or isinstance(
                self.sampler, BackendEstimatorV2
            ):
                raise RuntimeError(
                    "Setting Options is only possible for Qiskit Runtime Primtives!"
                )
            elif isinstance(self.estimator, RuntimeEstimatorV2):
                if hasattr(self.estimator, "options"):
                    self.estimator.options.update(**fields)
                    self._options_estimator = _convert_options_to_dict(self.estimator.options)
        else:
            raise RuntimeError("Unknown estimator type!")

    def set_options_sampler(self, **fields):
        """Set options values for the sampler (V1 and V2).

        Args:
            **fields: The fields to update the options
        """
        if isinstance(self.sampler, BaseSamplerV1):
            self.sampler.set_options(**fields)
            self._options_sampler = _convert_options_to_dict(self.sampler.options)
        elif isinstance(self.sampler, BaseSamplerV2):
            if isinstance(self.sampler, StatevectorSampler) or isinstance(
                self.sampler, BackendSamplerV2
            ):
                raise RuntimeError(
                    "Setting Options is only possible for Qiskit Runtime Primtives!"
                )
            elif isinstance(self.sampler, RuntimeSamplerV2):
                if hasattr(self.sampler, "options"):
                    self.sampler.options.update(**fields)
                    self._options_sampler = _convert_options_to_dict(self.sampler.options)
        else:
            raise RuntimeError("Unknown sampler type!")

    def set_primitive_options(self, **fields):
        """Set options values for the estimator and sampler primitive.

        Args:
            **fields: The fields to update the options
        """
        self.set_options_estimator(**fields)
        self.set_options_sampler(**fields)

    def set_seed_for_primitive(self, seed: int = 0):
        """Set options values for the estimator run.

        Args:
            **fields: The fields to update the options
        """
        self._set_seed_for_primitive = seed

    def select_backend(self, circuit, **options):
        """Selects the best backend for a given circuit and options.

        Args:
            circuit: Either a QuantumCircuit or an EncodingCircuitBase
            **options: Additional options for backend selection. Possible options:

                * min_num_qubits: Minimum number of qubits in the circuit (default: None)
                * max_num_qubits: Maximum number of qubits in the circuit (default: None)
                * cost_function: Cost function to use (default: None)
                * optimization_level: Optimization level (default: 3)
                * n_trials_transpile: Number of trials to transpile (default: 1)
                * call_limit: Call limit (default: int(3e7))
                * verbose: Whether to print information (default: False)
                * mode: Mode for the backend selection. Overwrites the option provided to the
                  constructor.
                * use_hqaa: Whether to use HQAA. Overwrites the option provided to the
                  constructor.

        Returns:
            A tuple containing the best backend and the transpiled circuit
        """
        from ..encoding_circuit.encoding_circuit_base import EncodingCircuitBase
        from ..encoding_circuit.transpiled_encoding_circuit import TranspiledEncodingCircuit

        min_num_qubits = options.get("min_num_qubits", None)
        max_num_qubits = options.get("max_num_qubits", None)
        cost_function = options.get("cost_function", None)
        optimization_level = options.get("optimization_level", 3)
        n_trials_transpile = options.get("n_trials_transpile", 1)
        call_limit = options.get("call_limit", int(3e7))
        verbose = options.get("verbose", False)
        logger = self._logger

        auto_selection_backend = AutomaticBackendSelection(
            backends_to_use=self.backend_list,
            min_num_qubits=min_num_qubits,
            max_num_qubits=max_num_qubits,
            cost_function=cost_function,
            optimization_level=optimization_level,
            n_trials_transpile=n_trials_transpile,
            call_limit=call_limit,
            verbose=verbose,
            logger=logger,
        )

        mode = options.get("mode", self._auto_backend_options["mode"])
        use_hqaa = options.get("use_hqaa", self._auto_backend_options["use_hqaa"])

        if isinstance(self._qpu_parallelization, int):
            if isinstance(circuit, QuantumCircuit):
                real_circuit = circuit

            elif isinstance(circuit, EncodingCircuitBase):
                x = ParameterVector("x", circuit.num_features)
                p = ParameterVector("p", circuit.num_parameters)
                real_circuit = circuit.get_circuit(x, p)
            else:
                raise ValueError("Circuit has to be a QuantumCircuit or EncodingCircuitBase")

            # create the circuit
            mapped_circuit = real_circuit.copy()

            # duplicate the circuit
            for _ in range(self._qpu_parallelization - 1):
                mapped_circuit.tensor(real_circuit, inplace=True)

            info, transpiled_circuit, backend = auto_selection_backend.evaluate(
                mapped_circuit, mode=mode, use_hqaa=use_hqaa
            )

            return_circ = transpiled_circuit

        else:
            if isinstance(circuit, QuantumCircuit):
                info, transpiled_circuit, backend = auto_selection_backend.evaluate(
                    circuit, mode=mode, use_hqaa=use_hqaa
                )
                return_circ = transpiled_circuit

            elif isinstance(circuit, EncodingCircuitBase):
                info = None
                transpiled_circuit = None
                backend = None

                def helper_function(qiskit_circuit, backend_dummy):
                    nonlocal info, transpiled_circuit, backend
                    info, transpiled_circuit, backend = auto_selection_backend.evaluate(
                        qiskit_circuit, mode=mode, use_hqaa=use_hqaa
                    )
                    return transpiled_circuit

                return_circ = TranspiledEncodingCircuit(circuit, backend, helper_function)

            else:
                raise ValueError("Circuit has to be a QuantumCircuit or EncodingCircuitBase")

        self.set_backend(backend)

        return return_circ, info

    def set_backend(self, backend: Backend):
        """Sets the backend that is used for the execution.

        Args:
            backend (Backend): Backend that is used for the execution.
        """

        shots = self.get_shots()
        self._backend = backend
        self._backend.options.shots = shots

        self._logger.info("Executor uses the backend: %s", str(self._backend))

        # Check if execution is on a remote backend
        if self.quantum_framework == "qiskit":
            if "ibm" in str(self._backend).lower() or "ibm" in str(self._backend_list).lower():
                # Sort out fake backends
                isfake = (
                    "fake" in str(self._backend).lower()
                    or "fake" in str(self._backend_list).lower()
                )
                self._remote_backend = not isfake
                self._ibm_quantum_backend = not isfake
            else:
                self._ibm_quantum_backend = False
                # Check if backend is a simulator
                self._remote_backend = not any(
                    str(substring) in str(self._backend) for substring in Aer.backends()
                )

    def unset_backend(self):
        """Unsets the backend that is used for the execution."""
        self._backend = None

    @property
    def backend_name(self) -> str:
        """Returns the name of the backend."""
        try:
            return self._backend.configuration().backend_name
        except AttributeError:
            try:
                return self._backend.name
            except AttributeError:
                return str(self._backend)

    @property
    def is_statevector(self) -> bool:
        """Returns True if the backend is a statevector simulator."""

        if self.quantum_framework == "qiskit":
            return "statevector" in self.backend_name.lower()
        elif self.quantum_framework == "pennylane":
            return any(
                name in self._pennylane_device.name.lower()
                for name in ["default.qubit", "default.clifford", "lightning.qubit"]
            )
        else:
            raise RuntimeError("Unknown quantum framework!")


class ExecutorEstimatorV2(BaseEstimatorV2):
    """
    Special Estimator V2 Primitive that uses the Executor service.

    Usefull for automatic restarting sessions and caching results.
    The object is created by the Executor method get_estimator()

    Args:
        executor (Executor): The executor service to use
        options: Options for the estimator
    """

    def __init__(self, executor: Executor):
        self._executor = executor

    def run(self, pubs: Iterable[EstimatorPubLike], *, precision: Union[float, None] = None):
        """
        Overwrites the estimator primitive run method, to evaluate circuits.
        Uses the Executor class for automatic session handling.

        Args:
            pubs: An iterable of pub-like objects, such as tuples ``(circuit, observables)``
                or ``(circuit, observables, parameter_values)``.
            precision: The target precision for expectation value estimates of each
                run Estimator Pub that does not specify its own precision. If None, the
                precision is set by the Executor's number of shots.

        Returns:
            A qiskit job containing the results of the run.
        """
        return self._executor.estimator_run_v2(
            pubs=pubs,
            precision=precision,
        )

    @property
    def options(self):
        """Return options values for the estimator.

        Returns:
            options
        """
        if hasattr(self._executor.estimator, "options"):
            return self._executor.estimator.options
        return None


class ExecutorSamplerV2(BaseSamplerV2):
    """
    Special Sampler V2 Primitive that uses the Executor service.

    Usefull for automatic restarting sessions and caching results.
    The object is created by the Executor method get_sampler()

    Args:
        executor (Executor): The executor service to use
    """

    def __init__(self, executor: Executor):
        self._executor = executor

    def run(self, pubs: Iterable[SamplerPubLike], *, shots: Union[int, None] = None):
        """
        Overwrites the sampler primitive run method, to evaluate circuits.
        Uses the Executor class for automatic session handling.

        Args:
            pubs: An iterable of pub-like objects, such as tuples ``(circuit,)``
                or ``(circuit, parameter_values)``.
            shots: The number of shots to use for each circuit.

        Returns:
            A qiskit job containing the results of the run.
        """
        return self._executor.sampler_run_v2(
            pubs=pubs,
            shots=shots,
        )

    @property
    def options(self):
        """Return options values for the sampler.

        Returns:
            options
        """
        if hasattr(self._executor.sampler, "options"):
            return self._executor.sampler.options
        return None


class ExecutorEstimatorV1(BaseEstimatorV1):
    """
    Special Estimator V1 Primitive that uses the Executor service.

    Usefull for automatic restarting sessions and caching results.
    The object is created by the Executor method get_estimator()

    Args:
        executor (Executor): The executor service to use
        options: Options for the estimator

    """

    def __init__(self, executor: Executor, options=None):
        super().__init__(options=_convert_options_to_dict(options))
        self._executor = executor

    def _call(
        self,
        circuits,
        observables,
        parameter_values=None,
        **run_options,
    ) -> EstimatorResult:
        """Has to be passed through, otherwise python will complain about the abstract method.
        Input arguments are the same as in Qiskit's estimator.call()
        """
        return self._executor.estimator._call(
            circuits, observables, parameter_values, **run_options
        )

    def _run(
        self,
        circuits,
        observables,
        parameter_values,
        **run_options,
    ) -> JobV1:
        """Has to be passed through, otherwise python will complain about the abstract method.
        Input arguments are the same as in Qiskit's estimator.run().
        """
        return self._executor.estimator_run_v1(
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
        Overwrites the sampler primitive run method, to evaluate expectation values.
        Uses the Executor class for automatic session handling.

        Input arguments are the same as in Qiskit's estimator.run()

        """
        return self._executor.estimator_run_v1(
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            **run_options,
        )

    @property
    def circuits(self):
        """Quantum circuits that represents quantum states.

        Returns:
            The quantum circuits.
        """
        return tuple(self._executor.estimator.circuits)

    @property
    def observables(self):
        """Observables to be estimated.

        Returns:
            The observables.
        """
        return tuple(self._executor.estimator.observables)

    @property
    def parameters(self):
        r"""Parameters of the quantum circuits.

        Returns:
            Parameters, where ``parameters[i][j]`` is the j-\ :spelling:word:`th` parameter of the
            i-th circuit.
        """
        return tuple(self._executor.estimator.parameters)

    @property
    def options(self) -> Options:
        """Return options values for the estimator.

        Returns:
            options
        """
        return self._executor.estimator.options

    def clear_cache(self):
        """
        Clears the cache of the estimator to prevent memory overflow.

        This function utilizes the executor's `clear_estimator_cache` method
        to reset any stored data related to the estimator's computations.
        """
        self._executor.clear_estimator_cache()

    def set_options(self, **fields):
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self._executor.estimator.set_options(**fields)
        self._executor._options_estimator = self._executor.estimator.options


class ExecutorSamplerV1(BaseSamplerV1):
    """
    Special Sampler V1 Primitive that uses the Executor service.

    Useful for automatic restarting sessions and caching the results.
    The object is created by the executor method get_sampler()

    Args:
        executor (Executor): The executor service to use
        options: Options for the sampler

    """

    def __init__(self, executor: Executor, options=None):
        super().__init__(options=_convert_options_to_dict(options))
        self._executor = executor

    def run(
        self,
        circuits,
        parameter_values=None,
        **run_options,
    ) -> JobV1:
        """
        Overwrites the sampler primitive run method, to evaluate circuits.
        Uses the Executor class for automatic session handling.

        Input arguments are the same as in Qiskit's sampler.run()

        """
        return self._executor.sampler_run_v1(
            circuits=circuits,
            parameter_values=parameter_values,
            **run_options,
        )

    def _run(
        self,
        circuits,
        parameter_values=None,
        **run_options,
    ) -> JobV1:
        """
        Overwrites the sampler primitive run method, to evaluate circuits.
        Uses the Executor class for automatic session handling.

        Input arguments are the same as in Qiskit's sampler.run()

        """
        return self._executor.sampler_run_v1(
            circuits=circuits,
            parameter_values=parameter_values,
            **run_options,
        )

    def _call(
        self,
        circuits,
        parameter_values=None,
        **run_options,
    ) -> SamplerResult:
        """Has to be passed through, otherwise python will complain about the abstract method"""
        return self._executor.sampler._call(circuits, parameter_values, **run_options)

    @property
    def circuits(self):
        """Quantum circuits to be sampled.

        Returns:
            The quantum circuits to be sampled.
        """
        return tuple(self._executor.sampler.circuits)

    @property
    def parameters(self):
        """Parameters of quantum circuits.

        Returns:
            List of the parameters in each quantum circuit.
        """
        return tuple(self._executor.sampler.parameters)

    @property
    def options(self) -> Options:
        """Return options values for the estimator.

        Returns:
            options
        """
        return self._executor.sampler.options

    def set_options(self, **fields):
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self._executor.sampler.set_options(**fields)
        self._executor._options_sampler = self._executor.sampler.options

    def clear_cache(self):
        """Clear the cache of the sampler primitive to avoid memory overflow.

        This method will be called automatically if a session is restarted.
        """
        self._executor.clear_sampler_cache()


class ExecutorEstimator:
    """
    A class that creates an estimator primitive that wraps a Primitives instance.

    Args:
        executor (Executor): The Primitives instance to wrap.
        options: Options for the estimator

    Returns:
        An estimator primitive that wraps the Primitives instance.
    """

    def __new__(
        cls, executor: Executor, options=None
    ) -> Union[ExecutorEstimatorV1, ExecutorEstimatorV2]:
        instance_estimator = executor.estimator
        if isinstance(instance_estimator, BaseEstimatorV1):
            return ExecutorEstimatorV1(executor=executor, options=options)
        if options:
            raise ValueError("Estimator options are not supported in V2")
        return ExecutorEstimatorV2(executor=executor)


class ExecutorSampler:
    """
    A class that creates a sampler primitive that wraps a Primitives instance.

    Args:
        executor (Executor): The Primitives instance to wrap.
        options: Options for the sampler

    Returns:
        A sampler primitive that wraps the Primitives instance.
    """

    def __new__(
        cls, executor: Executor, options=None
    ) -> Union[ExecutorSamplerV1, ExecutorSamplerV2]:
        instance_sampler = executor.sampler
        if isinstance(instance_sampler, BaseSamplerV1):
            return ExecutorSamplerV1(executor=executor, options=options)
        if options:
            raise ValueError("Sampler options are not supported in V2")
        return ExecutorSamplerV2(executor=executor)


class ExecutorCache:
    """Cache for jobs that are created by Primitives

    Args:
        folder (str): Folder to store the cache

    """

    def __init__(self, logger, folder: str = ""):
        self._folder = folder
        # Check if folder exist, creates the folder otherwise
        try:
            if not os.path.exists(self._folder):
                os.makedirs(self._folder)
        except:
            raise RuntimeError("Could not create folder for cache")

        self._logger = logger

    def hash_variable(self, variable: Any):
        """
        Creates a hash value for a list of circuits, parameters, operators.

        The hash value is used as the filename for the cached file.
        """

        def make_recursive_str(variable_):
            """creates a string from a list"""
            if type(variable_) == list or type(variable_) == tuple:
                text = ""
                for i in variable_:
                    text += make_recursive_str(i)
                return text
            else:
                return str(variable_)

        return blake2b(make_recursive_str(variable).encode("utf-8"), digest_size=20).hexdigest()

    def get_file(self, hash_value: str):
        """
        Searches for the cahced file and returns the file otherwise return None.

        Args:
            hash_value (str): Hash value of the file
        """

        try:
            file = Path(self._folder + "/" + str(hash_value) + ".p")
            if file.exists():
                file = open(self._folder + "/" + str(hash_value) + ".p", "rb")
                data = pickle.load(file)
                file.close()
                return data
            else:
                return None
        except Exception as e:
            raise e
            self._logger.info("Could not load job from cache!")
            self._logger.info("File: " + self._folder + "/" + str(hash_value) + ".p")
            return None

    def store_file(self, hash_value: str, job_data):
        """
        Store the data of a finsihed job.

        Args:
            hash_value (str): Hash value of the job that is used as a file name
            job_data: Data of the job
        """
        try:
            file = open(self._folder + "/" + str(hash_value) + ".p", "wb")
            pickle.dump(job_data, file)
            file.close()
        except:
            raise RuntimeError("Could not store job in cache")


def check_for_incircuit_measurements(circuit: QuantumCircuit, mode="all"):
    """
    Checks for measurements in the circuit, and returns True if there are measurements in the circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to check for measurements.

    Returns:
        True if there are measurements in the circuit.
    """

    for op in circuit.data:
        if mode == "all" or mode == "condition":
            if op.operation.condition:
                return True
        if mode == "all" or mode == "clbits":
            if len(op.clbits) > 0:
                return True
    return False


def _convert_options_to_dict(
    options: Union[Options, RuntimeOptionsV1, RuntimeOptionsV2, dict, None]
) -> dict:
    """Converts options to a dictionary."""

    if options is None:
        return None
    elif isinstance(options, dict):
        return options
    elif isinstance(options, RuntimeOptionsV1) or isinstance(options, RuntimeOptionsV2):
        return asdict(copy.deepcopy(options))
    else:
        return copy.deepcopy(options).__dict__
