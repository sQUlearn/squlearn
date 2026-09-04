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
import warnings

import dill as pickle
import numpy as np
from packaging import version
import pennylane as qml
from pennylane import __version__ as pennylane_version
from pennylane.devices import Device as PennylaneDevice
from qiskit import __version__ as qiskit_version
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.exceptions import QiskitError
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

from qc_executor import Executor as QcExecutorFactory

if version.parse(pennylane_version) < version.parse("0.39.0"):
    from pennylane import QubitDevice
else:
    from pennylane.devices import QubitDevice

from qiskit_algorithms.utils import algorithm_globals as qiskit_algorithm_globals

QISKIT_SMALLER_1_1 = version.parse(qiskit_version) < version.parse("1.1.0")
QISKIT_SMALLER_1_2 = version.parse(qiskit_version) < version.parse("1.2.0")
QISKIT_SMALLER_2_0 = version.parse(qiskit_version) < version.parse("2.0.0")

from qiskit.primitives import (
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

if QISKIT_SMALLER_1_1:

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

if QISKIT_SMALLER_2_0:
    # pylint: disable=ungrouped-imports
    from qiskit.primitives import (
        BackendEstimator as BackendEstimatorV1,
        BackendSampler as BackendSamplerV1,
        Estimator as PrimitiveEstimatorV1,
        Sampler as PrimitiveSamplerV1,
    )
else:

    class BackendEstimatorV1:
        """Dummy BackendEstimatorV1"""

    class BackendSamplerV1:
        """Dummy BackendSamplerV1"""

    class PrimitiveEstimatorV1:
        """Dummy PrimitiveEstimatorV1"""

    class PrimitiveSamplerV1:
        """Dummy PrimitiveSamplerV1"""


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
from .pennylane import PennyLaneCircuit
from .qulacs import QulacsCircuit


class SessionContextMisuseWarning(UserWarning):
    """Raised when a session is used outside its context manager."""


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

    **Important: Session Management**

    When using the Executor with IBM Quantum backends or Sessions, it is **strongly recommended**
    to use the Executor within a context manager (``with`` statement). This ensures that sessions
    are properly closed when you are done with the executor, avoiding unnecessary open sessions
    and preventing charges for unused sessions.

    .. code-block:: python

        from squlearn import Executor
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel="ibm_quantum_platform", token="INSERT_YOUR_TOKEN_HERE")

        # Recommended: Use context manager
        with Executor(service.backend('ibm_kingston'), caching=True,
                      cache_dir='cache', log_file="log.log") as executor:
            # Your quantum computations here
            pass
        # Session is automatically closed

    If you cannot use a context manager, ensure you manually close the session by calling
    :meth:`close_session` when you are done. Creating a session outside of a context manager
    will issue a warning.

    Args:
        execution (Union[str, Backend, List[Backend], QiskitRuntimeService, Session,BaseEstimatorV1, BaseSamplerV1, BaseEstimatorV2, BaseSamplerV2, PennylaneDevice]):
            The execution environment, possible inputs are:

                * A string, that specifics the simulator backend. For Qiskit this can be
                  ``"qiskit"``,``"statevector_simulator"`` or ``"qasm_simulator"``. For PennyLane
                  this can be ``"pennylane"``, ``"default.qubit"``. For Qulacs this can be
                  ``"qulacs"``.
                * A PennyLane device, to run the jobs with PennyLane (e.g. AWS Braket plugin
                  for PennyLane)
                * A Qiskit backend, to run the jobs on IBM Quantum systems or simulators
                * A list of Qiskit backends for automatic backend selection later on
                * A QiskitRuntimeService, to run the jobs on the Qiskit Runtime service.
                  In this case the backends are automatically selected based on the
                  available backends of the service, similar to providing a list of backends.
                * A Session, to run the jobs on the Qiskit Runtime service
                * A Estimator primitive (either simulator or Qiskit Runtime primitive - V1 or V2)
                * A Sampler primitive (either simulator or Qiskit Runtime primitive - V1 or V2)

            Default is the initialization with PennyLane's
            :class:`DefaultQubit <pennylane.devices.default_qubit.DefaultQubit>` simulator.
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

    **Example: Qulacs based initialization of the Executor**

    .. code-block:: python

        from squlearn import Executor

        # Executor with Qulacs backend
        executor = Executor("qulacs")

    **Example: Different Qiskit based initializations of the Executor**

    .. code-block:: python

       from squlearn import Executor
       from qiskit_ibm_runtime import QiskitRuntimeService

       # Executor with a ideal simulator backend
       exec = Executor("statevector_simulator")

       # Executor with a shot-based simulator backend and 1000 shots
       exec = Executor("qasm_simulator")
       exec.set_shots(1000)

       # Executor with a IBM Quantum backend (with context manager - recommended)
       # Session is automatically closed after the with block
       with Executor(service.backend('ibm_kingston'), caching=True,
                    cache_dir='cache', log_file="log.log") as executor:
           # Your quantum computations here
           pass

       # Executor with a IBM Quantum backend (without context manager - not recommended)
       service = QiskitRuntimeService(channel="ibm_quantum_platform", token="INSERT_YOUR_TOKEN_HERE")
       executor = Executor(service.backend('ibm_kingston'))

       # Make sure to close the session when done
       try:
           # Your code here
           pass
       finally:
           executor.close_session()

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
       service = QiskitRuntimeService(channel="ibm_quantum_platform", token="INSERT_YOUR_TOKEN_HERE")
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
        # The raw Estimator/Sampler/Session the caller passed directly as
        # execution= (not a backend). Only qc_executor's own primitive/
        # session-injection branches understand these objects, so they are
        # handed to it unchanged rather than introspected here.
        self._injected_primitive = None
        # The single qc_executor instance that owns backend resolution,
        # primitive construction, session lifecycle and execution for this
        # Executor - see _build_qc(). None until a concrete backend is
        # available (deferred for automatic backend selection).
        self._qc_executor = None
        # The current execution target for estimator_run_v1/v2 /
        # sampler_run_v1/v2 - the raw primitive qc_executor built, wrapped in
        # ParallelEstimator/ParallelSampler if qpu_parallelization is set.
        # Refreshed by _decorate_primitive() on every (re)build qc_executor
        # performs, including an IBM Quantum session renewal.
        self._inner_estimator = None
        self._inner_sampler = None
        self._execution_origin = ""
        self._context_managed = False

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
            elif execution in ["qulacs"]:
                self._quantum_framework = "qulacs"
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
            self._backend = execution
            self._execution_origin = "Backend"
        elif isinstance(execution, list):
            # Execution is a list of backends -> backands will be automatically selected
            if all(isinstance(exec, Backend) for exec in execution):
                self._backend = None
                self._backend_list = execution
                self._execution_origin = "BackendList"
            else:
                raise ValueError("Only list of backends are supported!")
        elif isinstance(execution, QiskitRuntimeService):
            self._backend = None
            self._backend_list = execution.backends()
            self._execution_origin = "QiskitRuntimeService"
        elif isinstance(execution, Session):
            self._injected_primitive = execution
            self._execution_origin = "Session"
        elif isinstance(execution, BaseEstimatorV1):
            self._injected_primitive = execution
            self._execution_origin = "Estimator"
        elif isinstance(execution, BaseSamplerV1):
            self._injected_primitive = execution
            self._execution_origin = "Sampler"
        elif isinstance(execution, BaseEstimatorV2):
            self._injected_primitive = execution
            self._execution_origin = "Estimator"
        elif isinstance(execution, BaseSamplerV2):
            self._injected_primitive = execution
            self._execution_origin = "Sampler"
        else:
            raise ValueError("Unknown execution type: " + str(type(execution)))

        if self.quantum_framework == "qiskit":
            if self._injected_primitive is not None:
                qc_target = self._injected_primitive
            elif self._backend is not None:
                if self.is_statevector and shots is None:
                    # Exact simulation: qc_executor's "statevector" alias uses
                    # Qiskit's reference primitives (StatevectorEstimator/
                    # StatevectorSampler), which are genuinely exact. A real
                    # Aer backend object here (even with .options.shots =
                    # None) does not give exact results - verified
                    # empirically against the analytic expectation value.
                    qc_target = "statevector"
                else:
                    qc_target = self._backend
            else:
                # No concrete backend yet (a backend list / QiskitRuntimeService) -
                # set_backend() builds self._qc_executor once automatic selection picks one.
                qc_target = None

            if qc_target is not None:
                self._build_qc_executor(qc_target, shots)
                # For an injected primitive, sQUlearn never resolves shots
                # itself - qc_executor reads it back from the primitive (see
                # QiskitExecutor.__init__). Re-sync the local variable so the
                # set_shots() call below doesn't clobber that with the still-
                # unresolved None it started as.
                shots = self._qc_executor.shots

            if self._backend_list is None:
                if self._backend is not None:
                    self._backend_list = [self._backend]
            else:
                if not self.IBMQuantum:
                    # If fake backends are given, automatic backend selection is supported
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
            self._qc_executor = QcExecutorFactory.create(self._pennylane_device)
        elif self.quantum_framework == "qulacs":
            self._qc_executor = QcExecutorFactory.create(
                "qulacs", shots=shots, seed=self._set_seed_for_primitive
            )
        else:
            raise RuntimeError("Unknown quantum framework!")

        if self._injected_primitive is not None and self._options_estimator is not None:
            self.set_options_estimator(**self._options_estimator)
        if self._injected_primitive is not None and self._options_sampler is not None:
            self.set_options_sampler(**self._options_sampler)

        # set initial shots
        self.set_shots(shots)
        self._inital_num_shots = self.get_shots()

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
        if self.quantum_framework == "qiskit" and self.session is not None:
            self._logger.info(
                f"Executor initialized with session: {{}}".format(self.session.session_id)
            )
        if self._injected_primitive is not None:
            self._logger.info(
                f"Executor initialized with injected primitive: {{}}".format(
                    self._injected_primitive
                )
            )
        self._logger.info(f"Executor intial shots: {{}}".format(self._inital_num_shots))

    def _build_qc_executor(self, target, shots) -> None:
        """Build (or rebuild) the single qc_executor instance that owns
        backend resolution, primitive construction, session lifecycle and
        execution for this Executor's qiskit framework - called from
        __init__ and set_backend().

        ``execution_mode="session"`` is passed unconditionally for
        backend-object/string targets: qc_executor only actually manages a
        session when it also recognizes real IBM Quantum hardware
        internally, so this is a no-op for local simulators and fake
        backends. It must be omitted for an injected primitive or
        Session, which qc_executor only accepts with the default "job" mode.
        """
        kwargs = {"shots": shots, "primitive_wrapper": self._decorate_primitive}
        if not isinstance(
            target, (BaseEstimatorV1, BaseSamplerV1, BaseEstimatorV2, BaseSamplerV2, Session)
        ):
            kwargs["execution_mode"] = "session"
        self._qc_executor = QcExecutorFactory.create(target, **kwargs)
        if self._backend is None:
            # Only for the injected-primitive/Session case: sQUlearn didn't
            # already resolve a concrete backend object itself. (The exact
            # statevector case intentionally keeps its own real Aer object
            # instead of qc_executor's - which is None there by design.)
            self._backend = self._qc_executor.backend

    def _decorate_primitive(self, raw, kind: str):
        """Registered as qc_executor's ``primitive_wrapper`` (see
        ``QiskitExecutor.__init__``): invoked for every primitive qc_executor
        (re)builds - initial construction, an IBM Quantum session renewal, or
        a deferred session's first real use.

        Wraps the raw primitive for QPU parallelization if requested
        (stored as ``self._inner_estimator``/``self._inner_sampler``, the
        target ``estimator_run_v1/v2``/``sampler_run_v1/v2`` actually call),
        and returns this Executor's own ``Executor*V1/V2`` wrapper. Its
        ``run()`` always resolves back through those methods to whatever is
        currently the correct execution target, so a caller holding it
        (qiskit-ml's ``ComputeUncompute``, qiskit-algorithms' QFI, sQUlearn's
        own OpTree fallback engine) survives future rebuilds transparently -
        it never holds a primitive itself.
        """
        inner = raw
        if self._qpu_parallelization is not None:
            if isinstance(self._qpu_parallelization, str):
                if self._qpu_parallelization != "auto":
                    raise ValueError(
                        "Unknown qpu_parallelization value: " + self._qpu_parallelization
                    )
                num_parallel = None
            elif isinstance(self._qpu_parallelization, int):
                num_parallel = self._qpu_parallelization
            else:
                raise TypeError(
                    "Unknown qpu_parallelization type: " + str(type(self._qpu_parallelization))
                )
            if kind == "estimator":
                inner = ParallelEstimator(raw, num_parallel=num_parallel)
            else:
                inner = ParallelSampler(raw, num_parallel=num_parallel)

        if kind == "estimator":
            self._inner_estimator = inner
            if isinstance(raw, BaseEstimatorV1):
                return ExecutorEstimatorV1(executor=self, options=self._options_estimator)
            return ExecutorEstimatorV2(executor=self)

        self._inner_sampler = inner
        if isinstance(raw, BaseSamplerV1):
            return ExecutorSamplerV1(executor=self, options=self._options_sampler)
        return ExecutorSamplerV2(executor=self)

    def __enter__(self):
        self._context_managed = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.IBMQuantum and self.session is not None:
            try:
                self.close_session()
            except Exception:
                pass

        self._context_managed = False

    @property
    def quantum_framework(self) -> str:
        """Return the quantum framework that is used in the executor."""
        return self._quantum_framework

    def qulacs_execute(
        self, qulacs_execution: callable, qulacs_circuit: QulacsCircuit, **kwargs
    ) -> np.ndarray:
        """
        Function for executing of Qulacs circuits with the Executor with caching

        Args:
            qulacs_execution (callable): The Qulacs execution function from qulacs_execution
            qulacs_circuit (QulacsCircuit): The Qulacs circuit data structure
            **kwargs: Parameter values of the qulacs circuit and observable, name must match
                the parameter names in the circuit and observable

        Returns:
            Numpy array: The result of the circuit execution
        """

        result = None
        cached = True
        hash_value = None

        # Check if the result of the qulacs execution is already cached
        if self._caching:

            # Get hash value of the circuit
            if hasattr(qulacs_execution, "__name__"):
                func_name = qulacs_execution.__name__
            else:
                raise ValueError("Unknown function specified as qulacs execution")
            hash_value = self._cache.hash_variable(
                ["qulacs", func_name, qulacs_circuit.hash, kwargs]
            )

            # Check if the result is already cached
            result = self._cache.get_file(hash_value)

        # If the result is not cached, execute the circuit
        if result is None:
            if self._caching:
                self._logger.info(
                    f"Execution of qulacs circuit with hash value: {{}}".format(hash_value)
                )
            else:
                self._logger.info(f"Execution of qulacs circuit")
            result = qulacs_execution(qulacs_circuit, **kwargs)
            cached = False
            self._logger.info(f"Execution of qulacs successful")
        elif self._caching:
            self._logger.info(f"Cached result found with hash value: {{}}".format(hash_value))

        # Store the result in the cache if caching is enabled and not already cached
        if self._caching and not cached:
            self._cache.store_file(hash_value, copy.copy(result))

        return result

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
        if isinstance(pennylane_circuit, PennyLaneCircuit):
            pennylane_circuit = pennylane_circuit.pennylane_circuit
            pennylane_circuit = qml.QNode(pennylane_circuit, self.backend, diff_method="best")
        if isinstance(pennylane_circuit, qml.QNode) and version.parse(
            pennylane_version
        ) >= version.parse("0.42.0"):
            pennylane_circuit = qml.set_shots(pennylane_circuit, shots=self.shots)

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
            circuit = pennylane_circuit[i]
            if isinstance(circuit, PennyLaneCircuit):
                circuit = circuit.pennylane_circuit

            circuit = qml.QNode(circuit, self.backend, diff_method="best")
            if version.parse(pennylane_version) >= version.parse("0.42.0"):
                circuit = qml.set_shots(circuit, shots=self.shots)
            circuit.construct(arg_tuple, kwargs)

            if hasattr(circuit, "hash"):
                hash_value += str(circuit.hash)
            else:
                hash_value += str(hash(circuit))

            batched_tapes.append(circuit._tape)

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
        elif self.quantum_framework == "qulacs":
            return None
        else:
            raise RuntimeError("Unknown quantum framework!")

    @property
    def remote(self) -> bool:
        """Returns a boolean if the execution is on a remote backend."""
        if self.quantum_framework == "qiskit":
            if self._qc_executor is not None:
                return self._qc_executor.remote
            # No concrete backend chosen yet (a backend list / QiskitRuntimeService) -
            # automatic backend selection is offered only for real/fake IBM backends.
            return self._backend_list is not None
        elif self.quantum_framework == "pennylane":
            return not any(
                substring in self._pennylane_device.name.lower()
                for substring in [
                    "default.qubit",
                    "default.mixed",
                    "default.clifford",
                    "lightning.qubit",
                    "lightning.gpu",
                ]
            )
        elif self.quantum_framework == "qulacs":
            return False
        else:
            raise RuntimeError("Unknown quantum framework!")

    @property
    def IBMQuantum(self) -> bool:
        """Returns a boolean if the execution is on a IBM Quantum backend."""
        if self.quantum_framework != "qiskit":
            return False
        if self._qc_executor is not None:
            return self._qc_executor.ibm_quantum
        isfake = "fake" in str(self._backend_list).lower()
        return self._backend_list is not None and not isfake

    @property
    def backend_list(self) -> List[Backend]:
        """Returns the backend list that is used in the executor."""
        return self._backend_list

    @property
    def backend_chosen(self) -> bool:
        """Returns true if the backend has been chosen."""
        if len(self._backend_list) > 1 and self.backend is None:
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
        if self.quantum_framework != "qiskit" or self._qc_executor is None:
            return None
        return self._qc_executor.session

    @property
    def estimator(self) -> Union[BaseEstimatorV1, BaseEstimatorV2]:
        """Returns the raw Qiskit estimator primitive this Executor currently
        executes through - undecorated (no retry/caching/parallelization).
        Use :meth:`get_estimator` for a stable, fully-decorated Estimator
        suitable for handing to third-party code.
        """
        if self.quantum_framework != "qiskit":
            raise RuntimeError("Estimator is only available for Qiskit backends")
        return self._qc_executor.raw_estimator

    def clear_estimator_cache(self) -> None:
        """Function for clearing the cache of the EstimatorV1 primitive to avoid memory overflow."""
        estimator = self.estimator
        if estimator is not None and (
            isinstance(estimator, PrimitiveEstimatorV1)
            or isinstance(estimator, BackendEstimatorV1)
        ):
            estimator._circuits = []
            estimator._observables = []
            estimator._parameters = []
            estimator._circuit_ids = {}
            estimator._observable_ids = {}

    @property
    def sampler(self) -> Union[BaseSamplerV1, BaseSamplerV2]:
        """Returns the raw Qiskit sampler primitive this Executor currently
        executes through - undecorated (no retry/caching/parallelization).
        Use :meth:`get_sampler` for a stable, fully-decorated Sampler
        suitable for handing to third-party code.
        """
        if self.quantum_framework != "qiskit":
            raise RuntimeError("Sampler is only available for Qiskit backends")
        return self._qc_executor.raw_sampler

    def clear_sampler_cache(self) -> None:
        """Function for clearing the cache of the SamplerV1 primitive to avoid memory overflow."""
        sampler = self.sampler
        if sampler is not None and (
            isinstance(sampler, PrimitiveSamplerV1) or isinstance(sampler, BackendSamplerV1)
        ):
            sampler._circuits = []
            sampler._parameters = []
            sampler._circuit_ids = {}
            sampler._qargs_list = []

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

        # No primitive swap needed when shots is set: is_statevector with shots
        # already constructs a real BackendEstimator/Sampler (see __init__),
        # which natively supports in-circuit measurements.
        if containes_incircuit_measurement and self.shots is None:
            raise ValueError(
                "In-circuit measurements with the Estimator are only possible with shots."
            )

        # Set seed for the primitive
        instance_estimator = self.estimator
        if isinstance(instance_estimator, BaseEstimatorV2):
            raise RuntimeError("Estimator is a BaseEstimatorV2, please use estimator_run_v2.")

        if isinstance(instance_estimator, BackendEstimatorV1):
            if self._set_seed_for_primitive is not None:
                kwargs["seed_simulator"] = self._set_seed_for_primitive
                self._set_seed_for_primitive += 1
        elif isinstance(instance_estimator, PrimitiveEstimatorV1):
            if self._set_seed_for_primitive is not None:
                instance_estimator.set_options(seed=self._set_seed_for_primitive)
                self._set_seed_for_primitive += 1

        def run():
            self.estimator  # refresh trigger for an IBM Quantum session, incl. on retries
            return self._inner_estimator.run(circuits, observables, parameter_values, **kwargs)

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

        # No primitive swap needed when shots is set: is_statevector with shots
        # already constructs a real BackendEstimator/Sampler (see __init__),
        # which natively supports in-circuit measurements.
        if containes_incircuit_measurement and self.shots is None:
            raise ValueError(
                "In-circuit measurements with the Estimator are only possible with shots."
            )

        # Set seed for the primitive
        instance_estimator = self.estimator
        if isinstance(instance_estimator, BaseEstimatorV1):
            raise RuntimeError("Estimator is a BaseEstimatorV1, please use estimator_run_v1.")

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
            self.estimator  # refresh trigger for an IBM Quantum session, incl. on retries
            return self._inner_estimator.run(pubs=pubs, precision=precision)

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
        # No primitive swap needed when shots is set: is_statevector with shots
        # already constructs a real BackendEstimator/Sampler (see __init__),
        # which natively supports conditioned gates.
        if circuits_contains_conditions and self.shots is None:
            raise ValueError("Conditioned gates on the Sampler are only possible with shots!")

        # Set seed for the primitive
        instance_sampler = self.sampler
        if isinstance(instance_sampler, BaseSamplerV2):
            raise RuntimeError("Sampler is a BaseSamplerV2, please use sampler_run_v2.")

        if isinstance(instance_sampler, BackendSamplerV1):
            if self._set_seed_for_primitive is not None:
                kwargs["seed_simulator"] = self._set_seed_for_primitive
                self._set_seed_for_primitive += 1
        elif isinstance(instance_sampler, PrimitiveSamplerV1):
            if self._set_seed_for_primitive is not None:
                instance_sampler.set_options(seed=self._set_seed_for_primitive)
                self._set_seed_for_primitive += 1

        def run():
            self.sampler  # refresh trigger for an IBM Quantum session, incl. on retries
            return self._inner_sampler.run(circuits, parameter_values, **kwargs)

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

        # No primitive swap needed when shots is set: is_statevector with shots
        # already constructs a real BackendEstimator/Sampler (see __init__),
        # which natively supports conditioned gates.
        if circuits_contains_conditions and self.shots is None and shots is None:
            raise ValueError("Conditioned gates on the Sampler are only possible with shots!")

        # Set seed for the primitive
        instance_sampler = self.sampler
        if isinstance(instance_sampler, BaseSamplerV1):
            raise RuntimeError("Sampler is a BaseSamplerV1, please use sampler_run_v1.")

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
            self.sampler  # refresh trigger for an IBM Quantum session, incl. on retries
            return self._inner_sampler.run(pubs=pubs, shots=shots)

        return self._primitive_run(run, "sampler_v2", hash_value)

    def get_estimator(self):
        """
        Returns a Estimator primitive that overwrites the Qiskit Estimator primitive.

        This Estimator runs all executions through the Executor and
        includes result caching, automatic session handling, and restarts of failed jobs.

        For Qiskit >= 1.2 the Estimator V2 is used, for Qiskit < 1.2 the Estimator V1 is returned.
        """

        if self.quantum_framework != "qiskit":
            raise RuntimeError("Estimator is only available for Qiskit backends")
        return self._qc_executor.estimator

    def get_sampler(self):
        """
        Returns a Sampler primitive that overwrites the Qiskit Sampler primitive.

        This Sampler runs all executions through the Executor and
        includes result caching, automatic session handling, and restarts of failed jobs.

        For Qiskit >= 1.2 the Sampler V2 is used, for Qiskit < 1.2 the Sampler V1 is returned.
        """
        if self.quantum_framework != "qiskit":
            raise RuntimeError("Sampler is only available for Qiskit backends")
        return self._qc_executor.sampler

    @property
    def optree_executor(self) -> str:
        """A string that indicates which executor is used for OpTree execution."""
        if self.quantum_framework == "qiskit" and self._qc_executor is not None:
            if self._qc_executor.raw_estimator is not None:
                return "estimator"
            if self._qc_executor.raw_sampler is not None:
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

    def expectation_value(self, circuit, observable, **parameters):
        """Evaluate the expectation value of *observable* on *circuit*
        directly through the underlying ``qc_executor`` instance - the
        native execution path used by ``LowLevelQNNUnified``. Supported for
        ``quantum_framework in ("qiskit", "pennylane", "qulacs")``.
        """
        return self._qc_executor.expectation_value(circuit, observable, **parameters)

    def expectation_value_derivatives(self, circuit, observable, *derivative, **parameters):
        """Evaluate derivatives of the expectation value of *observable* on
        *circuit* directly through the underlying ``qc_executor`` instance -
        see :meth:`expectation_value`.
        """
        return self._qc_executor.expectation_value_derivatives(
            circuit, observable, *derivative, **parameters
        )

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

        if self.quantum_framework == "qulacs":

            if num_shots != 0:
                raise RuntimeError(
                    "Qulacs does not support shot-based sampling;"
                    " it only supports statevector simulation."
                )

        elif self.quantum_framework == "pennylane":

            if (
                version.parse(pennylane_version) < version.parse("0.42.0")
                and self._pennylane_device is not None
            ):
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

            # Update shots on the backend object itself, for the shot-based
            # statevector case (kept manual, see __init__).
            if self._backend is not None and self.is_statevector:
                self._backend.options.shots = num_shots

            # Reconfigure the current raw primitives in place (mutates the
            # SAME object self._inner_estimator/sampler still hold - the
            # qc_executor shots setter now actually does this, see WP-3).
            if self._qc_executor is not None:
                self._qc_executor.shots = self._shots

            # ParallelEstimator/ParallelSampler track shots as their own
            # bookkeeping (used to compute per-call precision), independent
            # of the raw primitive's own options - update explicitly.
            if isinstance(self._inner_estimator, (ParallelEstimatorV1, ParallelEstimatorV2)):
                self._inner_estimator.shots = self._shots
            if isinstance(self._inner_sampler, (ParallelSamplerV1, ParallelSamplerV2)):
                self._inner_sampler.shots = self._shots
        else:
            raise RuntimeError("Unknown quantum framework!")

    def get_shots(self) -> int:
        """Getter for the number of shots.

        Returns:
            Returns the number of shots that are used for the current evaluation.
        """
        shots = self._shots

        if self.quantum_framework == "qulacs":

            return None

        elif self.quantum_framework == "pennylane":

            if (
                version.parse(pennylane_version) < version.parse("0.42.0")
                and self._pennylane_device is not None
            ):
                if isinstance(self._pennylane_device.shots, qml.measurements.Shots):
                    shots = self._pennylane_device.shots.total_shots
                elif (
                    isinstance(self._pennylane_device.shots, int)
                    or self._pennylane_device.shots is None
                ):
                    shots = self._pennylane_device.shots

        elif self.quantum_framework == "qiskit":

            # ParallelEstimator/ParallelSampler track shots as their own
            # bookkeeping (used to compute per-call precision), independent
            # of the raw primitive's own options - read from there first.
            if isinstance(self._inner_estimator, (ParallelEstimatorV1, ParallelEstimatorV2)):
                shots = self._inner_estimator.shots
            elif isinstance(self._inner_sampler, (ParallelSamplerV1, ParallelSamplerV2)):
                shots = self._inner_sampler.shots
            elif self._qc_executor is not None:
                shots = self._qc_executor.shots
            elif self._backend is not None and self.is_statevector:
                shots = self._backend.options.shots
            # else: no qc_executor instance and no (statevector) backend to read
            # from yet - e.g. a not-yet-chosen backend list, or set_backend()
            # calling this before _build_qc() runs. Fall back to the already-
            # initialized self._shots above rather than discarding it.
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
        """Creates a new session.

        **Warning**: Creating a session outside of a context manager will issue a warning.
        It is strongly recommended to use the Executor within a ``with`` statement when working
        with IBM Quantum backends or sessions to ensure proper cleanup.

        Raises:
            SessionContextMisuseWarning: If the session is created outside of a context manager.
            RuntimeError: If the session cannot be created due to a missing backend.

        See Also:
            :meth:`close_session`: For manually closing a session.
        """

        if self.quantum_framework != "qiskit":
            raise RuntimeError("Session can only be created for Qiskit framework!")

        if not self.IBMQuantum:
            raise RuntimeError("Sessions can only be created for IBM Quantum devices!")

        if not self._context_managed:
            warnings.warn(
                "\033[1;91mCreating a session outside of a context manager may lead to  unclosed "
                "sessions. It is recommended to use the Executor within a 'with' statement. At "
                "least make sure to call 'executor.close_session()' when you are done with the "
                "executor or make sure it is properly garbage collected.\033[0m",
                SessionContextMisuseWarning,
            )

        if self._backend is None:
            raise RuntimeError("Session can not started because of missing backend!")

        # Session creation, expiry detection and renewal are handed entirely
        # to self._qc_executor; its own lifetime (tied to this Executor) closes
        # the session on garbage collection, so no separate finalizer is needed.
        self._qc_executor.create_session()
        self._logger.info("Executor created a new session.")

    def close_session(self):
        """Closes the current session.

        This method should be called when you are done using the Executor with an IBM Quantum
        backend to avoid being charged for unused sessions. Alternatively, use the Executor
        within a context manager (``with`` statement) to ensure automatic cleanup.

        Raises:
            RuntimeError: If no session exists or the framework is not Qiskit.

        See Also:
            :meth:`create_session`: For creating a new session.
        """

        if self.quantum_framework != "qiskit":
            raise RuntimeError("Session can only be closed for Qiskit framework!")

        if self._qc_executor is None or self._qc_executor.session is None:
            raise RuntimeError("No session found!")
        self._logger.info(
            "Executor closed session: %s", self._qc_executor.session.session_id
        )
        self._qc_executor.close_session()

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

    def select_backend(self, circuit, num_features=None, **options):
        """Selects the best backend for a given circuit and options.

        Args:
            circuit: Either a QuantumCircuit or an EncodingCircuitBase
            num_features: Number of features, if None the TranspileEncodingCircuit won't be
                returned
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

        if isinstance(circuit, EncodingCircuitBase) and circuit.num_encoding_slots == np.inf:
            raise RuntimeError(
                f"""Automatic backend selection is not supported for {circuit.__name__}.\n 
                This circuit has an infinite number of encoding slots, which is not supported by the automatic backend selection."""
            )

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
                num_features_for_transpilation = (
                    num_features if num_features is not None else circuit.num_encoding_slots
                )
                x = ParameterVector("x", num_features_for_transpilation)
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
                num_features_for_transpilation = (
                    num_features if num_features is not None else circuit.num_encoding_slots
                )

                def helper_function(qiskit_circuit, backend_dummy):
                    nonlocal info, transpiled_circuit, backend
                    info, transpiled_circuit, backend = auto_selection_backend.evaluate(
                        qiskit_circuit, mode=mode, use_hqaa=use_hqaa
                    )
                    return transpiled_circuit

                return_circ = TranspiledEncodingCircuit(
                    circuit, backend, num_features_for_transpilation, helper_function
                )

            else:
                raise ValueError("Circuit has to be a QuantumCircuit or EncodingCircuitBase")

        self.set_backend(backend)

        if isinstance(circuit, EncodingCircuitBase) and num_features is None:
            return info
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

        if self.quantum_framework == "qiskit":
            self._build_qc_executor(backend, shots)

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
                for name in [
                    "default.qubit",
                    "default.mixed",
                    "default.clifford",
                    "lightning.qubit",
                    "lightning.gpu",
                ]
            )
        elif self.quantum_framework == "qulacs":
            return True
        else:
            raise RuntimeError("Unknown quantum framework!")


class ExecutorEstimatorV2(BaseEstimatorV2):
    """
    Special Estimator V2 Primitive that uses the Executor.

    Usefull for automatic restarting sessions and caching results.
    The object is created by the Executor method get_estimator()

    Args:
        executor (Executor): The executor to use
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

    @property
    def backend(self):
        """The backend this primitive executes on."""
        return self._executor.backend


class ExecutorSamplerV2(BaseSamplerV2):
    """
    Special Sampler V2 Primitive that uses the Executor.

    Usefull for automatic restarting sessions and caching results.
    The object is created by the Executor method get_sampler()

    Args:
        executor (Executor): The executor to use
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

    @property
    def backend(self):
        """The backend this primitive executes on."""
        return self._executor.backend


class ExecutorEstimatorV1(BaseEstimatorV1):
    """
    Special Estimator V1 Primitive that uses the Executor.

    Usefull for automatic restarting sessions and caching results.
    The object is created by the Executor method get_estimator()

    Args:
        executor (Executor): The executor to use
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

    @property
    def backend(self):
        """The backend this primitive executes on."""
        return self._executor.backend


class ExecutorSamplerV1(BaseSamplerV1):
    """
    Special Sampler V1 Primitive that uses the Executor.

    Useful for automatic restarting sessions and caching the results.
    The object is created by the executor method get_sampler()

    Args:
        executor (Executor): The executor to use
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

    @property
    def backend(self):
        """The backend this primitive executes on."""
        return self._executor.backend


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
            if hasattr(op.operation, "condition") and op.operation.condition:
                return True
        if mode == "all" or mode == "clbits":
            if len(op.clbits) > 0:
                return True
    return False


def _convert_options_to_dict(
    options: Union[Options, RuntimeOptionsV1, RuntimeOptionsV2, dict, None],
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
