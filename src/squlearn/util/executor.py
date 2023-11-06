import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from logging import handlers
import copy
from pathlib import Path
from hashlib import blake2b
from typing import Any, Union
import traceback
from dataclasses import asdict
import time
import dill as pickle

from qiskit.primitives import Estimator as qiskit_primitives_Estimator
from qiskit.primitives import BackendEstimator as qiskit_primitives_BackendEstimator
from qiskit.primitives import Sampler as qiskit_primitives_Sampler
from qiskit.primitives import BackendSampler as qiskit_primitives_BackendSampler
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.primitives.base import SamplerResult, EstimatorResult
from qiskit import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.providers import Options
from qiskit.providers import JobV1 as Job
from qiskit.providers.backend import Backend
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit_ibm_runtime import Estimator as qiskit_ibm_runtime_Estimator
from qiskit_ibm_runtime import Sampler as qiskit_ibm_runtime_Sampler
from qiskit_ibm_runtime.exceptions import IBMRuntimeError, RuntimeJobFailureError
from qiskit_ibm_runtime.options import Options as qiskit_ibm_runtime_Options
from qiskit.exceptions import QiskitError


class Executor:
    """
    A class for executing quantum jobs on IBM Quantum systems or simulators.

    The Executor class is the central component of sQUlearn, responsible for running quantum jobs.
    Both high- and low-level methods utilize the Executor class to execute jobs seamlessly.
    It automatically creates the necessary primitives when they are required in the sQUlearn
    sub-program. The Executor takes care about session handling, result caching, and automatic
    restarts of failed jobs.

    The Estimator can be initialized with various objects that specify the execution environment,
    as for example a Qiskit backend either from IBM Quantum or a Aer simulator.

    A detailed introduction to the Executor can be found in the
    :doc:`User Guide: The Executor Class </user_guide/executor>`

    Args:
        execution (Union[str, Backend, QiskitRuntimeService, Session, BaseEstimator, BaseSampler]): The execution environment, possible inputs are:

                                                                                                                     * A string, that specifics the simulator
                                                                                                                       backend (``"statevector_simulator"`` or ``"qasm_simulator"``)
                                                                                                                     * A Qiskit backend, to run the jobs on IBM Quantum
                                                                                                                       systems or simulators
                                                                                                                     * A QiskitRuntimeService, to run the jobs on the Qiskit Runtime service
                                                                                                                       In this case the backend has to be provided separately via ``backend=``
                                                                                                                     * A Session, to run the jobs on the Qiskit Runtime service
                                                                                                                     * A Estimator primitive (either simulator or Qiskit Runtime primitive)
                                                                                                                     * A Sampler primitive (either simulator or Qiskit Runtime primitive)

                                                                                                                     Default is the initialization with the :class:`StatevectorSimulator`.
        backend (Union[Backend, str, None]): The backend that is used for the execution.
                                             Only mandatory if a service is provided.
        options_estimator (Union[Options, Options, None]): The options for the created estimator
                                                           primitives.
        options_sampler (Union[Options, Options, None]): The options for the created sampler
                                                         primitives.
        log_file (str): The name of the log file, if empty, no log file is created.
        caching (Union[bool, None]): Whether to cache the results of the jobs.
        cache_dir (str): The directory where to cache the results of the jobs.
        max_session_time (str): The maximum time for a session, similar input as in Qiskit.
        max_jobs_retries (int): The maximum number of retries for a job
            until the execution is aborted.
        wait_restart (int): The time to wait before restarting a job in seconds.
        shots (Union[int, None]): The number of initial shots that is used for the execution.

    Attributes:
    -----------

    Attributes:
        execution (str): String of the execution environment.
        backend (Backend): The backend that is used in the Executor.
        session (Session): The session that is used in the Executor.
        service (QiskitRuntimeService): The service that is used in the Executor.
        estimator (BaseEstimator): The Qiskit estimator primitive that is used in the Executor.
                                   Different to :meth:`get_estimator`,
                                   which creates a new estimator object with overwritten methods
                                   that runs everything through the Executor with
                                   :meth:`estimator_run`.
        sampler (BaseSampler): The Qiskit sampler primitive that is used in the Executor.
                               Different to :meth:`get_sampler`,
                               which creates a new sampler object with overwritten methods
                               that runs everything through the Executor with
                               :meth:`estimator_run`.
        shots (int): The number of shots that is used in the Executor.

    See Also:
       * :doc:`User Guide: The Executor Class </user_guide/executor>`
       * `Qiskit Runtime <https://quantum-computing.ibm.com/lab/docs/iql/runtime>`_
       * `Qsikit Primitives <https://qiskit.org/documentation/apidoc/primitives.html>`_

    **Example: Different initializations of the Executor**

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
       executor = Executor(service.get_backend('ibm_nairobi'))

       # Executor with a IBM Quantum backend and caching and logging
       service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
       executor = Executor(service.get_backend('ibm_nairobi'), caching=True,
                            cache_dir='cache', log_file="log.log")

    **Example: Get the Executor based primitives**

    .. jupyter-execute::

       from squlearn import Executor

       # Initialize the Executor
       executor = Executor("statevector_simulator")

       # Get the Executor based Estimator - can be used as a normal Qiskit Estimator
       estimator = executor.get_estimator()

       # Get the Executor based Sampler - can be used as a normal Qiskit Sampler
       sampler = executor.get_sampler()

    .. jupyter-execute::

       # Run a circuit with the Executor based Sampler
       from qiskit.circuit.random import random_circuit
       circuit = random_circuit(2, 2, seed=1, measure=True).decompose(reps=1)
       job = sampler.run(circuit)
       result = job.result()


    Methods:
    --------
    """

    def __init__(
        self,
        execution: Union[
            str,
            Backend,
            QiskitRuntimeService,
            Session,
            BaseEstimator,
            BaseSampler,
        ] = "statevector_simulator",
        backend: Union[Backend, str, None] = None,
        options_estimator: Union[Options, qiskit_ibm_runtime_Options] = None,
        options_sampler: Union[Options, qiskit_ibm_runtime_Options] = None,
        log_file: str = "",
        caching: Union[bool, None] = None,
        cache_dir: str = "_cache",
        max_session_time: str = "8h",
        max_jobs_retries: int = 10,
        wait_restart: int = 1,
        shots: Union[int, None] = None,
    ) -> None:
        # Default values for internal variables
        self._backend = None
        self._session = None
        self._service = None
        self._estimator = None
        self._sampler = None
        self._remote = False
        self._session_active = False
        self._execution_origin = ""

        # Copy estimator options and make a dict
        self._options_estimator = options_estimator
        if self._options_estimator is None:
            self._options_estimator = {}

        # Copy sampler options and make a dict
        self._options_sampler = options_sampler
        if self._options_sampler is None:
            self._options_sampler = {}

        # Copy Executor options
        self._log_file = log_file
        self._caching = caching
        self._max_session_time = max_session_time
        self._max_jobs_retries = max_jobs_retries
        self._wait_restart = wait_restart

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

        if isinstance(execution, str):
            # Execution is a string -> get backend
            if execution == "statevector_simulator":
                self._backend = Aer.get_backend(execution)
            elif execution == "qasm_simulator":
                self._backend = Aer.get_backend(execution)
            elif "ibm" in execution:
                raise ValueError(
                    "IBM backend are not supported by string input, since credentials are missing "
                    + execution
                )
            else:
                raise ValueError("Unknown backend string: " + execution)
            self._execution_origin = "Simulator"
        elif isinstance(execution, Backend):
            # Execution is a backend class
            if hasattr(execution, "service"):
                self._service = execution.service
            self._backend = execution
            self._execution_origin = "Backend"
        elif isinstance(execution, QiskitRuntimeService):
            self._service = execution
            if isinstance(backend, str):
                self._backend = self._service.get_backend(backend)
            elif isinstance(backend, Backend):
                self._backend = backend
            elif backend is None:
                raise ValueError("Backend has to be specified for QiskitRuntimeService")
            else:
                raise ValueError("Unknown backend type: " + backend)
            self._execution_origin = "QiskitRuntimeService"
        elif isinstance(execution, Session):
            # Execution is a active? session
            self._session = execution
            self._service = self._session.service
            self._backend = self._session.service.get_backend(self._session.backend())
            self._session_active = True
            self._execution_origin = "Session"
        elif isinstance(execution, BaseEstimator):
            self._estimator = execution
            if isinstance(self._estimator, qiskit_primitives_Estimator):
                # this is only a hack, there is no real backend in the Primitive Estimator class
                self._backend = Aer.get_backend("statevector_simulator")
            elif isinstance(self._estimator, qiskit_primitives_BackendEstimator):
                self._backend = self._estimator._backend
            # Real Backend
            elif hasattr(self._estimator, "session"):
                self._session = self._estimator.session
                self._service = self._estimator.session.service
                self._backend = self._estimator.session.service.get_backend(
                    self._estimator.session.backend()
                )
                self._session_active = True
            else:
                raise RuntimeError("No backend found in the given Estimator Primitive!")

            if self._options_estimator is None:
                self._options_estimator = self._estimator.options
            else:
                self._estimator.options.update_options(**self._options_estimator)
            self._execution_origin = "Estimator"
        elif isinstance(execution, BaseSampler):
            self._sampler = execution

            if isinstance(self._sampler, qiskit_primitives_Sampler):
                # this is only a hack, there is no real backend in the Primitive Sampler class
                self._backend = Aer.get_backend("statevector_simulator")
            elif isinstance(self._sampler, qiskit_primitives_BackendSampler):
                self._backend = self._sampler._backend
            elif hasattr(self._sampler, "session"):
                self._session = self._sampler.session
                self._service = self._sampler.session.service
                self._backend = self._sampler.session.service.get_backend(
                    self._sampler.session.backend()
                )
                self._session_active = True
            else:
                raise RuntimeError("No backend found in the given Sampler Primitive!")

            if self._options_sampler is None:
                self._options_sampler = self._sampler.options
            else:
                self._sampler.options.update_options(**self._options_sampler)
            self._execution_origin = "Sampler"
        else:
            raise ValueError("Unknown execution type: " + str(type(execution)))

        # Check if execution is on a remote IBM backend
        if "ibm" in str(self._backend):
            self._remote = True
        else:
            self._remote = False

        if shots is None:
            self._inital_num_shots = self.get_shots()
        else:
            self._inital_num_shots = shots
            self.set_shots(shots)

        if self._caching is None:
            self._caching = self._remote

        if self._caching:
            self._cache = ExecutorCache(self._logger, cache_dir)

        self._logger.info(f"Executor initialized with backend: {{}}".format(self._backend))
        self._logger.info(f"Executor initialized with service: {{}}".format(self._service))
        if self._session is not None:
            self._logger.info(
                f"Executor initialized with session: {{}}".format(self._session.session_id)
            )
        else:
            self._logger.info(f"Executor initialized with session: {{}}".format(self._session))
        self._logger.info(f"Executor initialized with estimator: {{}}".format(self._estimator))
        self._logger.info(f"Executor initialized with sampler: {{}}".format(self._sampler))
        self._logger.info(f"Executor intial shots: {{}}".format(self._inital_num_shots))

    @property
    def execution(self) -> str:
        """Returns a string of the execution that is used to initialize the executor class."""
        return self._execution_origin

    @property
    def backend(self) -> Backend:
        """Returns the backend that is used in the executor."""
        return self._backend

    @property
    def session(self) -> Session:
        """Returns the session that is used in the executor."""
        return self._session

    @property
    def service(self) -> QiskitRuntimeService:
        """Returns the service that is used in the executor."""
        return self._service

    @property
    def estimator(self) -> BaseEstimator:
        """Returns the estimator primitive that is used for the execution.

        This function created automatically estimators and checks for an expired session and
        creates a new one if necessary.
        Note that the run function is the same as in the Qiskit primitives, and
        does not support caching and restarts
        For this use :meth:`sampler_run` or :meth:`get_sampler`.

        The estimator that is created depends on the backend that is used for the execution.
        """
        if self._estimator is not None:
            if self._session is not None and self._session_active is False:
                # Session is expired, create a new session and a new estimator
                self.create_session()
                self._estimator = qiskit_ibm_runtime_Estimator(
                    session=self._session, options=self._options_estimator
                )
            estimator = self._estimator
        else:
            # Create a new Estimator
            shots = self.get_shots()
            if self._session is not None:
                if self._session_active is False:
                    self.create_session()
                self._estimator = qiskit_ibm_runtime_Estimator(
                    session=self._session, options=self._options_estimator
                )
            elif self._service is not None:
                # No session but service -> create a new session
                self.create_session()
                self._estimator = qiskit_ibm_runtime_Estimator(
                    session=self._session, options=self._options_estimator
                )
            else:
                if "statevector_simulator" in str(self._backend):
                    # No session, no service, but state_vector simulator -> Estimator
                    self._estimator = qiskit_primitives_Estimator(options=self._options_estimator)
                else:
                    # No session, no service and no state_vector simulator -> BackendEstimator
                    self._estimator = qiskit_primitives_BackendEstimator(
                        backend=self._backend, options=self._options_estimator
                    )

            if not self._options_estimator:
                self.set_shots(shots)
            estimator = self._estimator

        return estimator

    def clear_estimator_cache(self) -> None:
        """Function for clearing the cache of the estimator primitive to avoid memory overflow."""
        if self._estimator is not None:
            if isinstance(self._estimator, qiskit_primitives_Estimator) or isinstance(
                self._estimator, qiskit_primitives_BackendEstimator
            ):
                self._estimator._circuits = []
                self._estimator._observables = []
                self._estimator._parameters = []
                self._estimator._circuit_ids = {}
                self._estimator._observable_ids = {}

    @property
    def sampler(self) -> BaseSampler:
        """Returns the sampler primitive that is used for the execution.

        This function created automatically estimators and checks for an expired session and
        creates a new one if necessary.

        Note that the run function is the same as in the Qiskit primitives, and
        does not support caching, session handing, etc.
        For this use :meth:`sampler_run` or :meth:`get_sampler`.

        The estimator that is created depends on the backend that is used for the execution.
        """
        if self._sampler is not None:
            if self._session is not None and self._session_active is False:
                # Session is expired, create a new one and a new estimator
                self.create_session()
                self._sampler = qiskit_ibm_runtime_Sampler(
                    session=self._session, options=self._options_sampler
                )
            sampler = self._sampler
        else:
            # Create a new Sampler
            shots = self.get_shots()
            if self._session is not None:
                if self._session_active is False:
                    self.create_session()
                self._sampler = qiskit_ibm_runtime_Sampler(
                    session=self._session, options=self._options_sampler
                )

            elif self._service is not None:
                # No session but service -> create a new session
                self.create_session()
                self._sampler = qiskit_ibm_runtime_Sampler(
                    session=self._session,
                    options=self._options_sampler,
                )
            else:
                if str(self._backend) == "statevector_simulator":
                    # No session, no service, but state_vector simulator -> Sampler
                    self._sampler = qiskit_primitives_Sampler(options=self._options_sampler)
                else:
                    # No session, no service and no state_vector simulator -> BackendSampler
                    self._sampler = qiskit_primitives_BackendSampler(
                        backend=self._backend, options=self._options_sampler
                    )
            if not self._options_sampler:
                self.set_shots(shots)
            sampler = self._sampler

        return sampler

    def clear_sampler_cache(self) -> None:
        """Function for clearing the cache of the sampler primitive to avoid memory overflow."""
        if self._sampler is not None:
            if isinstance(self._sampler, qiskit_primitives_Sampler) or isinstance(
                self._sampler, qiskit_primitives_BackendSampler
            ):
                self._sampler._circuits = []
                self._sampler._parameters = []
                self._sampler._circuit_ids = {}
                self._sampler._qargs_list = []

    def _primitive_run(
        self, run: callable, label: str, hash_value: Union[str, None] = None
    ) -> Job:
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
                    self._session_active = False
                    continue

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
                self._logger.info(f"Traceback: {{}}".format(traceback.print_exc()))

            # Wait for the job to complete
            if not cached:
                status = JobStatus.QUEUED
                last_status = None
            else:
                status = JobStatus.DONE
            while status not in JOB_FINAL_STATES:
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
                    self._logger.info(f"Traceback: {{}}".format(traceback.print_exc()))
                    break

                if self._remote:
                    time.sleep(1)
                else:
                    time.sleep(0.01)

            # Job is completed, check if it was successful
            if status == JobStatus.ERROR:
                self._logger.info(f"Failed executation of the job!")
                try:
                    self._logger.info(f"Error message: {{}}".format(job.error_message()))
                except Exception as e:
                    try:
                        job.result()
                    except Exception as e2:
                        pass
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
                        self._logger.info(f"Traceback: {{}}".format(traceback.print_exc()))
                    if result_success:
                        break
                    else:
                        self._logger.info(f"Retrying to get job result")
                        time.sleep(self._wait_restart)

            if success and result_success:
                break
            else:
                self._logger.info(f"Restarting " + label + f" run")
                success = False
                result_success = False

            if critical_error:
                self._logger.info(f"Critical error detected; abort execution")
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
            except QiskitError:
                job_pickle._backend = self.backend

            # overwrite result function with the obtained result
            def result_():
                return result

            job_pickle.result = result_
            self._cache.store_file(hash_value, job_pickle)
            self._logger.info(f"Stored job in cache with hash value: {{}}".format(hash_value))

        return job

    def estimator_run(self, circuits, observables, parameter_values=None, **kwargs: Any) -> Job:
        """
        Function similar to the Qiskit Sampler run function, but this one includes caching,
        automatic session handling, and restarts of failed jobs.

        Args:
            circuits: Quantum circuits to execute.
            observables: Observable to measure.
            parameter_values: Values for the parameters in circuits.
            kwargs (Any): Additional arguments that are passed to the estimator.

        Returns:
            A qiskit job containing the results of the run.
        """

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
                ]
            )
        else:
            hash_value = None

        return self._primitive_run(run, "estimator", hash_value)

    def sampler_run(self, circuits, parameter_values=None, **kwargs: Any) -> Job:
        """
        Function similar to the Qiskit Sampler run function, but this one includes caching,
        automatic session handling, and restarts of failed jobs.

        Args:
            circuits: Quantum circuits to execute.
            parameter_values: Values for the parameters in circuits.
            kwargs (Any): Additional arguments that are passed to the estimator.

        Returns:
            A qiskit job containing the results of the run.
        """

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
                ]
            )
        else:
            hash_value = None

        return self._primitive_run(run, "sampler", hash_value)

    def get_estimator(self):
        """
        Returns a Estimator primitive that overwrites the Qiskit Estimator primitive.
        This Estimator runs all executions through the Executor and
        includes result caching, automatic session handling, and restarts of failed jobs.
        """
        return ExecutorEstimator(executor=self, options=self._options_estimator)

    def get_sampler(self):
        """
        Returns a Sampler primitive that overwrites the Qiskit Sampler primitive.
        This Sampler runs all executions through the Executor and
        includes result caching, automatic session handling, and restarts of failed jobs.
        """
        return ExecutorSampler(executor=self, options=self._options_sampler)

    @property
    def optree_executor(self) -> str:
        """A string that indicates which executor is used for OpTree execution."""
        if self._estimator is not None:
            return "estimator"
        elif self._sampler is not None:
            return "sampler"
        else:  #  default if nothing is set -> use estimator
            return "estimator"

    def backend_run(self, run_input, **options):
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
        if num_shots is None:
            return None

        if str(self._backend) == "statevector_simulator":
            # Skip setting shots for statevector simulator
            return None

        # Update shots in estimator primitive
        if self._estimator is not None:
            try:
                execution = self._estimator.options.get("execution")
                execution["shots"] = num_shots
                self._estimator.set_options(execution=execution)
            except:
                try:
                    self._estimator.set_options(shots=num_shots)
                except:
                    pass

        # Update shots in sampler primitive
        if self._sampler is not None:
            try:
                execution = self._sampler.options.get("execution")
                execution["shots"] = num_shots
                self._sampler.set_options(execution=execution)
            except:
                try:
                    self._sampler.set_options(shots=num_shots)
                except:
                    pass

        # Update shots in estimator options for not yet created estimators
        if self._options_estimator is not None:
            try:
                self._options_estimator["execution"]["shots"] = num_shots
            except:
                try:
                    self._options_estimator["shots"] = num_shots
                except:
                    pass

        # Update shots in sampler options for not yet created samplers
        if self._options_sampler is not None:
            try:
                self._options_sampler["execution"]["shots"] = num_shots
            except:
                try:
                    self._options_sampler["shots"] = num_shots
                except:
                    pass

        # Update shots in backend
        if self._backend is not None:
            self._backend.options.shots = num_shots

        self._logger.info("Set shots to {}".format(num_shots))

    def get_shots(self) -> int:
        """Getter for the number of shots.

        Returns:
            Returns the number of shots that are used for the current evaluation.
        """
        shots = None
        if self._estimator is not None or self._sampler is not None:
            shots_estimator = 0
            shots_sampler = 0
            if self._estimator is not None:
                try:
                    execution = self._estimator.options.get("execution")
                    shots_estimator = execution["shots"]
                except:
                    try:
                        shots_estimator = self._estimator.options.get("shots", 0)
                    except:
                        shots_estimator = 0
            if self._sampler is not None:
                try:
                    execution = self._sampler.options.get("execution")
                    shots_sampler = execution["shots"]
                except:
                    try:
                        shots_sampler = self._sampler.options.get("shots", 0)
                    except:
                        shots_sampler = 0

            if self._estimator is not None and self._sampler is not None:
                if shots_estimator != shots_sampler:
                    raise ValueError(
                        "The number of shots of the given \
                                      Estimator and Sampler is not equal!"
                    )
        elif self._backend is not None:
            shots = self._backend.options.shots
            if "statevector_simulator" in str(self._backend):
                shots = 0
        else:
            return None  # No shots available

        if shots == 0:
            shots = None
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
        if self._service is not None:
            self._session = Session(
                self._service, backend=self._backend, max_time=self._max_session_time
            )
            self._session_active = True
            self._logger.info(f"Executor created a new session.")
        else:
            raise RuntimeError("Session can not started because of missing service!")

    def close_session(self):
        """Closes the current session, is called automatically."""
        if self._session is not None:
            self._logger.info(f"Executor closed session: {{}}".format(self._session.session_id))
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

    def set_options_estimator(self, **fields):
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self.estimator.set_options(**fields)
        self._options_estimator = self.estimator.options

    def set_options_sampler(self, **fields):
        """Set options values for the sampler.

        Args:
            **fields: The fields to update the options
        """
        self.sampler.set_options(**fields)
        self._options_sampler = self.sampler.options

    def reset_options_estimator(self, options: Union[Options, qiskit_ibm_runtime_Options]):
        """
        Overwrites the options for the estimator primitive.

        Args:
            options: Options for the estimator
        """
        self._options_estimator = options

        if isinstance(options, qiskit_ibm_runtime_Options):
            self.estimator._options = asdict(options)
        else:
            self.estimator._run_options = Options()
            self.estimator._run_options.update_options(**options)

    def reset_options_sampler(self, options: Union[Options, qiskit_ibm_runtime_Options]):
        """
        Overwrites the options for the sampler primitive.

        Args:
            options: Options for the sampler
        """
        self._options_sampler = options

        if isinstance(options, qiskit_ibm_runtime_Options):
            self.sampler._options = asdict(options)
        else:
            self.sampler._run_options = Options()
            self.sampler._run_options.update_options(**options)


class ExecutorEstimator(BaseEstimator):
    """
    Special Estimator Primitive that uses the Executor service.

    Usefull for automatic restarting sessions and caching results.
    The object is created by the Executor method get_estimator()

    Args:
        executor (Executor): The executor service to use
        options: Options for the estimator

    """

    def __init__(self, executor: Executor, options=None):
        if isinstance(options, Options) or isinstance(options, qiskit_ibm_runtime_Options):
            options_ini = asdict(copy.deepcopy(options))
        else:
            options_ini = options

        super().__init__(options=options_ini)
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
    ) -> Job:
        """Has to be passed through, otherwise python will complain about the abstract method.
        Input arguments are the same as in Qiskit's estimator.run().
        """
        return self._executor.estimator_run(
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
    ) -> Job:
        """
        Overwrites the sampler primitive run method, to evaluate expectation values.
        Uses the Executor class for automatic session handling.

        Input arguments are the same as in Qiskit's estimator.run()

        """
        return self._executor.estimator_run(
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
        """Parameters of the quantum circuits.

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
        self._executor.clear_estimator_cache()

    def set_options(self, **fields):
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self._executor.estimator.set_options(**fields)
        self._executor._options_estimator = self._executor.estimator.options


class ExecutorSampler(BaseSampler):
    """
    Special Sampler Primitive that uses the Executor service.

    Useful for automatic restarting sessions and caching the results.
    The object is created by the executor method get_sampler()

    Args:
        executor (Executor): The executor service to use
        options: Options for the sampler

    """

    def __init__(self, executor: Executor, options=None):
        if isinstance(options, Options) or isinstance(options, qiskit_ibm_runtime_Options):
            options_ini = asdict(copy.deepcopy(options))
        else:
            options_ini = options

        super().__init__(options=options_ini)
        self._executor = executor

    def run(
        self,
        circuits,
        parameter_values=None,
        **run_options,
    ) -> Job:
        """
        Overwrites the sampler primitive run method, to evaluate circuits.
        Uses the Executor class for automatic session handling.

        Input arguments are the same as in Qiskit's sampler.run()

        """
        return self._executor.sampler_run(
            circuits=circuits,
            parameter_values=parameter_values,
            **run_options,
        )

    def _run(
        self,
        circuits,
        parameter_values=None,
        **run_options,
    ) -> Job:
        """
        Overwrites the sampler primitive run method, to evaluate circuits.
        Uses the Executor class for automatic session handling.

        Input arguments are the same as in Qiskit's sampler.run()

        """
        return self._executor.sampler_run(
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
        self._executor.clear_sampler_cache()


class ExecutorCache:
    """Cache for jobs that are created by Primitives

    Args:
        folder (str): Folder to store the cache

    """

    def __init__(self, logger, folder: str = ""):
        self._folder = folder
        # Check if folder exist, creates the folder otherwise
        try:
            import os

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
            if type(variable_) == list:
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
        except:
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
