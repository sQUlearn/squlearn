.. _executor_user_guide:

.. currentmodule:: squlearn.Executor

##################
The Executor Class
##################

Overview
--------

The :class:`Executor <squlearn.Executor>` class is the central component of sQUlearn, responsible
for running all quantum jobs.
Both high- and low-level methods utilize the Executor class to execute circuits or other
quantum jobs.
The class provides a high-level interface to the Qiskit simulators and access to IBM
Quantum systems.
It features a variety of comfort features, such as automatic session handling, result caching,
restarts and it automatically creates the necessary Qiskit primitives when they are
required in the sub-program.
The Executor class is also responsible for handling the execution environment, and can be
initialized with a variety of objects that specify the execution environment (see figure below).
The following figure summarizes the structure of the Executor class; ingoing arrows indicate
that the Executor class can be initialized or adjusted with the corresponding object.
Outgoing arrows indicate that the Executor class can return the corresponding object.

.. image:: ../_static/util/executor.png
    :width: 525
    :align: center

Key Features of the Executor
----------------------------

The Executor class provides the following key comfort features when executing a quantum job:

- **Session handling:** Automatically manages the creation and handling of Qiskit sessions.
  If Sessions are time out, the Executor automatically creates a new session and re-executes the
  job.
- **Result caching:** Enables caching of results to avoid redundant job executions, and enable
  restarts of failed executions. Caching is enabled as default only for remote executions.
  The cached files are named after the hash out of different properties of the quantum job,
  that include the backend name, the circuit, the options of the primitive, etc.
  Before running a job, the Executor checks if a cached result exists and returns it if it does.
  The caching can be disabled by setting the ``caching`` argument to ``False``; the folder for
  the cached results can be specified by the ``cache_dir`` argument,
  (default folder: ``"_cache"``).
- **Automatic restarts:** In case the job execution fails or is canceled, the Executor
  automatically resubmits and restarts the job up to a specified number of times.
  The number of restarts can be specified via the ``max_jobs_retries`` argument, the pause
  between restarts can be adjusted by the ``wait_restart`` argument.
- **Logging:** The executor automatically logs all actions to a log file that ca be specified
  via the ``log_file`` argument.
- **Modified Qiskit Primitives:** The Executor allows the creation of modified Qiskit Primitives
  that function similarly to regular primitives but leverage the comfort features mentioned above.
  The primitives can be obtained utilizing the :meth:`get_estimator` and
  :meth:`get_sampler` methods. The modified primitives route all executions through the Executor
  class, and thus benefit from all comfort features. The primitives are fully compatible with
  the Qiskit framework, and can be used in the same way as regular primitives.
  The Executor primitives are automatically utilized in the sQUlearn
  sub-programs.

Initialization of the Executor class
------------------------------------

The Estimator can be initialized with various inputs (``execution=``) that specify the 
execution environment:

- A string specifying the simulator backend (e.g., ``"statevector_simulator"`` or
  ``"qasm_simulator"``).

  .. code-block:: python

      from squlearn import Executor
      executor = Executor(execution="statevector_simulator")

  The argument ``execution`` is omitted in the following examples since it is not necessary,
  if the environment is specified as the first argument.

- A backend following the Qiskit backend standard, e.g. a Qiskit Aer backend, a fake backend,
  a real IBM Quantum backend. This allows in principle the utilization
  of other quantum computing backends as IBM Quantum, as long as they provide a Qiskit backend
  class.

  .. code-block:: python

    from squlearn import Executor
    from qiskit_aer import Aer

    executor = Executor(Aer.get_backend("aer_simulator"))

- A backend from the Qiskit Runtime Service, which utilizes the execution of quantum jobs on
  IBM Quantum utilizing Sessions and Primitives.

  .. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
    executor = Executor(service.get_backend('ibm_nairobi'))

- A pre-initialized Session object, which can be used to execute quantum jobs on the Qiskit
  Runtime Service

  .. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
    session = service.create_session()
    executor = Executor(session)

- Pre-configured Primitive with options for transpiling and error mitigation. The Executor class
  utilizes the options of the inputted primitive, and automatically creates a new primitive with
  the same options if necessary. Note Options from an Estimator are not automatically copied to
  the Sampler, and vice versa.

  .. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Options

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")

    options = Options()
    options.execution.shots = 1000
    options.optimization_level = 0  # No optimization in transpilation
    options.resilience_level = 2  # ZNE

    estimator = Estimator(session=service.create_session(), options_estimator=options)
    executor = Executor(estimator)

- If only the ``backend.run`` execution is wanted, this can be achieved by utilizing the
  Qiskit IBM Provider package. However, most sQUlearn algorithms a build upon primitives.
  However, this access is not recommended, since it is likely to be deprecated in the future.

  .. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_provider import IBMProvider

    IBMProvider.save_account(token="INSERT_YOUR_TOKEN_HERE")
    provider = IBMProvider(instance="hub/group/project")
    executor = Executor(provider.get_backend("ibmq_qasm_simulator"))

An example for configuring the Executor class with a backend from the Qiskit Runtime Service
and setting options for caching, logging and restarts:

.. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService, Options

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")

    options = Options()
    options.execution.shots = 1000
    options.optimization_level = 0  # No optimization in transpilation
    options.resilience_level = 2  # ZNE

    estimator = Executor(service.get_backend('ibm_nairobi'), # Specify the backend
                         cache_dir='cache' # Set cache folder to "cache"
                         caching=True, # Enable caching default for remote executions
                         log_file="executor.log", # Set-up logging file
                         wait_restart=600,  # Set 10 min pause between restarts of Jobs
                         max_jobs_retries=10, # Set maximum number of restarts to 10 before aborting
                         options_estimator=options # Set options for the Estimator primitive
                         )


Utilizing Executor Primitives in Qiskit Routines
-------------------------------------------------

The Executor class provides an Estimator and Sampler primitive that are compatible with the
Qiskit framework. The primitives can be obtained by the :meth:`get_estimator` and
:meth:`get_sampler` methods of the Executor class. The primitives automatically utilized the parent
Executor class for all executions, and thus benefit from all comfort features of the Executor.

The following example shows, how to evaluate the Quantum Fisher Information utilizing the
Executor primitive (see `QFI in Qiskit <https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.gradients.QFI.html>`_)

  .. jupyter-execute::

      from squlearn import Executor
      from qiskit_algorithms.gradients import LinCombQGT, QFI
      from qiskit.quantum_info import Pauli
      # Executor intialization (other ways are possible, see above)
      executor = Executor(execution="statevector_simulator")
      # This creates the QFI primitive that utilizes the Estimator of the Executor class
      qfi = QFI(LinCombQGT(executor.get_estimator()))
      # Quantum Fischer Information can be evaluated as usual with qfi.run()

If only the run function of the Executor Primitive is wanted, this can be achieved by utilizing the
Executor class function :meth:`estimator_run` and :meth:`sampler_run`.

Note that the attributes :meth:`estimator` and :meth:`sampler` of the Executor class are
not creating or referring to the Executor primitives! Instead they refer to the
Qiskit Primitives used internally that do not utilize any caching, restarts, etc.


Setting Options for Primitives
------------------------------

Options for the Primitives can be provided through the ``options_estimator`` and
``options_sampler`` arguments, but they are also automatically copied from inputted primitives.

.. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService, Options

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")

    options = Options()
    options.optimization_level = 0 # No optimization in transpilation
    options.execution.shots = 5000
    options.resilience_level = 0 # No Mitigation

    executor = Executor(service.get_backend("ibm_nairobi"),options_estimator=options)

Options can be adjusted by the :meth:`set_options` method of the Primitives that are created by the
Executor class.

.. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService, Options

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")

    executor = Executor(service.get_backend("ibm_nairobi"))
    estimator = executor.get_estimator()
    estimator.set_options(resilience_level=2)

.. seealso::

   * :class:`Executor <squlearn.Executor>`
   * `Qiskit Runtime <https://quantum-computing.ibm.com/lab/docs/iql/runtime>`_
   * `Qsikit Primitives <https://qiskit.org/documentation/apidoc/primitives.html>`_
