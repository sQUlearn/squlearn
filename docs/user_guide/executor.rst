.. _executor_user_guide:

.. currentmodule:: squlearn.Executor

##################
The Executor Class
##################

Overview
--------

The :class:`Executor <squlearn.Executor>` class is the central component of sQUlearn, responsible
for running all quantum jobs.
Both high- and low-level methods utilize the Executor class to execute circuits or to run other
quantum jobs.
The class provides a high-level interface to the simulators of PennyLane and Qiskit,
as well as manages access to real QC hardware as for example IBM Quantum.
It features a variety of comfort features, such as result caching,
automatic restarts of failed jobs, logging of all actions, and Qiskit Session handling.
The Executor class is also responsible for handling the execution environment, and can be
initialized with a variety of objects that specify the execution environment (see figure below).
The following figure summarizes the structure of the Executor class; ingoing arrows indicate
that the Executor class can be initialized or adjusted with the corresponding object.
Outgoing arrows indicate that the Executor class can return the corresponding object.

.. image:: ../_static/util/executor.png
    :width: 600
    :align: center

Key Features of the Executor
----------------------------

The Executor class provides the following key comfort features when executing a quantum job:

- **Result caching:** Enables caching of results to avoid redundant job executions, and enable
  restarts of failed executions. Caching is enabled as default for remote executions, but can also
  be manually activated for local execution.
  The cached files are named after the hash out of different properties of the quantum job,
  that include the backend name, the circuit, the execution options, etc.
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
- **Random Seeds for shot-based simulators:** A random seeds can be specified for the PennyLane
  and Qiskit shot-based simulators to make the computations utilizing this Executor object
  reproducible. The random seeds can be set manually by specifying the ``seed`` argument.
- **Modified Qiskit Primitives:** The Executor allows the creation of modified Qiskit Primitives
  that function exactly as the Qiskit primitives but leverage the comfort features mentioned above.
  The primitives can be obtained utilizing the :meth:`get_estimator` and
  :meth:`get_sampler` methods. The modified primitives route all executions through the Executor
  class, and thus benefit from all comfort features. The primitives are fully compatible with
  the Qiskit framework, and can be used in the same way as regular primitives.
  The Executor primitives are automatically utilized in the sQUlearn
  sub-programs.
- **Qiskit Session handling:** Automatically manages the creation and handling of Qiskit sessions.
  If Sessions are time out, the Executor automatically creates a new session and re-executes the
  job.

Initialization of the Executor class
------------------------------------

The Estimator can be initialized with various inputs (``execution=``) that specify the
execution environment:

- Default initialization with no backend specified: If no backend is specified, the Executor
  class is initialized utilizing the fast PennyLane simulators.
  If no shots are specified (``shots=None``) the statevector simulator being utilized,
  which is the default. If shots are specified, the shot-based simulator is utilized.

  .. jupyter-execute::

      from squlearn import Executor
      # Initialize the Executor with the PennyLane statevector simulator
      # as default, shots=None (statevector simulator)
      executor = Executor()

      # Initialize the Executor with the PennyLane shot-based simulator
      executor = Executor(shots=1234)


- A string specifying the local simulator backend: Qiskit simulators are available by ``"qiskit"``,
  ``"statevector_simulator"`` and  ``"qasm_simulator"``; PennyLane simulators can be initialized by
  ``"pennylane"`` or ``"default.qubit"``.

  .. jupyter-execute::

      from squlearn import Executor
      # Initialize the Executor with the statevector simulator
      executor = Executor("statevector_simulator")

      # Initialize the Executor with the qasm simulator
      executor = Executor("qasm_simulator", shots=1234)

      # Initialize the Executor with qiksit is equivalent to "statevector_simulator"
      executor = Executor("qiskit")

      # Initialize the Executor with the PennyLane statevector simulator
      executor = Executor("pennylane")

      # Initialize the Executor with the PennyLane shot-based simulator
      executor = Executor("default.qubit", shots=1234)

- A backend following the Qiskit backend standard, e.g. a Qiskit Aer backend, a fake backend,
  a real IBM Quantum backend. This allows also the utilization
  of other quantum computing backends, as long as they provide a Qiskit backend
  class.

  .. jupyter-execute::

    from squlearn import Executor
    from qiskit_aer import Aer
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    # Executor with the Aer statevector simulator
    executor = Executor(Aer.get_backend("aer_simulator"))

    # Executor with the FakeManilaV2 backend
    executor = Executor(FakeManilaV2())

- A PennyLane device of a quantum computing backend. In the following example, the Executor is
  initialized with a AWS device. Note that this example requires the PennyLane AWS plugin to be
  installed (``pip install amazon-braket-pennylane-plugin``) and the AWS credentials to be
  configured.

  .. code-block:: python

    from squlearn import Executor
    import pennylane as qml

    # Initialize the Executor with the PennyLane default.qubit device
    dev = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", wires=2)
    executor = Executor(dev, shots = 1234)


- A backend from the Qiskit Runtime Service, which utilizes the execution of quantum jobs on
  IBM Quantum. Sessions and Primitives are automatically created and managed by the Executor class.

  .. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
    executor = Executor(service.get_backend('ibm_kyoto'))

  It is also possible to pass a list of backends from which the most suited backend is chosen
  automatically (see :ref:`Automatic backend selection <autoselect>`)

- Another way is to just pass the Qiskit Runtime Service to the executor. In this case, the backend
  will be chosen automatically, for more details see :ref:`Automatic backend selection <autoselect>`.

  .. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
    executor = Executor(service)

- A pre-initialized Session object, which can be used to execute quantum jobs on the Qiskit
  Runtime Service

  .. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
    session = service.create_session(backend = service.get_backend('ibm_kyoto'))
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

    session = service.create_session(backend = service.get_backend('ibm_kyoto'))
    estimator = Estimator(session=session, options_estimator=options)
    executor = Executor(estimator)

- If only the ``backend.run`` execution is wanted, this can be achieved by utilizing the
  Qiskit IBM Provider package. However, most sQUlearn algorithms a build upon primitives,
  and therefore, this access is not recommended, since it is likely to be deprecated in the future.

  .. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_provider import IBMProvider

    IBMProvider.save_account(token="INSERT_YOUR_TOKEN_HERE")
    provider = IBMProvider(instance="hub/group/project")
    executor = Executor(provider.get_backend("ibmq_qasm_simulator"))

The following code shows an example for configuring the Executor class with a
backend from the Qiskit Runtime Service and setting options for caching, logging and restarts:

.. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService, Options

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")

    options = Options()
    options.execution.shots = 1000
    options.optimization_level = 0  # No optimization in transpilation
    options.resilience_level = 2  # ZNE

    executor = Executor(service.get_backend('ibm_kyoto'), # Specify the backend
                         cache_dir='cache' # Set cache folder to "cache"
                         caching=True, # Enable caching default for remote executions
                         log_file="executor.log", # Set-up logging file
                         wait_restart=600,  # Set 10 min pause between restarts of Jobs
                         max_jobs_retries=10, # Set maximum number of restarts to 10 before aborting
                         options_estimator=options # Set options for the Estimator primitive
                         )

    executor.set_shots(1234) # Shots an be adjusted after initialization


Utilizing Executor Primitives in Qiskit Routines
-------------------------------------------------

The Executor class provides an Estimator and Sampler primitive that are compatible with the
Qiskit framework. This is only possible, in case the Executor class is initialized with a backend
compatible with Qiskit (and not PennyLane). The primitives can be obtained by the
:meth:`get_estimator` and :meth:`get_sampler` methods of the Executor class.
The primitives automatically utilized the parent Executor class for all executions,
and thus benefit from all comfort features of the Executor.

The following example shows, how to evaluate the Quantum Fisher Information utilizing the
Executor primitive (see `QFI in Qiskit <https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.gradients.QFI.html>`_)

  .. jupyter-execute::

      from squlearn import Executor
      from qiskit_algorithms.gradients import LinCombQGT, QFI
      from qiskit.quantum_info import Pauli
      # Executor initialization (other ways are possible, see above)
      executor = Executor(execution="statevector_simulator")
      # This creates the QFI primitive that utilizes the Estimator of the Executor class
      qfi = QFI(LinCombQGT(executor.get_estimator()))
      # Quantum Fischer Information can be evaluated as usual with qfi.run()

If only the run function of the Executor Primitive is wanted, this can be achieved by utilizing the
Executor class function :meth:`estimator_run` and :meth:`sampler_run`.

Note that the attributes :meth:`estimator` and :meth:`sampler` of the Executor class are
not creating or referring to the Executor primitives! Instead they refer to the
Qiskit Primitives used internally that do not utilize any caching, restarts, etc.


Setting Options for Qiskit Primitives
--------------------------------------

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

    executor = Executor(service.get_backend("ibm_kyoto"),options_estimator=options)

Options can be adjusted by the :meth:`set_options` method of the Primitives that are created by the
Executor class.

.. code-block:: python

    from squlearn import Executor
    from qiskit_ibm_runtime import QiskitRuntimeService, Options

    service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")

    executor = Executor(service.get_backend("ibm_kyoto"))
    estimator = executor.get_estimator()
    estimator.set_options(resilience_level=2)

.. _autoselect:

Automatic backend selection
---------------------------

sQUlearn offers automatic determination of the most suitable backend for the
Quantum Machine Learning (QML) problem. This is facilitated by initializing the Executor with a
list of supported backends, which includes real IBM backends or simulated fake backends.
Alternatively, users can pass a Service from Qiskit IBM Runtime, where all appropriate backends
are automatically considered. The selection process leverages the Mapomatic tool `[1]`_ and also
identifies the best transpilation for each backend.

Two modes for backend selection are currently implemented:

* ``"quality"``: This mode automatically selects the best backend, which is also the default
  setting. An estimation of the expected error is calculated for the optimal
  transpiled circuit of the QML model. Backend selection is performed by the
  mapomatic tool.

* ``"speed"``: In this mode, the backend with the smallest queue is automatically selected.

Once a backend is chosen, it remains fixed throughout the program unless explicitly changed.
It's worth noting that if other QML models are initialized, the chosen backend remains
consistent. The backend selection can be unset using the unset_backend method.
However, it's important to note that this action only triggers the selection process for
QML models initialized thereafter.

Below is an example demonstrating the selection of different simulated noisy IBM backends.
We set up a small Quantum Neural Network (QNN) example and train it on the most suitable backend:

.. jupyter-execute::

   import numpy as np
   from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeBelemV2, FakeAthensV2
   from squlearn.util import Executor
   from squlearn.qnn import QNNRegressor
   from squlearn.observables import SummedPaulis
   from squlearn.encoding_circuit import ChebyshevPQC
   from squlearn.optimizers import Adam
   from squlearn.qnn.loss import SquaredLoss

   backends = [FakeBelemV2(), FakeAthensV2(), FakeManilaV2()]
   executor = Executor(backends, shots=10000)
   qnn = QNNRegressor(
       ChebyshevPQC(2, 1),
       SummedPaulis(2),
       executor,
       SquaredLoss(),
       Adam({'maxiter':2}), # Two iteration for demonstration purposes only
       callback=None # Remove print of the progress bar for cleaner output
   )
   qnn.fit(np.array([[0.25],[0.75]]),np.array([0.25,0.75]))
   print("Chosen backend:", executor.backend)

In the following example, the service is used for initializing the Executor, and
the mode is switched to ``"speed"``. The executor is then utilized for running a
Quantum Kernel Ridge Regression (QKRR) in which the backend is chosen automatically.

.. code-block:: python

   import numpy as np
   from squlearn import Executor
   from qiskit_ibm_runtime import QiskitRuntimeService
   from squlearn.encoding_circuit import ChebyshevRx
   from squlearn.kernel import FidelityKernel, QKRR

   # Executor is initialized with a service, and considers all available backends
   # (except simulators)
   service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_YOUR_TOKEN_HERE")
   executor = Executor(service, auto_backend_mode="speed")

   # Create a QKRR model with a FidelityKernel and the ChebyshevRx encoding circuit
   qkrr = QKRR(FidelityKernel(ChebyshevRx(4,1),executor))

   # Backend is automatically selected based on the smallest queue
   # All the following functions will be executed on the selected backend
   X_train, y_train = np.array([[0.1],[0.2]]), np.array([0.1,0.2])
   qkrr.fit(X_train, y_train)


In-QPU parallelization
----------------------

The Executor class supports QPU (Quantum Processing Unit) parallelization, enabling simultaneous
measurements of the same quantum circuit on the quantum hardware by duplicating the circuit.
This feature significantly enhances the efficiency of quantum computation by reducing the number
of required shots.
However, it's essential to note that utilizing QPU parallelization may introduce additional noise
and hardware errors due to increased qubitd utilization and cross-talk.

The QPU parallelization parameter ``qpu_parallelization`` determines the number of parallel
evaluations of the Quantum Circuit on the QPU. When qpu_parallelization is set to an integer
value, it specifies the exact number of parallel executions, for instance:

.. code-block:: python

   executor = Executor(..., qpu_parallelization=4)

This configuration instructs the Executor to duplicate all circuit executions four times and
execute them concurrently on the QPU.

Alternatively, setting ``qpu_parallelization`` to ``"auto"`` enables automatic determination of
the parallelization level. In this mode, the Executor dynamically adjusts the degree of
parallelization to maximize the number of possible parallel circuit measurements, for example:

.. code-block:: python

   executor = Executor(..., qpu_parallelization="auto")

If activated, QPU parallelization is automatically applied to all Primitives created by the
Executor. By leveraging QPU parallelization, users can expedite the execution of quantum
circuits on real hardware. Nonetheless, it's crucial to weigh the benefits against the potential
drawbacks of increased noise and errors. By default, ``qpu_parallelization`` is set to ``None``,
implying no parallelization is considered unless explicitly specified.
The transpilation considers the duplicated circuits, hence the duplicated circuits are placed
accordingly to the qubit layout of the backend.
In principle, this feature can be used in combination with the automatic backend selection,
but the backend selection process might take a substantial time due to the increased number of
circuits that need to be transpiled.

.. seealso::

   * :class:`Executor <squlearn.Executor>`
   * `Qiskit Runtime <https://quantum-computing.ibm.com/lab/docs/iql/runtime>`_
   * `Qsikit Primitives <https://qiskit.org/documentation/apidoc/primitives.html>`_
   * `Mapomatic: Automatic mapping of compiled circuits to low-noise sub-graphs <https://github.com/qiskit-community/mapomatic>`_
   * `PennyLane Devices <https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html>`_

.. rubric:: References

_`[1]` P. D. Nation and M. Treinish "Suppressing quantum circuit errors due to system variability".
`PRX Quantum 4(1) 010327 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010327>`_ (2023)
