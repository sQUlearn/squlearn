.. _api_reference:

=============
API Reference
=============



QML Regressors
=====================================

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   kernel.ml.QSVR
   kernel.ml.QKRR
   kernel.ml.QGPR
   qnn.QNNRegressor


QML Classifiers
======================================

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   kernel.ml.QSVC
   kernel.ml.QGPC
   qnn.QNNClassifier


Circuit Design
====================================

.. _encoding_circuits:

Encoding Circuits
------------------------------------

.. automodule:: squlearn.encoding_circuit
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   encoding_circuit.YZ_CX_EncodingCircuit
   encoding_circuit.HighDimEncodingCircuit
   encoding_circuit.HubregtsenEncodingCircuit
   encoding_circuit.ChebyshevTower
   encoding_circuit.ChebyshevPQC
   encoding_circuit.MultiControlEncodingCircuit
   encoding_circuit.ChebyshevRx
   encoding_circuit.ParamZFeatureMap
   encoding_circuit.QiskitEncodingCircuit

Encoding Circuit Tools
------------------------------------

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   encoding_circuit.encoding_circuit_base.EncodingCircuitBase
   encoding_circuit.PrunedEncodingCircuit
   encoding_circuit.LayeredEncodingCircuit
   encoding_circuit.EncodingCircuitDerivatives
   encoding_circuit.TranspiledEncodingCircuit

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   encoding_circuit.automated_pruning
   encoding_circuit.pruning_from_QFI


.. _operators:

Operators
------------------------------------

.. automodule:: squlearn.observables
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   observables.SinglePauli
   observables.SummedPaulis
   observables.SingleProbability
   observables.SummedProbabilities
   observables.IsingHamiltonian
   observables.CustomObservable

Operator Tools
------------------------------------

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   observables.observable_base.ObservableBase
   observables.observable_derivatives.ObservableDerivatives


Execution Tools
===========================

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   Executor


Core
==============================

Quantum Kernel Core
------------------------------------

.. automodule:: squlearn.kernel.matrix
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   kernel.matrix.FidelityKernel
   kernel.matrix.ProjectedQuantumKernel

.. automodule:: squlearn.kernel.optimization
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   kernel.optimization.kernel_optimizer.KernelOptimizer
   kernel.optimization.negative_log_likelihood.NLL
   kernel.optimization.target_alignment.TargetAlignment

QNN Core
------------------------------------

.. automodule:: squlearn.qnn
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   qnn.qnn.QNN
   qnn.qnn.Expec
   qnn.loss.SquaredLoss
   qnn.loss.VarianceLoss
   qnn.loss.ParameterRegularizationLoss

Tools for training QNNs

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   qnn.get_variance_fac
   qnn.get_lr_decay
   qnn.ShotsFromRSTD
   qnn.training.train
   qnn.training.train_mini_batch


Implemented optimizers
------------------------------------

.. automodule:: squlearn.optimizers
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   optimizers.Adam
   optimizers.LBFGSB
   optimizers.SLSQP
   optimizers.SPSA

OpTree Data Structure
------------------------------------

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

    util.OpTree
    util.optree.OpTreeList
    util.optree.OpTreeSum
    util.optree.OpTreeCircuit
    util.optree.OpTreeOperator
    util.optree.OpTreeExpectationValue
    util.optree.OpTreeMeasuredOperator
    util.optree.OpTreeContainer
    util.optree.OpTreeValue


Base Classes
------------------------------------

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

    observables.observable_base.ObservableBase
    encoding_circuit.encoding_circuit_base.EncodingCircuitBase
    kernel.matrix.kernel_matrix_base.KernelMatrixBase
    kernel.optimization.kernel_loss_base.KernelLossBase
    kernel.optimization.kernel_optimization_base.KernelOptimizerBase
    optimizers.optimizer_base.OptimizerBase
    qnn.base_qnn.BaseQNN
    qnn.loss.LossBase






