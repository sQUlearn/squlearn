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

.. _feature_maps:

Feature Maps
------------------------------------

.. automodule:: squlearn.feature_map
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   feature_map.YZ_CX_FeatureMap
   feature_map.HighDimFeatureMap
   feature_map.QEKFeatureMap
   feature_map.ChebyshevTower
   feature_map.ChebPQC
   feature_map.HZCRxCRyCRz
   feature_map.ChebRx
   feature_map.ParamZFeatureMap
   feature_map.QiskitZFeatureMap
   feature_map.QiskitFeatureMap

Feature Map Tools
------------------------------------

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   feature_map.feature_map_base.FeatureMapBase
   feature_map.PrunedFeatureMap
   feature_map.LayeredFeatureMap
   feature_map.FeatureMapDerivatives
   feature_map.TranspiledFeatureMap

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   feature_map.automated_pruning
   feature_map.pruning_from_QFI


.. _operators:

Operators
------------------------------------

.. automodule:: squlearn.expectation_operator
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   expectation_operator.SinglePauli
   expectation_operator.SummedPaulis
   expectation_operator.SingleProbability
   expectation_operator.SummedProbabilities
   expectation_operator.IsingHamiltonian
   expectation_operator.CustomExpectationOperator

Operator Tools
------------------------------------

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   expectation_operator.expectation_operator_base.ExpectationOperatorBase
   expectation_operator.expectation_operator_derivatives.ExpectationOperatorDerivatives


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

.. automodule:: squlearn.optimizers
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   optimizers.adam.Adam
   optimizers.optimizers_wrapper.LBFGSB
   optimizers.optimizers_wrapper.SLSQP
   optimizers.optimizers_wrapper.SPSA

Base Classes
------------------------------------

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

    expectation_operator.expectation_operator_base.ExpectationOperatorBase
    feature_map.feature_map_base.FeatureMapBase
    kernel.matrix.kernel_matrix_base.KernelMatrixBase
    kernel.optimization.kernel_loss_base.KernelLossBase
    kernel.optimization.kernel_optimization_base.KernelOptimizerBase
    optimizers.optimizer_base.OptimizerBase
    qnn.base_qnn.BaseQNN
    qnn.loss.LossBase






