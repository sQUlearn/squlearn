.. _api_reference:

=============
API Reference
=============



Implemented High-Level QML Regressors
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


Implemented High-Level QML Classifiers
======================================

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   kernel.ml.QSVC
   kernel.ml.QGPC
   qnn.QNNClassifier


Implemented feature maps in squlearn
====================================

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

Implemented tools for feature maps
==================================

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   feature_map.FeatureMapBase
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


Implemented operators for expectation values
============================================

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

Tools for expectation operators
================================

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   expectation_operator.ExpectationOperatorBase
   expectation_operator.ExpectationOperatorDerivatives

Quantum Kernel implementations
==============================

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


Low-level QNN implementation
============================

.. automodule:: squlearn.qnn
    :no-members:
    :no-inherited-members:

.. currentmodule:: squlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   qnn.qnn.QNN
   qnn.qnn.expec
   qnn.loss.SquaredLoss
   qnn.loss.VarianceLoss
   qnn.loss.ParameterRegularizationLoss


Base classes of sQUlearn
========================

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






