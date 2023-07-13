##################################
Tutorial: Projected Quantum Kernel
##################################

.. plot::

   from squlearn.feature_map import ChebPQC
   pqc = ChebPQC(4, 1, 2, closed=False)
   pqc.draw()

This tutorial will show you how to use the :class:`~squlearn.kernel.matrix.ProjectedQuantumKernel` class.

Here are some information about the class:

.. autoclass:: squlearn.kernel.matrix.ProjectedQuantumKernel
  :members: