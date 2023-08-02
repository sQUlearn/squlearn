.. _quantum_kernel_methods:

.. currentmodule:: squlearn.kernel
.. currentmodule:: squlearn.kernel.ml 
.. currentmodule:: squlearn.kernel.matrix 

######################
Quantum Kernel Methods
######################

Quantum Kernel Methods are amongst the most promising approaches to (supervised) Quantum Machine 
Learning (QML) since it has been shown [REF] that they can be formally embedded into the rich 
mathematical framework of classical kernel theory. The key idea in kernel theory is to solve the 
general task of machine learning, i.e. finding and studying patterns in data, in a high dimensional 
feature space - the reproducing kernel Hilbert space (RKHS). The mapping from the original space to 
the RKHS (which in general can be infinite-dimensional) is performed by so called feature maps. 

:math:`K(x,y) = \Braket{\phi(x), \phi(y)}`

The key point of Quantum Kernel methods is that they can be fundamentally formulated as a classical kernel method whose kernel is computed by a quantum computer, i.e. therewith 
harnessing the purely quantum mechanical phenomena of *superposition* and *entanglement*.

High-Level methods that employ quantum kernels
----------------------------------------------

Classification
##############

.. autosummary::
    :nosignatures:

    QSVC
    QGPC

Regression
##########

.. autosummary::
    :nosignatures:

    QSVR
    QKRR
    QGPR


Methods to evaluate quantum kernels
-----------------------------------

The sQUlearn library provides two methods to evaluate quantum kernels, which also represent the standard approaches to quantum kernel methods in the literature.
link to feature map

Fidelity Quantum Kernel (FQK) via :class:`FidelityKernel`
#########################################################



Projected Quantum Kernel (PQK) via :class:`ProjectedQuantumKernel`
##################################################################



Training of quantum kernels
---------------------------


*References*
------------

