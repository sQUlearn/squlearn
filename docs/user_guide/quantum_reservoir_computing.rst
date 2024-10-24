.. _quantum_reservoir_computing:

.. currentmodule:: squlearn.qrc

===========================
Quantum Reservoir Computing
===========================

Reservoir Computing (RC) unlike conventional deep neural network architectures, which demand weight
optimization through all layers, utilizes a specific, fixed, connected network layer (usually randomized)
called the "reservoir". This reservoir allows for a dynamic, nonlinear transformation of the input data 
into a high-dimensional space. Quantum Reservoir Computing (QRC) tries to take advantage of the exponential
scaling dimension of the Hilbert space in the quantum computing context to prove an advantage over the 
classical RC. First the classical input data :math:`x=\lbrace x^{(i)}\rbrace_{i=1}^D` gets related to a unitary 
evolution :math:`U(x)`, called encoding circuit, that evolves the initial state of the hardware :math:`\rho_0` to 

.. math::
    \rho(x)=U(x)\rho_0U(x)^{\dagger}