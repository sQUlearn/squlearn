.. _install:

Installation
============

Prerequisites
-------------

sQUlearn requires a recent python3 (>=3.10) installation.
Additionally the following python packages are necessary: ::

    autograd<1.9,
    bayesian-optimization>=1.4.3,<2,
    dill>=0.3.5,
    mapomatic>=0.14.0,
    networkx>=3.0,
    numpy>=2.0.0,
    pennylane>=0.42.0,
    qc-executor>=0.2.1,
    qiskit>=2.5.2,<3.0.0,
    qiskit-aer>=0.17.0,
    qiskit-algorithms>=0.4.0,
    qiskit-ibm-runtime>=0.37.0,
    qiskit-machine-learning>=0.9.0,
    scipy>=1.14.0,
    scikit-learn>=1.4.2,
    sympy>=1.12.1,
    tqdm>=4.1.0,
    qulacs>=0.6.0,

The packages are automatically installed when installing sQUlearn with pip.

Stable Release
--------------

To install the stable release version of sQUlearn, run the following command:

.. code-block:: bash

    pip install squlearn


Bleeding-edge version
---------------------

To install the latest master version:

.. code-block:: bash

	pip install git+https://github.com/sQUlearn/squlearn.git


Development version
-------------------

To install the latest development version:

.. code-block:: bash

    pip install git+https://github.com/sQUlearn/squlearn.git@develop



Installation with optional dependencies:
----------------------------------------

There are several optional dependencies that can be installed with sQUlearn.
To install sQUlearn with the dependencies usefull for development, run the following command:

.. code-block:: bash

    pip install squlearn[dev]

To install sQUlearn with the dependencies necessary to run all examples, run the following command:

.. code-block:: bash

    pip install squlearn[examples]

And to install sQUlearn with the dependencies necessary to build the documentation,
run the following command:

.. code-block:: bash

    pip install squlearn[docs]