.. _install:

Installation
============

Prerequisites
-------------

sQUlearn requires a recent python3 (>=3.9) installation.
Additionally the following python packages are necessary: ::

    bayesian-optimization>=1.4.3,<2
    dill>=0.3.4
    mapomatic>=0.10.0
    networkx>=3.0
    numpy>=1.20
    pennylane>=0.34.0
    qiskit>=0.45.0
    qiskit-aer>=0.12.0
    qiskit-algorithms>=0.3.0
    qiskit-ibm-runtime>=0.18.0
    qiskit-machine-learning>=0.7.0
    scipy>=1.8.0
    scikit-learn>=1.2.0,<1.4.2
    tqdm>=4.1.0

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