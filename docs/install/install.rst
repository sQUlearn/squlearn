.. _install:

Installation
============

Prerequisites
-------------

sQUlearn requires a recent python3 (>=3.9) installation.
Additionally the following python packages are necessary: ::

    numpy>=1.17
    scipy>=1.5
    scikit-learn>=1.0
    qiskit>=0.42.1
    qiskit-aer>=0.12.0
    qiskit-machine-learning>=0.6.0
    qiskit-ibm-runtime>=0.9
    dill>=0.3

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
