# sQUlearn

## Prerequisites

The package is written and tested in python 3.10
The following python packages are required:

```bash
pip install qiskit
pip install numpy
```
## Installation

If access to the repository is granted the package can be installed via:

https access:
```bash
pip install git+https://gitlab.cc-asp.fraunhofer.de/cci/code/quantum/squlearn
```

or with ssh access:
```bash
pip install git+ssh://git@gitlab.cc-asp.fraunhofer.de/cci/code/quantum/squlearn.git 
```

## Uninstall

The package can be removed by the following command:

```bash
pip uninstall squlearn
```

## Update

The package can be update by

https access:
```bash
pip install git+https://gitlab.cc-asp.fraunhofer.de/cci/code/quantum/squlearn --upgrade
```
or for ssh access:
```bash
pip install git+ssh://git@gitlab.cc-asp.fraunhofer.de/cci/code/quantum/squlearn.git --upgrade
```

## Examples

There are several examples available in the folder ``./examples`` which displaying the features of this package.

## Documentation:

TODO

## Contribution
If you plan to contribute to this project, please read this section carefully and check that your contribution fits the desired process and style.

### Devtools
Install the recommended tools with
```bash
pip install -r requirements_dev.txt
```

### Style Guide
#### Code Style
We try to match all of our python code closely to the [PEP8 style guide](https://pep8.org/) with the exception of using a line length of `99`. To ensure this we use the [Black formater](https://black.readthedocs.io/en/stable/index.html) with the configuration specified in [pyproject.toml](`pyproject.toml`). To format your code, run
```bash
black path/to/folder/or/file.py
```
from the top level directory to ensure the use of the `pyproject.toml` configuration file.

We don't enforce but highly encourage the use of the [pylint linter](https://docs.pylint.org/) to check newly written code and adapt accordingly. To run the linter with the desired configuration run
```bash
pylint path/to/folder/or/file.py
```
from the top level directory to again ensure the use of the `pyproject.toml` configuration file. Running pylint before and after the contribution shouldn't add violations and/or lower the overall code score.

#### Docstrings
We furthermore desire to use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for our Docstrings. Generally we at least expect
 - A one-line summary, terminated by a period, and
 - a description of the class/method.

For classes we furthermore expect
 - A list of attributes.

For methods we furthermore expect
 - A list of arguments, and
 - A list of return values.

### Git Flow (starting with GitHub)
We use the Git Flow branch structure specified [here](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) except for release branches.

## TODOs

### Known Issues:
- Merge Frederic's ADAM with David's ADAM
- Unify data preprocessing (stacking of variables)
- Add more explanations in the examples
- better treatment of variable groups in LayeredFeatureMap (store copy within the class) -> DKR-AK
- Wrapper for all qiskit feature maps
- clean up kernel examples (e.g. there are some files name test...)
- qgpr_optimization_workflow.ipynb has error messages in the notbeook (only KeyboardInterrupts, check anyway)
- replace pqk_impedance.ipynb and pqk_impedance_real_backend.ipynb notebooks with better examples (rely on measurement data) -> JSL
- installation does not install required dependencys (qiskit_machine_learning did not update to the required version)

### Bigger Projects
- Sklearn interfaces
    - batch based training for QNNs -> MOW
    - wrapper for kernel methods
- Unified documentation and documentation framework (https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html, https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Testjobs + pipeline for testing
- Distribution via pip
- Extend PQK class -> JSL
    - k-RDMs (check if compatible with DKRs code)
    - Flexibility in definition
    - implement shadow kernel
- Neural tangent kernels may be worth to work on
