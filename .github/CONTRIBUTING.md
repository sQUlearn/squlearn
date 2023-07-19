# Contributing
If you plan to contribute to this project, please read this section carefully and check that your contribution fits the desired process and style.

## Install development version
To contribute to sQUlearn install the package from the source code:

```bash
git clone https://github.com/sQUlearn/squlearn.git && cd squlearn
git checkout develop
pip install -e .
```

> Note that editable installing as above might require an updated version of pip `pip install pip --upgrade` 

## Devtools
Install the recommended tools with
```bash
pip install -e .[dev]
```

## Style Guide
### Code Style
We try to match all of our python code closely to the [PEP8 style guide](https://pep8.org/) with the exception of using a line length of `99`. To ensure this we use the [Black formater](https://black.readthedocs.io/en/stable/index.html) with the configuration specified in [pyproject.toml](https://github.com/sQUlearn/squlearn/blob/main/pyproject.toml). To format your code, run
```bash
black path/to/folder/or/file.py
```
from the top level directory to ensure the use of the `pyproject.toml` configuration file.

We don't enforce but highly encourage the use of the [pylint linter](https://docs.pylint.org/) to check newly written code and adapt accordingly. To run the linter with the desired configuration run
```bash
pylint path/to/folder/or/file.py
```
from the top level directory to again ensure the use of the `pyproject.toml` configuration file. Running pylint before and after the contribution shouldn't add violations and/or lower the overall code score.

### Docstrings
We furthermore desire to use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for our Docstrings. Generally we at least expect
 - A one-line summary, terminated by a period, and
 - a description of the class/method.

For classes we furthermore expect
 - A list of attributes.

For methods we furthermore expect
 - A list of arguments, and
 - A list of return values.

## Git Flow
We use the Git Flow branch structure specified [here](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) except for release branches.
