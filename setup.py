from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().strip().split("\n")

setup(
    name="squlearn",
    version="0.0.1",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    author="Fraunhofer IPA",
    description="A library for quantum machine learning following the sklearn standard.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sQUlearn/squlearn",
    install_requires=requirements,
    python_requires=">=3.9",
)
