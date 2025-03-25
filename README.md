<p align="center">
  <img alt="sQUlearn" src="https://raw.githubusercontent.com/sQUlearn/squlearn/main/docs/_static/logo.png" />
</p>

sQUlearn is a user-friendly, NISQ-ready Python library for quantum machine learning (QML), designed for seamless integration with classical machine learning tools like scikit-learn. The library's dual-layer architecture serves both QML researchers and practitioners, enabling efficient prototyping, experimentation, and pipelining. sQUlearn provides a comprehensive tool set that includes both quantum kernel methods and quantum neural networks, along with features like customizable data encoding strategies, automated execution handling, and specialized kernel regularization techniques. By focusing on NISQ-compatibility and end-to-end automation, sQUlearn aims to bridge the gap between current quantum computing capabilities and practical machine learning applications.

sQUlearn offers scikit-learn compatible high-level interfaces for various kernel methods, QNNs and quantum reservoir computing. They build on top of the low-level interfaces of the QNN engine and the quantum kernel engine. The executor is used to run experiments on simulated and real backends of the Qiskit or PennyLane frameworks.

<p align="center">
  <img width=600px alt="sQUlearn schematic" src="https://raw.githubusercontent.com/sQUlearn/squlearn/main/docs/_static/schematic.png" />
</p>

---

## Prerequisites

The package requires **at least Python 3.9**.
## Install sQUlearn

### Stable Release

To install the stable release version of sQUlearn, run the following command:
```bash
pip install squlearn
```

Alternatively, you can install sQUlearn directly from GitHub via
```bash
pip install git+ssh://git@github.com:sQUlearn/squlearn.git
```

## Examples
There are several more elaborate examples available in the folder ``./examples`` which display the features of this package.
Tutorials for beginners can be found at ``./examples/tutorials``.

To install the required packages, run
```bash
pip install .[examples]
```

## Contribute to sQUlearn
Thanks for considering contributing to sQUlearn! Please read our [contribution guidelines](https://github.com/sQUlearn/squlearn/blob/main/.github/CONTRIBUTING.md) before you submit a pull request.

---
## License

sQUlearn is released under the [Apache License 2.0](https://github.com/sQUlearn/squlearn/blob/main/LICENSE.txt)

## Cite sQUlearn
If you use sQUlearn in your work, please cite our paper:

> Kreplin, D. A., Willmann, M., Schnabel, J., Rapp, F., Hagelüken, M., & Roth, M. (2023). sQUlearn - A Python Library for Quantum Machine Learning. [https://doi.org/10.48550/arXiv.2311.08990](https://doi.org/10.48550/arXiv.2311.08990)

## Contact
This project is maintained by the quantum computing group at the Fraunhofer Institute for Manufacturing Engineering and Automation IPA.

[http://www.ipa.fraunhofer.de/quantum](http://www.ipa.fraunhofer.de/quantum)

For general questions regarding sQUlearn, use the [GitHub Discussions](https://github.com/sQUlearn/squlearn/discussions) or feel free to contact [sQUlearn@gmail.com](mailto:sQUlearn@gmail.com).

---
## Acknowledgements

This project was supported by the German Federal Ministry of Economic Affairs and Climate Action through the projects AutoQML (grant no. 01MQ22002A) and AQUAS (grant no. 01MQ22003D), as well as the German Federal Ministry of Education and Research through the project H2Giga Degrad-EL3 (grant no. 03HY110D).

---
