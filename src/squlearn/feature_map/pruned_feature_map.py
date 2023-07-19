import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit.parameterexpression import ParameterExpression

import warnings
from typing import Union, Tuple
from qiskit import QuantumCircuit

from .feature_map_base import FeatureMapBase

from ..util.data_preprocessing import adjust_input
from ..util.qfi import get_quantum_fisher
from ..util.executor import Executor


class PrunedFeatureMap(FeatureMapBase):

    """
    Class for pruning redundant parameter of feature maps.

    This class is designed to accept a feature map and selectively prune parameters
    based on a provided list of indices. The pruned feature map can be used as a usual feature map.

    **Example: Pruned QEK Feature Map**

    .. code-block:: python

       from squlearn.feature_map import QEKFeatureMap,PrunedFeatureMap
       fm = QEKFeatureMap(4,2,2)
       fm.draw()
       PrunedFeatureMap(fm,[4,7,11,15]).draw()

    .. plot::

       from squlearn.feature_map import QEKFeatureMap,PrunedFeatureMap
       fm = QEKFeatureMap(4,2,2)
       plt = fm.draw()
       plt.text(0.55,0.88,'QEK Feature Map',fontsize=14,ha='center',va='center')
       plt
       plt2 = PrunedFeatureMap(fm,[4,7,11,15]).draw()
       plt2.text(0.55,0.88,'Pruned Feature Map',fontsize=14,ha='center',va='center')
       plt2

    Args:
        feature_map (FeatureMapBase): FeatureMap from which the parameters are removed
        pruned_parameters (list): list with indices of the redundant parameters

    """

    def __init__(self, feature_map: FeatureMapBase, pruned_parameters: list):
        self._feature_map = feature_map
        self._pruned_parameters = pruned_parameters

        # Read in the original pqc
        self._x_base = ParameterVector("x_base", self._feature_map.num_features)
        self._p_base = ParameterVector("p_base", self._feature_map.num_parameters)
        self._base_circuit = self._feature_map.get_circuit(self._x_base, self._p_base)

        # create pruned circuit
        del_list = []
        for i in range(len(self._p_base)):
            if i in self._pruned_parameters:
                del_list.append(self._p_base[i])

        # Create circuit from the base circuit, but empty
        self._pruned_circuit = self._base_circuit.copy()
        self._pruned_circuit.clear()

        # Loop through all gates in the circuits and copy the ones which are kept
        for i in range(len(self._base_circuit._data)):
            if len(self._base_circuit._data[i].operation.params) == 1:
                # Compare pruning list with parameters in gate
                if isinstance(
                    self._base_circuit._data[i].operation.params[0], ParameterExpression
                ) or isinstance(
                    self._base_circuit._data[i].operation.params[0], ParameterVectorElement
                ):
                    if (
                        len(
                            set(del_list).intersection(
                                self._base_circuit._data[i].operation.params[0].parameters
                            )
                        )
                        <= 0
                    ):
                        self._pruned_circuit.append(self._base_circuit._data[i])
                else:
                    self._pruned_circuit.append(self._base_circuit._data[i])
            else:
                self._pruned_circuit.append(self._base_circuit._data[i])

        # Parameter indexing is not the same, since ordering can change -> renumber variables
        # Get all used parameters in the pruned circuit
        used_param = self._pruned_circuit._parameter_table.get_keys()

        # Renumber parameters
        used_old_param = [p for p in self._p_base if p in used_param]
        self._p = ParameterVector("p_", len(used_old_param))
        exchange_dict_p = dict(zip(used_old_param, self._p))

        # Renumber x vector
        used_old_x = [x for x in self._x_base if x in used_param]
        self._x = ParameterVector("x_", len(used_old_x))
        exchange_dict_x = dict(zip(used_old_x, self._x))

        # Replace variables by the relabeled ones
        exchange_both = exchange_dict_x
        exchange_both.update(exchange_dict_p)
        self._pruned_circuit.assign_parameters(exchange_both, inplace=True)
        self._num_qubits = self._feature_map.num_qubits
        self._num_features = len(self._x)
        if self._num_features != self._feature_map._num_features:
            warnings.warn("Number of features changed in the pruning process!", RuntimeWarning)

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the pruned pqc"""
        return len(self._p)

    @property
    def origin_feature_map(self) -> FeatureMapBase:
        """Feature map that is pruned"""
        return self._feature_map

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Generates and returns the circuit of the pruned feature map.

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained

        Return:
            The circuit in Qiskit's QuantumCircuit format of the pruned feature map.
        """

        if len(features) != len(self._x):
            raise ValueError("Number of features does not match the number of input features!")
        if len(parameters) != len(self._p):
            raise ValueError("Number of parameters does not match the number of input parameters!")

        exchange_dict_x = dict(zip(self._x, features))
        exchange_dict_p = dict(zip(self._p, parameters))
        exchange_both = exchange_dict_x
        exchange_both.update(exchange_dict_p)
        return self._pruned_circuit.assign_parameters(exchange_both)


def automated_pruning(
    feature_map: FeatureMapBase,
    executor: Executor,
    n_sample: int = 1,
    pruning_thresh: float = 1e-10,
    x_lim: Union[Tuple[float, float], None] = None,
    p_lim: Union[Tuple[float, float], None] = None,
    x_val: Union[Tuple[float, float], None] = None,
    p_val: Union[Tuple[float, float], None] = None,
    verbose: int = 1,
    seed: Union[int, None] = None,
) -> PrunedFeatureMap:
    """
    Function for automated pruning of the parameters in the inputted parameterized quantum circuit.

    The algorithms for the automated pruning is based on
    https://doi.org/10.1103/PRXQuantum.2.040309.

    Args:
        feature_map (FeatureMapBase): Parameterized quantum circuit that will be pruned.
                                      Has to be in the feature_map format of quantum_fit!
        QI (QuantumInstance): Quantum Instance for evaluating the Quantum Fisher Information Matrix
        n_sample (int): Number of random parameter values and input data that is generated.
                               (if ``x_val=None`` and ``p_val=None``, the number of
                               evaluated Fisher matrices is ``n_sample*n_sample``)
        pruning_thresh (float): Threshold for pruning, eigenvalues lower that value are considered
                                to be redundant.
        x_lim (Union[Tuple[float, float],None]): Limits for the random input data values;
                                                 default limits are :math:`(-\pi,\pi)`.
        p_lim (Union[Tuple[float, float],None]): Limits for the random parameter values;
                                                 default limits are :math:`(-\pi,\pi)`.
        x_val (Union[Tuple[float, float],None]): Array with input data values. If specified,
                                                 no random input data is generated.
        p_val (Union[Tuple[float, float],None]): Array with parameter values. If specified,
                                                 no random parameter values are generated.
        verbose (int): Verbosity of the algorithm, as default only the redundant parameters
                       are printed.
        seed (int): Seed for the random input data and parameters generation.

    Returns:
        Pruned feature_map as a PrunedFeatureMap class object.
    """

    # Set-up default limits
    if x_lim is None:
        x_lim = (-np.pi, np.pi)
    if p_lim is None:
        p_lim = (-np.pi, np.pi)

    # Set-up numpy seed (backup old one)
    if seed is not None:
        seed_backup = np.random.get_state()
        np.random.seed(seed)

    # Process x-values
    if x_val is not None:
        x_val, multi = adjust_input(x_val, feature_map.num_features)
        if not isinstance(x_val, np.ndarray):
            x = np.array(x_val)
        else:
            x = x_val
        if x.shape[1] != feature_map.num_features:
            raise ValueError("Wrong size of the input x_val")
    else:
        x = np.random.uniform(
            low=x_lim[0], high=x_lim[1], size=(n_sample, feature_map.num_features)
        )

    # Process p-values
    if p_val is not None:
        p_val, multi = adjust_input(p_val, feature_map.num_parameters)
        if not isinstance(p_val, np.ndarray):
            p = np.array(p_val)
        else:
            p = p_val
        if p.shape[1] != feature_map.num_parameters:
            raise ValueError("Wrong size of the input p_val")
    else:
        p = np.random.uniform(
            low=p_lim[0], high=p_lim[1], size=(n_sample, feature_map.num_parameters)
        )

    # Reset numpy random state
    if seed is not None:
        np.random.set_state(seed_backup)

    # Calculate QIF for all x and parameter combinations and average over all fishers
    qfis = get_quantum_fisher(feature_map, x, p, executor, mode="p")
    fishers = np.zeros((feature_map.num_parameters, feature_map.num_parameters))
    count = 0
    for ix in range(len(x)):
        for ip in range(len(p)):
            fishers += qfis[ix][ip]
            count += 1
    if count > 0:
        fishers = fishers / count

    # Build pruning list
    pruning_list = pruning_from_QFI(fishers, pruning_thresh)
    if verbose >= 1:
        print("Pruned parameters:", np.sort(pruning_list))

    # Return pruned feature_map
    return PrunedFeatureMap(feature_map, pruning_list)


def pruning_from_QFI(QFI: np.ndarray, pruning_thresh: float = 1e-10) -> list:
    """
    Algorithm for determining the redundant parameters from the Quantum Fischer Information.

    Implementation of the method proposed in https://doi.org/10.1103/PRXQuantum.2.040309.

    Args:
        QFI (np.ndarray): Quantum Fisher Information Matrix as numpy matrix.
        pruning_thresh (float): threshold for pruning, eigenvalues lower that value are
                                 considered to be redundant.
    Returns:
        List of indices of redundant parameters.
    """

    # Symmetrization
    QFI_val = 0.5 * (QFI + QFI.transpose())
    # Eigenvalue decomposition
    W, V = np.linalg.eigh(QFI_val)
    pruning_list = []
    index_list = np.arange(0, len(W))
    # Pruning algorithm from doi:10.1103/PRXQuantum.2.040309
    while abs(W[0]) <= pruning_thresh:
        vec = np.zeros(len(W))
        icount = 0
        while abs(W[icount]) <= pruning_thresh:
            vec = vec + np.square(V[:, icount])
            icount = icount + 1
            if icount >= len(W):
                break

        i = np.argmax(vec)
        pruning_list.append(index_list[i])
        index_list = np.delete(index_list, i, 0)
        QFI_val = np.delete(QFI_val, i, 0)
        QFI_val = np.delete(QFI_val, i, 1)
        if QFI_val.ndim <= 0:
            break
        W, V = np.linalg.eigh(QFI_val)
    return pruning_list
