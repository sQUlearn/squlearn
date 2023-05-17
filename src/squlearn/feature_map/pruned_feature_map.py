import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.utils import QuantumInstance
import warnings
from typing import Union
from qiskit import QuantumCircuit

from .feature_map_base import FeatureMapBase

from ..util.data_preprocessing import adjust_input
from ..util.quantum_fisher import get_quantum_fisher


class PrunedFeatureMap(FeatureMapBase):

    """
    Class for generating a new feature map from an existing one without the pruned parameters.
    Relabels the parameters.
    """

    def __init__(self, feature_map: FeatureMapBase, pruned_parameters):
        """
        Constructor of the pruned pqc

        Args:
            feature_map : pqc from which the parameters are removed
            pruned_parameters: list with indices of the redundant parameters
        """

        self.feature_map = feature_map
        self.pruned_parameters = pruned_parameters

        # Read in the original pqc
        self.x_base = ParameterVector("x_base", self.feature_map.num_features)
        self.p_base = ParameterVector("p_base", self.feature_map.num_parameters)
        self.base_pqc = self.feature_map.get_circuit(self.x_base, self.p_base)

        # create pruned circuit
        del_list = []
        for i in range(len(self.p_base)):
            if i in self.pruned_parameters:
                del_list.append(self.p_base[i])

        # Create circuit from the base circuit, but empty
        self.pruned_pqc = self.base_pqc.copy()
        self.pruned_pqc.clear()

        # Loop through all gates in the circuits and copy the ones which are kept
        for i in range(len(self.base_pqc._data)):
            if len(self.base_pqc._data[i].operation.params) == 1:
                # Compare pruning list with parameters in gate
                if isinstance(
                    self.base_pqc._data[i].operation.params[0], ParameterExpression
                ) or isinstance(
                    self.base_pqc._data[i].operation.params[0], ParameterVectorElement
                ):
                    if (
                        len(
                            set(del_list).intersection(
                                self.base_pqc._data[i].operation.params[0].parameters
                            )
                        )
                        <= 0
                    ):
                        self.pruned_pqc.append(self.base_pqc._data[i])
                else:
                    self.pruned_pqc.append(self.base_pqc._data[i])
            else:
                self.pruned_pqc.append(self.base_pqc._data[i])

        # Parameter indexing is not the same, since ordering can change -> renumber variables
        # Get all used parameters in the pruned circuit
        used_param = self.pruned_pqc._parameter_table.get_keys()

        # Renumber parameters
        used_old_param = [p for p in self.p_base if p in used_param]
        self.p = ParameterVector("p_", len(used_old_param))
        exchange_dict_p = dict(zip(used_old_param, self.p))

        # Renumber x vector
        used_old_x = [x for x in self.x_base if x in used_param]
        self.x = ParameterVector("x_", len(used_old_x))
        exchange_dict_x = dict(zip(used_old_x, self.x))

        # Replace variables by the relabeled ones
        exchange_both = exchange_dict_x
        exchange_both.update(exchange_dict_p)
        self.pruned_pqc.assign_parameters(exchange_both, inplace=True)
        self._num_qubits = self.feature_map.num_qubits
        self._num_features = len(self.x)
        if self._num_features != self.feature_map._num_features:
            warnings.warn("Number of features changed in the pruning process!", RuntimeWarning)

    @property
    def num_parameters(self) -> int:
        """Returns number of parameters in the pruned pqc"""
        return len(self.p)

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """Returns the circuit of the pruned pqc"""
        exchange_dict_x = dict(zip(self.x, features))
        exchange_dict_p = dict(zip(self.p, parameters))
        exchange_both = exchange_dict_x
        exchange_both.update(exchange_dict_p)
        return self.pruned_pqc.assign_parameters(exchange_both)


def automated_pruning(
    feature_map: FeatureMapBase,
    QI: QuantumInstance,
    n_sample=1,
    pruning_thresh=1e-10,
    x_lim=None,
    p_lim=None,
    x_val=None,
    p_val=None,
    verbose=1,
    seed=0,
) -> FeatureMapBase:
    """
    Function for automated pruning of the parameters in the inputted parameterized quantum circuit.

    Args:
        feature_map : Parameterized quantum circuit that will be pruned. Has to be in the feature_map format of quantum_fit!
        QI : Quantum Instance for evaluating the Quantum Fisher Information Matrix
        n_sample=1 (optional) : Number of random parameter values and input data that is generated.
                               (if x_val=None and p_val=None, the number of evaluated Fisher matrices is n_sample^2)
        pruning_thresh=1e-10 (optional) : threshold for pruning, eigenvalues lower that value are considered to be redundant.
        x_lim=None (optional) : Limits for the random input data values; default limits are (-pi,pi).
        p_lim=None (optional) : Limits for the random parameter values; default limits are (-pi,pi).
        x_val=None (optional) : Array with input data values. If specified, no random input data is generated.
        p_val=None (optional) : Array with parameter values. If specified, no random parameter values are generated.
        verbose=1 (optional) : Verbosity of the algorithm, as default only the redundant parameters are printed.
        seed=0 (optional) : Seed for the random input data and parameters generation.

    Returns:
        Pruned feature_map in the feature_map format
    """

    # Set-up default limits
    if x_lim is None:
        x_lim = (-np.pi, np.pi)
    if p_lim is None:
        p_lim = (-np.pi, np.pi)

    # Set-up numpy seed (backup old one)
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
    np.random.set_state(seed_backup)

    # Calculate QIF for all x and parameter combinations and average over all fishers
    fishers = np.zeros([feature_map.num_parameters, feature_map.num_parameters])
    count = 0
    for x_ in x:
        for p_ in p:
            if verbose >= 2:
                print("Calc fisher number ", count + 1)
            fishers += get_quantum_fisher(feature_map, x_, p_, QI)
            count += 1
    if count > 0:
        fishers = fishers / count

    # Build pruning list
    pruning_list = pruning_from_QFI(fishers, pruning_thresh)
    if verbose >= 1:
        print("Pruned parameters:", np.sort(pruning_list))

    # Return pruned feature_map
    return PrunedFeatureMap(feature_map, pruning_list)


def pruning_from_QFI(QFI, pruning_thresh=1e-10):
    """
    Algorithm for determining the redundant parameters from the QFI.
    (see doi:10.1103/PRXQuantum.2.040309)

    Args:
        QFI: Quantum Fisher Information Matrix as numpy matrix.
        pruning_thresh=1e-10 (optional) : threshold for pruning, eigenvalues lower that value are considered to be redundant.
    Returns:
        List of redundant parameters (as indices)
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
