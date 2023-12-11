from qiskit.visualization import plot_coupling_map, plot_gate_map
from qiskit import QuantumCircuit
from qiskit.providers import Backend
import matplotlib.pyplot as plt
from qiskit.converters import circuit_to_dag
from .qubit_coordinates import qubit_coordinates_map
from typing import Union, Optional


def plot_map_circuit_on_backend(circuit: QuantumCircuit,
                                backend: Backend,
                                return_fig: bool = False
                                ) -> Optional[Union[plt.Figure, plt.Axes]]:
    """Visualize the gate map of a backend and the qubits used by a circuit on the same plot.

    Args:
        circuit (QuantumCircuit): Circuit to be visualized.
        backend (Backend): Backend on which the circuit is to be run.
        return_fig (bool, optional): If True, return the matplotlib figure and axes instances.
                                     Otherwise, it returns None. Defaults to False.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot: Figure and axes instances.
        None: If `return_fig` is False.
    """
    fig, ax = plt.subplots()
    ax.axis('off')  # using axis('off')
    plot_gate_map(backend, ax=ax)  # this is needed to visualize the connection unused by the circuit map, could be removed if not needed
    coup_map, qubit_cord, qubit_involved = coupling_map_and_qubit_coordinates_from_circ(circuit, backend)
    coloring = ['#648fff'] * backend.configuration().n_qubits
    for ii in qubit_involved:
        coloring[ii] = 'red'
    plot_coupling_map(backend.configuration().n_qubits,
                      qubit_cord,
                      coup_map,
                      qubit_color=coloring,
                      line_color=['red'] * len(coup_map),
                      ax=ax)
    if return_fig:
        return fig, ax
    else:
        return None


def coupling_map_and_qubit_coordinates_from_circ(circuit: QuantumCircuit, backend: Backend):
    """Extract the coupling map and qubit coordinates from the circuit.

    Args:
        circuit (QuantumCircuit): The Quantum Circuit.
        backend (Backend): The backend for the circuit.

    Returns:
        tuple: The coupling map, qubit coordinates and qubits involved in the circuit.
    """
    dag = circuit_to_dag(circuit)
    qubit_indices = {qubit: index for index, qubit in enumerate(dag.qubits)}
    interactions = []
    qubits_involved = set()
    for node in dag.op_nodes(include_directives=False):
        len_args = len(node.qargs)
        if len_args == 2:
            interactions.append((qubit_indices[node.qargs[0]], qubit_indices[node.qargs[1]]))
            qubits_involved.add(node.qargs[0])
            qubits_involved.add(node.qargs[1])
        elif len_args == 1:
            qubits_involved.add(node.qargs[0])
    coupling_map = [list(tup) for tup in set(interactions)]
    qubit_coord = qubit_coordinates_map[backend.configuration().n_qubits]
    # Gets the list of unique qubits involved in the coupling map
    qubits_involved = [circuit.find_bit(q).index for q in qubits_involved]

    return coupling_map, qubit_coord, qubits_involved
