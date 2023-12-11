from qiskit.converters import circuit_to_dag
from typing import List, Tuple, Dict, Set
from qiskit import QuantumCircuit


def create_graph(edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """
    Create a graph from a list of edges.

    Args:
        edges (List[Tuple[int, int]]): A list of tuples where each tuple represents an edge between two nodes.

    Returns:
        Dict[int, List[int]]: A dictionary representing the graph, with each key being a node and the value being a list of adjacent nodes.
    """
    graph = {}
    for edge in edges:
        a, b = edge
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)
    return graph


def dfs(graph: Dict[int, List[int]], start: int, visited: Set[int]):
    """
    Perform a Depth First Search on the graph from a starting node.

    Args:
        graph (Dict[int, List[int]]): The graph in which to perform DFS.
        start (int): The starting node for the DFS.
        visited (Set[int]): A set to keep track of visited nodes.
    """
    visited.add(start)
    for neighbour in graph[start]:
        if neighbour not in visited:
            dfs(graph, neighbour, visited)


def find_connected_components(circuit: QuantumCircuit) -> List[Set[int]]:
    """
    Find connected components in the graph representation of a quantum circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to analyze.

    Returns:
        List[Set[int]]: A list of sets, each set containing nodes that form a connected component in the circuit's graph.
    """
    edges = extract_coupling_map_from_circ(circuit)
    graph = create_graph(edges)
    visited = set()
    components = []

    for node in graph:
        if node not in visited:
            component = set()
            dfs(graph, node, component)
            components.append(component)
            visited.update(component)

    return components


def extract_coupling_map_from_circ(circuit: QuantumCircuit) -> List[Tuple[int, int]]:
    """
    Extract the coupling map from a quantum circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit from which to extract the coupling map.

    Returns:
        List[Tuple[int, int]]: A list of tuples representing the interactions between qubits in the circuit.
    """
    dag = circuit_to_dag(circuit)
    qubit_indices = {qubit: index for index, qubit in enumerate(dag.qubits)}
    interactions = []
    for node in dag.op_nodes(include_directives=False):
        len_args = len(node.qargs)
        if len_args == 2:
            interactions.append((qubit_indices[node.qargs[0]],
                                 qubit_indices[node.qargs[1]]))

    coupling_map = [tup for tup in set(interactions)]
    return coupling_map
