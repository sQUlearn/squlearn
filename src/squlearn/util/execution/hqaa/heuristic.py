from typing import Dict, Optional, List, Callable

import networkx as nx
from networkx import Graph


def default_print(*args, **kwargs):
    """Default print function."""


def heuristic(noise: Graph, traffic: Graph, print_function: Optional[Callable] = default_print):
    """
    Performs a heuristic for allocating qubits to nodes in a noisy graph.

    This heuristic first looks for loops in the graph and assigns the qubits to the loop with the minimum noise.
    If no loops are found, it uses the allocate function to allocate the qubits.

    Args:
        noise (Graph): The graph representing the connection between qubits and noise.
        traffic (Graph): The graph representing the qubits and their connections.
        print_function (Optional[Callable], optional): A print function to use for printing information. Defaults to default_print.

    Returns:
        Dict[int, int]: A mapping of qubits to nodes in the graph.
    """
    edges = {k: v["weight"] for k, v in noise.edges().items()}
    inv_edges = {v["weight"]: k for k, v in noise.edges().items()}
    num_qubits = len(traffic.nodes)
    loop_mapping = find_loops(num_qubits, noise)
    if loop_mapping:
        print_function(f"Found {len(loop_mapping)} loops, skipping the rest of the allocation.")
        # Calculate the noise for each loop
        loop_noises = []
        for loop in loop_mapping:
            loop_nodes = [loop[k] for k in range(num_qubits)]
            loop_pairs = zip(loop_nodes, loop_nodes[1:] + [loop_nodes[0]])
            noise = sum(edges.get((u, v), 0) for u, v in loop_pairs)
            loop_noises.append(noise)
        # Pick the loop with the minimum noise
        min_noise_index = loop_noises.index(min(loop_noises))
        final_mapping = loop_mapping[min_noise_index]

        return final_mapping
    return allocate(traffic, noise, inv_edges, print_function)


def find_loops(num_qubits: int, noise_graph: Graph) -> List[Dict[int, int]]:
    """
    Finds all possible recursive looping paths for a given number of qubits.

    Args:
        num_qubits (int): The number of qubits to find looping paths for.
        noise_graph (Graph): The graph representing the connection between qubits and noise.

    Returns:
        List[Dict[int, int]]: A list of mappings representing looping paths.
    """
    source_nodes = [
        node
        for node, degree in noise_graph.in_degree()
        if degree == 0 and noise_graph.out_degree(node) > 0
    ]
    convergence_nodes = [node for node, degree in noise_graph.in_degree() if degree > 1]

    # Pre-calculate paths from source nodes to convergence nodes and vice versa
    paths_from_source_to_convergence = {}
    paths_from_convergence_to_source = {}
    for source_node in source_nodes:
        for convergence_node in convergence_nodes:
            paths_from_source_to_convergence[(source_node, convergence_node)] = list(
                nx.all_simple_paths(noise_graph, source_node, convergence_node)
            )
            paths_from_convergence_to_source[(convergence_node, source_node)] = list(
                nx.all_simple_paths(noise_graph.reverse(), convergence_node, source_node)
            )

    # Recursive function to find looping paths
    def recursive_search(current_path: List[int], visited_nodes: set) -> None:
        if len(current_path) > num_qubits + 1:
            return
        if len(current_path) == num_qubits + 1 and current_path[0] == current_path[-1]:
            result.append(dict(zip(range(num_qubits), current_path[:-1])))
            return

        last_node = current_path[-1]
        next_paths = [
            path
            for key, paths in paths_from_source_to_convergence.items()
            if key[0] == last_node
            for path in paths
        ] + [
            path
            for key, paths in paths_from_convergence_to_source.items()
            if key[0] == last_node
            for path in paths
        ]

        for path in next_paths:
            new_nodes = set(path[1:])
            if not new_nodes.intersection(visited_nodes):
                recursive_search(current_path + path[1:], visited_nodes.union(new_nodes))

    result = []
    # Run the recursive function starting from each source node
    for source_node in source_nodes:
        recursive_search([source_node], set())

    return result


def calculate_normalized_traf_coefs(traffic: Graph):
    """Calculate and return normalized traffic coefficients based on frequencies."""
    frequencies = [traffic.nodes[i]["single"] for i in range(len(traffic))]
    # Other calculations for frequencies
    traf_coefs = [1 - (1 / r) for r in frequencies]
    if sum(traf_coefs) > 0:
        normalization = 1 / sum(traf_coefs)
        traf_coefs = [coef * normalization for coef in traf_coefs]
    # calculation of traffic coefficents
    traf_qubits_and_coefs = {i: coef for i, coef in enumerate(traf_coefs)}

    return traf_qubits_and_coefs


# calculate the traffic coefficient and allocate the first 2 qubits
def allocate(
    traffic: Graph,
    noise: Graph,
    inv_edges: Dict[int, int],
    print_function: Optional[Callable] = default_print,
):
    """Perform the allocation of qubits using the heuristic method."""
    normalized_traf_coefs = calculate_normalized_traf_coefs(traffic)

    qubit_key_list = list(range(len(normalized_traf_coefs)))
    excluded_nodes = []

    for _ in range(1000):
        final_mapping = {}
        i, last_qubit = 0, None
        restart_needed = False

        while i < len(qubit_key_list):
            qubit = qubit_key_list[i]
            if len(final_mapping) == 0:
                initial_node = get_best_initial_node(
                    noise, inv_edges, exclude_nodes=excluded_nodes, print_function=print_function
                )
                final_mapping.update({qubit: initial_node})
                excluded_nodes.append(initial_node)
                last_qubit = qubit
            else:
                next_node = find_next_node(qubit, qubit_key_list, final_mapping, noise)
                if next_node is not None:
                    next_in_line_idx = qubit_key_list.index(last_qubit) + 1
                    i = next_in_line_idx
                    if next_in_line_idx >= len(qubit_key_list):
                        return  # Completed the mapping
                    final_mapping[next_in_line_idx] = next_node
                    last_qubit = next_in_line_idx
                else:
                    print_function(f"No available nodes for qubit {qubit}. Restarting.")
                    restart_needed = True
                    break
            i += 1

        if not restart_needed:
            break  # If no restart needed, then break the infinite loop

    return final_mapping


def find_next_node(
    current_qubit: int, qubit_key_list: List[int], final_mapping: Dict[int, int], noise: Graph
) -> Optional[int]:
    """
    Find the next node to be allocated for a given qubit.

    Parameters:
        current_qubit (int): The current qubit being processed.
        qubit_key_list (list): A list of keys representing qubits.

    Returns:
        int: Identifier of the next node to be allocated.
    """
    if current_qubit == 0:
        return None
    prev_qubit = qubit_key_list[current_qubit - 1]
    prev_node = final_mapping[prev_qubit]
    # print(f"{prev_node=}", f"{final_mapping=}")
    cutoff = 10
    path_physical = nx.single_source_dijkstra_path_length(noise, prev_node, cutoff=cutoff)
    path_physical = {k: v for k, v in sorted(path_physical.items(), key=lambda item: item[1])}
    available_nodes = [node for node in path_physical.keys() if node not in final_mapping.values()]
    if available_nodes:
        next_node = available_nodes[0]
        return next_node
    else:
        return find_next_node(prev_qubit, qubit_key_list, final_mapping, noise)


def get_best_initial_node(
    noise: Graph,
    inv_edges: Dict[int, int],
    exclude_nodes: List[int] = None,
    print_function: Optional[Callable] = default_print,
):
    """
    Find the best initial node to be allocated for the first qubit.

    Args:
        qubit (int): The qubit for which to find the initial node.
        exclude_nodes (List[int], optional): List of nodes to exclude from consideration.

    Returns:
        int: Identifier of the best initial node.
    """
    if exclude_nodes is None:
        exclude_nodes = []

    degrees_noise = dict(sorted(noise.out_degree(), key=lambda item: item[1], reverse=True))
    # print(f"{degrees_noise=}")
    best_nodes = [
        k for k, v in degrees_noise.items() if v >= max(degrees_noise.values())
    ]  # this can be changed
    best_nodes = [node for node in best_nodes if node not in exclude_nodes]
    # print(f"{best_nodes=}")
    if not best_nodes:
        print_function("No suitable initial nodes remaining.")
        return None

    error_edges = list(noise.edges(best_nodes))
    min_edge_rate = min(noise[edge[0]][edge[1]]["weight"] for edge in error_edges)
    mapp_noise_qubit = inv_edges[min_edge_rate]

    target_node = mapp_noise_qubit[0] if mapp_noise_qubit[0] in best_nodes else mapp_noise_qubit[1]

    return target_node
