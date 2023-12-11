import networkx as nx
from networkx import Graph
from typing import Dict, Optional, List


class heuristic:
    """A heuristic-based algorithm for solving a problem involving noise and traffic graphs."""
    
    def __init__(self, noise: Graph, traffic: Graph, verbose: bool = False):
        """
        Initialize the heuristic instance with noise and traffic graphs.

        Args:
            noise (Graph): A graph representing the noise.
            traffic (Graph): A graph representing the traffic.
            verbose (bool, optional): Flag to enable verbose output. Defaults to False.
        """
        self.verbose = verbose
        self.noise = noise
        self.traffic = traffic
        self.create_node_and_edge_dicts()
        num_qubits = len(self.traffic.nodes)
        loop_mapping = self.find_recursive_looping_paths(num_qubits)
        if loop_mapping:
            self.print(f"Found {len(loop_mapping)} loops, skipping the rest of the allocation.")
            # Calculate the noise for each loop
            loop_noises = []
            for loop in loop_mapping:
                loop_nodes = [loop[k] for k in range(num_qubits)]
                loop_pairs = zip(loop_nodes, loop_nodes[1:] + [loop_nodes[0]])
                noise = sum(self.edges.get((u, v), 0) for u, v in loop_pairs)
                loop_noises.append(noise)
            # Pick the loop with the minimum noise
            min_noise_index = loop_noises.index(min(loop_noises))
            self.final_mapping = loop_mapping[min_noise_index]

            return
        self.allocate()

    def print(self, message: str):
        """
        Print a message if verbose mode is enabled.

        Args:
            message (str): The message to be printed.
        """
        if self.verbose:
            print(message)

    def find_recursive_looping_paths(self, num_qubits: int) -> List[Dict[int, int]]:
        """
        Find all possible recursive looping paths for a given number of qubits.

        Args:
            num_qubits (int): The number of qubits to find looping paths for.

        Returns:
            List[Dict[int, int]]: A list of mappings representing looping paths.
        """
        all_mappings = []
        source_points = [n for n, deg in self.noise.in_degree() if deg == 0 and self.noise.out_degree(n) > 0]
        convergence_points = [n for n, deg in self.noise.in_degree() if deg > 1]
        
        # Pre-calculate paths from source points to convergence points and vice versa
        paths_from_source_to_cp = {}
        paths_from_cp_to_source = {}
        for sp in source_points:
            for cp in convergence_points:
                paths_from_source_to_cp[(sp, cp)] = list(nx.all_simple_paths(self.noise, sp, cp))
                paths_from_cp_to_source[(cp, sp)] = list(nx.all_simple_paths(self.noise.reverse(), cp, sp))
                
        # Recursive function to find looping paths
        def recursive_search(current_path, visited_nodes):
            if len(current_path) > num_qubits + 1:
                return
            if len(current_path) == num_qubits + 1 and current_path[0] == current_path[-1]:
                all_mappings.append(dict(zip(range(num_qubits), current_path[:-1])))
                return

            last_node = current_path[-1]
            next_paths = [p for key, paths in paths_from_source_to_cp.items() if key[0] == last_node for p in paths] + \
                         [p for key, paths in paths_from_cp_to_source.items() if key[0] == last_node for p in paths]
            
            for path in next_paths:
                new_nodes = set(path[1:])
                if not new_nodes.intersection(visited_nodes):
                    recursive_search(current_path + path[1:], visited_nodes.union(new_nodes))

        # Run the recursive function starting from each source point
        for sp in source_points:
            recursive_search([sp], set())
            
        return all_mappings

    def create_node_and_edge_dicts(self):
        """Create dictionaries to map nodes and edges to their respective attributes."""
        self.edges = {k: v['weight'] for k, v in self.noise.edges().items()}
        self.inv_edges = {v['weight']: k for k, v in self.noise.edges().items()}
        self.nodes = dict(self.noise.nodes(data='weight', default=1))
        self.inv_nodes = {v: k for k, v in self.nodes.items()}
    
    def calculate_normalized_traf_coefs(self):
        """Calculate and return normalized traffic coefficients based on frequencies."""
        frequencies = [self.traffic.nodes[i]['single'] for i in range(len(self.traffic))]
        # Other calculations for frequencies
        traf_coefs = [1 - (1 / r) for r in frequencies]
        normalization = 1 / sum(traf_coefs)
        self.normalized_traf_coefs = [normalization * coef for coef in traf_coefs]
        # calculation of traffic coefficents
        self.traf_qubits_and_coefs = {}
        self.traf_qubits_and_coefs_inverse = {}
        for i in range(0, len(traf_coefs)):
            self.traf_qubits_and_coefs.update({i: self.normalized_traf_coefs[i]})
        for key, value in self.traf_qubits_and_coefs.items():
            if value not in self.traf_qubits_and_coefs_inverse:
                self.traf_qubits_and_coefs_inverse[value] = [key]
            else:
                self.traf_qubits_and_coefs_inverse[value].append(key)

    # calculate the traffic coefficient and allocate the first 2 qubits     
    def allocate(self):
        """Perform the allocation of qubits using the heuristic method."""
        MAX_TRIALS = 1000
        self.calculate_normalized_traf_coefs()
        
        qubit_key_list = list(range(len(self.normalized_traf_coefs)))
        excluded_initial_nodes = []

        while True:
            self.final_mapping = {}
            i, trial, last_qubit = 0, 0, None
            restart_needed = False

            while i < len(qubit_key_list):
                trial += 1
                if trial > MAX_TRIALS:
                    self.print("Too many iterations!")
                    return
                qubit = qubit_key_list[i]
                if len(self.final_mapping) == 0:
                    self.get_best_initial_node(qubit, exclude_nodes=excluded_initial_nodes)
                    # self.print('Starting from ', best_initial_node)
                    
                    last_qubit = qubit
                else:
                    next_node = self.find_next_node(qubit, qubit_key_list)
                    if next_node is not None:
                        next_in_line_idx = qubit_key_list.index(last_qubit) + 1
                        i = next_in_line_idx
                        if next_in_line_idx >= len(qubit_key_list):
                            return  # Completed the mapping
                        self.final_mapping[next_in_line_idx] = next_node
                        last_qubit = next_in_line_idx
                    else:
                        self.print(f"No available nodes for qubit {qubit}. Restarting.")
                        restart_needed = True
                        break
                i += 1

            if not restart_needed:
                break  # If no restart needed, then break the infinite loop

    def find_next_node(self, current_qubit: int, qubit_key_list: List[int]) -> Optional[int]:
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
        prev_node = self.final_mapping[prev_qubit]
        # self.print(f"{prev_node=}", f"{self.final_mapping=}")
        cutoff = 10         
        path_physical = nx.single_source_dijkstra_path_length(self.noise, prev_node, cutoff=cutoff)
        path_physical = {k: v for k, v in sorted(path_physical.items(), key=lambda item: item[1])}
        available_nodes = [node for node in path_physical.keys() if node not in self.final_mapping.values()]
        if available_nodes:
            next_node = available_nodes[0]
            return next_node
        else:
            return self.find_next_node(prev_qubit, qubit_key_list)
    
    def get_best_initial_node(self, qubit: int, exclude_nodes: List[int] = []):
        """
        Find the best initial node to be allocated for the first qubit.

        Args:
            qubit (int): The qubit for which to find the initial node.
            exclude_nodes (List[int], optional): List of nodes to exclude from consideration.

        Returns:
            int: Identifier of the best initial node.
        """
        degrees_noise = {k: v for k, v in sorted(self.noise.out_degree(), key=lambda item: item[1], reverse=True)}
        # self.print(f"{degrees_noise=}")
        best_nodes = [k for k, v in degrees_noise.items() if v >= max(degrees_noise.values())]  # this can be changed
        best_nodes = [node for node in best_nodes if node not in exclude_nodes]
        # self.print(f"{best_nodes=}")
        if not best_nodes:
            self.print("No suitable initial nodes remaining.")
            return None

        error_edges = list(self.noise.edges(best_nodes))
        min_edge_rate = min(self.noise[edge[0]][edge[1]]['weight'] for edge in error_edges)
        mapp_noise_qubit = self.inv_edges[min_edge_rate]

        target_node = mapp_noise_qubit[0] if mapp_noise_qubit[0] in best_nodes else mapp_noise_qubit[1]
        self.final_mapping.update({qubit: target_node})
        
        exclude_nodes.append(target_node)


