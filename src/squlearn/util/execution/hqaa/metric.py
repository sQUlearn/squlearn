import networkx as nx
from networkx import Graph
from typing import Dict


class metric:
    def __init__(self, noise: Graph, traffic: Graph, final_mapping: Dict):
        """
        Initialize the metric instance with noise and traffic graphs, and a final mapping.

        Args:
            noise (Graph): A graph representing the noise.
            traffic (Graph): A graph representing the traffic.
            final_mapping (Dict): A mapping of qubits or nodes to their respective locations.
        """

        self.noise = noise
        self.traffic = traffic
        self.final_mapping = final_mapping
        self.create_node_and_edge_dicts()

        self.calculate_metric()

    def calculate_metric(self):
        """
        Calculate the overall metric for the quantum circuit based on single and double gate errors.
        """
        self.single_gate_product_metric()
        self.SWAP_double_gate_product_metric()
        # self.calculate_normalization()

    def create_node_and_edge_dicts(self):
        """
        Create dictionaries to map nodes and edges to their respective attributes.
        """
        self.edges = dict(self.noise.edges())
        set_1 = list(self.edges.keys())
        set_2 = list(self.edges.values())
        set_3 = {}
        inv_set_3 = {}

        for i in range(0, len(set_2)):
            set_3.update({set_1[i]: set_2[i]["weight"]})

        for j in range(0, len(set_2)):
            inv_set_3.update({set_2[j]["weight"]: set_1[j]})

        self.edges = set_3
        self.inv_edges = inv_set_3
        self.nodes = dict(self.noise.nodes(data="weight", default=1))

        self.inv_nodes = {}
        set_1 = list(self.nodes.keys())
        set_2 = list(self.nodes.values())
        for n in range(0, len(set_1)):
            self.inv_nodes.update({set_2[n]: set_1[n]})

    def single_gate_product_metric(self):
        """
        Calculate the product metric for single gate errors.
        """
        # dictionary of form {algorithm:single_gate_frequency...}
        single_invocation = nx.get_node_attributes(self.traffic, "single")
        # print("single_invocation = ",single_invocation)
        # single error rate for the product metric
        self.single_product = 1
        # cycles through the dictionary single_invocation
        for key, val in single_invocation.items():
            # extracts the error rate for a node
            error_rate_single = self.nodes[self.final_mapping[key]]
            self.single_product = self.single_product * (
                (1 - error_rate_single) ** single_invocation[key]
            )

    def SWAP_double_gate_product_metric(self):
        """
        Calculate the product metric for SWAP and double gate errors.
        """
        # set initial overall double-gate and SWAP errors
        double_invocation = nx.get_edge_attributes(self.traffic, "double")
        # outputs list form of the algorithm edges in tuples
        algorithmic_edges = list(double_invocation.keys())
        # set summations for the gate invocations
        current_swap_invocation_number_sum = 0
        double_gate_invocations_sum = 0
        # set the error rate for each portion of the product metric
        self.double_product = 1
        self.swap_product = 1
        self.final_metric_product = 1
        # cycle through list of tuples and find their positions on physical lattice
        for j in range(0, len(algorithmic_edges)):
            # find the distance between the two nodes á la Dijkstra
            distance = nx.shortest_path(
                self.noise,
                source=self.final_mapping[algorithmic_edges[j][0]],
                target=self.final_mapping[algorithmic_edges[j][1]],
            )
            # if the size of distance is 1 - then it is a double-gate error problem
            if len(distance) == 2:
                # find the error rate for physical edge used
                error_rate_product = self.noise[distance[0]][distance[1]]["weight"]
                # find number of double gate iterations
                double_gate_invocations = self.traffic[algorithmic_edges[j][0]][
                    algorithmic_edges[j][1]
                ]["double"]
                # multiply by the swap and double-gate error rate
                self.double_product = self.double_product * (
                    (1 - error_rate_product) ** double_gate_invocations
                )
            # if the size of distance is not 1 - then it is a SWAP-gate error problem
            elif len(distance) > 2:
                list_error_rates_product = []
                # cycle through the list of physical qubits used and find error rates
                for i in range(0, len(distance) - 1):
                    # find the error rate for physical edge used
                    list_error_rates_product.append([distance[i], distance[i + 1]])
                    list_error_rates_product = list_error_rates_product[0]
                    # adds the physical qubits to new variable for only the corresponding error rates
                    list_error_rates_metric = self.noise[list_error_rates_product[0]][
                        list_error_rates_product[1]
                    ]
                    list_error_rates_product = []
                    # multiply the error rates together
                    list_error_rates_metric_base = 1
                    list_error_rates_metric_base = (
                        1 - list_error_rates_metric["weight"]
                    ) * list_error_rates_metric_base
                    success_rates = 1
                    success_rates = (
                        1 - list_error_rates_metric["weight"]
                    ) * list_error_rates_metric_base
                # find the number of double-gate invocations needed for the SWAP route - multiply everything by this number
                current_swap_invocation_number = self.traffic[algorithmic_edges[j][0]][
                    algorithmic_edges[j][1]
                ]["double"]
                # final calculation
                self.swap_product = self.swap_product * (
                    success_rates ** (current_swap_invocation_number * 2)
                )

        # final calculation for the overall error rate for single, double, and swap gates
        self.final_metric_product = self.swap_product * self.double_product * self.single_product
        # print("error_rate_before_adding_measurement_errors = ",self.final_metric_product)

        # incorporate measurement errors
        read_out_invocation = nx.get_node_attributes(self.noise, "read_out")
        self.read_out_error = 1
        # cycles through the dictionary read_out_invocation
        for key, val in self.final_mapping.items():
            # extracts the error rate for a node
            error_rate_read_out = read_out_invocation[self.final_mapping[key]]
            self.read_out_error = self.read_out_error * (1 - error_rate_read_out)
        self.final_metric_product = self.final_metric_product * self.read_out_error
