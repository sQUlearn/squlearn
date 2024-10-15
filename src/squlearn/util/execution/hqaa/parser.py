import networkx as nx
import re


def parse_openqasm(lines: str, number_qubits: int) -> nx.Graph:
    """
    Parses OPENQASM code to generate a traffic graph representing single and double qubit gate frequencies.

    Args:
        lines (str): The OPENQASM code as a string.
        number_qubits (int): The number of qubits in the quantum circuit.

    Returns:
        nx.Graph: A NetworkX graph representing the traffic of the quantum circuit, with nodes for qubits and edges indicating gate operations.
    """
    traffic = nx.Graph()

    for i in range(0, number_qubits):
        traffic.add_node(i, single=0)
    for j in range(0, number_qubits):
        for k in range(number_qubits - 1, 0, -1):
            if j != k:
                traffic.add_edge(j, k, double=0)

    # Update the lists for OPENQASM 3
    list_s_gates = ["x", "h", "rx", "ry", "rz", "sx"]
    list_m_gates = ["cx", "cz", "swap", "ccx", "ecr"]

    for j in range(0, len(list_s_gates)):
        for line in lines.split("\n"):
            if line.startswith(list_s_gates[j]):
                match = re.search(r"q\[(\d+)\]", line)
                if match:
                    number_single_gate = int(match.group(1))
                    traffic.nodes[number_single_gate]["single"] += 1

    for k in range(0, len(list_m_gates)):
        for line in lines.split("\n"):
            if line.startswith(list_m_gates[k]):
                match = re.findall(r"q\[(\d+)\]", line)
                if match:
                    qubits = list(map(int, match))
                    for i in range(len(qubits) - 1):
                        traffic[qubits[i]][qubits[i + 1]]["double"] += 1

    # Remove edges with "double" = 0
    double_0_new = [(i, j) for i, j, data in traffic.edges(data=True) if data["double"] == 0]
    for i, j in double_0_new:
        traffic.remove_edge(i, j)

    return traffic
