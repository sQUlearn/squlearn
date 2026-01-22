"""Utility function to generate random Pauli strings."""

import numpy as np


def random_pauli_list(pauli_string_length: int, num_samples: int, seed: int = 42) -> list[str]:
    """
    Sample num_samples unique random Pauli strings of length pauli_string_length without replacement.

    Args:
        pauli_string_length: Length of each Pauli string (number of qubits).
        num_samples: Number of unique strings to sample (must be <= 4**pauli_string_length).
        seed: RNG seed for reproducibility.

    Returns:
        List of num_samples unique Pauli strings like ['IXYZI', 'YYIII', ...].
    """
    total_possible = 4**pauli_string_length
    if num_samples > total_possible:
        raise ValueError(f"num_samples={num_samples} exceeds total possible {total_possible}")

    pauli_symbols = np.array(["I", "X", "Y", "Z"])
    rng = np.random.default_rng(seed)

    # Sample unique indices from the full space
    unique_indices = rng.choice(total_possible, size=num_samples, replace=False)

    # Convert indices to Pauli strings via base-4
    pauli_strings = []
    for index in unique_indices:
        bits = []
        temp_index = index
        for _ in range(pauli_string_length):
            bits.append(pauli_symbols[temp_index % 4])
            temp_index //= 4
        pauli_strings.append("".join(reversed(bits)))

    return pauli_strings
