import csv
import math
from typing import List, Optional, Sequence, Tuple
from constants import CELL_SIZE, PROOF_SIZE, N_SIZE, SECONDS_PER_ROUND, T_ROUNDS, DELTA_OVERLAP_ROUNDS, DELTA_OVERLAP_MIN_ROUNDS, DELTA_OVERLAP_MIN_SYNC_ROUNDS

def binary_entropy(epsilon: float) -> float:
    """
    Calculates the binary entropy function h(ε) = -ε log₂(ε) - (1-ε)log₂(1-ε)
    """
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be between 0 and 1")

    if epsilon < 1e-10:
        return -(1 - epsilon) * math.log2(1 - epsilon)
    if epsilon > 1 - 1e-10:
        return -epsilon * math.log2(epsilon)
    return -epsilon * math.log2(epsilon) - (1 - epsilon) * math.log2(1 - epsilon)


def _alpha_and_decay(M_hon: int, k2: int, k_chunks: int) -> Optional[Tuple[float, float]]:
    """Returns (alpha, decay_factor) for the RLNC bound, or None if invalid."""
    if 1 - (k_chunks * k2 / M_hon) <= 0 or 1 - (k_chunks * k2 / M_hon) >= 1:
        return None

    alpha = M_hon / (k2 * k_chunks) - 1
    if alpha <= 0:
        return None

    decay = (alpha * alpha) / (2 * (alpha + 1) * (alpha + 1))
    return alpha, decay


def calculate_rhs(
    delta_sd: float,
    T: int,
    delta_overlap: float,
    delta_overlap_min: float,
    k1: int,
    k2: int,
    epsilon: float,
    M_hon: int,
    k_chunks: int,
) -> float:
    """
    Calculates:
    δ ≤ δ_SD + ⌈(T+2)/(Δ_overlap - Δ_overlap,min + 1)⌉ · (k₁2^(h(ε)k₂)e^(-εM_hon/k₁) + k₂e^(-((α^2)/(2(α+1)^2))·M_hon/k₂))
    with μ = (1+α)k and μ = M_hon/k₂ ⇒ α determined by (M_hon, k₂, k).
    """
    coeffs = _alpha_and_decay(M_hon, k2, k_chunks)
    if coeffs is None:
        return math.inf

    _, decay = coeffs

    ceiling_term = math.ceil((T + 2) / (delta_overlap - delta_overlap_min + 1))

    h_epsilon = binary_entropy(epsilon)
    term1_base2 = pow(2, h_epsilon * k2)
    term1_exp_e = math.exp(-epsilon * M_hon / k1)
    exp_term1 = k1 * term1_base2 * term1_exp_e
    exp_term2 = k2 * math.exp(-decay * M_hon / k2)

    return delta_sd + ceiling_term * (exp_term1 + exp_term2)


def find_max_k1(
    k2: int,
    epsilon: float,
    M_hon: int,
    k_chunks: int,
    target_delta: float = 1e-9,
) -> Optional[int]:
    """
    Finds the largest value of k1 that keeps the estimate below target_delta using binary search.
    """
    left, right = 1, 10000
    best_k1 = None

    while left <= right:
        mid = (left + right) // 2
        result = calculate_rhs(
            delta_sd=0.0,
            T=T_ROUNDS,
            delta_overlap=DELTA_OVERLAP_ROUNDS,
            delta_overlap_min=DELTA_OVERLAP_MIN_ROUNDS,
            k1=mid,
            k2=k2,
            epsilon=epsilon,
            M_hon=M_hon,
            k_chunks=k_chunks,
        )

        if result <= target_delta:
            best_k1 = mid
            left = mid + 1
        else:
            right = mid - 1

    return best_k1


def find_max_k2_with_k1_1(
    epsilon: float,
    M_hon: int,
    k_chunks: int,
    target_delta: float = 1e-9,
) -> Optional[int]:
    """
    Finds the largest k2 satisfying the bound when k1=1 for the supplied k_chunks.
    """
    right = 2000
    left = 1
    best_k2 = None

    while left <= right:
        mid = (left + right) // 2
        result = calculate_rhs(
            delta_sd=0.0,
            T=T_ROUNDS,
            delta_overlap=DELTA_OVERLAP_ROUNDS,
            delta_overlap_min=DELTA_OVERLAP_MIN_SYNC_ROUNDS,
            k1=1,
            k2=mid,
            epsilon=epsilon,
            M_hon=M_hon,
            k_chunks=k_chunks,
        )

        if result <= target_delta:
            best_k2 = mid
            left = mid + 1
        else:
            right = mid - 1

    return best_k2

def calculate_historical_sync_cost(k1: int, k2: int, m_hon_max: int, k_chunks: int) -> float:
    """
    Calculates historical synchronization cost of the Join operation.

    Args:
        k1: k₁ parameter
        k2: k₂ parameter
        m_hon_max: total number of honest nodes

    Formula:
        hon_nodes_in_column * coded_part_size

    Returns:
        Historical synchronization cost of one part in an erasure_block
    """
    hon_nodes_in_column= m_hon_max / k2 # honest in a column
    chunks_in_part = N_SIZE / k2
    coded_part_size = (CELL_SIZE / k_chunks + PROOF_SIZE) * chunks_in_part
    return hon_nodes_in_column * coded_part_size

def calculate_probagation_cost(k1: int, k2: int, m_max: int, m_hon_max: int, k_chunks: int) -> float:
    """
    Calculates the expected cost when probagating an erasure block to all nodes.

    Formula:
        Probagation_cost_one_part = nodes_in_cell * (part_size) + hon_nodes_in_cell * nodes_in_column * (coded_part_size)
    """
    nodes_in_cell = m_max / (k1 * k2) # honest in a cell
    nodes_in_column = m_max / k2 # honest in a column
    hon_nodes_in_cell = m_hon_max / (k1 * k2) # honest in a cell
    chunks_in_part = N_SIZE / k2

    chunk_size = (CELL_SIZE + PROOF_SIZE) * N_SIZE
    part_size = chunk_size * chunks_in_part # chunk size = (CELL_SIZE + PROOF_SIZE) * N_SIZE

    coded_chunk_size = (CELL_SIZE / k_chunks + PROOF_SIZE) * N_SIZE
    coded_part_size = coded_chunk_size * chunks_in_part

    total_part = k2
    return (nodes_in_cell * (part_size) + hon_nodes_in_cell * nodes_in_column * (coded_part_size)) * total_part

def generate_rows(
    epsilon: float,
    M: int,
    k_chunks: int,
    target_delta: float = 1e-9,
) -> List[Tuple[int, float, int, int, int]]:
    """
    Returns tuples describing the (k1, k2) pairs for a given epsilon, M, and k.
    Each tuple is (k, epsilon, N, k2, k1).
    """
    m_max = M
    m_hon_max = M / 2

    max_k2 = find_max_k2_with_k1_1(epsilon, m_hon_max, k_chunks, target_delta)
    if max_k2 is None:
        return []

    rows = []
    for k2 in range(max_k2, 0, -1):
        k1 = find_max_k1(k2, epsilon, m_hon_max, k_chunks, target_delta)
        if k1 is None:
            continue
        replication_factor = m_hon_max / (k2 * k_chunks)
        historical_synchronization_complexity = calculate_historical_sync_cost(k1, k2, m_hon_max, k_chunks)
        probagation_complexity = calculate_probagation_cost(k1, k2, m_max, m_hon_max, k_chunks)
        rows.append((k_chunks, epsilon, M, k2, k1, replication_factor, probagation_complexity, historical_synchronization_complexity))

    return rows


def write_rows(
    rows: Sequence[Tuple[int, float, int, int, int, float, float]],
    filename: str,
) -> None:
    headers = ["k", "epsilon", "M", "k2", "k1", "replication_factor", "probagation_complexity", "historical_synchronization_complexity"]
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    M_values = [5000, 10000, 20000]
    epsilon_nominator_values = [5, 10]
    target_delta = 1e-9
    k_values = [8, 16]

    for k_chunks in k_values:
        for eps_nom in epsilon_nominator_values:
            epsilon = eps_nom / 100
            for M in M_values:
                rows = generate_rows(epsilon, M, k_chunks, target_delta)
                if not rows:
                    continue

                filename = f"results/cda/estimates_k_{k_chunks}_eps{eps_nom}_M{M}.csv"
                write_rows(rows, filename)
                print(f"Data written to {filename}")
