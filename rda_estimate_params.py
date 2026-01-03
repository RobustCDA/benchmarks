import math
import csv
from constants import CELL_SIZE, PROOF_SIZE, N_SIZE, SECONDS_PER_ROUND, T_ROUNDS, DELTA_OVERLAP_ROUNDS, DELTA_OVERLAP_MIN_ROUNDS, DELTA_OVERLAP_MIN_SYNC_ROUNDS

def binary_entropy(epsilon: float) -> float:
    """
    Calculates the binary entropy function h(ε) = -ε log₂(ε) - (1-ε)log₂(1-ε)

    Args:
        epsilon: A value between 0 and 1

    Returns:
        The binary entropy value
    """
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be between 0 and 1")

    # Handle the case where epsilon is very close to 0 or 1 to avoid log(0)
    if epsilon < 1e-10:
        return -(1-epsilon) * math.log2(1-epsilon)
    if epsilon > 1 - 1e-10:
        return -epsilon * math.log2(epsilon)

    return -epsilon * math.log2(epsilon) - (1-epsilon) * math.log2(1-epsilon)

def calculate_rhs(
    delta_sd: float,
    T: int,
    delta_overlap: float,
    delta_overlap_min: float,
    k1: int,
    k2: int,
    epsilon: float,
    M_hon: int
) -> float:
    """
    Calculates the right hand side of the inequality:
    δ ≤ δ_SD + ⌈(T+3)/(Δ_overlap - Δ_overlap,min + 1)⌉ · (k₁2^(h(ε)k₂)e^(-εM_hon/k₁) + k₂e^(-M_hon/k₂))

    Args:
        delta_sd: The δ_SD value
        T: Time parameter
        delta_overlap: Δ_overlap value
        delta_overlap_min: Δ_overlap,min value
        k1: k₁ parameter
        k2: k₂ parameter
        epsilon: ε parameter
        M_hon: M_hon parameter

    Returns:
        The calculated right hand side value
    """
    import math

    # Calculate the ceiling term
    ceiling_term = math.ceil((T + 2) / (delta_overlap - delta_overlap_min + 1))

    # Calculate h(ε)
    h_epsilon = binary_entropy(epsilon)

    # Calculate the exponential terms
    term1_base2 = pow(2, h_epsilon * k2)
    term1_exp_e = math.exp(-epsilon * M_hon / k1)
    exp_term1 = k1 * term1_base2 * term1_exp_e

    exp_term2 = k2 * math.exp(-M_hon / k2)

    # Put it all together
    return delta_sd + ceiling_term * (exp_term1 + exp_term2)

def find_max_k1(k2: int, epsilon: float, M_hon: int, target_delta: float = 1e-9) -> int:
    """
    Finds the largest value of k1 that keeps the estimate below target_delta using binary search.

    Args:
        k2: k₂ parameter
        epsilon: ε parameter
        M_hon: M_hon parameter
        target_delta: Target upper bound (default: 10^-9)

    Returns:
        The largest valid k1 value
    """
    # Binary search for k1
    left, right = 1, 10000  # Reasonable range for k1
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
            M_hon=M_hon
        )

        if result <= target_delta:
            # This k1 works, try to find a larger one
            best_k1 = mid
            left = mid + 1
        else:
            # This k1 is too large
            right = mid - 1

    return best_k1

def find_max_k2_with_k1_1(epsilon: float, M_hon: int, target_delta: float = 1e-9) -> int:
    """
    Finds the largest value of k2 that keeps the estimate below target_delta when k1=1 using binary search.

    Args:
        epsilon: ε parameter
        M_hon: M_hon parameter
        target_delta: Target upper bound (default: 10^-9)

    Returns:
        The largest valid k2 value
    """
    # Binary search for k2
    left, right = 1, 2000  # Increased range for k2
    best_k2 = None

    while left <= right:
        mid = (left + right) // 2
        result = calculate_rhs(
            delta_sd=0.0,
            T=T_ROUNDS,
            delta_overlap=DELTA_OVERLAP_ROUNDS,
            delta_overlap_min=DELTA_OVERLAP_MIN_SYNC_ROUNDS,
            k1=1,  # Fixed to 1
            k2=mid,
            epsilon=epsilon,
            M_hon=M_hon
        )

        if result <= target_delta:
            # This k2 works, try to find a larger one
            best_k2 = mid
            left = mid + 1
        else:
            # This k2 is too large
            right = mid - 1

    return best_k2

def calculate_historical_sync_cost(k1: int, k2: int, m_hon_max: int) -> float:
    """
    Calculates historical synchronization cost of the Join operation.

    Args:
        k1: k₁ parameter
        k2: k₂ parameter
        m_hon_max: total number of honest nodes

    Formula:
        hon_nodes_in_column * part_size

    Returns:
        Historical synchronization cost of one part in an erasure_block
    """
    hon_nodes_in_column= m_hon_max / k2 # honest in a column
    chunks_in_part = N_SIZE / k2
    part_size = (CELL_SIZE + PROOF_SIZE) * chunks_in_part
    return hon_nodes_in_column * part_size

def calculate_probagation_cost(k1: int, k2: int, m_max: int, m_hon_max: int) -> float:
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
    
    total_part = k2
    return (nodes_in_cell * (part_size) + hon_nodes_in_cell * nodes_in_column * (part_size)) * total_part

def generate_rows(epsilon: float, M: int, target_delta: float = 1e-9):
    """
    Generates a list of tuples (k2, k1, data complexity, join complexity, get complexity, store complexity), such that
    (k1,k2) yield the target error probability delta and the given complexities.
    For each k2 from max possible down to 1, the function takes the maximum k1
    that yields the desired error.
    """
    m_max = M
    m_hon_max = M / 2

    # First find the maximum k2 possible with k1=1
    max_k2 = find_max_k2_with_k1_1(epsilon, m_hon_max, target_delta)
    if max_k2 is None:
        return []

    rows = []
    # For each k2 value from max down to 1
    for k2 in range(max_k2, 0, -1):
        k1 = find_max_k1(k2, epsilon, m_hon_max, target_delta)
        if k1 is not None:
            replication_factor = m_hon_max / (k2)
            historical_synchronization_complexity = calculate_historical_sync_cost(k1, k2, m_hon_max)
            probagation_complexity = calculate_probagation_cost(k1, k2, m_max, m_hon_max)
            rows.append((epsilon, M, k2, k1, replication_factor, probagation_complexity, historical_synchronization_complexity))
    return rows

if __name__ == "__main__":
    headers = ["epsilon", "M", "k2", "k1", "replication_factor", "probagation_complexity", "historical_synchronization_complexity"]
    M_values = [5000, 10000, 20000]
    epsilon_nominator_values = [5, 10]
    target_delta = 1e-9

    for eps_nom in epsilon_nominator_values:
        for M in M_values:
            eps = eps_nom / 100
            rows = generate_rows(eps, M, target_delta)

            if not rows:
                print(f"[WARN] No data for eps={eps}, M={M}")
                continue

            filename = f"results/rda/estimates_data_eps{eps_nom}_M{M}.csv"

            clean_rows = [
                row for row in rows
                if row is not None
                and len(row) == 7
                and all(v is not None for v in row)
            ]

            with open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(clean_rows)

            print(f"[OK] Data written to {filename} ({len(clean_rows)} rows)")