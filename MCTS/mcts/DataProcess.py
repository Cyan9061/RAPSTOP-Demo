import numpy as np
import random
import torch

# --------------------------------------------------------------------------------
# Utility functions for generating permutations, rankings, and manipulating vote matrices
# --------------------------------------------------------------------------------

def random_permutation(N):
    """
    Generate a random permutation of the integers from 1 to N (inclusive).

    Args:
        N (int): Upper bound of the permutation range.

    Returns:
        list[int]: A shuffled list containing each integer from 1 to N exactly once.
    """
    numbers = list(range(1, N + 1))  # Create list [1, 2, ..., N]
    random.shuffle(numbers)           # Shuffle in-place using Python's random module
    return numbers


def rank_of_elements(nums):
    """
    Compute the rank of each element in a list, where the largest element receives rank 1.

    Args:
        nums (list[float] or list[int]): Input sequence of numeric values.

    Returns:
        list[int]: A list of the same length as `nums`, where each position gives
                   the rank of the corresponding input element (1 = highest value).
    """
    # Pair each value with its original index and sort descending by value
    sorted_with_index = sorted(
        enumerate(nums), key=lambda x: x[1], reverse=True
    )

    ranks = [0] * len(nums)
    # Assign ranks starting at 1 for the largest element
    for rank, (original_index, _) in enumerate(sorted_with_index, start=1):
        ranks[original_index] = rank

    return ranks


def seq2Matrix(sequences):
    """
    Convert a list of ranking sequences into their corresponding pairwise comparison matrices.

    Each sequence of length M produces an M x M matrix where entry (i,j) is:
      - -1 if sequence[i] > sequence[j]
      - +1 if sequence[i] < sequence[j]
      -  0 if sequence[i] == sequence[j]

    Args:
        sequences (list[list[int]]): A list of T sequences, each of length M.

    Returns:
        list[np.ndarray]: A list of T numpy arrays of shape (M, M) encoding pairwise comparisons.
    """
    M = len(sequences[0])  # Number of items in each sequence
    T = len(sequences)     # Number of sequences
    Matrixs = []           # Will hold T comparison matrices

    for i in range(T):
        curMatrix = np.zeros((M, M), dtype=float)
        for j in range(M):
            for k in range(M):
                v1 = sequences[i][j]
                v2 = sequences[i][k]
                if v1 > v2:
                    curMatrix[j, k] = -1
                elif v1 < v2:
                    curMatrix[j, k] = 1
                else:
                    curMatrix[j, k] = 0
        Matrixs.append(curMatrix)

    return Matrixs


def swap_votes(matrices, k=0.05):
    """
    Randomly swap a proportion of asymmetric vote entries in each matrix.

    For each of N matrices of size M x M, select approximately k * [M(M-1)/2] index pairs (i<j)
    where the entries (i,j) and (j,i) differ, then swap their values.

    Args:
        matrices (torch.Tensor): Tensor of shape (N, M, M) containing vote matrices.
        k (float): Fraction of possible asymmetric pairs to swap (default 0.05).

    Returns:
        torch.Tensor: A cloned tensor with specified pairs swapped.
    """
    N, M, _ = matrices.shape
    modified_matrices = matrices.clone()  # Copy to avoid in-place side-effects
    print("RandomSwapVotes Processing...")

    for n in range(N):
        # Determine how many swaps to perform (at least one)
        total_pairs = M * (M - 1) / 2
        num_swaps = max(1, int(k * total_pairs))

        # Identify indices (i,j) in the upper triangle where values differ
        possible_indices = [
            (i, j)
            for i in range(M)
            for j in range(i + 1, M)
            if matrices[n, i, j] != matrices[n, j, i]
        ]

        # Randomly choose up to num_swaps index pairs to swap
        swaps = random.sample(possible_indices, min(num_swaps, len(possible_indices)))
        for i, j in swaps:
            # Swap the two asymmetric entries
            temp = modified_matrices[n, i, j].clone()
            modified_matrices[n, i, j] = modified_matrices[n, j, i]
            modified_matrices[n, j, i] = temp

    print("RandomSwapVotes Over!")
    return modified_matrices


def genTrainingData(M, T):
    """
    Generate synthetic vote matrices and normalized rank targets for training.

    Each of T samples is a random permutation of 1..M, converted into both:
      - An M x M vote matrix (1 for v1>v2, 0 otherwise)
      - A normalized rank vector of length M (rank/M)

    Args:
        M (int): Number of items per sequence.
        T (int): Number of samples to generate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
          - Matrixs: FloatTensor of shape (T, M, M) with vote matrices.
          - ranks: FloatTensor of shape (T, M) with normalized ranks.
    """
    sequences = []
    Matrixs = []
    ranks = []

    # Generate random sequences and their integer ranks
    for i in range(T):
        vals = random_permutation(M)
        sequences.append(vals)
        ranks.append(rank_of_elements(vals))

    # Build vote matrices: 1 if element j > k, else 0
    for i in range(T):
        curMatrix = np.zeros((M, M), dtype=float)
        for j in range(M):
            for k in range(M):
                v1 = sequences[i][j]
                v2 = sequences[i][k]
                if v1 > v2:
                    curMatrix[j, k] = 1
                else:
                    curMatrix[j, k] = 0
        Matrixs.append(curMatrix)

    # Convert to numpy arrays and normalize ranks to [0,1]
    Matrixs = np.array(Matrixs)
    ranks = (np.array(ranks, dtype=float) + 0.0) / M

    # Convert to PyTorch tensors for model consumption
    Matrixs = torch.tensor(Matrixs).float()
    ranks = torch.tensor(ranks).float()
    return Matrixs, ranks


def randomDelete(matrix, k=0.3):
    """
    Randomly set k fraction of symmetric vote pairs to zero in-place.

    Args:
        matrix (torch.Tensor): Square tensor of shape (M, M) to modify.
        k (float): Fraction of upper-triangle pairs to zero out.
    """
    # Number of items M in square matrix
    n = matrix.shape[1]

    # Get indices of upper-triangle off-diagonal positions
    indices = torch.triu_indices(n, n, offset=1)
    num_pairs = indices.shape[1]

    # Determine how many pairs to zero
    num_to_zero = int(k * num_pairs)
    random_indices = torch.randperm(num_pairs)[:num_to_zero]

    # Zero out both (i,j) and (j,i) for selected pairs
    selected_indices = indices[:, random_indices]
    for i, j in zip(selected_indices[0], selected_indices[1]):
        matrix[i, j] = 0
        matrix[j, i] = 0


def randomDelete_N(inputs, N, max_k=0.8):
    """
    Apply randomDelete to each of N vote matrices with a random k up to max_k.

    Args:
        inputs (torch.Tensor): Tensor of shape (N, M, M) containing vote matrices.
        N (int): Number of matrices to process (should match inputs.shape[0]).
        max_k (float): Maximum fraction of pairs to delete per matrix.
    """
    print("RandomDelete Processing...")
    for i in range(N):
        # Periodically print progress every ~5% of data
        if i % max(int(N/20), 1) == 0:
            print(f"Data Processing {100 * i / N:.1f}%")
        # Choose a random deletion fraction for this sample
        random_k = torch.rand(1).item() * max_k
        randomDelete(inputs[i], random_k)
