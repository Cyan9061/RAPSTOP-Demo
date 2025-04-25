import random
from networkx import strongly_connected_components
import networkx as nx
import torch
from math import pow
import numpy as np

# --------------------------------------------------------------------------------
# Cycle Counting and Graph Construction Utilities
# --------------------------------------------------------------------------------

def count_cycles_matrix_power(A, min_length=2, max_length=20, device='cuda:0'):
    """
    Estimate the number of closed walks (cycles) in a directed graph using matrix powers.

    For each k in [min_length, max_length], computes the trace of A^k and accumulates
    a decayed sum: trace(A^k) / 2^(k-1). This count includes non-simple cycles
    (walks that may revisit nodes), serving as an upper bound on the number of
    simple cycles.

    Args:
        A (torch.Tensor): Adjacency matrix of size M x M (positive entries indicate edges).
        min_length (int): Minimum length of walks to count (default 2).
        max_length (int): Maximum walk length to consider (default 20).
        device (str): CUDA device identifier to use (e.g., 'cuda:0').

    Returns:
        float: Decayed sum of traces, representing a rough cycle count estimate.
    """
    # Binarize the adjacency matrix: any positive entry becomes 1, others 0
    A = (A > 0).float()

    # Move tensor to GPU if available, else fallback to CPU
    if torch.cuda.is_available():
        device = torch.device(device)
        A = A.to(device)
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
        A = A.to(device)

    # Initialize variables
    M = A.shape[0]
    count = 0.0
    Ak = A.clone()  # Will hold A^k iteratively

    # Compute powers of A and accumulate decayed trace values
    for k in range(1, max_length + 1):
        if k > 1:
            Ak = torch.matmul(Ak, A)
        if k >= min_length:
            diag_sum = torch.trace(Ak)
            # Apply exponential decay 1 / 2^(k-1)
            count += diag_sum.item() / pow(2, k - 1)

    return count


def estimate_cycles(A, N=6, sample_size=None):
    """
    Approximate the number of simple cycles of length up to N using DFS sampling.

    Args:
        A (torch.Tensor): Weighted adjacency matrix (positive entries indicate edges).
        N (int): Maximum cycle length to explore (default 6).
        sample_size (int, optional): Number of start nodes to sample; if provided,
                                     results are scaled to the full graph size.
    Returns:
        int: Estimated count of simple cycles (each cycle counted once).
    """
    # Convert to NumPy for adjacency list construction
    mat = A.cpu().detach().numpy()
    M = mat.shape[0]
    G = {i: [] for i in range(M)}
    # Build adjacency list: only include edges with weight > 0
    for i in range(M):
        for j in range(M):
            if i != j and mat[i][j] > 0:
                G[i].append(j)

    def dfs_unique(start, current, depth, visited):
        """
        Recursive DFS to count unique simple cycles starting and ending at `start`.
        Only counts a cycle when `start` is the minimal node in the cycle to avoid duplicates.
        """
        count = 0
        if depth > N:  # Stop if exceeding max length
            return 0
        for neighbor in G.get(current, []):
            if neighbor == start and depth >= 3:
                # Count cycle only if start is smallest in visited set
                if start == min(visited):
                    count += 1
            elif neighbor not in visited:
                count += dfs_unique(start, neighbor, depth + 1, visited | {neighbor})
        return count

    # Choose start nodes: either all or a random subset
    all_nodes = list(range(M))
    if sample_size is not None and sample_size < M:
        start_nodes = random.sample(all_nodes, sample_size)
    else:
        start_nodes = all_nodes

    # Accumulate cycle counts from each start node
    total_cycles = 0
    for node in start_nodes:
        total_cycles += dfs_unique(node, node, 1, {node})

    # Scale result if using sampling
    if sample_size is not None and sample_size < M:
        total_cycles = total_cycles * M / sample_size

    return int(total_cycles)


# --------------------------------------------------------------------------------
# Parallelized version using thread pool for DFS sampling
# --------------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor

def estimate_cycles_parallel(A, N=6, sample_size=None, num_threads=4):
    """
    Parallelized approximation of simple cycle count up to length N.

    Args:
        A (torch.Tensor): Weighted adjacency matrix.
        N (int): Maximum cycle length (default 6).
        sample_size (int, optional): Number of start nodes to sample.
        num_threads (int): Number of threads for parallel DFS.
    Returns:
        int: Estimated cycle count.
    """
    # Build adjacency list from tensor
    mat = A.cpu().detach().numpy()
    M = mat.shape[0]
    G = {i: [] for i in range(M)}
    for i in range(M):
        for j in range(M):
            if i != j and mat[i][j] > 0:
                G[i].append(j)

    def dfs_unique(start, current, depth, visited):
        count = 0
        if depth > N:
            return 0
        for neighbor in G.get(current, []):
            if neighbor == start and depth >= 3:
                if start == min(visited):
                    count += 1
            elif neighbor not in visited:
                count += dfs_unique(start, neighbor, depth + 1, visited | {neighbor})
        return count

    # Select start nodes
    all_nodes = list(range(M))
    if sample_size is not None and sample_size < M:
        start_nodes = random.sample(all_nodes, sample_size)
    else:
        start_nodes = all_nodes

    # Execute DFS in parallel threads
    total_cycles = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(dfs_unique, node, node, 1, {node}) for node in start_nodes]
        for future in futures:
            total_cycles += future.result()

    # Scale if sampling used
    if sample_size is not None and sample_size < M:
        total_cycles = total_cycles * M / sample_size

    return int(total_cycles)

# --------------------------------------------------------------------------------
# Graph construction utilities using NetworkX
# --------------------------------------------------------------------------------

def construct_graph(votes_matrix):
    """
    Convert a vote matrix into a weighted NetworkX DiGraph.

    Args:
        votes_matrix (np.ndarray or torch.Tensor): Adjacency-like matrix where
        entry (i,j) > 0 indicates an edge i->j with weight = entry value.
    Returns:
        networkx.DiGraph: Directed graph with weighted edges.
    """
    # Convert to NumPy if given as tensor
    if isinstance(votes_matrix, torch.Tensor):
        mat = votes_matrix.cpu().detach().numpy()
    else:
        mat = np.array(votes_matrix)
    M = mat.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(M))
    for i in range(M):
        for j in range(M):
            if i != j and mat[i][j] > 0:
                G.add_edge(i, j, weight=mat[i][j])
    return G


def construct_graph_no_weight(votes_matrix):
    """
    Convert a vote matrix into an unweighted NetworkX DiGraph.

    Args:
        votes_matrix (np.ndarray or torch.Tensor): Input adjacency matrix.
    Returns:
        networkx.DiGraph: Directed graph without edge weights.
    """
    # Ensure NumPy array representation
    if isinstance(votes_matrix, torch.Tensor):
        mat = votes_matrix.cpu().detach().numpy()
    else:
        mat = np.array(votes_matrix)
    M = mat.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(M))
    for i in range(M):
        for j in range(M):
            if i != j and mat[i][j] > 0:
                G.add_edge(i, j)
    return G


def is_graph_acyclic(votes_matrix):
    """
    Determine if the directed graph implied by `votes_matrix` is acyclic.

    Args:
        votes_matrix (np.ndarray or torch.Tensor): Adjacency matrix representation.
    Returns:
        bool: True if the graph is a DAG (no cycles), False otherwise.
    """
    G = construct_graph(votes_matrix)
    return nx.is_directed_acyclic_graph(G)
