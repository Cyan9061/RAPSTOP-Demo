import argparse
import numpy as np
import torch
import os
from math import sqrt

# --------------------------------------------------------------------------------
# Import key MCTS and sorting utilities from the 'mcts' module
# --------------------------------------------------------------------------------
# MCTS_Debug: core Monte Carlo Tree Search debugging function that returns a processed
#             matrix and the corresponding cycle count for analysis.
# genTrainingData: generates synthetic training data for the MCTS based ranking (may be unused here).
from mcts.MCTS_MinimizeCycles import MCTS_debug, genTrainingData
# genTrainingData: alternative data generation method for sorting attention model.
# randomDelete_N: function to randomly delete N evidence entries for perturbation.
# swap_evidences: function to swap evidence entries between positions in the matrix.
from mcts.Sort_attention import genTrainingData, randomDelete_N
# greedyPolicy: heuristic policy to quickly generate a sort order without search.
# preprocess_evidence_matrix: preprocess raw evidence matrix into graph state form.
from mcts.SortingGraphState_and_Policy import greedyPolicy
# SortingGraphState_MinimizeCycles: graph state representation specialized for minimizing cycles.
from mcts.SortingGraphState_and_Policy import SortingGraphState_MinimizeCycles as SortingGraphState

# --------------------------------------------------------------------------------
# Set random seeds for reproducibility across NumPy and PyTorch (CPU and GPU)
# --------------------------------------------------------------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def MCTSDebugImplement(matrices, PrecisionList, p_list, explor_list, epsilon_list, evalFunc='Cycle'):
    """
    Wrapper to tune MCTS parameters (p, explorationConstant, epsilon) on a batch of evidence matrices.

    Args:
        matrices (np.ndarray or torch.Tensor): K x M x M input vote/evidence matrices.
        PrecisionList (list of float): List of per-model precision probabilities.
        p_list (list of float): Candidate values of budget parameter p to test.
        explor_list (list of float): Candidate exploration constants for UCT.
        epsilon_list (list of float): Candidate epsilon values for error tolerance.
        evalFunc (str): Evaluation criterion (default 'Cycle' for cycle minimization).

    Returns:
        torch.Tensor: The best transformed evidence matrices under optimal parameters.
    """
    # Ensure matrices are in PyTorch tensor format, float type
    if not isinstance(matrices, torch.Tensor):
        matrices = torch.tensor(matrices).float()
    else:
        matrices = matrices.float()

    # Zero out any negative entries to maintain valid evidence format
    matrices[matrices < 0] = 0

    # Clone original input to reset for each parameter combination
    matrices_original = matrices.clone()

    best_final_cycle_counts = None
    best_matrices_New = None
    best_params = None

    # Grid search over all combinations of p, explorationConstant, and epsilon
    for p_val in p_list:
        for explor_val in explor_list:
            for eps_val in epsilon_list:
                # Reset randomness for each trial to ensure fair comparison
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                # Work on a fresh copy of the original matrices
                matrices_run = matrices_original.clone()

                # Run the MCTS debug process to obtain a new matrix and cycle count
                matrices_New, final_cycle_counts = MCTS_debug(
                    matrices_run,
                    explorationConstant=explor_val,
                    epsilon=eps_val,
                    p=p_val,
                    accuracies=PrecisionList
                )

                # Report performance for this parameter setting
                print(f"Params: p={p_val}, explorationConstant={explor_val}, epsilon={eps_val}  "
                      f"->  final_cycle_counts = {final_cycle_counts}")

                # Update best result if cycle count improved (i.e., decreased)
                if best_final_cycle_counts is None or final_cycle_counts < best_final_cycle_counts:
                    best_final_cycle_counts = final_cycle_counts
                    best_matrices_New = matrices_New.clone()
                    best_params = (p_val, explor_val, eps_val)

    # Display the best-found parameter combination and its cycle count
    print("=" * 20)
    print(f"Best parameters: p={best_params[0]}, explorationConstant={best_params[1]}, epsilon={best_params[2]}")
    print(f"Final cycle counts = {best_final_cycle_counts}")
    print("=" * 20)

    # Post-process the best matrix: enforce skew symmetry by reflecting zero entries
    M_dim = best_matrices_New.shape[1]
    K_dim = matrices_original.shape[0]
    for i in range(M_dim):
        for j in range(M_dim):
            if best_matrices_New[i, j].item() == 0:
                # For each zero entry, set opposite entry to negative of original
                for k in range(K_dim):
                    matrices_original[k, i, j] = -matrices_original[k, j, i]

    # Return the adjusted original matrices as the final output
    return matrices_original


if __name__ == "__main__":
    # --------------------------------------------------------------------------------
    # Main script: parse arguments, load data, run debug implementation, and save result
    # --------------------------------------------------------------------------------
    # Determine script and project root paths
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    data_default = os.path.join(project_root, "data")

    # Configure command-line arguments for debugging parameters and file paths
    parser = argparse.ArgumentParser(description='Args on Real MCTS Debug')
    parser.add_argument('--budgetP', type=float, default=0.01,
                        help='Budget parameter p for MCTS')
    parser.add_argument('--explorationConstant', type=float, default=1 / sqrt(2),
                        help='Exploration constant for UCT in MCTS')
    parser.add_argument('--epsilon', type=float, default=1e-5,
                        help='Epsilon threshold for action selection')
    parser.add_argument('--evidenceLoadPath', type=str, default="evidenceMatrix.npy",
                        help='Filename of input evidence matrix (npy)')
    parser.add_argument('--savePath', type=str, default="evidenceMatrix_debug.npy",
                        help='Filename to save the debugged matrix')
    parser.add_argument('--PrecisionList', nargs='+', type=float,
                        default=[0.66,0.22,0.33,0.9,0.66,0.4,0.66,0.33,0.33],
                        help='List of precision values for each model')
    args = parser.parse_args()

    # Build full file paths for loading and saving
    evidenceLoadPath = os.path.join(data_default, args.evidenceLoadPath)
    savePath = os.path.join(data_default, args.savePath)

    # Load the raw evidence matrix from disk
    init_matrix = np.load(evidenceLoadPath)

    # Print configuration summary to console
    print("=" * 20)
    print(f"PrecisionList: {args.PrecisionList}")
    print(f"explorationConstant: {args.explorationConstant}")
    print(f"epsilon: {args.epsilon}")
    print(f"evidence load path: {evidenceLoadPath}")
    print(f"evidence save path: {savePath}")
    print(f"evidence budget: {args.budgetP}")
    print("=" * 20)

    # Execute the MCTS debugging and parameter tuning procedure
    best_matrix = MCTSDebugImplement(
        init_matrix,
        args.PrecisionList,
        [args.budgetP],
        [args.explorationConstant],
        [args.epsilon]
    )

    # Output the best-found matrix and save to disk
    print(best_matrix)
    np.save(savePath, best_matrix)
    print(f"Best matrix saved to {savePath}")