# %%
# This program implements a method to maximize the number of broken cycles (MinimizeCycles)
# under a given deletion budget using Monte Carlo Tree Search (MCTS)

import random
import torch
import numpy as np
import networkx as nx
from mcts.mcts import mcts
from mcts.DataProcess import genTrainingData, randomDelete_N, swap_votes
from mcts.SortingGraphState_and_Policy import greedyPolicy, preprocess_vote_matrix
from mcts.SortingGraphState_and_Policy import SortingGraphState_MinimizeCycles as SortingGraphState

# Set random seed for reproducibility
SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Function to compute a weighted vote matrix using LLM accuracies
def get_weighted_vote_matrix(vote_matrices, accuracies):
    """
    Args:
      vote_matrices: torch_tensor, shape = (K, M, M), where K is the number of LLMs and M is the number of elements
      accuracies: list of length K, each entry is the accuracy of an LLM

    Returns:
      weighted_matrix: torch_tensor of shape (M, M), aggregated vote matrix weighted by accuracy
    """
    K, M, _ = vote_matrices.shape
    weighted_matrix = torch.zeros((M, M))
    for k in range(K):
        weighted_matrix += accuracies[k] * vote_matrices[k]
    return weighted_matrix

# Function for running MCTS with debug output, used to minimize cycles
def MCTS_debug(matrices, explorationConstant=1 / np.sqrt(2), epsilon=1e-5, p=0.2, accuracies=None):
    """
    Args:
        matrices: torch tensor of shape (K, M, M), input vote matrices
        explorationConstant: MCTS exploration parameter
        epsilon: rollout probability
        p: proportion of uncertain votes to be preserved
        accuracies: list of LLM accuracies; if None, assumed to be 1.0

    Returns:
        Updated vote matrix after MCTS
        Estimated number of cycles removed
    """
    K = matrices.shape[0]
    M = matrices.shape[1]
    if accuracies is None:
        accuracies = [1 for _ in range(K)]
    weightedVoteMatrix = get_weighted_vote_matrix(matrices, accuracies)

    # Initialize initial MCTS state
    initialState = SortingGraphState(
        remaining_budget=None,
        K=K,
        M=M,
        accuracies=accuracies,
        p=p,
        votes_matrix=weightedVoteMatrix
    )

    # Create MCTS instance with greedy rollout policy
    mcts_instance = mcts(timeLimit=2000, explorationConstant=explorationConstant, rolloutPolicy=greedyPolicy, epsilon=epsilon)

    currentState = initialState
    Init_cycle_count = currentState.initial_cycle_count
    iteration = 0

    # Iterate while not terminal and within iteration budget
    while not currentState.isTerminal() and iteration < 1000:
        best_action = mcts_instance.search(currentState)
        currentState = currentState.takeAction(best_action)
        Now_cycle_counts = Init_cycle_count - currentState.getReward()
        if Now_cycle_counts == 0:
            break
        iteration += 1

    # Print final results
    print("Final deleted edges weight:", currentState.removed_weight)
    print("Final deleted edges counts:", iteration)
    print("Final reward (cycle reduction):", currentState.getReward())
    return currentState.votes_matrix, Init_cycle_count - currentState.getReward()


# %%
# ----------------------------
# Main function to run the full process

if __name__ == "__main__":
    K = 10  # Number of LLMs
    M = 10  # Number of items to rank

    accuracies = [0.8] * K  # Simulated accuracy for each LLM

    # Generate synthetic vote matrix and ground truth ranking
    print("Making ranking data")
    matrices_original, rankings = genTrainingData(M, 1)

    # Stack the same matrix K times to simulate K LLMs
    matrices_original_stack = torch.stack([matrices_original[0]] * K, dim=0)

    # Randomly delete entries from the vote matrix
    matrices_delete = matrices_original_stack.clone()
    randomDelete_N(matrices_delete, 1, max_k=0.6)

    # Randomly flip votes in the matrix
    matrices = swap_votes(matrices_delete, k=0.2)
    print("rankings", rankings * M)

    # Calculate the weighted vote matrix using accuracies
    weightedVoteMatrix = get_weighted_vote_matrix(matrices, accuracies)

    # Preprocess to remove conflicting votes and measure their total weight
    initVoteMatrix, count_weight_delete = preprocess_vote_matrix(weightedVoteMatrix, accuracies, hope_confidence=0)

    print("Initial votes matrix:\n", initVoteMatrix)
    print("Weight of deleted votes:", count_weight_delete)

    # Determine remaining budget for MCTS after preprocessing
    p = 0.2
    testState = SortingGraphState(
        remaining_budget=None,
        K=K,
        M=M,
        accuracies=accuracies,
        p=p,
        votes_matrix=weightedVoteMatrix,
    )
    B = testState.remaining_budget - count_weight_delete
    print("Budget", B)

    # Create initial state for MCTS after vote preprocessing
    initialState = SortingGraphState(
        remaining_budget=B,
        K=K,
        M=M,
        accuracies=accuracies,
        p=p,
        votes_matrix=initVoteMatrix
    )

    # Initialize MCTS
    mcts_instance = mcts(timeLimit=10000, rolloutPolicy=greedyPolicy, epsilon=1e-5)
    currentState = initialState
    Init_cycle_count = currentState.initial_cycle_count
    print(f"Init cycle counts:{Init_cycle_count}")
    iteration = 0

    # Run MCTS until terminal state is reached
    while not currentState.isTerminal():
        print(f"\nIteration {iteration + 1}:")
        best_action = mcts_instance.search(currentState)
        print(f"Discard edge: {best_action}")
        currentState = currentState.takeAction(best_action)
        Now_cycle_counts = Init_cycle_count - currentState.getReward()
        print("Now cycle counts:(estimated)", Now_cycle_counts)
        if Now_cycle_counts == 0:
            break
        iteration += 1

    print("\nFinal votes matrix:")
    print(currentState.votes_matrix)
    print("Final cycle counts:(estimated)", Init_cycle_count - currentState.getReward())
    print("Final deleted edges weight:", currentState.removed_weight)
    print("Budget", B)
    print("Final deleted edges counts:", iteration)
    print("Final reward (cycle reduction):", currentState.getReward())

    # %%
    # Load and test deep learning model (AttentionModel) to evaluate final ranking results

    from mcts.Sort_attention import AttentionModel, rank_of_elements, normalize_symmetric_elements

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained attention model
    model = AttentionModel(embed_dim=M, num_heads=2).to(device)
    model.load_state_dict(torch.load('final_model_M=10_k1=0.6_k2=0.05.pth', map_location=device))
    model.eval()

    with torch.no_grad():
        difNumber1 = 0
        difNumber0 = 0
        difNumber01 = 0
        verified_input = matrices_original
        # print(verified_input.shape)
        verified_input = normalize_symmetric_elements(verified_input)
        verified_input = verified_input.to(device)

        input1 = (currentState.votes_matrix).unsqueeze(0)
        input1 = normalize_symmetric_elements(input1)
        input1 = input1.to(device)  # 将输入数据移动到设备上

        input0 = weightedVoteMatrix.unsqueeze(0)
        input0 = normalize_symmetric_elements(input0)
        input0 = input0.to(device)

        input01 = initVoteMatrix.unsqueeze(0)
        input01 = normalize_symmetric_elements(input01)
        input01 = input01.to(device)

        target = torch.tensor(rankings)
        target = target.to(device)  # 将目标数据移动到设备上

        verified_output, _ = model(verified_input)
        output1, _ = model(input1)
        output0, _ = model(input0)
        output01, _ = model(input01)
        # 将数据移回CPU进行rank_of_elements操作（如果rank_of_elements不支持GPU）
        # print(output1)

        v_verified = 1 - verified_output.cpu().detach().numpy()[0]
        v1 = 1 - output1.cpu().numpy()[0]
        v0 = 1 - output0.cpu().numpy()[0]
        v01 = 1 - output01.cpu().numpy()[0]
        # print("v1",v1)
        r_verified = rank_of_elements(v_verified)
        r1 = rank_of_elements(v1)
        r0 = rank_of_elements(v0)
        r01 = rank_of_elements(v01)
        ranking = (rankings[0] * M).tolist()

        print("原顺序", ranking)
        print("检测Model是否正确（如果和原顺序相同则正确）", r_verified)
        print("丢弃争议投票前预测顺序", r0)
        print("丢弃争议投票后预测顺序", r01)
        print("丢弃争议投票并且debug后预测顺序", r1)
        for j in range(M):
            # print(r0[j],ranking[j])
            if r0[j] != ranking[j]:
                difNumber0 += 1
        for j in range(M):
            if r01[j] != ranking[j]:
                difNumber01 += 1
        for j in range(M):
            if r1[j] != ranking[j]:
                difNumber1 += 1

        print(f"丢弃争议投票前预测顺序difNumber:{difNumber0},accuracy:{1 - difNumber0 / M}")
        print(f"丢弃争议投票后预测顺序difNumber:{difNumber01},accuracy:{1 - difNumber01 / M}")
        print(f"丢弃争议投票并且debug后预测顺序difNumber:{difNumber1},accuracy:{1 - difNumber1 / M}")

