import random
import torch
from copy import deepcopy
from mcts.GraphCount_and_Operate import count_cycles_matrix_power, estimate_cycles_parallel
import math
import numpy as np


# 3. Definition of State class
# This implements cycle breaking under a total deletion budget (MinimizeCycles)

class SortingGraphState_MinimizeCycles:
    def __init__(self, remaining_budget, K, M, accuracies, p, votes_matrix=None,
                 initial_total_weight=None, initial_cycle_count=None, max_delete_edge=1, delete_edge=0,
                 # true_init_cycle_count=None,
                 removed_weight=0.0):
        """
        Modified constructor:
        remaining_budget: now represents the remaining total weight budget for edge deletions
        Added parameter:
          initial_total_weight: the total weight of all edges in the initial graph
        """
        self.K = K
        self.M = M
        self.accuracies = accuracies
        self.p = p

        # Handle vote matrix
        if votes_matrix is None:
            self.votes_matrix = torch.zeros((M, M))
        else:
            self.votes_matrix = votes_matrix.clone() if isinstance(votes_matrix, torch.Tensor) \
                else torch.tensor(votes_matrix, dtype=torch.float)

        device = torch.device("cuda:0")
        self.votes_matrix = self.votes_matrix.to(device)

        # Compute initial total weight of edges
        if initial_total_weight is None:
            self.initial_total_weight = torch.sum(torch.abs(self.votes_matrix)).item()
        else:
            self.initial_total_weight = initial_total_weight

        # Initialize deletion budget
        if remaining_budget is None:
            # Automatically compute budget as p * total weight
            if self.M > 500:
                self.remaining_budget = min(p * self.initial_total_weight,
                                            4 * self.initial_total_weight / (self.M * self.M))
            else:
                self.remaining_budget = p * self.initial_total_weight
        else:
            self.remaining_budget = remaining_budget

        # Cycle counting (unchanged)
        if initial_cycle_count is None:
            self.initial_cycle_count = self.compute_cycle_count()
        else:
            self.initial_cycle_count = initial_cycle_count

        # Deletion tracking (based on weights)
        self.removed_weight = removed_weight  # total weight of deleted edges

        # True cycle count (for testing)
        # if true_init_cycle_count is None:
        #     self.true_init_cycle_count = compute_cycle_count_Johnson(votes_matrix)
        # else:
        #     self.true_init_cycle_count = true_init_cycle_count

    def takeAction(self, action):
        """
        Modified action function:
        The consumed budget is now equal to the weight of the deleted edge
        """
        i, j = action
        edge_weight = self.votes_matrix[i][j].item()

        new_state = SortingGraphState_MinimizeCycles(
            remaining_budget=self.remaining_budget - edge_weight,
            K=self.K,
            M=self.M,
            accuracies=self.accuracies,
            p=self.p,
            votes_matrix=deepcopy(self.votes_matrix),
            initial_total_weight=self.initial_total_weight,
            initial_cycle_count=self.initial_cycle_count,
            removed_weight=self.removed_weight + edge_weight
        )

        new_state.votes_matrix[i][j] = 0.0
        return new_state

    def isTerminal(self):
        """
        Modified terminal condition:
        1. Remaining budget is not sufficient to delete any edge (<=0)
        2. Deleted weight reaches allowed maximum (1 - p) * total weight
        """
        if self.remaining_budget <= 0:
            return True
        if (self.removed_weight / self.initial_total_weight) >= (1 - self.p):
            return True
        return False

    def compute_cycle_count(self):
        # Non-parallel version
        estimate_cycles_counts = count_cycles_matrix_power(self.votes_matrix, min_length=2, max_length=10)
        return estimate_cycles_counts

    def getPossibleActions(self):
        """
        Return all possible actions (non-zero edges).
        Uses GPU acceleration and zero-copy conversion to a list of (i,j) tuples.
        """
        if self.votes_matrix.device.type != 'cuda':
            self.votes_matrix = self.votes_matrix.to(device='cuda:4')

        non_zero_indices = (self.votes_matrix != 0).nonzero(as_tuple=False)

        if non_zero_indices.is_cuda:
            non_zero_indices = non_zero_indices.cpu()

        np_indices = np.ascontiguousarray(non_zero_indices.numpy())
        answer = np_indices.view([('i', np.int64), ('j', np.int64)]).ravel().tolist()
        return answer

    def getReward(self):
        current_cycle_count = self.compute_cycle_count()
        return self.initial_cycle_count - current_cycle_count


# This class targets making the final graph acyclic while minimizing the total deletion cost
from mcts.GraphCount_and_Operate import is_graph_acyclic, construct_graph


class SortingGraphState_AcyclicOptimize:
    def __init__(self, remaining_budget, K, M, accuracies, p, votes_matrix=None,
                 initial_edge_count=None, removed_count=0, removed_weight=0.0):
        """
        Parameters:
          remaining_budget: deletion budget (each edge deletion costs its weight)
          K, M, accuracies: same as above
          p: retention ratio, used to compute initial budget but not for terminal condition
          votes_matrix: M×M weighted vote matrix (torch tensor or numpy array)
          initial_edge_count: number of initial effective edges (optional)
          removed_count: number of deleted edges (incremented by 1 per deletion)
          removed_weight: total weight of deleted edges
        """
        if remaining_budget is None:
            G = construct_graph(votes_matrix)
            total_weight = sum(weights for _, _, weights in G.edges(data='weight'))
            self.remaining_budget = (p * total_weight)
        else:
            self.remaining_budget = remaining_budget

        self.K = K
        self.M = M
        self.accuracies = accuracies
        self.p = p  # retention ratio (kept for reference)

        # Use deep copy of votes_matrix
        if votes_matrix is None:
            self.votes_matrix = torch.zeros((M, M))
        else:
            if isinstance(votes_matrix, torch.Tensor):
                self.votes_matrix = votes_matrix.clone()
            else:
                self.votes_matrix = torch.tensor(votes_matrix, dtype=torch.float)

        # Count initial number of edges (non-zero values)
        if initial_edge_count is None:
            count = 0
            mat = self.votes_matrix.cpu().detach().numpy()
            for i in range(self.M):
                for j in range(self.M):
                    if mat[i][j] != 0:
                        count += 1
            self.initial_edge_count = count
        else:
            self.initial_edge_count = initial_edge_count

        self.removed_count = removed_count  # number of removed edges
        self.removed_weight = removed_weight  # total weight of removed edges

    def getPossibleActions(self):
        """
        Return all current candidate actions, i.e., all existing directed edges (i, j) with non-zero weight.
        """
        actions = []
        mat = self.votes_matrix.cpu().detach().numpy()
        for i in range(self.M):
            for j in range(self.M):
                if mat[i][j] != 0:
                    actions.append((i, j))
        return actions

    def takeAction(self, action):
        """
        Perform edge deletion. Given action = (i, j), set edge (i, j) to 0.
        Budget is reduced by the edge weight, and deletion statistics are updated.
        Returns a new state.
        """
        i, j = action
        edge_weight = self.votes_matrix[i][j].item()
        new_state = SortingGraphState_AcyclicOptimize(
            remaining_budget=self.remaining_budget - edge_weight,
            K=self.K,
            M=self.M,
            accuracies=self.accuracies,
            p=self.p,
            votes_matrix=deepcopy(self.votes_matrix),
            initial_edge_count=self.initial_edge_count,
            removed_count=self.removed_count + 1,
            removed_weight=self.removed_weight + edge_weight
        )

        # Set the deleted edge weight to 0 in the new state
        new_state.votes_matrix[i][j] = 0.0
        return new_state

    def isTerminal(self):
        """
        Terminal condition for the state:
        1. If the graph becomes acyclic, the process is done.
        2. If the budget is exhausted (≤ 0), the process is done.
        """
        if is_graph_acyclic(self.votes_matrix):
            return True
        if self.remaining_budget <= 0:
            return True
        return False

    def getReward(self):
        """
        Compute the reward based on how many edges were preserved.
        Reward is defined as: the proportion of edges preserved in the acyclic graph.
        """
        preserved_edges = self.initial_edge_count - self.removed_count
        return preserved_edges / self.initial_edge_count


def greedyPolicy(state):
    #print("begin greedyPolicy")
    while not state.isTerminal():
        #print("begin getPossibleActions")
        actions = state.getPossibleActions()
        #print("over getPossibleActions")
        if not actions:
            break

        if random.random() < 0.5:
            chosen_action = random.choice(actions)
        else:
            mat = state.votes_matrix.cpu().detach().numpy()
            accuracy = min(state.accuracies)

            def candidate_key(action):
                i, j = action
                a = mat[i][j]
                b = mat[j][i]
                confidence = calculate_confidence(accuracy, a, b)
                return (confidence, a)

            chosen_action = min(actions, key=candidate_key)
        #print("begin takeAction")
        state = state.takeAction(chosen_action)
        #print("over takeAction")
    #print("over rollout")
    #print("begin getReward")
    reward = state.getReward()
    #print("over getReward")
    return reward  # [0]

# %%
# Confidence computation
def calculate_confidence(correct_rate, correct_count, incorrect_count):
    """
    Calculate the confidence (i.e., probability that the majority is correct).

    Args:
        correct_rate: Accuracy of the voters (assumed the same for all).
        correct_count: Number of votes in favor (can be float).
        incorrect_count: Number of votes against (can be float).

    Returns:
        probability: Posterior probability that the decision is correct.
    """
    total_count = correct_count + incorrect_count

    # Assume prior probabilities: P(H) = 0.5 and P(~H) = 0.5
    prior_H = 0.5
    prior_not_H = 0.5

    # Compute likelihoods P(data | H) and P(data | ~H)
    # Since we allow float counts, we approximate using continuous correction
    # Normalizing term approximates continuity
    likelihood_H = (
        (correct_rate ** correct_count) *
        ((1 - correct_rate) ** incorrect_count) *
        (1 / math.sqrt(2 * math.pi * correct_count))  # approximate continuous correction
    )
    likelihood_not_H = (
        ((1 - correct_rate) ** correct_count) *
        (correct_rate ** incorrect_count) *
        (1 / math.sqrt(2 * math.pi * correct_count))  # approximate continuous correction
    )

    # Compute posterior probability using Bayes' Rule
    posterior_H = (likelihood_H * prior_H) / (likelihood_H * prior_H + likelihood_not_H * prior_not_H)

    return posterior_H


def preprocess_vote_matrix(votes_matrix, accuracies, hope_confidence = 0.001):
    """
    Preprocess the initial vote matrix: remove edges with confidence below a threshold.

    Args:
        votes_matrix: Original vote matrix (shape M×M), can be torch.Tensor or numpy array.
        accuracies: List of individual accuracies; use the lowest one for conservative estimation.
        hope_confidence: Threshold of minimum required confidence to retain an edge.

    Returns:
        new_matrix: The vote matrix after removing low-confidence edges (same type as input).
        count_weight_delete: Sum of weights of all deleted edges.
    """
    # Use the minimum accuracy for conservative confidence calculation
    correct_rate = min(accuracies)

    # Convert vote matrix to numpy array for processing
    if isinstance(votes_matrix, torch.Tensor):
        mat = votes_matrix.cpu().detach().numpy().copy()
    else:
        mat = np.array(votes_matrix, dtype=float)

    M = mat.shape[0]
    #count_all = 0
    #count_need = 0
    count_weight_delete = 0
    for i in range(M):
        for j in range(M):
            if i == j or mat[i][j] == 0.:
                continue
            # Only consider edge (i,j) if the reverse edge (j,i) exists
            #count_all += 1
            if mat[j][i] == 0.:
                continue
            #count_need += 1
            if mat[i][j] > 0:
                # Number of correct votes = weight in (i,j)
                correct_count = mat[i][j]
                # Number of incorrect votes = weight in (j,i)
                incorrect_count = mat[j][i]
                # Compute confidence for edge (i,j)
                confidence = calculate_confidence(correct_rate, correct_count, incorrect_count)
                # Remove edge if confidence is too low
                if confidence < hope_confidence:
                    count_weight_delete += mat[i][j]
                    mat[i][j] = 0.0

    # Return processed matrix (same type as input), and total deleted weight
    return torch.tensor(mat, dtype=votes_matrix.dtype) if isinstance(votes_matrix, torch.Tensor) else mat ,count_weight_delete
