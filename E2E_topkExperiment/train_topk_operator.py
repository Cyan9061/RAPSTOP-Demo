#train_topk_operator.py

import numpy as np
import torch
import random
import os
import torch.optim as optim
import torch.nn as nn
# --------------------------------------------------------------------------------
# Utility Functions for Reproducibility and Data Preparation
# --------------------------------------------------------------------------------

# Function to set random seeds for reproducibility across Python, NumPy, and PyTorch
# Ensures deterministic behavior for operations like convolution by setting appropriate flags

def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # guarantee consistent results for convolution operations

# Function to generate a random permutation of integers 1 through N
# Returns a shuffled list of numbers [1, 2, ..., N]

def random_permutation(N):
    numbers = list(range(1, N + 1))  # generate list from 1 to N
    random.shuffle(numbers)           # shuffle in-place
    return numbers

# Function to compute the rank positions of elements in a list
# Higher values receive lower numeric ranks (1 is the highest-ranked element)
# Returns a list of ranks corresponding to original positions

def rank_of_elements(nums):
    # Pair each value with its original index, sort by value descending
    sorted_with_index = sorted(
        enumerate(nums), key=lambda x: x[1], reverse=True
    )

    ranks = [0] * len(nums)
    # Assign rank based on sorted order, starting at 1 for the largest element
    for rank, (original_index, _) in enumerate(sorted_with_index, start=1):
        ranks[original_index] = rank

    return ranks

# Function to create a skew-symmetric matrix from a given matrix
def make_skew_symmetric(A):
    K, M, _ = A.shape
    result = np.zeros_like(A)
    
    for k in range(K):
        upper_tri = np.triu(A[k])
        
        lower_tri = -upper_tri.T
        
        result[k] = upper_tri + lower_tri
    
    return result

# Generate training data using vectorized methods (faster, uses probabilities)
def genTrainingDataFaster(M, T, ModelPrecisionList, EmptyRatioList):
    """
    Generate training data for attention model using vectorized operations.
    
    Args:
        M (int): Number of elements to rank.
        T (int): Number of samples to generate.
        ModelPrecisionList (list): Precision values for each simulated model (0~1).
        EmptyRatioList (list): Probability of each model skipping a comparison.
    
    Returns:
        Tuple[Tensor, Tensor]: Pair of (Matrixs, ranks) used as training input and target.
    """
    sequences = np.random.rand(T, M).argsort(axis=1)
    ranks = sequences.argsort(axis=1)
    
    num_models = len(ModelPrecisionList)
    correct_probs = np.array(ModelPrecisionList)[:, np.newaxis, np.newaxis]
    skip_probs = np.array(EmptyRatioList)[:, np.newaxis, np.newaxis]
    
    Matrixs = []
    for i in range(T):
        if i % 1000 == 1:
            print(f"Processing: {i/T*100:.2f}%", flush=True)
        
        vals = sequences[i]
        correct_matrix = np.sign(vals[:, None] - vals[None, :])
        
        skip_rand = np.random.rand(num_models, M, M)
        correct_rand = np.random.rand(num_models, M, M)
        
        skip_mask = skip_rand < skip_probs
        correct_mask = correct_rand < correct_probs
        
        correct_matrix_expanded = correct_matrix[np.newaxis, :, :]
        
        model_matrix = np.where(
            skip_mask,
            0,
            np.where(correct_mask, correct_matrix_expanded, -correct_matrix_expanded)
        )
        
        Matrixs.append(make_skew_symmetric(model_matrix))
    
    Matrixs = np.array(Matrixs)
    Matrixs = Matrixs.reshape(Matrixs.shape[0], -1, Matrixs.shape[-1])
    
    ranks = (ranks.astype(np.float32) + 1) / M
    Matrixs = torch.tensor(Matrixs).float()
    ranks = torch.tensor(ranks).float()
    
    return Matrixs, ranks


from scipy.stats import spearmanr

# Function to compute Spearman rank correlation coefficient between two rankings
def compute_spearman(true_ranks, predicted_ranks):
    true_ranks = np.array(true_ranks)
    predicted_ranks = np.array(predicted_ranks)
    spearman_corr, _ = spearmanr(true_ranks, predicted_ranks)
    return spearman_corr

# Function to compute Recall@k and pairwise accuracy ACC@k for top-k elements
# true_ranks: ground-truth rank list
# predicted_ranks: predicted score list (higher means more likely to be top)
# k: number of top items to consider
# Returns: recall (proportion of true top-k recovered) and ACC@k (pairwise ordering accuracy)
def compute_recall_acc(true_ranks, predicted_ranks, k):
    # Identify the top-k predicted scores
    top_k_values = sorted(predicted_ranks, reverse=True)[:k]
    top_k_set = set(top_k_values)

    A = []
    indices = []
    for i, v in enumerate(predicted_ranks):
        if v in top_k_set:
            A.append(v)
            indices.append(i)
            if len(A) == k:
                break

    # Gather corresponding true ranks
    T = [true_ranks[i] for i in indices]

    intersection = set(A) & set(T)
    recall = len(intersection) / k

    # Build position maps for computing pairwise accuracy
    pred_pos = {obj: idx + 1 for idx, obj in enumerate(predicted_ranks)}
    true_pos = {obj: idx + 1 for idx, obj in enumerate(true_ranks)}

    count = 0
    # Count correctly ordered pairs among the intersection
    for o_i in intersection:
        for o_j in intersection:
            if o_j != o_i:
                a_i = pred_pos[o_i]
                a_j = pred_pos[o_j]
                t_i = true_pos[o_i]
                t_j = true_pos[o_j]
                if a_i < a_j and t_i < t_j:
                    count += 1

    denominator = k * (k - 1) / 2
    acc_k = count / denominator if denominator != 0 else 0.0
    return recall, acc_k

# Wrapper function to compute and print metrics for multiple k values and full-length
# estranks: estimated rank scores (higher means more likely top)
# TestRank: ground-truth ranks
# klist: list of k values to evaluate

def compute_metrices(estranks, TestRank, klist):
    M = len(estranks)
    for k in klist:
        recall_k, acc_k = compute_recall_acc(TestRank, estranks, k)
        print(f"Recall@{k}: {recall_k:.4f}, ACC^{k}: {acc_k:.4f}")

    # Evaluate at full length k = M
    recall_M, acc_k_M = compute_recall_acc(TestRank, estranks, M)
    print(f"Recall@{M}: {recall_M:.4f}, ACC^{M}: {acc_k_M:.4f}")

    # Compute Spearman correlation
    print("Spearman", compute_spearman(TestRank, estranks))

# --------------------------------------------------------------------------------
# Dataset and Model Definitions
# --------------------------------------------------------------------------------

from torch.utils.data import Dataset, DataLoader, random_split

# Custom PyTorch Dataset for inputs and targets
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Multi-head attention based model for ranking
# embed_dim: dimension of input embeddings (should match M)
class MultiAttentionModel(nn.Module):
    def __init__(self, embed_dim):
        super(MultiAttentionModel, self).__init__()
        # Single-head attention (num_heads=1) applied batch-first
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        ])
        # Final linear layer (not used here but reserved for extension)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Apply each attention layer sequentially
        for attention in self.attention_layers:
            attn_output, _ = attention(x, x, x)
            x = attn_output

        # Aggregate by mean pooling across sequence dimension
        global_features = torch.mean(x, dim=1)

        return global_features

# --------------------------------------------------------------------------------
# File Matching and Matrix Utilities
# --------------------------------------------------------------------------------

# Function to pair realOrder and sortMatrix files based on naming conventions
# Returns list of tuples (real_order_filename, sort_matrix_filename)
def match_files(path):
    files = os.listdir(path)
    real_order_files = []
    sort_matrix_files = []

    for file in files:
        if 'realOrder' in file:
            real_order_files.append(file)
        elif 'sortMatrix' in file:
            sort_matrix_files.append(file)

    matched_pairs = []
    for real_file in real_order_files:
        real_key = real_file.replace('realOrder', '')
        for sort_file in sort_matrix_files:
            sort_key = sort_file.replace('sortMatrix', '')
            if real_key == sort_key:
                matched_pairs.append((real_file, sort_file))
                break

    return matched_pairs

# Function to convert a sequence of values into a pairwise comparison matrix
# Returns MxM matrix with -1, 0, or 1 indicating ordering between each pair
def seq2Matrix(sequences):
    M = len(sequences)
    curMatrix = np.zeros((M, M))
    for j in range(M):
        for k in range(M):
            v1 = sequences[j]
            v2 = sequences[k]
            if v1 > v2:
                curMatrix[j, k] = -1
            elif v1 < v2:
                curMatrix[j, k] = 1
            else:
                curMatrix[j, k] = 0
    return curMatrix

# Function to compute empty ratio and precision list from predicted and real matrices
# sortMatrix_file: path to .npy file of shape (K,M,M) with predicted comparisons
# realOrder_file: path to .npy file with true sequence order of length M
# Returns two lists:
#   empty_ratio_list: fraction of comparisons skipped per model
#   precise_list: accuracy on non-zero comparisons per model
def getEmptyRatio_PreciseList(sortMatrix_file, realOrder_file):
    predict_sort_matrix = np.load(sortMatrix_file)
    real_order = np.load(realOrder_file)
    real_sort_matrix = seq2Matrix(real_order.tolist())

    K, M, _ = predict_sort_matrix.shape
    total_num = (M * (M - 1))

    # Compute proportion of zero entries per model
    empty_ratio_list = [
        1 - np.count_nonzero(predict_sort_matrix[i]) / total_num
        for i in range(K)
    ]

    precise_list = []
    for i in range(K):
        pred_matrix = predict_sort_matrix[i]
        nonzero_mask = pred_matrix != 0
        nonzero_count = np.count_nonzero(nonzero_mask)

        if nonzero_count > 0:
            correct_count = np.sum(
                pred_matrix[nonzero_mask] == real_sort_matrix[nonzero_mask]
            )
            accuracy = correct_count / nonzero_count
        else:
            accuracy = 0.0  # define accuracy as 0 if no comparisons made
        precise_list.append(accuracy)

    return empty_ratio_list, precise_list

if __name__ == "__main__":
    # Determine the absolute path of the current script file
    current_script_path = os.path.abspath(__file__)

    # Derive the project root directory by going two levels up from the script location
    project_root = os.path.dirname(os.path.dirname(current_script_path))  # Project root (up two levels)

    # Define the data directory within the project structure
    data_dir = os.path.join(project_root, 'data')

    # Argument parsing setup: define CLI options for training configuration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=16,
                        help='Number of items to rank per sample')
    parser.add_argument('--N', type=int, default=5000,
                        help='Number of training samples to generate')
    parser.add_argument('--PrecisionList', nargs='+', type=float,
                        default=[0.66,0.22,0.33,0.9,0.66,0.4,0.66,0.33,0.33],
                        help='List of precision probabilities for each simulated model')
    parser.add_argument('--GPU', type=int, default=0,
                        help='CUDA device index to use (if available)')
    parser.add_argument('--evidenceMatrix', type=str, default="evidenceMatrix.npy",
                        help='Filename of the input evidence matrix stored in data directory')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--bs', type=int, default=32,
                        help='Batch size for DataLoader')
    parser.add_argument('--trainEpochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--savingModel', type=int, default=1,
                        help='Flag to save the trained model (1: save, 0: skip)')
    parser.add_argument('--isDebug', type=int, default=False,
                        help='Debug mode flag (currently unused)')
    parser.add_argument('--model_dir', type=str, default="models/",
                        help='Directory to save model checkpoints')
    args = parser.parse_args()

    # Construct the full path to the evidence matrix file
    TestMatrixInput_file = os.path.join(data_dir, args.evidenceMatrix)

    # Print a separator and the parsed configuration for verification
    print("=" * 20)
    print(f"M: {args.M}")
    print(f"N: {args.N}")
    print(f"PrecisionList: {args.PrecisionList}")
    print(f"GPU: {args.GPU}")
    print(f"evidenceMatrix: {TestMatrixInput_file}")
    print(f"lr: {args.lr}")
    print(f"bs: {args.bs}")
    print(f"trainEpochs: {args.trainEpochs}")
    print(f"seed: {args.seed}")
    print(f"savingModel: {args.savingModel}")
    print(f"isDebug: {args.isDebug}", flush=True)

    # Initialize random seeds for reproducibility
    setup_seed(args.seed)

    # Define a mission name for model checkpoint naming
    mission = "Visual"

    # Recompute project and models directories for clarity
    models_dir = os.path.join(project_root, 'models')
    # Construct the path where the model will be saved, encoding key hyperparameters
    modelSavePath = os.path.join(
        models_dir,
        f"{mission}_K{len(args.PrecisionList)}_M{args.M}_N{args.N}_bs{args.bs}_lr{args.lr}"
    )

    # If saving is enabled, display the target save path
    if args.savingModel == 1:
        print(modelSavePath)
    print("=" * 20)

    # Select computation device: GPU if available and specified, otherwise CPU
    device = torch.device(
        f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu"
    )

    # Shorthand local variables for key parameters
    M = args.M
    N = args.N
    ModelPrecisionList = args.PrecisionList

    # Load the precomputed evidence matrix from .npy file
    print(f"Using device: {device}", flush=True)
    TestMatrixInput = np.load(TestMatrixInput_file)

    # Determine the current matrix dimension (M2) from the loaded tensor
    m2 = TestMatrixInput.shape[2]

    # Compute each model's empty ratio: fraction of zero (skipped) comparisons
    EmptyRatioList = [
        float(1 - np.count_nonzero(np.abs(TestMatrixInput[i])) / (m2 * (m2 - 1)))
        for i in range(len(ModelPrecisionList))
    ]

    # If loaded matrix is smaller than expected M, augment it to full size
    if m2 < M:
        print("Augmenting the testing matrix")
        K = TestMatrixInput.shape[0]
        augMatrix = np.zeros((K, M, M))
        for i in range(K):
            for j in range(M):
                for k in range(M):
                    if j < m2 and k < m2:
                        augMatrix[i, j, k] = TestMatrixInput[i, j, k]
                    else:
                        # Fill diagonal with 0, lower with -1, upper with +1
                        augMatrix[i, j, k] = 0 if j == k else (-1 if j > k else 1)
        TestMatrixInput = augMatrix

    # Generate synthetic training data based on DeepSeek method
    print("Generating Training Data")
    inputs, targets = genTrainingDataFaster(M, N, ModelPrecisionList, EmptyRatioList)
    print("Done Training Data", flush=True)

    # Prepare PyTorch datasets and split into training/validation sets
    dataset = CustomDataset(inputs, targets)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = MultiAttentionModel(embed_dim=M).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # === Training and Evaluation Loop ===
    num_epochs = args.trainEpochs
    for epoch in range(num_epochs):
        print(f"epoch {epoch} begin")
        model.train()
        total_train_loss = 0

        # --- Training Phase ---
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_inputs.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Batch Loss: {loss.item():.4f}, "
              f"Avg Train Loss: {avg_train_loss:.4f}")

        # --- Evaluation Phase ---
        model.eval()
        with torch.no_grad():
            total_test_loss = 0
            difNumber = 0
            eval_count = 0
            for batch_inputs, batch_targets in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                total_test_loss += loss.item()

                # Calculate cumulative difference in rank positions
                for i in range(outputs.size(0)):
                    true_vals = batch_targets[i].cpu().numpy()
                    pred_vals = outputs[i].cpu().numpy()
                    true_ranks = rank_of_elements(true_vals)
                    pred_ranks = rank_of_elements(pred_vals)
                    eval_count += 1
                    difNumber += sum(
                        abs(true_ranks[j] - pred_ranks[j])
                        for j in range(len(pred_ranks))
                    )

            avg_diff = difNumber / eval_count
            avg_test_loss = total_test_loss / len(test_loader)
            print(f"All difNumber {difNumber}, "
                  f"Average difNumber {avg_diff:.4f}")
            print(f"Test Loss: {avg_test_loss:.4f}")

            # Perform final ranking on full input matrix for a quick sanity check
            TensorMatrixInput = torch.tensor(TestMatrixInput).to(device).float().view(1, -1, M)
            final_outputs = model(TensorMatrixInput)
            estranks = rank_of_elements(final_outputs[0].cpu().numpy()[:m2])
            print("Estimated Ranks on Full Matrix:", estranks)

        # Save model checkpoint if enabled
        if args.savingModel == 1:
            torch.save(model, f"{modelSavePath}.pt")
            print(f"Model saved to {modelSavePath}.pt")
