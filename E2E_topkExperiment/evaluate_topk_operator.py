# evaluate_topk_operator.py

import numpy as np
import torch
import random
import os
import csv
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

# Function to generate synthetic training data for the top-k operator
# M: number of items per sequence
# T: number of sequences to generate
# ModelPrecisionList: list of probabilities that each model predicts correctly
# EmptyRatioList: list of probabilities that each model skips a comparison (outputs zero)
# Returns:
#   - Matrixs: tensor of shape (T, numModels*M, M) representing evidence matrices flattened
#   - ranks: tensor of shape (T, M) of normalized ground-truth ranks

def genTrainingData(M, T, ModelPrecisionList, EmptyRatioList):
    sequences = []
    Matrixs = []
    ranks = []

    # Generate T random sequences and their ground-truth ranks
    for i in range(T):
        vals = random_permutation(M)
        sequences.append(vals)
        ranks.append(rank_of_elements(vals))

    # For each sequence, build a 3D evidence matrix per model
    for i in range(T):
        if i % 100 == 1:
            print("genTrainingData Processing:", i / T * 100, "%")

        numModels = len(ModelPrecisionList)
        AllModels = np.zeros((numModels, M, M))

        # For each model, fill its pairwise comparison matrix
        for m in range(numModels):
            correctProb = ModelPrecisionList[m]
            skipProb = EmptyRatioList[m]
            for j in range(M):
                for k in range(M):
                    v1 = sequences[i][j]
                    v2 = sequences[i][k]

                    # Model may skip this comparison
                    if random.random() < skipProb:
                        AllModels[m, j, k] = 0
                        continue

                    # With probability correctProb, model predicts correctly
                    if random.random() < correctProb:
                        if v1 > v2:
                            AllModels[m, j, k] = -1
                        elif v1 < v2:
                            AllModels[m, j, k] = 1
                        else:
                            AllModels[m, j, k] = 0
                    else:
                        # Incorrect prediction flips the sign
                        if v1 > v2:
                            AllModels[m, j, k] = 1
                        elif v1 < v2:
                            AllModels[m, j, k] = -1
                        else:
                            AllModels[m, j, k] = 0

        Matrixs.append(AllModels)

    # Convert lists to arrays/tensors and normalize ranks
    Matrixs = np.array(Matrixs)
    ranks = (np.array(ranks) + 0.0) / M

    # Print diagnostic shapes
    print(Matrixs.shape)
    print(ranks.shape)

    # Flatten and convert to PyTorch tensors
    Matrixs = Matrixs.reshape(Matrixs.shape[0], -1, Matrixs.shape[-1])
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

if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--M',  type=int,default=16)
    parser.add_argument('--N', type=int,default=5000)
    parser.add_argument('--PrecisionList',nargs='+', type=float,default=[0.66,0.22,0.33,0.9,0.66,0.4,0.66,0.33,0.33])
    parser.add_argument('--GPU', type=int,default=0)
    parser.add_argument('--evidenceMatrix', type=str,default="evidenceMatrix_debug.npy")
    parser.add_argument('--realOrder', type=str,default="realOrder.npy")
    parser.add_argument('--seed', type=int,default=42)
    parser.add_argument('--lr', type=float,default=1e-5)
    parser.add_argument('--bs', type=int,default=32)
    parser.add_argument('--trainEpochs', type=int,default=1)
    parser.add_argument('--model_path', type=str,default="Visual_K9_M16_N5000_bs32_lr0.0001.pt")
    parser.add_argument('--result_path', type=str,default="evalResult.csv")
    args = parser.parse_args()
    args.N = 0
    args.trainEpochs = 1

    current_script_path = os.path.abspath(__file__)       
    project_root = os.path.dirname(os.path.dirname(current_script_path)) 

    # dataPath define
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root)
    TestMatrixInput = os.path.join(data_dir, args.evidenceMatrix)
    TestLabel = os.path.join(data_dir, args.realOrder) 
    ModelPath = os.path.join(models_dir, args.model_path)
    pathOfExpResult = os.path.join(results_dir, args.result_path)
    curLen =  np.load(TestLabel).shape[0]


    print("="*20)

    # print args
    print(f"M: {args.M}")
    print(f"N: {args.N}")
    print(f"PrecisionList: {args.PrecisionList}")
    print(f"TestMatrixInput: {TestMatrixInput}")
    print(f"TestRank: {TestLabel }")
    print(f"seed: {args.seed}")
    print('pathOfExpResult',pathOfExpResult,flush=True)
    setup_seed(args.seed)

    print("="*20)
    csv_filename = pathOfExpResult
    k_values = [i +1 for i in range(curLen)]

    # write header to csv
    header = ["epoch", "TrainLoss", "TestLoss"]
    for k in k_values:
        header.append(f"Recall@{k}")
    for k in k_values:
        header.append(f"ACC@{k}")
    header += ["difNum", "Spearman"]

    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    M = args.M
    N = args.N
    ModelPrecisionList = args.PrecisionList

    print(f"Using device: {device}",flush=True)
    TestMatrixInput  =  np.load(TestMatrixInput)
    TestLabel  =  np.load(TestLabel )
    TestRank = rank_of_elements(TestLabel)

    m2 = TestMatrixInput.shape[2]
    EmptyRatioList = [ float(1- (sum(sum(abs(TestMatrixInput[i]))))/((m2)*(m2-1) )) for i  in range(len(args.PrecisionList)) ]

    print("Empty Ratio:",EmptyRatioList)
    # -------------------- Evaluation Branching --------------------
    # There are three scenarios based on input matrix size m2 vs. args.M:
    # 1) m2 < args.M: Augment smaller matrices by padding default comparisons
    # 2) m2 > args.M: Perform iterative K-way merge join using the model
    # 3) m2 == args.M: Direct single-batch model evaluation
    if m2 < args.M:
        print("Augmenting the testing matrix")
        K = TestMatrixInput.shape[0]
        augMatrix = np.zeros((K,M,M))
        # for i in range(M-m2):
        #     TestRank.append(max(TestRank)+1)
        print(f"OKay1",K,M,M,K*M*M,flush=True)
        for i in range(K):
            for j in range(M):
                for k in range(M):
                    if j < m2 and k < m2:
                        augMatrix[i,j,k] = TestMatrixInput[i,j,k]
                    else:
                        if j==k:
                            augMatrix[i,j,k] = 0
                        elif j> k:
                            augMatrix[i,j,k] = -1
                        else:
                            augMatrix[i,j,k] = 1
        # print(augMatrix)
        TestMatrixInput=augMatrix
    if m2 > args.M:
        print("Start to K-way Merge Join")
        model = torch.load(
            ModelPath,
            map_location=lambda storage, loc: storage.cuda(0)
        )
        model.to(device)
        # Assume M2=t*M
        print(m2,args.M)
        t =int( (m2+0.0)/args.M)
        print(t)
        mergeSubPart = {}
        for i in range(t):
            mergeSubPart[i] = []
        for i in range(m2):
            mergeSubPart[i % t].append(i)
        for k in mergeSubPart.keys():
            Idx = mergeSubPart[k]
            selectedMtr = []
            for i in range(TestMatrixInput.shape[0]):
                selectedMtr.append(TestMatrixInput[i][np.ix_(Idx, Idx)])
            selectedMtr = np.array(selectedMtr)
            TensorMatrixInput = torch.tensor(selectedMtr).to(device).float().view(-1,M,M)
            outputs = model(TensorMatrixInput)
            estranks = rank_of_elements(-outputs[0].cpu().detach().numpy())
            sorted_pairs = sorted(zip(Idx, estranks), key=lambda x: x[1])
            sorted_list1 = [item for item, _ in sorted_pairs]
            mergeSubPart[k] = np.array(sorted_list1)

        outputsL = []
        for i in range(m2):
            curTop = []
            for k in mergeSubPart.keys():
                if len(mergeSubPart[k])<1:
                    continue
                curTop.append([mergeSubPart[k][0],k])
            while len(curTop)< M:
                for k in mergeSubPart.keys():
                    if len(mergeSubPart[k])>=1:
                        for j in range(len(mergeSubPart[k])):
                            curTop.append([mergeSubPart[k][j],k])
                            if len(curTop)>=M:
                                break
                        if len(curTop)>=M:
                                break

            # sorting
            Idx = [value for value,k  in curTop]
            ks = [k for value,k  in curTop]
            selectedMtr = []
            for i in range(TestMatrixInput.shape[0]):
                selectedMtr.append(TestMatrixInput[i][np.ix_(Idx, Idx)])
            selectedMtr = np.array(selectedMtr)
            TensorMatrixInput = torch.tensor(selectedMtr).to(device).float().view(-1,M,M)
            outputs = model(TensorMatrixInput)
            estranks = rank_of_elements(-outputs[0].cpu().detach().numpy())


            for ien in range(len(estranks)):
                if estranks[ien] == 1:
                    value = Idx[ien]
                    kdl =  ks[ien]
                    break

            lenSum = 0
            for kiii in mergeSubPart.keys():
                lenSum+=len(mergeSubPart[kiii])
            if lenSum == M:
                for v in np.array(Idx)[np.array(estranks)-1]:
                    outputsL.append(v)
                break
            outputsL.append(value)
            mergeSubPart[kdl]= mergeSubPart[kdl][mergeSubPart[kdl]!=value]

        estranks = outputsL

        print("Real Rank:",TestRank)
        print("Est Rank:",estranks)
        difN = 0
        for j in range(len(estranks)):
            if TestRank[j] != estranks[j]:
                difN+=abs(TestRank[j] - estranks[j])
        # Compute Recall and ACCk and Spearman
        compute_metrices(estranks,TestRank,k_values)
        print("Diff Num:",difN)
        recalls, accs = [], []
        for k in k_values:
            recall, acc = compute_recall_acc(TestRank, estranks, k)
            recalls.append(recall)
            accs.append(acc)
        spearman = compute_spearman(TestRank, estranks)
        row = [0, 0,0]
        row += recalls + accs + [difN, spearman]
        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("Done Written into:",csv_filename)

    else:

        print("Generating Training Data")
        print("Done Training Data")

        #initialize model

        model = torch.load(
            ModelPath,
            map_location=lambda storage, loc: storage.cuda(0)
        )
        model.to(device)
        TensorMatrixInput = torch.tensor(TestMatrixInput).to(device).float().view(1,-1,M)

        outputs = model(TensorMatrixInput)
        estranks = rank_of_elements(outputs[0].cpu().detach().numpy()[:m2])
        print("Real Rank:",TestRank)
        print("Est Rank:",estranks)
        difN = 0
        for j in range(len(estranks)):
            if TestRank[j] != estranks[j]:
                difN+=abs(TestRank[j] - estranks[j])
        # Compute Recall and ACCk and Spearman
        compute_metrices(estranks,TestRank,k_values)
        print("Diff Num:",difN)
        recalls, accs = [], []
        for k in k_values:
            recall, acc = compute_recall_acc(TestRank, estranks, k)
            recalls.append(recall)
            accs.append(acc)
        spearman = compute_spearman(TestRank, estranks)
        # write into csv
        row = [0, 0,0]
        row += recalls + accs + [difN, spearman]
        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("Done Written into:",csv_filename)
