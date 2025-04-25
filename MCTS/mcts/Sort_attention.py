import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from mcts.DataProcess import *

# Custom dataset class wrapping input-output pairs for training/testing
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Define the attention model using PyTorch's MultiheadAttention
class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)  # Linear layer for dimension adjustment

    def forward(self, x):
        # x shape: [N, M, M]
        attn_output, attn_weights = self.attention(x, x, x)  # Self-attention mechanism
        output = self.fc(attn_output)  # Pass attention output through a linear layer
        return output[:, :, 0], attn_weights  # Return selected output and attention weights

# Compute entropy of attention weights to quantify uncertainty
def compute_attention_entropy(attn_weights):
    # attn_weights shape: [N, num_heads, M, M]
    attn_avg = attn_weights.mean(dim=1)  # Average over heads → shape: [N, M, M]

    # Entropy for each row in attention matrix
    attn_entropy = -torch.sum(attn_avg * torch.log(attn_avg + 1e-9), dim=-1)  # Shape: [N, M]

    # Mean entropy per matrix (optional)
    avg_entropy = attn_entropy.mean(dim=-1)  # Shape: [N]

    return attn_entropy, avg_entropy

# Placeholder function for computing rank difference (not implemented)
def evalDifference(label1, label2):
    rank1 = rank_of_elements(label1)
    rank2 = rank_of_elements(label2)

# Symmetrically normalize elements a_ij and a_ji in an [N, M, M] tensor
def normalize_symmetric_elements(tensor):
    """
    Normalize symmetric off-diagonal elements in each MxM matrix in a [N, M, M] tensor.
    For a pair a_ij and a_ji (i≠j), normalize as: a_ij / (a_ij + a_ji), and vice versa.

    Args:
        tensor (torch.Tensor): Tensor of shape [N, M, M]

    Returns:
        normalized_tensor (torch.Tensor): Normalized tensor
    """
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be 3-dimensional (N x M x M)")
    N, M, _ = tensor.shape

    # Mask for non-diagonal symmetric elements
    mask = torch.ones((M, M), dtype=bool)
    mask = mask ^ torch.eye(M, dtype=bool)
    mask_upper = torch.triu(mask, diagonal=1)

    normalized_tensor = tensor.clone()

    # Normalize each matrix individually
    for i in range(N):
        mat = normalized_tensor[i]
        upper_indices = torch.where(mask_upper)
        for row, col in zip(upper_indices[0], upper_indices[1]):
            a_ij = mat[row, col]
            a_ji = mat[col, row]
            total = a_ij + a_ji
            if total == 0:
                continue
            mat[row, col] = a_ij / total
            mat[col, row] = a_ji / total
        normalized_tensor[i] = mat

    return normalized_tensor

if __name__ == "__main__":
    # Select device: use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("In")
    M = 50  # Size of each vote matrix (MxM)
    N = 5000  # Number of samples
    inputs, targets = genTrainingData(M, N)  # Generate synthetic training data
    print(inputs[0])
    randomDelete_N(inputs, N, max_k=0.6)  # Randomly delete up to 60% of votes in each matrix
    inputs = swap_votes(inputs, k=0.1)  # Introduce noise by randomly swapping 10% of votes

    # Wrap the data in a custom PyTorch dataset
    dataset = CustomDataset(inputs, targets)
    train_size = int(0.7 * len(dataset))  # 70% for training
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = AttentionModel(embed_dim=M, num_heads=2).to(device)
    criterion = nn.MSELoss()  # Use mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    num_epochs = 80
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs, attn_weights = model(inputs)
            attn_entropy, avg_entropy = compute_attention_entropy(attn_weights)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Attention_entropy: {float(avg_entropy):.4f}")

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            total_loss = 0
            difNumber = 0
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs, attn_weights = model(inputs)
                attn_entropy, avg_entropy = compute_attention_entropy(attn_weights)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                # Compare predicted vs. target rankings
                r, c = outputs.shape  # r = batch size, c = M
                for i in range(r):
                    v1 = targets[i].cpu().numpy()
                    v2 = outputs[i].cpu().numpy()
                    r1 = rank_of_elements(v1)
                    r2 = rank_of_elements(v2)
                    for j in range(c):
                        if r1[j] != r2[j]:
                            difNumber += 1

            print(f"difNumber:{difNumber}, accuracy:{1 - difNumber/(test_size * (M*M - M))}")

            avg_loss = total_loss / len(test_loader)
            print(f"Test Loss: {avg_loss:.4f}, Attention_entropy: {float(avg_entropy):.4f}")

        # Save model checkpoint every 40 epochs
        if (epoch + 1) % 40 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
            print(f"Model for epoch {epoch + 1} saved.")

    # Save final model
    torch.save(model.state_dict(), 'final_model_M=50_k1=0.6_k2=0.1.pth')
    print("Final model saved.")
