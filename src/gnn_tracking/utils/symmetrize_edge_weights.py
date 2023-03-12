import torch
from utils.hash_tensors import hash_tensors

def symmetrize_edge_weights(edge_index: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
    """Symmetrize edge weights by averaging the weights of the symmetric edges.

    Args:
        edge_index (Tensor): Edge indices (2, N)
        edge_weights (Tensor): Edge weights (N)

    Returns:
        Tensor: Symmetrized edge weights (N)
    """

    # Compute the edge hash using hash_tensors
    edge_hash = hash_tensors(edge_index[0, :], edge_index[1, :])

    # Find unique elements in edge_hash and their corresponding indices using torch.unique
    unique, indices = torch.unique(edge_hash, return_inverse=True)

    # Calculate the sum of weights and count of occurrences for each unique element using torch.bincount
    sum_weights = torch.bincount(
        indices, weights=edge_weights, minlength=len(unique))
    count = torch.bincount(indices, minlength=len(unique))

    # Calculate the average weight for each unique element using torch.where
    avg_weight = torch.where(count == 1, sum_weights, sum_weights / count)

    # Assign the average weights to the output tensor using indexing with the indices tensor
    return avg_weight[indices]
