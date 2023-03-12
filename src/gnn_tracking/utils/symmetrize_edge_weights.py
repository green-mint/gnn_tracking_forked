import torch
from torch import Tensor as T

def __hash_tensors(a: T, b: T) -> T:
    """
    Hashes two 1D tensors of same size, N, into a single 1D tensor of size N.
    Such that hash(a[i], b[j]) == hash(b[j], a[i])

    Args:
        a (T): 1D tensor of size N
        b (T): 1D tensor of size N

    Raises:
        AssertionError: If both the tensors are not 1D or of different size

    Returns:
        T: hashed 1D tensor of size N
    
    Example:

    >>> hash_tensors(torch.tensor([1, 4, 3, 6]), torch.tensor([4, 1, 6, 3]))
    tensor([16, 16, 48, 48])
    """    
    # check if both the tensors are of 1D and of same size
    try:
        assert a.dim() == 1 and b.dim() == 1
        assert a.size() == b.size()
    except AssertionError:
        raise AssertionError("Both the tensors must be 1D and of same size")

    s = a + b
    return torch.floor_divide(s * (s + 1), 2) + torch.minimum(a, b)

def symmetrize_edge_weights(edge_index: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
    """Symmetrize edge weights by averaging the weights of the symmetric edges.

    Args:
        edge_index (Tensor): Edge indices (2, N)
        edge_weights (Tensor): Edge weights (N)

    Returns:
        Tensor: Symmetrized edge weights (N)
    """

    # Compute the edge hash using hash_tensors
    edge_hash = __hash_tensors(edge_index[0, :], edge_index[1, :])

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
