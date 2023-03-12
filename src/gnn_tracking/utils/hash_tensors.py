import torch
from torch import Tensor as T

def hash_tensors(a: T, b: T) -> T:
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