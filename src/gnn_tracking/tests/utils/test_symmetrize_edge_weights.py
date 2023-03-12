from gsoc.symmetrize_edge_weights import symmetrize_edge_weights
from torch import Tensor as T
from torch.testing import assert_close

def test_symmetrize_edge_weights_1():
    edge_index = T([[1, 2], [2, 1]])
    edge_weights = T([1, 3])

    # Call the function
    symmetrized_weights = symmetrize_edge_weights(edge_index.T, edge_weights)
    # Check the output
    assert_close(symmetrized_weights, T([2, 2]))

def test_symmetrize_edge_weights_2():
    edge_index = T([[1, 2], [3, 4], [2, 1]])
    edge_weights = T([1, 2, 3])

    # Call the function
    symmetrized_weights = symmetrize_edge_weights(edge_index.T, edge_weights)
    # Check the output
    assert_close(symmetrized_weights, T([2, 2, 2]))



def test_symmetrize_edge_weights_3():
    edge_index = T([[1, 2], [3, 4], [2, 1], [4, 3], [1, 3], [2, 4], [4, 2]])
    edge_weights = T([1, 2, 3, 4, 9, 1, 7])

    # Call the function
    symmetrized_weights = symmetrize_edge_weights(edge_index.T, edge_weights)
    # Check the output
    assert_close(symmetrized_weights, T([2., 3., 2., 3., 9., 4., 4.]))

