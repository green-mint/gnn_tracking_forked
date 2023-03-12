import torch
from ..datasets.CustomDataset import CustomDataset


def calc_class_weights(dataset: CustomDataset):
    """
    Calculates the class weights in a dataset to counter class imbalance

    Args:
        dataset (CustomDataset): A dataset object that must be a torch.utils.data.Dataset or inherit from it

    Returns:
        Tensor: A tensor of size C where C are the number of classes
    """    

    labels = []
    for data in dataset:
        labels += data.y.tolist()

    # Calculate class frequencies
    class_freqs = torch.bincount(torch.tensor(labels).to(torch.int32)).float()

    # Calculate class weights
    total_samples = class_freqs.sum()
    return total_samples / (class_freqs + 1e-6)
