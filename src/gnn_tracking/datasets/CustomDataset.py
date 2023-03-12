import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_files):
        super().__init__()
        self.graphs = data_files
    
    def __getitem__(self, idx):
        return torch.load(self.graphs[idx])
    
    def __len__(self) -> int:
        return len(self.graphs)