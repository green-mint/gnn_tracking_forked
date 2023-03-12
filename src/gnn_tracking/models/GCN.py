import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TransformerConv, PNAConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    """A noobs implementation of a GCN model for edge classification."""

    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        # TODO: Add edge features, TransformerConv & PNA can incorporate edge features as well.
        # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        # edge index is a 2xN tensor representing the edges edge_index[0,i] -> edge_index[1,i]
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
