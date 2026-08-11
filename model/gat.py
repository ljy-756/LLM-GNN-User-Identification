#model\gat.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, heads=4):
        super(GAT, self).__init__()

        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=0.6
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            num_classes,
            heads=1,
            concat=False,
            dropout=0.6
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x = self.gat2(x, edge_index)

        return x