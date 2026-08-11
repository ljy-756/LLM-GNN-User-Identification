# model\fusion_gcn_bert.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FusionGCN(torch.nn.Module):


    def __init__(self, bert_dim=768, hidden_dim=64, num_classes=2, dropout=0.5):
        super(FusionGCN, self).__init__()
        self.conv1 = GCNConv(bert_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
