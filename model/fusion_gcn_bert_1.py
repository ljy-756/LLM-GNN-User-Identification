# model\fusion_gcn_bert_1.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class FusionGCN1(nn.Module):
    def __init__(self, bert_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        # 两层 GCN
        self.gcn1 = GCNConv(bert_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        # 投影到 BERT 维度
        self.gnn_proj = nn.Linear(hidden_dim, bert_dim)
        self.dropout = nn.Dropout(dropout)
        # 输出层
        self.fc = nn.Linear(bert_dim, num_classes)
        # α 初始化（后续训练脚本会覆盖）
        self.alpha = nn.Parameter(torch.tensor(0.5))
        

    def forward(self, data):
        x = data.x  # BERT embedding
        edge_index = data.edge_index

        # GCN 分支
        gnn_h = torch.relu(self.gcn1(x, edge_index))
        gnn_h = torch.relu(self.gcn2(gnn_h, edge_index))
        gnn_h = self.gnn_proj(gnn_h)

        # 加权融合
        alpha = torch.sigmoid(self.alpha)  # 保证在0~1之间
        fused = alpha * x + (1 - alpha) * gnn_h
        fused = self.dropout(fused)

        # 输出预测
        out = self.fc(fused)
        return out