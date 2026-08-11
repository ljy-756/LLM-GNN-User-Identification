# model\fusion_gcn_bert_2.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    def __init__(self, text_dim, graph_dim, hidden_dim, num_classes=2, dropout=0.3):
        super(FusionModel, self).__init__()

        # === 1. 投影层（统一维度）===
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)

        # === 2. Attention层（核心）===
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # === 3. 分类器 ===
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_text, h_graph, return_alpha=False):
        """
        h_text:  [batch, text_dim]
        h_graph: [batch, graph_dim]
        """

        # === Step 1：对齐维度 ===
        h_text = self.text_proj(h_text)
        h_graph = self.graph_proj(h_graph)

        h_text = self.dropout(h_text)
        h_graph = self.dropout(h_graph)

        # === Step 2：拼接 ===
        h_concat = torch.cat([h_text, h_graph], dim=-1)

        # === Step 3：计算 attention 权重 ===
        alpha = self.attention(h_concat)   # [batch, 1]

        # === Step 4：融合 ===
        h = alpha * h_text + (1 - alpha) * h_graph

        # === Step 5：分类 ===
        logits = self.classifier(h)

        if return_alpha:
            return logits, alpha
        return logits