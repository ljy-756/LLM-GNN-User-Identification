import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    def __init__(self, text_dim, graph_dim, hidden_dim, num_classes=2, dropout=0.3):
        super(FusionModel, self).__init__()

        # ===== 1. projection =====
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)

        # ===== 2. disagreement-aware attention =====
        # 输入变成 3 * hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # ===== 3. classifier =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_text, h_graph, return_alpha=False):

        # Step1: projection
        h_text = self.text_proj(h_text)
        h_graph = self.graph_proj(h_graph)

        h_text = self.dropout(h_text)
        h_graph = self.dropout(h_graph)

        # Step2: disagreement feature
        diff = torch.abs(h_text - h_graph)

        # Step3: concat (关键区别)
        h_concat = torch.cat([h_text, h_graph, diff], dim=-1)

        # Step4: attention
        alpha = self.attention(h_concat)

        # Step5: fusion
        h = alpha * h_text + (1 - alpha) * h_graph

        # Step6: classifier
        logits = self.classifier(h)

        if return_alpha:
            return logits, alpha

        return logits