import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    def __init__(self, text_dim, graph_dim, hidden_dim, num_classes=2, dropout=0.3):
        super(FusionModel, self).__init__()

        # ===== 1. projection =====
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)

        # ===== 2. individual classifiers =====
        self.text_classifier = nn.Linear(hidden_dim, num_classes)
        self.graph_classifier = nn.Linear(hidden_dim, num_classes)

        # ===== 3. confidence gate =====
        self.gate = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # ===== 4. final classifier =====
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_text, h_graph, return_alpha=False):

        # projection
        h_text = self.text_proj(h_text)
        h_graph = self.graph_proj(h_graph)

        h_text = self.dropout(h_text)
        h_graph = self.dropout(h_graph)

        # individual prediction
        logits_text = self.text_classifier(h_text)
        logits_graph = self.graph_classifier(h_graph)

        # confidence
        prob_text = F.softmax(logits_text, dim=-1)
        prob_graph = F.softmax(logits_graph, dim=-1)

        gate_input = torch.cat([prob_text, prob_graph], dim=-1)

        alpha = self.gate(gate_input)

        # fusion
        h = alpha * h_text + (1 - alpha) * h_graph

        # final
        logits = self.final_classifier(h)

        if return_alpha:
            return logits, alpha

        return logits