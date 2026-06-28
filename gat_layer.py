import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

    def forward(self, target_feat, neighbor_feat):
        target_transformed = self.W(target_feat)
        neighbor_transformed = self.W(neighbor_feat)

        num_neighbors = neighbor_transformed.shape[1]
        target_expanded = target_transformed.unsqueeze(1).expand(-1, num_neighbors, -1)

        combined = torch.cat([target_expanded, neighbor_transformed], dim=-1)
        raw_scores = self.a(combined)

        raw_scores = F.leaky_relu(raw_scores)
        attention_weights = F.softmax(raw_scores, dim=1)

        weighted_sum = (attention_weights * neighbor_transformed).sum(dim=1)
        return weighted_sum