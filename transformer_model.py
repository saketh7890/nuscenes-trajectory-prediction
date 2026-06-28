import torch
import torch.nn as nn
from gat_layer import GraphAttentionLayer

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_size=2, d_model=256,
                 past_frames=4, future_frames=6,
                 num_heads=8, map_points=30, num_neighbors=5, K=5):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.map_points = map_points
        self.num_neighbors = num_neighbors
        self.K = K

        self.input_embedding = nn.Linear(input_size, d_model)
        self.gat = GraphAttentionLayer(in_features=input_size, out_features=d_model)

        self.positional_encoding = nn.Embedding(
            past_frames + map_points + 1, d_model
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.decoder = nn.Linear(d_model, K * future_frames * 2)

    def forward(self, x, map_features, neighbor_features):
        batch_size = x.shape[0]

        x_emb = self.input_embedding(x)
        m_emb = self.input_embedding(map_features)

        neighbor_reshaped = neighbor_features.view(batch_size, self.num_neighbors, 4, 2)
        neighbor_summary = neighbor_reshaped[:, :, -1, :]

        target_state = x[:, -1, :]

        n_emb_single = self.gat(target_state, neighbor_summary)
        n_emb = n_emb_single.unsqueeze(1)

        combined = torch.cat([x_emb, m_emb, n_emb], dim=1)

        positions = torch.arange(
            self.past_frames + self.map_points + 1,
            device=x.device
        )
        pos_enc = self.positional_encoding(positions)
        combined = combined + pos_enc

        combined = self.transformer_encoder(combined)
        features = combined[:, self.past_frames - 1, :]
        output = self.decoder(features)
        output = output.view(-1, self.K, self.future_frames, 2)
        return output

if __name__ == '__main__':
    model = TrajectoryTransformer()
    print(model)
    past = torch.randn(32, 4, 2)
    map_feat = torch.randn(32, 30, 2)
    neighbor_feat = torch.randn(32, 20, 2)
    out = model(past, map_feat, neighbor_feat)
    print(f"Input:    {past.shape}")
    print(f"Map:      {map_feat.shape}")
    print(f"Neighbor: {neighbor_feat.shape}")
    print(f"Output:   {out.shape}")
    print("Model works!")