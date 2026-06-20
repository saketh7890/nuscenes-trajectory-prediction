import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_size=2, d_model=256,
                 past_frames=4, future_frames=6, 
                 num_heads=8, map_points=30,K=5):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.map_points = map_points
        self.K = K

        self.input_embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Embedding(past_frames + map_points, d_model)

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

    def forward(self, x, map_features):
        x_emb = self.input_embedding(x)
        m_emb = self.input_embedding(map_features)
        combined = torch.cat([x_emb, m_emb], dim=1)

        positions = torch.arange(self.past_frames + self.map_points, device=x.device)
        pos_enc = self.positional_encoding(positions)
        combined = combined + pos_enc

        combined = self.transformer_encoder(combined)
        features = combined[:, self.past_frames - 1, :]
        output = self.decoder(features)
        output = output.view(-1,self.K, self.future_frames, 2)
        return output

if __name__ == '__main__':
    model = TrajectoryTransformer()
    print(model)
    past = torch.randn(32, 4, 2)
    map_feat = torch.randn(32, 10, 2)
    out = model(past, map_feat)
    print(f"Input:  {past.shape}")
    print(f"Map:    {map_feat.shape}")
    print(f"Output: {out.shape}")
    print("Model works!")