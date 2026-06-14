import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_size=2, d_model=256,
                 past_frames=4, future_frames=6, num_heads=8):
        super().__init__()

        self.past_frames = past_frames
        self.future_frames = future_frames

        self.input_embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Embedding(past_frames, d_model)

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

        self.decoder = nn.Linear(d_model, future_frames * 2)

    def forward(self, x):
        x = self.input_embedding(x)

        positions = torch.arange(self.past_frames, device=x.device)
        pos_enc = self.positional_encoding(positions)
        x = x + pos_enc

        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.decoder(x)
        x = x.view(-1, self.future_frames, 2)
        return x


if __name__ == '__main__':
    model = TrajectoryTransformer()
    print(model)
    fake_input = torch.randn(32, 4, 2)
    output = model(fake_input)
    print(f"Input:  {fake_input.shape}")
    print(f"Output: {output.shape}")
    print("Model works!")