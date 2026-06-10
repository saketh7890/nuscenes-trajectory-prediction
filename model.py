import torch
import torch.nn as nn

class TrajectoryPredictor(nn.Module):
    
    def __init__(self, input_size=2, hidden_size=128, output_size=2, 
                 past_frames=4, future_frames=6):
        super(TrajectoryPredictor, self).__init__()
        
        self.past_frames = past_frames
        self.future_frames = future_frames
        
        # Encoder — reads past trajectory
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.dropout=nn.Dropout(0.2)
        
        # Decoder — predicts future trajectory
        self.decoder = nn.Linear(hidden_size, future_frames * output_size)
    
    def forward(self, x):
        # x shape: (batch, past_frames, 2)
        _, (hidden, _) = self.encoder(x)
        
        # Use last hidden state
        features = hidden[-1]
        
        # Predict future
        output = self.decoder(features)
        
        # Reshape to (batch, future_frames, 2)
        output = output.view(-1, self.future_frames, 2)
        
        return output


# Test the model
if __name__ == '__main__':
    model = TrajectoryPredictor()
    print(model)
    
    # Test with fake data
    fake_input = torch.randn(32, 4, 2)
    output = model(fake_input)
    print(f"\nInput shape:  {fake_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model works!")