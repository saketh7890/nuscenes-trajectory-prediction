# nuScenes Trajectory Prediction

LSTM and Transformer-based trajectory prediction pipeline trained on the full nuScenes autonomous driving dataset, deployed as a real-time ROS2 publisher-subscriber system.

## Results

| Model | Dataset | minADE | Val Samples |
|-------|---------|--------|-------------|
| LSTM | Mini (10 scenes) | 2.26m | 1,767 |
| Transformer | Full (850 scenes metadata) | 1.56m | 107,923 |

![Predictions](predictions.png)

## Architecture

**LSTM Baseline:**
- 2-layer LSTM encoder (hidden=64)
- Linear decoder → 6 future frames
- Input: ego-relative (x,y) coordinates, normalized

**Transformer (Phase 2):**
- Input embedding: Linear(2, 256)
- Positional encoding: nn.Embedding(4, 256)
- Transformer encoder: 2 layers, 8 heads, d_model=256
- Linear decoder → 6 future frames
- Training: 50 epochs, Adam lr=0.0001, best model saving

## ROS2 Deployment

Deployed Transformer as a real-time ROS2 pipeline:
- Publisher node: loads nuScenes val samples, runs Transformer, publishes predictions over /trajectory topic at 100Hz
- Subscriber node: receives predictions, calculates running minADE across all 107,923 validation samples

## Dataset

- Full nuScenes dataset (850 scenes metadata)
- 33,319 trajectories extracted
- 26,655 train trajectories / 6,664 val trajectories
- 404,128 training samples / 107,923 validation samples
- Categories: cars, pedestrians, trucks, motorcycles, bicycles, buses

## Pipeline