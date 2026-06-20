# nuScenes Trajectory Prediction

LSTM and Transformer-based trajectory prediction pipeline trained on the full nuScenes autonomous driving dataset, deployed as a real-time ROS2 publisher-subscriber system.

## Results

| Model | Dataset | minADE@5 | Val Samples |
|-------|---------|----------|-------------|
| LSTM | Mini (10 scenes) | 2.26m | 1,767 |
| Transformer (no map) | Full (850 scenes metadata) | 1.56m | 107,923 |
| Transformer + top-3-lane map + K=5 + cosine LR | Full | **1.01m** | 107,915 |

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

**Transformer + Map Features + K=5 (Phase 3):**
- HD map integration via nuScenes Map API: top-3 nearest lane centerlines per agent, ego-relative, normalized, interpolated to 10 points each (30 map tokens total)
- K=5 multimodal trajectory generation via winner-takes-all (WTA) loss
- Positional encoding: nn.Embedding(34, 256) — 4 past frames + 30 map tokens
- Cosine annealing LR schedule (lr=0.00005 → 1e-6 over 100 epochs)
- GPU-accelerated training (PyTorch MPS on Apple Silicon)
- Result: **1.01m minADE@5** on 107,915 validation samples (35% improvement over map-agnostic baseline)

## ROS2 Deployment

Deployed Transformer as a real-time ROS2 pipeline:
- Publisher node: loads nuScenes val samples, runs Transformer, publishes predictions over /trajectory topic at 100Hz
- Subscriber node: receives predictions, calculates running minADE across all validation samples

## Dataset

- Full nuScenes dataset (850 scenes metadata)
- 33,319 trajectories extracted
- 26,655 train trajectories / 6,664 val trajectories
- 404,128 training samples / 107,923 validation samples
- Categories: cars, pedestrians, trucks, motorcycles, bicycles, buses

## Pipeline
## Setup

```bash
git clone https://github.com/saketh7890/nuscenes-trajectory-prediction
cd nuscenes-trajectory-prediction
python3 -m venv venv
source venv/bin/activate
pip install torch nuscenes-devkit numpy matplotlib
```

Download nuScenes full dataset from nuscenes.org and place in `data/nuscenes/`.

## Run

```bash
python3 prepare_data.py
python3 build_samples.py
python3 train.py
python3 transformer_train.py
python3 evaluate.py
python3 transformer_evaluate.py
python3 visualize.py
```

## ROS2 Deployment

```bash
docker run -it -v ~/nuscenes-project:/nuscenes osrf/ros:humble-desktop
source /opt/ros/humble/setup.bash
python3 trajectory_publisher_real.py
python3 trajectory_subscriber_real.py
```

## Files

| File | Description |
|------|-------------|
| explore.py | Dataset exploration |
| prepare_data.py | Trajectory + map feature extraction |
| map_features.py | Top-3-lane HD map feature extraction |
| build_samples.py | Dataset construction |
| model.py | LSTM architecture |
| transformer_model.py | Transformer architecture (K=5, map-aware) |
| train.py | LSTM training |
| transformer_train.py | Transformer training (K=5, cosine LR, GPU) |
| evaluate.py | LSTM evaluation |
| transformer_evaluate.py | Transformer evaluation (minADE@5) |
| visualize.py | Prediction visualization |
| trajectory_publisher_real.py | ROS2 publisher node |
| trajectory_subscriber_real.py | ROS2 subscriber node |

## Next Steps

- Extract remaining nuScenes blobs for larger training set
- FDE (final displacement error) loss term
- Social agent interaction modeling
- Leaderboard submission
