import pickle
import numpy as np

with open('data/trajectories.pkl', 'rb') as f:
    all_trajectories = pickle.load(f)

with open('data/stats.pkl', 'rb') as f:
    stats = pickle.load(f)

mean = stats['mean']
std  = stats['std']

PAST_FRAMES   = 4
FUTURE_FRAMES = 6

# Split trajectories FIRST — no leakage
split = int(0.8 * len(all_trajectories))
train_trajs = all_trajectories[:split]
val_trajs   = all_trajectories[split:]

def build_samples(trajectories):
    samples = []
    for traj in trajectories:
        positions = traj['positions']
        category  = traj['category']

        # Normalize
        norm = [((x - mean[0]) / std[0],
                 (y - mean[1]) / std[1])
                for x, y in positions]

        for i in range(len(norm) - PAST_FRAMES - FUTURE_FRAMES + 1):
            past   = norm[i : i + PAST_FRAMES]
            future = norm[i + PAST_FRAMES : i + PAST_FRAMES + FUTURE_FRAMES]
            samples.append({
                'past':     np.array(past),
                'future':   np.array(future),
                'category': category
            })
    return samples

train_samples = build_samples(train_trajs)
val_samples   = build_samples(val_trajs)

print(f"Train trajectories: {len(train_trajs)}")
print(f"Val   trajectories: {len(val_trajs)}")
print(f"Train samples: {len(train_samples)}")
print(f"Val   samples: {len(val_samples)}")

with open('data/train_samples.pkl', 'wb') as f:
    pickle.dump(train_samples, f)

with open('data/val_samples.pkl', 'wb') as f:
    pickle.dump(val_samples, f)

print("Saved train_samples.pkl and val_samples.pkl")