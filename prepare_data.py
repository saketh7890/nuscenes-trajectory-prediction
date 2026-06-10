from nuscenes.nuscenes import NuScenes
from collections import Counter
import numpy as np
import pickle
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
def get_ego_pose(nusc,sample_token):
    sample=nusc.get('sample',sample_token)
    lidar_token=sample['data']['LIDAR_TOP']
    lidar_data=nusc.get('sample_data',lidar_token)
    ego_pose=nusc.get('ego_pose',lidar_data['ego_pose_token'])
    x=ego_pose['translation'][0]
    y=ego_pose['translation'][1]
    return x,y
INTERESTING_CATEGORIES = {
    'vehicle.car',
    'vehicle.truck',
    'vehicle.bus.rigid',
    'vehicle.bus.bendy',
    'vehicle.motorcycle',
    'vehicle.bicycle',
    'vehicle.trailer',
    'vehicle.construction',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
}

MIN_PAST_FRAMES = 4
MIN_FUTURE_FRAMES = 6
MIN_TOTAL_FRAMES = MIN_PAST_FRAMES + MIN_FUTURE_FRAMES

all_trajectories = []

for scene in nusc.scene:
    scene_name = scene['name']
    sample_token = scene['first_sample_token']
    instance_tokens_seen = set()

    while sample_token:
        sample = nusc.get('sample', sample_token)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            instance_tokens_seen.add(ann['instance_token'])
        sample_token = sample['next']

    for instance_token in instance_tokens_seen:
        instance = nusc.get('instance', instance_token)
        category = nusc.get('category', instance['category_token'])['name']

        if category not in INTERESTING_CATEGORIES:
            continue

        positions = []
        ann_token = instance['first_annotation_token']

        while ann_token:
            ann = nusc.get('sample_annotation', ann_token)
            x = ann['translation'][0]
            y = ann['translation'][1]
            sample_token=ann['sample_token']
            ego_x,ego_y=get_ego_pose(nusc,sample_token)
            rel_x=x-ego_x
            rel_y=y-ego_y
            positions.append((rel_x,rel_y))
            ann_token = ann['next']

        if len(positions) < MIN_TOTAL_FRAMES:
            continue

        all_trajectories.append({
            'scene': scene_name,
            'category': category,
            'positions': positions
        })

# Verify ego-relative conversion
sample_traj = all_trajectories[0]
print(f"\nSample trajectory: {sample_traj['category']}")
print(f"Number of frames: {len(sample_traj['positions'])}")
print(f"First 3 positions (ego-relative):")
for i, (x, y) in enumerate(sample_traj['positions'][:3]):
    print(f"  Frame {i}: ({x:.2f}, {y:.2f}) meters from ego car")
with open('data/trajectories.pkl','wb') as f:
    pickle.dump(all_trajectories,f)
print(f"\nSaved {len(all_trajectories)} trajectories to data/trajectories")
# Calculate mean and std for normalization
all_positions = []
for traj in all_trajectories:
    all_positions.extend(traj['positions'])

all_positions = np.array(all_positions)
mean = all_positions.mean(axis=0)
std = all_positions.std(axis=0)

print(f"\nMean: {mean}")
print(f"Std:  {std}")

# Save normalization stats
stats = {'mean': mean, 'std': std}
with open('data/stats.pkl', 'wb') as f:
    pickle.dump(stats, f)

print("Saved normalization stats")