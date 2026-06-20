import numpy as np
from nuscenes.map_expansion import arcline_path_utils

def get_map_features(agent_x, agent_y, nusc_map, num_lanes=3, num_points=10):
    all_lane_features = []
    radius = 10
    lane_records = nusc_map.get_records_in_radius(
        agent_x, agent_y, radius, ['lane', 'lane_connector']
    )
    lane_tokens = (lane_records['lane'] + lane_records['lane_connector'])

    if len(lane_tokens) == 0:
        return np.zeros((num_lanes * num_points, 2))

    distances = []
    for token in lane_tokens:
        arcline = nusc_map.get_arcline_path(token)
        poses = arcline_path_utils.discretize_lane(arcline, resolution_meters=1)
        poses = np.array(poses)[:, :2]
        dist = np.sqrt((poses[:, 0] - agent_x)**2 + (poses[:, 1] - agent_y)**2).min()
        distances.append((dist, token))

    distances.sort(key=lambda x: x[0])
    top_lanes = distances[:num_lanes]

    for _, token in top_lanes:
        arcline = nusc_map.get_arcline_path(token)
        poses = arcline_path_utils.discretize_lane(arcline, resolution_meters=1)
        poses = np.array(poses)[:, :2]

        dists = np.sqrt((poses[:, 0] - agent_x)**2 + (poses[:, 1] - agent_y)**2)
        nearby = poses[dists < 20]

        if len(nearby) == 0:
            nearby = poses[:1]

        nearby = nearby.copy()
        nearby[:, 0] -= agent_x
        nearby[:, 1] -= agent_y

        indices = np.linspace(0, len(nearby)-1, num_points).astype(int)
        sampled = nearby[indices]
        all_lane_features.append(sampled)

    while len(all_lane_features) < num_lanes:
        all_lane_features.append(np.zeros((num_points, 2)))

    return np.vstack(all_lane_features)