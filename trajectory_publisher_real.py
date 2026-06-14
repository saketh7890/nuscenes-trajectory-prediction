import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import torch
import pickle
import numpy as np
import sys
sys.path.append('/nuscenes')
from transformer_model import TrajectoryTransformer

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher_real')
        self.publisher = self.create_publisher(String, 'trajectory', 10)
        self.timer = self.create_timer(0.01, self.publish_trajectory)
        self.index = 0

        with open('/nuscenes/data/val_samples.pkl', 'rb') as f:
            self.samples = pickle.load(f)

        with open('/nuscenes/data/stats.pkl', 'rb') as f:
            stats = pickle.load(f)
        self.mean = stats['mean']
        self.std  = stats['std']

        self.model = TrajectoryTransformer()
        self.model.load_state_dict(
            torch.load('/nuscenes/data/transformer_model_best.pth',
                      map_location='cpu')
        )
        self.model.eval()
        self.get_logger().info(f"Loaded {len(self.samples)} val samples")

    def publish_trajectory(self):
        if self.index >= len(self.samples):
            self.get_logger().info("Done! Published all samples")
            rclpy.shutdown()
            return

        sample = self.samples[self.index]
        past = torch.FloatTensor(sample['past']).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(past)

        past_m   = sample['past']               * self.std + self.mean
        future_m = sample['future']             * self.std + self.mean
        pred_m   = prediction.squeeze().numpy() * self.std + self.mean

        diff = pred_m - future_m
        ade  = np.sqrt((diff ** 2).sum(axis=-1)).mean()

        data = {
            'index':         self.index,
            'category':      sample['category'],
            'past':          past_m.tolist(),
            'actual_future': future_m.tolist(),
            'predicted':     pred_m.tolist(),
            'ade':           float(ade),
            'total':         len(self.samples)
        }

        msg = String()
        msg.data = json.dumps(data)
        self.publisher.publish(msg)

        if self.index % 1000 == 0:
            self.get_logger().info(
                f"Published {self.index}/{len(self.samples)} | ADE: {ade:.2f}m"
            )
        self.index += 1

def main():
    rclpy.init()
    node = TrajectoryPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
