import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import numpy as np

class TrajectorySubscriber(Node):
    def __init__(self):
        super().__init__('trajectory_subscriber_real')
        self.subscription = self.create_subscription(
            String, 'trajectory', self.callback, 10)
        self.ade_list = []
        self.get_logger().info("Subscriber ready — waiting...")

    def callback(self, msg):
        data = json.loads(msg.data)
        self.ade_list.append(data['ade'])
        mean_ade = np.mean(self.ade_list)

        if len(self.ade_list) % 1000 == 0:
            self.get_logger().info(
                f"=== {len(self.ade_list)}/{data['total']} | "
                f"Running minADE: {mean_ade:.4f}m ==="
            )

        if len(self.ade_list) >= data['total']:
            self.get_logger().info(
                f"\nFINAL minADE: {mean_ade:.4f} meters"
                f"\nTotal samples: {len(self.ade_list)}"
            )
            rclpy.shutdown()

def main():
    rclpy.init()
    node = TrajectorySubscriber()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
