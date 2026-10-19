import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
from numpy.typing import NDArray

from transform_helpers.utils import rotmat2q

# Modified DH Params for the Franka FR3 robot arm
# https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
# meters
a_list = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
d_list = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]

# radians
alpha_list = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
theta_list = [0] * len(alpha_list)

DH_PARAMS = np.array([a_list, d_list, alpha_list, theta_list]).T

BASE_FRAME = "base"
FRAMES = ["fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "fr3_link8"]

def get_transform_n_to_n_minus_one(n: int, theta: float) -> NDArray:
    # this function calculates the transform to go from n to n-1 
    # using modified denavit hartenberg parameters

    transform_matrix = np.zeros((4,4))

    n_minus_one = n - 1

    a = DH_PARAMS[n_minus_one, 0]
    d = DH_PARAMS[n_minus_one,1]
    alpha = DH_PARAMS[n_minus_one,2]

    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    # first row
    transform_matrix[0, 0] = ct
    transform_matrix[0, 1] = -st
    transform_matrix[0, 2] = 0
    transform_matrix[0, 3] = a

    # second row
    transform_matrix[1, 0] = st*ca
    transform_matrix[1, 1] = ct*ca
    transform_matrix[1, 2] = -sa
    transform_matrix[1, 3] = -d*sa

    # third row
    transform_matrix[2, 0] = st*sa
    transform_matrix[2, 1] = ct*sa
    transform_matrix[2, 2] = ca
    transform_matrix[2, 3] = d*ca

    # forth row
    transform_matrix[3, 0] = 0
    transform_matrix[3, 1] = 0
    transform_matrix[3, 2] = 0
    transform_matrix[3, 3] = 1

    return transform_matrix

class ForwardKinematicCalculator(Node):

    def __init__(self):
        super().__init__('fk_calculator')

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.publish_transforms,
            10
        )
        self.joint_sub  # prevent unused variable warning

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # self.prefix = ""
        self.prefix = "my_robot/"

    def publish_transforms(self, msg: JointState):
        self.get_logger().debug(str(msg))

        # note our frames list is longer than the number of joints, so some special handling is required
        for i in range(len(FRAMES) - 1, -1, -1):
            frame_id = self.prefix + FRAMES[i]
            if i != 0:
                parent_id = self.prefix + FRAMES[i - 1]
            else:
                parent_id = self.prefix + BASE_FRAME
            theta = None
            if i != len(FRAMES) - 1 and i != 0:
                # joint msg has 7 entries, not base or static flange
                # 'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
                theta = msg.position[i - 1]
            elif i == len(FRAMES) - 1:
                # flange joint with the static transform and theta of zero
                theta = 0
            else:
                theta = 0

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent_id
            t.child_frame_id = frame_id

            if i != 0:
                transform = get_transform_n_to_n_minus_one(i, theta)
            else:
                transform = np.eye(4)

            quat = rotmat2q(transform[:3, :3])

            t.transform.translation.x = transform[0,3]
            t.transform.translation.y = transform[1,3]
            t.transform.translation.z = transform[2,3]

            t.transform.rotation.x = quat.x
            t.transform.rotation.y = quat.y
            t.transform.rotation.z = quat.z
            t.transform.rotation.w = quat.w

            self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    fk_calculator = ForwardKinematicCalculator()
    rclpy.spin(fk_calculator)
    fk_calculator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()