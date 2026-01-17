import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

import numpy as np
import mujoco
# import mediapy as media
import mujoco.viewer

from src.quat_helper import quat_to_rpy, rpy_to_quat, quaternion_error
from src.config_loader import load_robot_config


class MuJoCoSimNode(Node):
    def __init__(self, model, data):
        super().__init__('panda_sim_server')
        self.model = model
        self.data = data
        self.bridge = CvBridge()
        
        # 1. Feed: Publish where the robot is
        self.joint_pub = self.create_publisher(JointState, '/panda/joint_states', 10)
        self.ee_pub = self.create_publisher(Point, '/panda/ee_pose', 10)
        
        # 2. Control: Listen for joint targets
        self.create_subscription(Float64MultiArray, '/panda/joint_cmds', self.cmd_cb, 10)

    def cmd_cb(self, msg):
            # Directly apply incoming ROS commands to MuJoCo actuators
            self.data.ctrl[:7] = msg.data[:7]
    
    def publish_feeds(self):
        # Publish coordinates of the 'attachment_site' (hand tip)
        ee_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand")]
        self.ee_pub.publish(Point(x=ee_pos[0], y=ee_pos[1], z=ee_pos[2]))
        
        # Publish all joint positions
        js = JointState()
        js.position = self.data.qpos[:7].tolist()
        self.joint_pub.publish(js)


def main():
    
    cfg = load_robot_config("./config/config.yaml")

    model = mujoco.MjModel.from_xml_path(cfg.mujoco_config.model)

    model = mujoco.MjModel.from_xml_path(cfg.mujoco_config.model)
    data = mujoco.MjData(model)
    node = MuJoCoSimNode(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # data.qpos[:cfg.robot_config.num_dof] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        

        while viewer.is_running(): 
            rclpy.spin_once(node, timeout_sec=0)
            # data.qpos[:cfg.robot_config.num_dof] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
            mujoco.mj_step(model, data)
            # data.ctrl[:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
            # mujoco.mj_forward(model, data)
            node.publish_feeds()
            viewer.sync()
            # viewer.sync()



if __name__ == "__main__":
    rclpy.init()
    main()
