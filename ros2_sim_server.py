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
    def __init__(self, model, data, renderer):
        super().__init__('panda_sim_server')
        self.model = model
        self.data = data
        self.renderer = renderer
        self.bridge = CvBridge()

        
        self.joint_pub = self.create_publisher(JointState, '/panda/joint_states', 10)
        self.ee_pub = self.create_publisher(Point, '/panda/ee_pose', 10)
        self.camera = self.create_publisher(Image,'camera/image_raw', 10)
        
        self.create_subscription(Float64MultiArray, '/panda/joint_cmds', self.cmd_cb, 10)

    def cmd_cb(self, msg):
            self.data.ctrl[:7] = msg.data[:7]
    
    def publish_feeds(self):
        ee_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand")]
        self.ee_pub.publish(Point(x=ee_pos[0], y=ee_pos[1], z=ee_pos[2]))
        
        js = JointState()
        js.position = self.data.qpos[:7].tolist()
        self.joint_pub.publish(js)

        self.renderer.update_scene(self.data, camera="demo_cam")  # to change the camera demo_cam or top_cam , or we can add more camera in mjx_single_cube.xml 
        pixels = self.renderer.render()
        img_msg = self.bridge.cv2_to_imgmsg(pixels, encoding="rgb8")
        self.camera.publish(img_msg)




def main():
    
    cfg = load_robot_config("./config/config.yaml")

    model = mujoco.MjModel.from_xml_path(cfg.mujoco_config.model)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    node = MuJoCoSimNode(model, data, renderer)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        while viewer.is_running(): 
            rclpy.spin_once(node, timeout_sec=0)
            mujoco.mj_step(model, data)
            # data.ctrl[:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
            # mujoco.mj_forward(model, data)
            node.publish_feeds()
            viewer.sync()



if __name__ == "__main__":
    rclpy.init()
    main()
