import time
import mujoco
import mujoco_viewer
import numpy as np

# Path to your Panda XML file (adjust as needed)
model_path = "/home/saitama/Documents/panda_robot_learning/franka_emika_panda/mjx_single_cube.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# data.qpos[:] = model.key_qpos[0]
data.ctrl[:] = model.key_ctrl[2] 

mujoco.mj_forward(model, data)

viewer = mujoco_viewer.MujocoViewer(model, data)

# We'll print joint angles to the console at this interval
PRINT_INTERVAL = 0.5
last_print_time = time.time()

try:
    while True:
        # Step the simulation
        mujoco.mj_step(model, data)
        viewer.render()
        
        # If viewer is closed, break
        if not viewer.is_alive:
            break
        
        # Print angles to console every 0.5s
        current_time = time.time()
        if (current_time - last_print_time) > PRINT_INTERVAL:
            joint_angles_degrees = np.degrees(data.qpos[:7])
            print("Joint angles (deg):", np.round(joint_angles_degrees, 2))
            last_print_time = current_time

except KeyboardInterrupt:
    print("Simulation interrupted by user.")

finally:
    viewer.close()
