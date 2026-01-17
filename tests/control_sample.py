import mujoco
# import mujoco_viewer
import mujoco.viewer
import numpy as np
import os

# Load the Panda model from the specified XML file
model_path = os.path.join('./franka_emika_panda', 'mjx_single_cube.xml')  # Make sure this points to the correct path of the XML file
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Create the viewer for visualization
viewer = mujoco.viewer.MujocoViewer(model, data)
# viewer = mujoco_viewer.launch_passive(model, data)

# Target joint positions for the Panda arm (7 joints)
target_positions = np.array([0.0, 0.5, 0.0, -1.57, 0.0, 1.57, -0.78])

# A simple proportional controller (P-controller)
Kp = 50.0  # Proportional gain, you can tweak this for faster/slower control

try:
    # Simulation loop
    while True:
        # Calculate control inputs (desired joint torques or forces)
        joint_error = target_positions - data.qpos[:7]  # Error between current and target joint positions
        data.ctrl[:7] = Kp * joint_error  # Apply proportional control

        # Step the simulation forward
        mujoco.mj_step(model, data)

        # Update the viewer to visualize the motion
        viewer.render()

except KeyboardInterrupt:
    print("Simulation interrupted.")

# Close the viewer when done
viewer.close()
