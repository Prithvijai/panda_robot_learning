import mujoco
import mujoco_viewer
import numpy as np
import os
import pygame  # Import pygame

# --- MuJoCo Setup (same as before) ---
model_path = os.path.join('./franka_emika_panda', 'mjx_single_cube.xml')
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

# --- Pygame Initialization ---
pygame.init()
pygame.joystick.init()

# Check for joysticks
joystick_count = pygame.joystick.get_count()
if joystick_count > 0:
    joystick = pygame.joystick.Joystick(0)  # Use the first joystick
    joystick.init()
    print(f"Joystick detected: {joystick.get_name()}")
else:
    joystick = None
    print("No joystick detected. Using keyboard control only.")

# --- Control Parameters ---
Kp = 50.0
joint_velocities = np.zeros(7)  # Store desired joint velocities
velocity_scale = 0.1  # Control how fast the joints move
deadzone = 0.1  # Joystick deadzone to prevent drift

# --- Keyboard Mapping (Example - Customize as needed) ---
key_mapping = {
    pygame.K_1: (0, 1),  # Joint 0, positive direction
    pygame.K_q: (0, -1), # Joint 0, negative direction
    pygame.K_2: (1, 1),
    pygame.K_w: (1, -1),
    pygame.K_3: (2, 1),
    pygame.K_e: (2, -1),
    pygame.K_4: (3, 1),
    pygame.K_r: (3, -1),
    pygame.K_5: (4, 1),
    pygame.K_t: (4, -1),
    pygame.K_6: (5, 1),
    pygame.K_y: (5, -1),
    pygame.K_7: (6, 1),
    pygame.K_u: (6, -1),
}

# --- Main Simulation Loop ---
try:
    while True:
        # --- Handle Events (Joystick and Keyboard) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise KeyboardInterrupt  # Exit cleanly

            # --- Joystick Input ---
            if joystick:
                if event.type == pygame.JOYAXISMOTION:
                    # Example: Map joystick axes to joint velocities (adjust as needed)
                    for i in range(min(joystick.get_numaxes(), 7)):  # Limit to 7 joints
                        axis_value = joystick.get_axis(i)
                        if abs(axis_value) > deadzone:
                            joint_velocities[i] = axis_value * velocity_scale
                        else:
                            joint_velocities[i] = 0.0

            # --- Keyboard Input ---
            if event.type == pygame.KEYDOWN:
                if event.key in key_mapping:
                    joint_index, direction = key_mapping[event.key]
                    joint_velocities[joint_index] = direction * velocity_scale
            if event.type == pygame.KEYUP:
                if event.key in key_mapping:
                    joint_index, _ = key_mapping[event.key]
                    joint_velocities[joint_index] = 0.0


        # --- Apply Control (Proportional Control with Velocity Input) ---
        # Instead of directly setting target positions, we're using the joint velocities
        # to *incrementally* change the desired position.  This is a simple form of
        # velocity control.
        target_positions = data.qpos[:7] + joint_velocities  # Integrate velocity
        joint_error = target_positions - data.qpos[:7]
        data.ctrl[:7] = Kp * joint_error
        # Limit Ctrl value to avoid system damage.
        data.ctrl[:7] = np.clip(data.ctrl[:7], -20, 20)  # Example limits

        # --- Step Simulation and Render ---
        mujoco.mj_step(model, data)
        viewer.render()


except KeyboardInterrupt:
    print("Simulation interrupted.")

finally:
    if 'viewer' in locals(): # Close viewer in case of exception
      viewer.close()
    pygame.quit()