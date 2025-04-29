import threading
import time
import mujoco
import mujoco_viewer
import numpy as np

# Path to your Panda XML with position actuators
MODEL_PATH = "franka_emika_panda/mjx_single_cube.xml"

# Load model and data
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# We'll store the current target angles (in radians) here
target_angles = data.ctrl[:7].copy()  # initialize to whatever is in ctrl

# A flag to tell the simulation thread when to exit
exit_flag = False

def simulation_thread():
    """Continuously runs the simulation and renders the viewer."""
    global exit_flag

    viewer = mujoco_viewer.MujocoViewer(model, data)

    try:
        while not exit_flag:
            # The main control logic: position actuators want target_angles
            data.ctrl[:7] = target_angles

            # Step simulation
            mujoco.mj_step(model, data)

            # Render the viewer
            viewer.render()

            # Check if the viewer was closed
            if not viewer.is_alive:
                exit_flag = True
                break

            # Small delay to avoid maxing CPU
            time.sleep(0.005)
    finally:
        viewer.close()

# Start the simulation in a separate thread
sim_thread = threading.Thread(target=simulation_thread, daemon=True)
sim_thread.start()

try:
    while not exit_flag:
        # Prompt user for new angles
        user_input = input(
            "\nEnter 7 joint angles (degrees) separated by spaces (or 'q' to quit): "
        )

        if user_input.lower() == 'q':
            exit_flag = True
            break

        # Parse angles
        angle_strs = user_input.split()
        if len(angle_strs) != 7:
            print("Please enter exactly 7 joint angles.")
            continue

        try:
            angles_deg = [float(a) for a in angle_strs]
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            continue

        # Convert to radians
        angles_rad = np.radians(angles_deg)

        # Update the global target_angles
        target_angles[:] = angles_rad
        print(f"Moving to (radians): {np.round(angles_rad, 3)}")

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    exit_flag = True
    sim_thread.join()
    print("Simulation terminated.")
