import mujoco
print("Python mujoco module version:", mujoco.__version__)

# Try printing engine version if available (some versions support this):
try:
    engine_version = mujoco.mj_versionString()
    print("Engine version string:", engine_version)
except:
    print("Could not get engine version string.")
