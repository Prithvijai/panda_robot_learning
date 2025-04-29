import threading
import time
import mujoco
import mujoco_viewer
import numpy as np
import yaml
import os

# Custom helper functions
from src.quat_helper import quat_to_rpy, rpy_to_quat, quaternion_error

# Load config file which contains robot and IK parameters
config_path = os.path.join(os.path.dirname(__file__), 'src/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# USER CONFIG
MODEL_PATH =  "/home/saitama/Documents/panda_robot_learning/franka_emika_panda/mjx_single_cube.xml"
EE_BODY    = config['robot_config']['ee_body']      # end-effector body in MJCF
BASE_BODY  = config['robot_config']['base_body']    # robot base body in MJCF
GRIP_JOINT = config['robot_config']['grip_joint']   # gripper joint
NV         = config['robot_config']['num_dof']      # number of arm DOFs (no gripper)

# IK + PD gains
MAX_ITERS  = config['ik_config']['max_iters']
TOL_POS    = config['ik_config']['tol_pos']
TOL_ORI    = config['ik_config']['tol_ori']
DAMP       = config['ik_config']['damp']
ORI_WEIGHT = config['ik_config']['ori_weight']

lock  = threading.Lock()
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# IDs
ee_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  EE_BODY)
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  BASE_BODY)
grip_j  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIP_JOINT)
assert ee_id>=0 and base_id>=0 and grip_j>=0, "Check your body/joint names"

# start in a “home” joint configuration
home_q = np.array([0.0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, 0])
with lock:
    data.qpos[:NV] = home_q
    data.ctrl[:NV] = home_q   # desired positions for PD servos
    data.ctrl[NV]  = 0.0      # open gripper
    mujoco.mj_forward(model, data)

def solve_6d_ik(target_pos, target_rpy, seed_q):
    """6-DOF Damped Least-Squares IK (pos+ori)."""
    q = seed_q.copy()
    q_des_quat = rpy_to_quat(target_rpy)
    W = np.diag([1,1,1, ORI_WEIGHT, ORI_WEIGHT, ORI_WEIGHT])
    for _ in range(MAX_ITERS):
        with lock:
            data.qpos[:NV] = q
            mujoco.mj_forward(model, data)
        # current pose
        pos_cur  = data.xpos[ee_id].copy()
        quat_cur = data.xquat[ee_id].copy()
        # errors
        err_p = target_pos - pos_cur
        err_r = quaternion_error(quat_cur, q_des_quat)
        if np.linalg.norm(err_p)<TOL_POS and np.linalg.norm(err_r)<TOL_ORI:
            return q
        err6 = np.hstack((err_p, err_r))
        # Jacobian
        Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
        mujoco.mj_jacBody(model, data, Jp, Jr, ee_id)
        J6 = np.vstack((Jp[:,:NV], Jr[:,:NV]))
        JW = W @ J6
        JJt = JW @ JW.T + (DAMP**2)*np.eye(6)
        dq  = JW.T @ np.linalg.solve(JJt, W @ err6)
        q  += dq
    raise RuntimeError(f"IK failed (pos_err={np.linalg.norm(err_p):.3f}, ori_err={np.linalg.norm(err_r):.3f})")

# start sim + viewer thread
exit_flag = False
def sim_thread():
    v = mujoco_viewer.MujocoViewer(model, data, hide_menus=False)
    while not exit_flag and v.is_alive:
        with lock:
            mujoco.mj_step(model, data)
        v.render()
        time.sleep(0.005)
    v.close()

thread = threading.Thread(target=sim_thread, daemon=True)
thread.start()

# Interactive control loop
current_q = home_q.copy()
try:
    while True:
        # print current EE pose + ori
        with lock:
            mujoco.mj_forward(model, data)
            pos   = data.xpos[ee_id].copy()
            rpy   = quat_to_rpy(data.xquat[ee_id])
            grip  = data.ctrl[NV]
        print(
            f"\nCURRENT → X Y Z = {pos.round(3)}, "
            f"Roll Pitch Yaw = {np.degrees(rpy).round(1)}°, "
            f"Gripper = {'open' if grip<0.5 else 'closed'}"
        )

        inp = input("Enter x y z roll pitch yaw (deg) gripper(0/1) or 'q':\n> ").strip()
        if inp.lower()=='q':
            break
        vals = list(map(float, inp.split()))
        if len(vals)!=7:
            print("ERROR: need 7 values"); continue

        tgt_pos = np.array(vals[:3])
        tgt_rpy = np.deg2rad(vals[3:6])
        grip_open = vals[6]<0.5

        # solve IK
        try:
            sol_q = solve_6d_ik(tgt_pos, tgt_rpy, current_q)
        except Exception as e:
            print("IK ERROR:", e)
            continue
        print("IK → q =", np.round(sol_q,3))

        # smooth joint-space PD‐controlled trajectory
        steps = 100
        for i in range(steps):
            alpha = (i+1)/steps
            q_cmd = (1-alpha)*current_q + alpha*sol_q
            with lock:
                data.ctrl[:NV] = q_cmd    # desired positions
                data.ctrl[NV]  = 0.0 if grip_open else 1.0
            time.sleep(0.005)

        current_q = sol_q.copy()
        print("Motion complete.")

except KeyboardInterrupt:
    pass

exit_flag = True
thread.join()
print("Exited.")
