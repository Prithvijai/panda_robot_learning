import threading
import time
import mujoco
import mujoco_viewer
import numpy as np

# — USER CONFIG — 
MODEL_PATH = "/home/saitama/Documents/panda_robot_learning/franka_emika_panda/mjx_single_cube.xml"   # or full path to your MJCF
EE_BODY    = "hand"                  # MuJoCo body name of the Panda end-effector
BASE_BODY  = "link0"                 # MuJoCo body name of the Panda base
GRIP_JOINT = "finger_joint1"         # which joint you use to open/close
NV         = 7                       # number of Panda arm DOFs (no gripper)
# — end USER CONFIG — 

# load model + data
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# find IDs once
ee_body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
grip_jid     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIP_JOINT)

assert ee_body_id   >= 0, f"Unknown body: {EE_BODY}"
assert base_body_id >= 0, f"Unknown body: {BASE_BODY}"
assert grip_jid     >= 0, f"Unknown joint: {GRIP_JOINT}"

# start in a “home” pose
home_q = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, 0.0])
with threading.Lock():
    data.qpos[:NV] = home_q
    mujoco.mj_forward(model, data)

# IK parameters
MAX_ITERS = 100
TOL       = 1e-3
DAMP      = 1e-2

def quat_to_rpy(wxyz):
    """ Convert (w,x,y,z) to roll,pitch,yaw """
    w,x,y,z = wxyz
    # from standard formulas
    sinr = 2*(w*x + y*z); cosr = 1-2*(x*x+y*y)
    roll  = np.arctan2(sinr,cosr)
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(np.clip(sinp,-1,1))
    siny = 2*(w*z + x*y); cosy = 1-2*(y*y+z*z)
    yaw   = np.arctan2(siny,cosy)
    return np.array([roll,pitch,yaw])

def rpy_to_quat(rpy):
    """ Convert roll,pitch,yaw to a unit quaternion (w,x,y,z) """
    r,p,y = rpy
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w,x,y,z])

def quaternion_error(q_cur, q_des):
    """ Returns axis-angle error vector: axis*angle """
    # q_err = q_des * q_cur^{-1}
    w0,x0,y0,z0 = q_cur; w1,x1,y1,z1 = q_des
    # inverse of cur = (w0, -x0, -y0, -z0)
    # quaternion multiply
    we = w1*w0 + x1*(-x0) + y1*(-y0) + z1*(-z0)
    xe = w1*(-x0)+ x1*w0   + y1*(-z0)+ z1*y0
    ye = w1*(-y0)+ y1*w0   + z1*(-x0)+ x1*z0
    ze = w1*(-z0)+ z1*w0   + x1*(-y0)+ y1*x0
    # map to axis-angle: v*(2*acos(w))
    angle = 2*np.arccos(np.clip(we,-1,1))
    if abs(angle)<1e-6: return np.zeros(3)
    axis = np.array([xe,ye,ze])/np.sin(angle/2)
    return axis * angle

def solve_ik(target_pos, target_rpy, seed_q):
    """ Simple Damped Least-Squares IK using mujoco.jacBody """
    q = seed_q.copy()
    q_des_quat = rpy_to_quat(target_rpy)
    for i in range(MAX_ITERS):
        # FK
        with threading.Lock():
            data.qpos[:NV] = q
            mujoco.mj_forward(model, data)

        # current EE pose
        pos_cur  = data.xpos[ee_body_id].copy()
        quat_cur = data.xquat[ee_body_id].copy()

        # position and orientation error
        err_p = target_pos - pos_cur
        err_r = quaternion_error(quat_cur, q_des_quat)
        err   = np.hstack((err_p, err_r))
        if np.linalg.norm(err) < TOL:
            return q

        # Jacobian (3×nv transl + 3×nv rotational)
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, Jp, Jr, ee_body_id)
        J6 = np.vstack((Jp[:,:NV], Jr[:,:NV]))  # only first NV columns

        # Damped least-squares step
        JJt = J6@J6.T
        dq  = J6.T @ np.linalg.solve(JJt + (DAMP**2)*np.eye(6), err)
        q  += dq

    raise RuntimeError("IK failed: err=%.4f"%np.linalg.norm(err))

# start simulation+render thread
exit_flag = False
def simloop():
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=False)
    while not exit_flag and viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
        time.sleep(0.005)
    viewer.close()

thread = threading.Thread(target=simloop, daemon=True)
thread.start()

# interactive loop
current_q = home_q.copy()
try:
    while True:
        # print current EE pose & gripper
        with threading.Lock():
            data.qpos[:NV] = current_q
            mujoco.mj_forward(model, data)
            p = data.xpos[ee_body_id]
            rpy = quat_to_rpy(data.xquat[ee_body_id])
            grip = data.qpos[grip_jid]   # or data.ctrl[7]
        print(f"\nCURRENT →  x,y,z = {p.round(3)}   rpy(deg) = {np.degrees(rpy).round(1)}   gripper = {grip:.2f}")

        cmd = input("Enter x y z roll pitch yaw  (deg)  gripper(0=open,1=close), or 'q':\n> ")
        if cmd.lower()=='q': break
        vals = cmd.split()
        if len(vals)!=7:
            print("Need 7 values!"); continue

        x,y,z,rd,pd,yd,gf = map(float, vals)
        target_pos  = np.array([x,y,z])
        target_rpy  = np.deg2rad([rd,pd,yd])
        grip_target = (gf>0.5)

        try:
            sol = solve_ik(target_pos, target_rpy, current_q)
        except RuntimeError as e:
            print("IK ERROR:", e)
            continue

        current_q[:] = sol
        print("IK → q =", np.round(sol,3))

        # apply into simulation
        with threading.Lock():
            data.ctrl[:NV] = sol
            # set gripper joint position
            data.qpos[grip_jid] = 1.0 if grip_target else 0.0

        time.sleep(0.5)
finally:
    exit_flag = True
    thread.join()
    print("Exited.")
