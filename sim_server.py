import os 
import yaml
import time
import threading

import numpy as np
import mujoco
import mediapy as media
import mujoco.viewer

from src.helper_functions.quat_helper import quat_to_rpy, rpy_to_quat, quaternion_error
from src.helper_functions.config_loader import load_robot_config
from src.helper_functions.ik_solver import InverseKinematicsSolver


cfg = load_robot_config("./config/config.yaml")

model = mujoco.MjModel.from_xml_path(cfg.mujoco_config.model)

ik = InverseKinematicsSolver(cfg)

# ik.printer()
# ik.solve_6d_ik([0,0,0,0,0,0])

def main():

    model = mujoco.MjModel.from_xml_path(cfg.mujoco_config.model)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # data.qpos[:cfg.robot_config.num_dof] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        mujoco.mj_forward(model, data)
        while viewer.is_running(): 
            # data.qpos[:cfg.robot_config.num_dof] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
            mujoco.mj_step(model, data)
            # data.ctrl[:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
            # mujoco.mj_forward(model, data)
            # print("a",data.qpos)
            viewer.sync()



if __name__ == "__main__":
    main()
# model.geom()

# id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger_pad")

# print(data.geom())



# # ——— Build model + data ———
# model = mujoco.MjModel.from_xml_path(MODEL_PATH)
# data  = mujoco.MjData(model)
# lock  = threading.Lock()

# # ——— IDs ———
# ee_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  EE_BODY)
# base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  BASE_BODY)
# grip_j  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIP_JOINT)
# assert ee_id>=0 and base_id>=0 and grip_j>=0

# # ——— Home pose ———
# home_q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, 0])
# with lock:
#     data.qpos[:NV] = home_q
#     data.ctrl[:NV] = home_q
#     data.ctrl[NV]  = 0.0
#     mujoco.mj_forward(model, data)
# current_q = home_q.copy()

# # ——— IK solver ———
# scratch = mujoco.MjData(model)
# def solve_6d_ik(tp, tr, seed_q):
#     q = seed_q.copy()
#     qd = rpy_to_quat(tr)
#     W  = np.diag([1,1,1,ORI_WEIGHT,ORI_WEIGHT,ORI_WEIGHT])
#     for _ in range(MAX_ITERS):
#         scratch.qpos[:NV] = q
#         mujoco.mj_forward(model, scratch)
#         p, r = scratch.xpos[ee_id], scratch.xquat[ee_id]
#         ep, er = tp - p, quaternion_error(r,qd)
#         if np.linalg.norm(ep)<TOL_POS and np.linalg.norm(er)<TOL_ORI:
#             return q
#         Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
#         mujoco.mj_jacBody(model, scratch, Jp, Jr, ee_id)
#         J6 = np.vstack((Jp[:,:NV], Jr[:,:NV]))
#         JW = W @ J6
#         JJt = JW @ JW.T + (DAMP**2)*np.eye(6)
#         dq = JW.T @ np.linalg.solve(JJt, W @ np.hstack((ep,er)))
#         q += dq
#     raise RuntimeError("IK failed")

# # ——— ZMQ setup ———
# ctx    = zmq.Context()
# socket = ctx.socket(zmq.REP)
# socket.bind("tcp://127.0.0.1:5555")
# print("Server listening on tcp://127.0.0.1:5555")

# # ——— Capture infrastructure ———
# want_capture = threading.Event()
# capture_buf  = None
# capture_lock = threading.Lock()

# def sim_thread():
#     global capture_buf, current_q
#     viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=False)
#     win    = viewer.window

#     while viewer.is_alive:
#         with lock:
#             mujoco.mj_step(model, data)
#         viewer.render()
#         time.sleep(0.005)

#         # handle capture
#         if want_capture.is_set():
#             want_capture.clear()
#             time.sleep(2.0)  # settle
#             glfw.make_context_current(win)
#             viewer.render()
#             GL.glFlush(); GL.glFinish()
#             w,h = glfw.get_framebuffer_size(win)
#             raw = GL.glReadPixels(0,0,w,h,GL.GL_RGB,GL.GL_UNSIGNED_BYTE)
#             arr = np.frombuffer(raw, np.uint8).reshape(h,w,3)[::-1,:,:]
#             buf = io.BytesIO()
#             Image.fromarray(arr).save(buf, format="PNG")
#             with capture_lock:
#                 capture_buf = buf.getvalue()

#     viewer.close()

# threading.Thread(target=sim_thread, daemon=True).start()
# time.sleep(0.2)

# # ——— Main ZMQ Loop ———
# try:
#     while True:
#         msg = socket.recv_json()
#         cmd = msg.get("cmd","")

#         if cmd == "move":
#             tp        = np.array(msg["target_pos"])
#             tr        = np.array(msg["target_rpy"])
#             grip_open = bool(msg.get("grip_open", False))

#             # IK + PD move in main thread
#             sol_q = solve_6d_ik(tp, tr, current_q)
#             for α in np.linspace(0,1,30)[1:]:
#                 q_cmd = (1-α)*current_q + α*sol_q
#                 with lock:
#                     data.ctrl[:NV] = q_cmd
#                     data.ctrl[NV]  = 0.0 if grip_open else 1.0
#                 time.sleep(0.005)
#             current_q = sol_q.copy()
#             socket.send_json({"status":"moved", "q": sol_q.tolist()})

#         elif cmd == "capture":
#             want_capture.set()
#             # wait for sim thread to fill buffer
#             while True:
#                 with capture_lock:
#                     if capture_buf is not None:
#                         png = capture_buf
#                         capture_buf = None
#                         break
#                 time.sleep(0.01)
#             socket.send(png, zmq.SNDMORE)
#             socket.send_json({"time": data.time})
        

#         elif cmd == "quit":
#             socket.send_json({"status":"shutting down"})
#             break

#         else:
#             socket.send_json({"error":"unknown command"})

# finally:
#     print("Shutting down...")
#     socket.close()
#     ctx.term()
#     sys.exit(0)
