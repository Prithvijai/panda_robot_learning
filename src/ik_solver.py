import mujoco
import numpy as np

from src.quat_helper import quat_to_rpy, rpy_to_quat, quaternion_error


class InverseKinematicsSolver():
    def __init__(self, cfg):
        self.cfg = cfg
        pass

    def printer(self):
        print(self.cfg)

    def solve_6d_ik(self, target):
        model = mujoco.MjModel.from_xml_path(self.cfg.mujoco_config.model)
        print(model)
        pass
# model = mujoco.MjModel.from_xml_path(MODEL_PATH)


# ee_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  EE_BODY)
# base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  BASE_BODY)
# grip_j  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIP_JOINT)

# def solve_6d_ik(target_pos, target_rpy, seed_q, model):
#     q = seed_q.copy()
#     q_des_quat = rpy_to_quat(target_rpy)
#     W = np.diag([1,1,1, ORI_WEIGHT, ORI_WEIGHT, ORI_WEIGHT])
#     scratch = mujoco.MjData(model)
#     for _ in range(MAX_ITERS):
#         # ----------------------------
#         # 1) load your guess into scratch
#         scratch.qpos[:NV] = q
#         mujoco.mj_forward(model, scratch)

#         # 2) read pose & Jacobian from scratch
#         pos_cur  = scratch.xpos[ee_id].copy()
#         quat_cur = scratch.xquat[ee_id].copy()
#         err_p = target_pos - pos_cur
#         err_r = quaternion_error(quat_cur, q_des_quat)
#         if np.linalg.norm(err_p)<TOL_POS and np.linalg.norm(err_r)<TOL_ORI:
#             return q

#         err6 = np.hstack((err_p, err_r))
#         Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
#         mujoco.mj_jacBody(model, scratch, Jp, Jr, ee_id)
#         J6 = np.vstack((Jp[:,:NV], Jr[:,:NV]))

#         # 3) Damped‐LS step
#         JW = W @ J6
#         JJt = JW @ JW.T + (DAMP**2)*np.eye(6)
#         dq  = JW.T @ np.linalg.solve(JJt, W @ err6)
#         q  += dq
#     raise RuntimeError("IK failed …")

