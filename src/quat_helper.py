import numpy as np

def quat_to_rpy(q):
    w,x,y,z = q
    sinr = 2*(w*x + y*z); cosr = 1-2*(x*x+y*y)
    roll  = np.arctan2(sinr, cosr)
    sinp  = 2*(w*y - z*x); pitch = np.arcsin(np.clip(sinp,-1,1))
    siny = 2*(w*z + x*y); cosy = 1-2*(y*y+z*z)
    yaw   = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw])

def rpy_to_quat(rpy):
    r,p,y = rpy
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])

def quaternion_error(q_cur, q_des):
    # Ensure inputs are numpy arrays for consistency and vectorization
    q_cur = np.array(q_cur)
    q_des = np.array(q_des)

    # Ensure they are unit quaternions (optional but recommended)
    q_cur = q_cur / np.linalg.norm(q_cur)
    q_des = q_des / np.linalg.norm(q_des)

    # Calculate the conjugate of q_cur
    q_cur_conj = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])

    # Calculate the error quaternion q_err = q_des * q_cur_conj
    # Standard quaternion multiplication: (w1, v1) * (w2, v2) = (w1w2 - v1.v2, w1v2 + w2v1 + v1 x v2)
    w0, x0, y0, z0 = q_cur_conj
    w1, x1, y1, z1 = q_des

    # *** CORRECTED SCALAR PART ***
    we = w1*w0 - x1*x0 - y1*y0 - z1*z0 # This is w1*w0 - v1.v2

    # Vector part: w1*v2 + w2*v1 + v1 x v2
    # Note: The original code's calculation for xe, ye, ze was actually correct for q_des * q_cur_conj
    xe = w1*x0 + x1*w0 + y1*z0 - z1*y0
    ye = w1*y0 - x1*z0 + y1*w0 + z1*x0
    ze = w1*z0 + x1*y0 - y1*x0 + z1*w0

    # Ensure shortest path rotation (we >= 0)
    # If we is negative, the angle is > pi. Use -q_err instead.
    if we < 0:
        we = -we
        xe = -xe
        ye = -ye
        ze = -ze

    # Convert error quaternion to axis-angle
    angle = 2 * np.arccos(np.clip(we, -1.0, 1.0)) # Ensure we is clipped

    # Handle the case of almost zero angle
    if abs(angle) < 1e-9 or abs(np.sin(angle / 2)) < 1e-9 :
         return np.zeros(3) # No rotation needed

    # Calculate axis
    axis_norm = np.sqrt(xe**2 + ye**2 + ze**2) # Should be sin(angle/2)
    # axis = np.array([xe, ye, ze]) / np.sin(angle / 2) # Can be unstable near angle=0 or 2pi
    axis = np.array([xe, ye, ze]) / axis_norm # More stable calculation

    return axis * angle
