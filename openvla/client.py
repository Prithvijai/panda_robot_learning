#!/usr/bin/env python3
"""
client.py

Usage:
    python client.py <image_path> "<prompt>"

Example:
    python client.py "C:/Users/krisk/Downloads/openvla/openvla/img1.jpg" \
        "pick up the green block"
"""

import sys
import os
import getpass
import json
import paramiko  # pip install paramiko

# ─── Configuration (edit to match your setup) ────────
USERNAME      = "kvinod"
LOGIN_HOST    = "login.sol.rc.asu.edu"
GPU_HOST      = "sg010"
REMOTE_BASE   = "/scratch/kvinod/openvla_server"
REMOTE_SCRIPT = REMOTE_BASE + "/remote_infer.py"
PYTHON_BIN    = "/home/kvinod/.conda/envs/openvla/bin/python"
# ─────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    local_img = sys.argv[1]
    prompt    = sys.argv[2]

    if not os.path.isfile(local_img):
        print(f"Error: local image not found: {local_img}")
        sys.exit(1)

    password = getpass.getpass(f"Password for {USERNAME}@{LOGIN_HOST}: ")

    # 1) SSH → login node
    ssh_login = paramiko.SSHClient()
    ssh_login.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to login node {LOGIN_HOST}…")
    ssh_login.connect(LOGIN_HOST, username=USERNAME, password=password)
    print("✔ Connected to login node")

    # 2) Upload image via SFTP
    remote_img = f"{REMOTE_BASE}/{os.path.basename(local_img)}"
    print(f"Uploading {local_img} → {remote_img} …")
    sftp = ssh_login.open_sftp()
    try:
        sftp.mkdir(REMOTE_BASE)
    except IOError:
        pass  # already exists
    sftp.put(local_img, remote_img)
    sftp.close()
    print("✔ Image uploaded")

    # 3) Tunnel SSH → GPU node
    print(f"Tunneling to GPU node {GPU_HOST}…")
    transport = ssh_login.get_transport()
    chan = transport.open_channel("direct-tcpip", (GPU_HOST, 22), ("127.0.0.1", 0))
    ssh_gpu = paramiko.SSHClient()
    ssh_gpu.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_gpu.connect(GPU_HOST, username=USERNAME, password=password, sock=chan)
    print("✔ Connected to GPU node")

    # 4) Run remote_infer.py on the GPU node
    cmd = f"{PYTHON_BIN} {REMOTE_SCRIPT} --image {remote_img} --prompt \"{prompt}\""
    print("Running inference:", cmd)
    stdin, stdout, stderr = ssh_gpu.exec_command(cmd)
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()

    if code != 0:
        print("❌ Remote error:\n", err)
        sys.exit(1)

    try:
        vec = json.loads(out)
        print("✅ 7-DoF action vector:", vec)
    except json.JSONDecodeError:
        print("⚠️ Could not parse JSON:\n", out)

    ssh_gpu.close()
    ssh_login.close()

if __name__ == "__main__":
    main()
