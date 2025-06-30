import cv2
import numpy as np
import csv
from ultralytics import YOLO
from math import radians, sin, cos

# === CONFIG ===
VID1, VID2, VID3 = "back.mp4", "left.mp4", "right.mp4"
MODEL_PATH = "CBDbest.pt"

IMG_W, IMG_H = 7680, 4320
F_MM, SENSOR_W_MM = 24.0, 36.0
F_PX = F_MM / (SENSOR_W_MM / IMG_W)
CX, CY = IMG_W / 2, IMG_H / 2

BALL_DIAM = 0.0713
MIN_BBOX = 8
WEIGHT_BY = "conf"  # or "size"

cam_positions = [
    np.array([0.062624, -25.5260, 1.0606]),
    np.array([4.6071, 7.2320, 3.2221]),
    np.array([-5.3876, 7.2320, 3.2221]),
]
cam_eulers = [
    (90, 0, 0),
    (90, 0, 180),
    (90, 0, 180),
]

# Build intrinsics matrix
K = np.array([[F_PX, 0, CX],
              [0, F_PX, CY],
              [0, 0, 1]])

# Axis flip for Blender → OpenCV convention
M_flip = np.diag([1, -1, -1])

def euler_to_mat(rx, ry, rz):
    """Blender XYZ Euler angles to rotation matrix."""
    x, y, z = map(radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

# Precompute projection matrices and rotation matrices
Ps, R_mats = [], []
for C, E in zip(cam_positions, cam_eulers):
    R = euler_to_mat(*E).T
    R = M_flip @ R
    t = -R @ C
    Ps.append(K @ np.hstack((R, t.reshape(3, 1))))
    R_mats.append(R)

# Load YOLO model
model = YOLO(MODEL_PATH)

def detect_ball(img):
    res = model(img)[0]
    if not res.boxes:
        return None
    b = max(res.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
    w = x2 - x1
    if w < MIN_BBOX:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (cx, cy, w, float(b.conf))

def single_view_3d(cx, cy, bbox_w, cam_idx):
    Z = (BALL_DIAM * F_PX) / bbox_w
    v_flip = IMG_H - cy
    dir_cam = np.array([(cx - CX) / F_PX, (v_flip - CY) / F_PX, -1.0])
    dir_cam /= np.linalg.norm(dir_cam)
    ray = R_mats[cam_idx] @ dir_cam
    return cam_positions[cam_idx] + ray * Z

def triangulate(obs):
    A = []
    for u, v, P, w, _, _ in obs:
        A.append((u * P[2] - P[0]) * w)
        A.append((v * P[2] - P[1]) * w)
    A = np.vstack(A)
    _, _, VT = np.linalg.svd(A)
    X = VT[-1]
    X /= X[3]
    return X[:3]

# === MAIN LOOP ===
caps = [cv2.VideoCapture(p) for p in (VID1, VID2, VID3)]
fps = caps[0].get(cv2.CAP_PROP_FPS)
w, h = int(caps[0].get(3)), int(caps[0].get(4))
out = cv2.VideoWriter("tracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

trajectory = []
frame_idx = 0

while True:
    rets = [cap.read() for cap in caps]
    if not all(r for r, _ in rets):
        break
    frames = [img for _, img in rets]

    # 1) Ball detection in all cameras
    dets = [detect_ball(f) for f in frames]

    # 2) Build observations (u,v,P,weight,index,bbox_width)
    obs = []
    for i, (d, P) in enumerate(zip(dets, Ps)):
        if d:
            u, v, bw, conf = d
            weight = conf if WEIGHT_BY == "conf" else bw
            obs.append((u, v, P, weight, i, bw))

    # 3) Triangulate or estimate
    if len(obs) >= 2:
        X3d = triangulate(obs)
    elif len(obs) == 1:
        u, v, P, wt, i, bw = obs[0]
        X3d = single_view_3d(u, v, bw, i)
    else:
        X3d = np.array([np.nan, np.nan, np.nan])

    # 4) Save to trajectory with detection count
    trajectory.append((frame_idx, *X3d, len(obs)))

    # 5) Annotate and save video
    vis = frames[0].copy()
    if dets[0]:
        u, v, _, _ = dets[0]
        cv2.circle(vis, (int(u), int(v)), 12, (0, 0, 255), -1)
    text = f"F{frame_idx}: X={X3d[0]:.2f}, Y={X3d[1]:.2f}, Z={X3d[2]:.2f}, Cams={len(obs)}"
    cv2.putText(vis, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    out.write(vis)
    frame_idx += 1

# Release everything
for cap in caps:
    cap.release()
out.release()

# Save CSV
with open("trajectory.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "X", "Y", "Z", "cams_detected"])
    writer.writerows(trajectory)

print(" Done: saved tracked.mp4 + trajectory.csv")
