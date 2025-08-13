from apps.CameraTransformManager import CameraTransformManager
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import mcubes
import json
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def quaternion_to_rotation_matrix(q):
    """Converts a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

def load_cameras_from_json(json_path, device="cpu", use_blender_to_cv=True):
    with open(json_path, 'r') as f:
        cam_params = json.load(f)

    # only flip the camera Y axis (so that world‐up → +Y becomes camera‐down → +v)
    B2C = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0],
                                  dtype=torch.float32,
                                  device=device))

    cameras = []
    for cam_data in cam_params:
        loc  = np.array(cam_data["location"], dtype=np.float32)
        quat = cam_data["quaternion"]

        # build Blender camera→world
        R_bw = quaternion_to_rotation_matrix(quat)
        pose_c2w = np.eye(4, dtype=np.float32)
        pose_c2w[:3, :3] = R_bw
        pose_c2w[:3,  3] = loc
        pose_c2w = torch.tensor(pose_c2w, device=device)

        # invert → world→camera (in Blender cam coords)
        pose_w2c = torch.linalg.inv(pose_c2w)

        # apply only the Y‐axis flip for OpenCV
        if use_blender_to_cv:
            pose_w2c = B2C @ pose_w2c

        # split extrinsics
        R = pose_w2c[:3, :3]
        t = pose_w2c[:3,  3]

        # build K (with correct 512→pixel scaling)
        w, h = cam_data["image_size"]
        sx, sy = 512.0 / w, 512.0 / h
        f = cam_data["focal_length_px"]
        K = torch.tensor([
            [f * 0.28,   0.0, 256.0],
            [  0.0, f * 0.28, 256.0],
            [  0.0,   0.0,   1.0],
        ], dtype=torch.float32, device=device)

        cameras.append(Camera(R, t, K))
        # sanity‐check
        C = (-R.T @ t).cpu().numpy()
        print(f"Camera {cam_data['frame']} center: {C}, JSON loc: {loc}")
    
    return cameras

class Camera(nn.Module):
    """Represents a camera with optimizable extrinsic parameters."""
    def __init__(self, R, t, K):
        super(Camera, self).__init__()
        self.R = nn.Parameter(R)
        self.t = nn.Parameter(t)
        self.K = K.to(device)
        self.offset_2d = nn.Parameter(torch.zeros(1, 2, device=R.device), requires_grad=True)

    def generate_rays(self, H, W, pixels_uv):
        u, v = pixels_uv[:, 0], pixels_uv[:, 1]
        dirs_cam = torch.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], torch.ones_like(u)], -1)
        rays_d = F.normalize(torch.matmul(dirs_cam, self.R.T), dim=-1)
        rays_o = -torch.matmul(self.R.T, self.t).expand_as(rays_d)
        return rays_o, rays_d


def apply_homogeneous_transform(x, T):
        """
        Applies a 4x4 homogeneous transformation matrix `T` to a [B, N, 3] tensor `x`.
        """
        if x.dim() == 2:  # [N, 3]
            x = x.unsqueeze(0)  # make it [1, N, 3]
    
        B, N, _ = x.shape
        homo = torch.cat([x, torch.ones(B, N, 1).to(x.device)], dim=-1)  # [B, N, 4]
        return torch.matmul(homo, T[:3, :].T)  # [B, N, 3]

def compute_head_roll_loss(head_rotmat, up_direction="world_z"):
    """
    Computes a loss that penalizes the roll (side-tilt) of the head to encourage a neutral, forward-facing posture.
    """
    head_up = head_rotmat[:, :, 1]  # shape: [B, 3]

    # Define target up direction
    if isinstance(up_direction, str):
        if up_direction == "world_z":
            target_up = torch.tensor([0.0, 0.0, 1.0], device=head_up.device)
        elif up_direction == "view_y":
            target_up = torch.tensor([0.0, 1.0, 0.0], device=head_up.device)
        else:
            raise ValueError(f"Unknown up_direction string: {up_direction}")
    elif isinstance(up_direction, torch.Tensor):
        target_up = F.normalize(up_direction, dim=0)
    else:
        raise TypeError("up_direction must be a string or a torch.Tensor")

    # Normalize head up vectors
    head_up = F.normalize(head_up, dim=1)

    # Compute cosine loss
    loss = 1 - torch.sum(head_up * target_up.unsqueeze(0), dim=1)  # [B]
    return loss.mean()
