import numpy as np
import pyexr
import json
import torch
from scipy.spatial.transform import Rotation as R

def save_normal_map_exr(normal_tensor, save_path):
    """Save a (1, 3, H, W) normal tensor as a float32 .exr file."""
    normal_np = normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pyexr.write(save_path, normal_np.astype(np.float32))

def convert_camera_json_to_npz(json_path, output_path='cameras_supernormal.npz'):
    """Convert camera_parameters.json to SuperNormal-compatible .npz format with proper scale normalization."""
    with open(json_path, 'r') as f:
        cameras = json.load(f)

    pose_all = []
    intrinsic_all = []
    cam_positions = []

    for cam in cameras:
        fx = fy = cam["focal_length_px"]
        w, h = cam["image_size"]
        cx = w / 2
        cy = h / 2

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])
        intrinsic_all.append(K)

        q = cam["quaternion"]  # [w, x, y, z]
        r = R.from_quat([q[1], q[2], q[3], q[0]])  # convert to [x, y, z, w]
        R_mat = r.as_matrix()

        t = np.array(cam["location"]).reshape(3, 1)
        cam2world = np.eye(4)
        cam2world[:3, :3] = R_mat
        cam2world[:3, 3] = t[:, 0]
        pose_all.append(cam2world)
        cam_positions.append(t[:, 0])

    pose_all = np.stack(pose_all)
    intrinsic_all = np.stack(intrinsic_all)
    cam_positions = np.stack(cam_positions)

    # Estimate bounding box and compute normalization scale
    center = cam_positions.mean(axis=0)
    radius = np.linalg.norm(cam_positions - center, axis=1).max()
    scale = 1.0 / (radius * 1.2)  # ensure fit inside [-1,1]^3

    # Apply scale to all translations in poses
    for i in range(len(pose_all)):
        pose_all[i][:3, 3] = (pose_all[i][:3, 3] - center) * scale

    # Save to npz
    np.savez(output_path, pose=pose_all, intrinsic=intrinsic_all)
    print(f"Saved to {output_path}")

def save_normal_map_exr_camera_to_world(normal_tensor, rotation_matrix, save_path):
    """
    Convert normal map from camera space to world space and save as .exr
    - normal_tensor: (1, 3, H, W)
    - rotation_matrix: (3, 3) numpy array
    - save_path: path to .exr output
    """
    normal_np = normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    normal_world = normal_np @ rotation_matrix.T  # rotate into world space
    pyexr.write(save_path, normal_world.astype(np.float32))
    return normal_world


def save_normal_map_exr(normal_tensor, save_path):
    """
    normal_tensor: torch.Tensor of shape (1, 3, H, W) in range [-1, 1]
    save_path: path to save the .exr file
    """
    normal_np = normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    pyexr.write(save_path, normal_np.astype(np.float32))

def rotate_normals_to_world(normal_map: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotates a normal map from frame space to world space.

    Args:
        normal_map (torch.Tensor): Tensor of shape (1, 3, H, W), normal map in frame space.
        R (torch.Tensor): Tensor of shape (3, 3), rotation matrix from frame to world space.

    Returns:
        torch.Tensor: Rotated normal map of shape (1, 3, H, W).
    """
    if normal_map.shape[1] != 3:
        raise ValueError(f"Expected normal_map shape (1, 3, H, W), got {normal_map.shape}")
    if R.shape != (3, 3):
        raise ValueError(f"Expected R shape (3, 3), got {R.shape}")

    B, C, H, W = normal_map.shape
    # (1, 3, H, W) → (H*W, 3)
    normal_flat = normal_map.view(3, -1).permute(1, 0)  # shape (H*W, 3)

    # Rotate normals
    rotated_flat = torch.matmul(normal_flat, R.T)

    # Normalize to unit vectors
    rotated_flat = torch.nn.functional.normalize(rotated_flat, dim=1)

    # Reshape back to (1, 3, H, W)
    rotated = rotated_flat.permute(1, 0).view(1, 3, H, W)

    return rotated