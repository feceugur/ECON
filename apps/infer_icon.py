# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
from collections import defaultdict
from apps.SDFNetwork import Camera, extract_mesh_from_sdf, load_cameras_from_json, optimize_sdf, quaternion_to_rotation_matrix
from lib.common.smpl_eval import SMPLMetricsCalculator
import pyexr
import logging
import warnings
import os
import os.path as osp
from lib.common import BNI
from pytorch3d.loss.chamfer import chamfer_distance

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import trimesh
from pytorch3d.ops import SubdivideMeshes
from termcolor import colored
from tqdm.auto import tqdm
import numpy as np
import smplx
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd

from apps.IFGeo import IFGeo
from apps.Normal import Normal
from apps.sapiens import ImageProcessor
from apps.clean_mesh import MeshCleanProcess
from apps.SMPLXJointAligner import SMPLXJointAligner
from apps.CameraTransformManager import CameraTransformManager
from apps.texture_utils import TextureGenerator, generate_texture_map

from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.common.config import cfg
from lib.common.imutils import blend_rgb_norm, load_img, transform_to_tensor, wrap
from lib.common.local_affine import LocalAffine, register, trimesh2meshes
from lib.common.render import Render, query_color, query_normal_color
from lib.common.train_util import Format, init_loss
from lib.common.voxelize import VoxelGrid
from lib.dataset.mesh_util import *
from lib.dataset.TestDataset import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis, rotmat_to_rot6d

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_fixed_vertex_mask(initial_mesh, front_view_id, back_view_id, transform_manager, device, visibility_threshold=0.1):
    """
    Identifies vertices visible from the front or back views to be anchored.
    (Robust version)
    """
    print("Identifying vertices to anchor...")
    
    # --- Robustly get vertices and normals, removing any batch dimension of 1 ---
    initial_verts = initial_mesh.verts_packed().squeeze(0)      # Squeeze potential batch dim
    initial_normals = initial_mesh.verts_normals_packed().squeeze(0) # Squeeze potential batch dim

    total_verts = initial_verts.shape[0]
    if total_verts <= 1:
        raise ValueError(f"Error in get_fixed_vertex_mask: Found only {total_verts} vertices. Check initial_mesh.")

    # Get camera transformations from canonical space to view space
    T_front = transform_manager.get_transform_to_target(front_view_id).to(device)
    T_back = transform_manager.get_transform_to_target(back_view_id).to(device)

    # Transform vertices and normals into the front view's camera space
    verts_in_front_view = apply_homogeneous_transform(initial_verts, T_front.T)
    normals_in_front_view = torch.matmul(initial_normals, T_front[:3, :3].T)
    
    # A vertex is visible if its normal points towards the camera
    front_visibility = torch.sum(F.normalize(normals_in_front_view, dim=-1) * F.normalize(-verts_in_front_view, dim=-1), dim=-1)
    front_mask = front_visibility > visibility_threshold

    # Repeat for the back view
    verts_in_back_view = apply_homogeneous_transform(initial_verts, T_back.T)
    normals_in_back_view = torch.matmul(initial_normals, T_back[:3, :3].T)
    back_visibility = torch.sum(F.normalize(normals_in_back_view, dim=-1) * F.normalize(-verts_in_back_view, dim=-1), dim=-1)
    back_mask = back_visibility > visibility_threshold

    # The final mask includes vertices visible from either the front OR the back
    fixed_mask = torch.logical_or(front_mask, back_mask)
    
    num_fixed = fixed_mask.sum().item()
    
    print(colored(f"Anchoring {num_fixed} / {total_verts} vertices ({num_fixed / total_verts:.2%}).", "yellow"))
    
    return fixed_mask.to(device)


def semantic_texture_from_multi_view(
    final_mesh, 
    multi_view_data, 
    transform_manager, 
    view_indices, 
    device, 
    visibility_threshold=0.05,
    blend_iterations=3
):
    """
    Computes per-vertex colors using a semantic hierarchy (front, back, sides).

    Args:
        final_mesh (trimesh.Trimesh): The final, refined 3D mesh.
        multi_view_data (list): List of data dictionaries for each view.
        transform_manager (CameraTransformManager): Your camera manager.
        view_indices (dict): Dict mapping 'front', 'back', 'left', 'right' to frame indices.
        device (torch.device): The GPU device.
        visibility_threshold (float): Dot product threshold for a surface to be considered visible.
        blend_iterations (int): Number of iterations for seam blending.

    Returns:
        np.ndarray: An array of vertex colors of shape (num_vertices, 4) in uint8 format.
    """
    print("Performing SEMANTIC multi-view texturing...")
    
    verts = torch.tensor(final_mesh.vertices, dtype=torch.float32, device=device)
    vert_normals = torch.tensor(final_mesh.vertex_normals, dtype=torch.float32, device=device)
    num_verts = verts.shape[0]

    # --- 1. Pre-calculate Projection Data for All Views ---
    projection_data = {}
    semantic_order = ['front', 'back', 'left', 'right']

    for view_name in semantic_order:
        view_idx = view_indices.get(view_name)
        if view_idx is None:
            continue
        
        view_data = multi_view_data[view_idx]
        frame_id = int(view_data['name'].split("_")[1])
        T_canon_to_view = transform_manager.get_transform_to_target(frame_id).to(device)
        
        verts_in_view = apply_homogeneous_transform(verts, T_canon_to_view.T).squeeze(0)  # Remove batch dimension
        normals_in_view = torch.matmul(vert_normals, T_canon_to_view[:3, :3].T)
        
        quality_scores = torch.sum(F.normalize(normals_in_view, dim=1) * F.normalize(-verts_in_view, dim=1), dim=1)
        uv_coords = verts_in_view[:, :2] # Y-flip may be needed here

        # Mask for valid vertices in this view
        visible_mask = quality_scores > visibility_threshold
        in_frame_mask = (uv_coords[:, 0].abs() < 1.0) & (uv_coords[:, 1].abs() < 1.0)
        valid_mask = visible_mask & in_frame_mask
        
        img_tensor = view_data["img_icon"].to(device) * 0.5 + 0.5
        
        projection_data[view_name] = {
            'uvs': uv_coords,
            'scores': quality_scores,
            'valid_mask': valid_mask,
            'image': img_tensor
        }

    # --- 2. Hierarchical Coloring ---
    vertex_colors = torch.zeros(num_verts, 3, device=device)
    # -1: uncolored, 0: front, 1: back, 2: left, 3: right
    ownership = torch.full((num_verts,), -1, dtype=torch.long, device=device) 
    
    # Layer 1 & 2: Front and Back
    for i, view_name in enumerate(['front', 'back']):
        if view_name not in projection_data: continue
        data = projection_data[view_name]
        
        # Vertices that are valid in this view AND are not yet colored
        mask = data['valid_mask'] & (ownership == -1)
        if mask.sum() == 0: continue
        
        uvs_to_sample = data['uvs'][mask].unsqueeze(0).unsqueeze(2)
        colors = F.grid_sample(data['image'], uvs_to_sample, mode='bilinear', align_corners=False)
        vertex_colors[mask] = colors.squeeze(3).squeeze(0).T
        ownership[mask] = i

    # Layer 3: Left vs Right for remaining vertices
    if 'left' in projection_data and 'right' in projection_data:
        left_data = projection_data['left']
        right_data = projection_data['right']
        
        # Vertices that are still uncolored
        uncolored_mask = (ownership == -1)
        # Vertices valid in left view AND uncolored
        left_candidate_mask = left_data['valid_mask'] & uncolored_mask
        # Vertices valid in right view AND uncolored
        right_candidate_mask = right_data['valid_mask'] & uncolored_mask

        # Decide between left and right based on score
        left_scores = torch.where(left_candidate_mask, left_data['scores'], -1.0)
        right_scores = torch.where(right_candidate_mask, right_data['scores'], -1.0)
        
        assign_left_mask = (left_scores >= right_scores) & left_candidate_mask
        assign_right_mask = (right_scores > left_scores) & right_candidate_mask
        
        # Apply left colors
        if assign_left_mask.sum() > 0:
            uvs_to_sample = left_data['uvs'][assign_left_mask].unsqueeze(0).unsqueeze(2)
            colors = F.grid_sample(left_data['image'], uvs_to_sample, mode='bilinear', align_corners=False)
            vertex_colors[assign_left_mask] = colors.squeeze(3).squeeze(0).T
            ownership[assign_left_mask] = 2
            
        # Apply right colors
        if assign_right_mask.sum() > 0:
            uvs_to_sample = right_data['uvs'][assign_right_mask].unsqueeze(0).unsqueeze(2)
            colors = F.grid_sample(right_data['image'], uvs_to_sample, mode='bilinear', align_corners=False)
            vertex_colors[assign_right_mask] = colors.squeeze(3).squeeze(0).T
            ownership[assign_right_mask] = 3

    # --- 4. Fill any remaining uncolored vertices (e.g., bottom of feet) by nearest neighbor ---
    uncolored_final_mask = (ownership == -1)
    if uncolored_final_mask.sum() > 0:
        print(f"Filling {uncolored_final_mask.sum()} uncolored vertices by nearest neighbor...")
        colored_verts = verts[~uncolored_final_mask]
        uncolored_verts = verts[uncolored_final_mask]
        
        # Simple KNN with k=1
        distances = torch.cdist(uncolored_verts, colored_verts)
        nearest_indices = torch.argmin(distances, dim=1)
        
        vertex_colors[uncolored_final_mask] = vertex_colors[~uncolored_final_mask][nearest_indices]

    # --- 5. Seam Blending ---
    print(f"Blending seams over {blend_iterations} iterations...")
    edges_np = final_mesh.edges
    edges = torch.tensor(edges_np.T, dtype=torch.long, device=device) # Shape: [2, num_edges]

    for _ in tqdm(range(blend_iterations), desc="Blending"):
        # Find vertices at the seams
        neighbor_ownership = ownership[edges[1]].view(edges.shape[1])
        is_seam = (ownership[edges[0]] != neighbor_ownership).any()
        seam_vertex_indices = torch.unique(edges[0][is_seam])
        
        # Average their colors with neighbors
        new_colors = vertex_colors.clone()
        for v_idx in seam_vertex_indices:
            neighbor_indices = final_mesh.vertex_neighbors[v_idx]
            neighbor_indices_tensor = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
            # Include the vertex itself in the average
            all_indices = torch.cat([torch.tensor([v_idx], device=device), neighbor_indices_tensor])
            new_colors[v_idx] = torch.mean(vertex_colors[all_indices], dim=0)
        vertex_colors = new_colors

    # Convert to numpy and return
    final_colors_np = (vertex_colors.cpu().numpy() * 255).astype(np.uint8)
    return np.hstack([final_colors_np, np.full((num_verts, 1), 255, dtype=np.uint8)])


def save_normal_comparison(pred_norm, gt_norm, path):
    """
    pred_norm and gt_norm: (1, 3, H, W) torch tensors in [-1, 1]
    path: file path to save the image
    """
    # Convert to (H, W, 3) and [0, 255]
    def tensor_to_img(tensor):
        img = tensor[0].permute(1, 2, 0)  # (H, W, 3)
        img = ((img + 1.0) / 2.0 * 255).clamp(0, 255).byte().cpu().numpy()
        return img[..., ::-1]  # Convert RGB to BGR for OpenCV

    pred_img = tensor_to_img(pred_norm)
    gt_img = tensor_to_img(gt_norm)

    # Add labels above images
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # Add text: "PRED" and "GT"
    pred_img = cv2.copyMakeBorder(pred_img, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gt_img = cv2.copyMakeBorder(gt_img, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    cv2.putText(pred_img, "PRED", (10, 20), label_font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(gt_img, "GT", (10, 20), label_font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Concatenate horizontally
    combined = np.concatenate([pred_img, gt_img], axis=1)

    # Save image
    cv2.imwrite(path, combined)


def rotate_normal_map(normal_map: torch.Tensor,
                      R: torch.Tensor,
                      y_down=True):
    """
    Rotate a normal map with shape (B, 3, H, W).

    Args
    ----
    normal_map : tensor in [-1, 1], view-space.
    R          : (3,3) or (B,3,3) rotation from *view* → *target* frame.
    y_down     : if True, flip the Y component because image-space Y axis
                 points down, whereas 3-D space Y usually points up.

    Returns
    -------
    rotated_map : (B, 3, H, W) in the *target* frame, still in [-1, 1].
    """
    B, C, H, W = normal_map.shape
    n = normal_map.permute(0, 2, 3, 1).reshape(-1, 3)          # (BHW, 3)

    if y_down:                                                 # ① undo screen-space Y
        n[:, 1] = -n[:, 1]

    R_flat = R.view(-1, 3, 3).repeat_interleave(H * W, dim=0)  # broadcast
    n_rot  = torch.bmm(R_flat, n.unsqueeze(-1)).squeeze(-1)    # (BHW, 3)

    n_rot  = F.normalize(n_rot, dim=1)                         # keep unit length
    if y_down:
        n_rot[:, 1] = -n_rot[:, 1]                             # flip back

    return n_rot.view(B, H, W, 3).permute(0, 3, 1, 2)          # (B,3,H,W)

def extract_canonical_bni_surface(depth, normal, K, T, mask=None):
    depth = np.squeeze(depth)  # (512, 512)
    normal = np.squeeze(normal)  # (512, 512, 3)
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    z = depth
    x = (i - K[0, 2]) * z / K[0, 0]
    y = (j - K[1, 2]) * z / K[1, 1]

    points_cam = np.stack((x, y, z), axis=-1)  # (H, W, 3)
    points_cam_flat = points_cam.reshape(-1, 3)
    ones = np.ones((points_cam_flat.shape[0], 1))
    points_cam_homo = np.concatenate([points_cam_flat, ones], axis=1).T  # (4, N)

    # Transform to canonical (world) space
    points_world_flat = (T @ points_cam_homo).T[:, :3]
    points_world = points_world_flat.reshape(H, W, 3)

    # Rotate normals to canonical space
    R = T[:3, :3]
    normal_flat = normal.reshape(-1, 3)
    normal_rotated = (R @ normal_flat.T).T
    normal_rotated /= np.linalg.norm(normal_rotated, axis=1, keepdims=True)
    normal_world = normal_rotated.reshape(H, W, 3)

    if mask is None:
        mask = (depth > 0)

    valid_points = points_world[mask]
    valid_normals = normal_world[mask]

    return valid_points, valid_normals

def get_silhouette_from_normal(normal_map):
    """
    Given a normal map tensor (B, 3, H, W) in [-1, 1], return silhouette mask (B, 1, H, W)
    where valid normals (not background) exist.
    """
    magnitude = normal_map.norm(dim=1, keepdim=True)  # (B, 1, H, W)
    mask = (magnitude <= 1e-3).float()  # Background (zero normals) will be 1, valid points will be 0
    return mask


def get_intrinsics_matrix(camera_info):
    fx = fy = camera_info["focal_length_px"]
    width, height = camera_info["image_size"]
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx,  0, cx],
        [0,  fy, cy],
        [0,   0,  1]
    ], dtype=np.float32)
    return K

def run_ifnet_surface_reconstruction(ifnet, in_tensor, cfg):
    with torch.no_grad():
        features, inter = ifnet.netG.filter(in_tensor, return_inter=True)
        sdf = ifnet.reconEngine(
            opt=cfg,
            netG=ifnet.netG,
            features=features,
            proj_matrix=None
        )
        verts_pr, faces_pr = ifnet.reconEngine.export_mesh(sdf)
        return verts_pr, faces_pr, inter
    
def visualize_landmarks_detailed(
    image, smpl_verts, smpl_faces, smpl_lmks, ghum_lmks, ghum_conf, save_path, nose_idx=331, chest_idx=5492
):
    img_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
    smpl_lmks_np = smpl_lmks[0].detach().cpu().numpy()
    ghum_lmks_np = ghum_lmks[0].detach().cpu().numpy()
    conf_np = ghum_conf[0].detach().cpu().numpy()
    H, W = img_np.shape[:2]

    plt.figure(figsize=(10, 10))
    plt.imshow(img_np * 0.5 + 0.5)

    # Plot SMPL mesh
    # verts_2d = smpl_verts[0, :, :2].detach().cpu().numpy()
    # plt.scatter(verts_2d[:, 0] * W, verts_2d[:, 1] * H, c='orange', s=1, alpha=0.3, label='SMPL Body')

    # Plot landmarks as before...
    valid = conf_np > 0.5
    plt.scatter(smpl_lmks_np[valid, 0] * W, smpl_lmks_np[valid, 1] * H, c='red', s=120, edgecolors='black', linewidths=2, label='SMPL Landmarks')
    plt.scatter(ghum_lmks_np[valid, 0] * W, ghum_lmks_np[valid, 1] * H, c='cyan', s=120, edgecolors='black', linewidths=2, label='GT Landmarks')
    for i in range(len(smpl_lmks_np)):
        if conf_np[i] > 0.5:
            plt.plot([smpl_lmks_np[i, 0] * W, ghum_lmks_np[i, 0] * W],
                     [smpl_lmks_np[i, 1] * H, ghum_lmks_np[i, 1] * H],
                     color='yellow', alpha=0.7, linewidth=2)
            # Add index for SMPL landmark
            plt.text(smpl_lmks_np[i, 0] * W + 5, smpl_lmks_np[i, 1] * H - 5, f"S{i}", color='red', fontsize=10, weight='bold')
            # Add index for GT landmark
            plt.text(ghum_lmks_np[i, 0] * W + 5, ghum_lmks_np[i, 1] * H + 15, f"G{i}", color='cyan', fontsize=10, weight='bold')

    plt.legend(loc='upper right')
    plt.title('SMPL Body and Landmark Visualization\nRed: SMPL, Cyan: Ground Truth', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_landmarks_with_true_indices(
    image, smpl_verts, smpl_faces, smpl_lmks, ghum_lmks, ghum_conf, save_path,
    smpl_indices, ghum_indices, nose_idx=331, chest_idx=5492
):
    
    img_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
    smpl_lmks_np = smpl_lmks[0].detach().cpu().numpy()
    ghum_lmks_np = ghum_lmks[0].detach().cpu().numpy()
    conf_np = ghum_conf[0].detach().cpu().numpy()
    H, W = img_np.shape[:2]
    valid = conf_np > 0.5

    verts_2d = smpl_verts[0, :, :2].detach().cpu().numpy()
    verts_2d = (verts_2d + 1.0) * 0.5  # normalize if needed

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # --- SMPL body and SMPL landmarks with GT/SMPL indices ---
    axs[0].scatter(verts_2d[:, 0] * W, verts_2d[:, 1] * H, c='orange', s=1, alpha=0.3, label='SMPL Body')
    axs[0].scatter(smpl_lmks_np[valid, 0] * W, smpl_lmks_np[valid, 1] * H, 
                   c='red', s=120, edgecolors='black', linewidths=2, label='SMPL Landmarks')
    for i, (gt_idx, smpl_idx) in enumerate(zip(ghum_indices, smpl_indices)):
        if conf_np[i] > 0.5:
            axs[0].text(
                smpl_lmks_np[i, 0] * W + 5, smpl_lmks_np[i, 1] * H - 5,
                f"G{gt_idx}/S{smpl_idx}", color='red', fontsize=10, weight='bold'
            )
    axs[0].set_title('SMPL Body + SMPL Landmarks (Gx/Sy)')
    axs[0].set_xlim(0, W)
    axs[0].set_ylim(H, 0)
    axs[0].axis('off')
    axs[0].legend(loc='upper right')

    # --- GT image and GT landmarks with GT/SMPL indices ---
    axs[1].imshow(img_np * 0.5 + 0.5)
    axs[1].scatter(ghum_lmks_np[valid, 0] * W, ghum_lmks_np[valid, 1] * H, 
                   c='cyan', s=120, edgecolors='black', linewidths=2, label='GT Landmarks')
    for i, (gt_idx, smpl_idx) in enumerate(zip(ghum_indices, smpl_indices)):
        if conf_np[i] > 0.5:
            axs[1].text(
                ghum_lmks_np[i, 0] * W + 5, ghum_lmks_np[i, 1] * H + 15,
                f"G{gt_idx}/S{smpl_idx}", color='cyan', fontsize=10, weight='bold'
            )
    axs[1].set_title('GT Image + GT Landmarks (Gx/Sy)')
    axs[1].set_xlim(0, W)
    axs[1].set_ylim(H, 0)
    axs[1].axis('off')
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def convert_rot_matrix_to_angle_axis(rot_matrix):
    """
    Helper function to convert rotation matrices to angle-axis format
    while handling different input shapes correctly.
    
    Args:
        rot_matrix (torch.Tensor): Rotation matrix of shape (B, J, 3, 3) or (B, 3, 3)
        
    Returns:
        torch.Tensor: Angle-axis representation
    """
    input_shape = rot_matrix.shape
    
    if len(input_shape) == 4:  # (B, J, 3, 3)
        B, J = input_shape[:2]
        rot_matrix_flat = rot_matrix.reshape(-1, 3, 3)
        
        # Add homogeneous coordinate for rotation_matrix_to_angle_axis
        hom = torch.zeros(rot_matrix_flat.shape[0], 3, 1, device=rot_matrix.device)
        rot_matrix_flat = torch.cat([rot_matrix_flat, hom], dim=-1)  # (B*J, 3, 4)
        
        # Convert to angle-axis
        angle_axis = rotation_matrix_to_angle_axis(rot_matrix_flat)  # (B*J, 3)
        
        # Reshape back to input format
        return angle_axis.reshape(B, J, 3)
    
    elif len(input_shape) == 3:  # (B, 3, 3)
        # Add homogeneous coordinate
        hom = torch.zeros(input_shape[0], 3, 1, device=rot_matrix.device)
        rot_matrix = torch.cat([rot_matrix, hom], dim=-1)  # (B, 3, 4)
        
        # Convert to angle-axis
        return rotation_matrix_to_angle_axis(rot_matrix)  # (B, 3)
    
    else:
        raise ValueError(f"Unsupported rotation matrix shape: {input_shape}")


def apply_homogeneous_transform(x, T):
        """
        Applies a 4x4 homogeneous transformation matrix `T` to a [B, N, 3] tensor `x`.
        """
        if x.dim() == 2:  # [N, 3]
            x = x.unsqueeze(0)  # make it [1, N, 3]
    
        B, N, _ = x.shape
        homo = torch.cat([x, torch.ones(B, N, 1).to(x.device)], dim=-1)  # [B, N, 4]
        return torch.matmul(homo, T[:3, :].T)  # [B, N, 3]

def rotation_matrix_from_vectors(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    v = torch.cross(a, b)
    c = torch.sum(a * b, dim=-1, keepdim=True)
    s = v.norm(dim=-1, keepdim=True)

    kmat = torch.zeros((3, 3), device=a.device)
    kmat[0, 1], kmat[0, 2] = -v[2], v[1]
    kmat[1, 0], kmat[1, 2] = v[2], -v[0]
    kmat[2, 0], kmat[2, 1] = -v[1], v[0]

    rotation = torch.eye(3, device=a.device) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-8))
    return rotation

def compute_head_roll_loss(head_rotmat, up_direction="world_z"):
    """
    Computes a loss that penalizes roll (tilt) of the head by aligning the head's
    up vector (typically Y-axis of the rotation matrix) with a given reference direction.

    Args:
        head_rotmat (torch.Tensor): Rotation matrix of head joint. Shape: [B, 3, 3]
        up_direction (str or torch.Tensor): "world_z", "view_y", or a custom tensor of shape [3]

    Returns:
        torch.Tensor: Scalar roll loss
    """
    # Extract head "up" vector: usually the 2nd column (Y-axis) of rotation matrix
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

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=35)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ.yaml")
    parser.add_argument("-multi", action="store_false")
    parser.add_argument("-novis", action="store_true")
    parser.add_argument("-front_view", "--front_view", type=int, default=2)
    parser.add_argument("-back_view", "--back_view", type=int, default=6)

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymafx/configs/pymafx_config.yaml")
    device = torch.device(f"cuda:{args.gpu_device}")

    # setting for testing on in-the-wild images
    cfg_show_list = [
        "test_gpus", [args.gpu_device], "mcube_res", 512, "clean_mesh", True, "test_mode", True,
        "batch_size", 1
    ]
 
    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()
    normal_path = "/home/ubuntu/Data/Fulden/ckpt/normal.ckpt"
    # load normal model
    normal_net = Normal.load_from_checkpoint(
        cfg=cfg, checkpoint_path=normal_path, map_location=device, strict=False
    )
    normal_net = normal_net.to(device)
    normal_net.netG.eval()
    print(
        colored(
            f"Resume Normal Estimator from {Format.start} {cfg.normal_path} {Format.end}", "green"
        )
    )

    if cfg.sapiens.use:
        sapiens_normal_net = ImageProcessor(device=device)

    # SMPLX object
    SMPLX_object = SMPLX()

    lmk_ids = np.load("/home/ubuntu/Data/Fulden/smpl_related/smplx_vertex_lmkid.npy")  # shape: [N]

    dataset_param = {
        "image_dir": args.in_dir,
        "seg_dir": args.seg_dir,
        "use_seg": True,    # w/ or w/o segmentation
        "hps_type": cfg.bni.hps_type,    # pymafx/pixie
        "vol_res": cfg.vol_res,
        "single": args.multi,
    }

    if cfg.bni.use_ifnet:
        # load IFGeo model
        ifnet = IFGeo.load_from_checkpoint(
            cfg=cfg, checkpoint_path=cfg.ifnet_path, map_location=device, strict=False
        )
        ifnet = ifnet.to(device)
        ifnet.netG.eval()

        print(colored(f"Resume IF-Net+ from {Format.start} {cfg.ifnet_path} {Format.end}", "green"))
        print(colored(f"Complete with {Format.start} IF-Nets+ (Implicit) {Format.end}", "green"))
    else:
        print(colored(f"Complete with {Format.start} SMPL-X (Explicit) {Format.end}", "green"))

    dataset = TestDataset(dataset_param, device)

    front_view = args.front_view
    back_view = args.back_view

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)

    for data in pbar:
        losses = init_loss()

        pbar.set_description(f"{data['name']}")

        # final results rendered as image (PNG)
        # 1. Render the final fitted SMPL (xxx_smpl.png)
        # 2. Render the final reconstructed clothed human (xxx_cloth.png)
        # 3. Blend the original image with predicted cloth normal (xxx_overlap.png)
        # 4. Blend the cropped image with predicted cloth normal (xxx_crop.png)

        os.makedirs(osp.join(args.out_dir, cfg.name, "png"), exist_ok=True)

        # final reconstruction meshes (OBJ)
        # 1. SMPL mesh (xxx_smpl_xx.obj)
        # 2. SMPL params (xxx_smpl.npy)
        # 3. d-BiNI surfaces (xxx_BNI.obj)
        # 4. seperate face/hand mesh (xxx_hand/face.obj)
        # 5. full shape impainted by IF-Nets+ after remeshing (xxx_IF.obj)
        # 6. sideded or occluded parts (xxx_side.obj)
        # 7. final reconstructed clothed human (xxx_full.obj)

        os.makedirs(osp.join(args.out_dir, cfg.name, "obj"), exist_ok=True)
        # Save first canonical SMPL body
        in_tensor = {}

        smpl_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_00.obj"

        # sapiens inference for current batch data

        if cfg.sapiens.use:
            sapiens_normal = sapiens_normal_net.process_image(
                Image.fromarray(
                    data["img_raw"].squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                ), "1b", cfg.sapiens.seg_model
            )
            print(colored("Estimating normal maps from input image, using Sapiens-normal", "green"))

            sapiens_normal_square_lst = []
            for idx in range(len(data["img_icon"])):
                sapiens_normal_square_lst.append(wrap(sapiens_normal, data["uncrop_param"], idx))
            sapiens_normal_square = torch.cat(sapiens_normal_square_lst)
        
        # smpl optimization
        loop_smpl = tqdm(range(args.loop_smpl))
        
        multi_view_data = []
        cam_param_path = os.path.join(args.in_dir, "cam_params", "camera_parameters.json")
        
        for data in pbar:
            # Collect multiple views
            frame_id = int(data["name"].split("_")[1])
            in_tensor[f"view_{frame_id}"] = {
                "smpl_faces": data["smpl_faces"], 
                "image": data["img_icon"].to(device), 
                "mask": data["img_mask"].to(device)
            }
            data["image"] = data["img_icon"].to(device)
            multi_view_data.append(data)

            if len(multi_view_data) < len(dataset):  
                continue

            print(colored(f"Optimizing Canonical SMPL Body using {len(multi_view_data)} views...", "blue"))

        losses = init_loss()
        front_data = multi_view_data[front_view]
        back_data = multi_view_data[back_view]
        first_data = multi_view_data[0]
        in_tensor = {
            "smpl_faces": first_data["smpl_faces"],
        }
        
        # === Collect all initial parameters across views ===
        pose_list = []
        trans_list = []
        betas_list = []
        exp_list = []
        jaw_list = []
        rotated_global_orients = []

        # --- Get first view transformation ---
        first_frame_id = int(multi_view_data[0]["name"].split("_")[1])
        cameras = load_cameras_from_json(cam_param_path, device)
        for i, cam in enumerate(cameras):
            C = -cam.R.T @ cam.t
            print(f"Camera {i} center reconstructed = {C}")
        transform_manager = CameraTransformManager(cam_param_path, target_frame=first_frame_id, device=device, debug=False, use_blender_to_cv=False)
        T_first = transform_manager.get_transform_to_target(first_frame_id)  # 4x4 matrix

        # === Load camera parameters ===
        with open(cam_param_path, "r") as f:
            cam_params = json.load(f)
 
        # --- For each view ---
        for data in multi_view_data:
            frame_id = int(data["name"].split("_")[1])

            # Get transformation of current frame
            T_view = transform_manager.get_transform_to_target(frame_id)  # 4x4 matrix
    
            # Compute relative rotation
            R_relative = T_view[:3, :3] @ T_first[:3, :3].T  # [3,3]

            # Global orient: from 6D to rotation matrix
            global_orient_mat = rot6d_to_rotmat(data["global_orient"].view(-1, 6)).squeeze(0)  # [3,3]

            # Rotate orientation
            corrected_orient_mat = R_relative @ global_orient_mat

            # Back to 6D
            corrected_orient_6d = rotmat_to_rot6d(corrected_orient_mat.unsqueeze(0)).squeeze(0)

            rotated_global_orients.append(corrected_orient_6d)

            # Collect other parameters
            pose_list.append(data["body_pose"])
            trans_list.append(data["trans"])
            betas_list.append(data["betas"])
            exp_list.append(data["exp"])
            jaw_list.append(data["jaw_pose"])

        # compute losses for each view before mean computation
        before_mean_losses = defaultdict(list)
        for view_data in multi_view_data:
            frame_id = int(view_data["name"].split("_")[1])
            view_data["T_normal_F"], view_data["T_normal_B"] = dataset.render_normal(
                view_data["smpl_verts"] * torch.tensor([-1.0, -1.0, 1.0]).to(device),
                view_data["smpl_faces"],
            )
            view_data["T_mask_F"], view_data["T_mask_B"] = dataset.render.get_image(type="mask")
            view_data["normal_F"], view_data["normal_B"] = normal_net.netG(view_data)
            smpl_arr = torch.cat([view_data["T_mask_F"], view_data["T_mask_B"]], dim=-1)
            gt_arr = view_data["img_mask"].to(device).repeat(1, 1, 2)
            diff_S = torch.abs(smpl_arr - gt_arr)
            sil_loss = diff_S.mean()
            normal_loss = (torch.abs(view_data["normal_F"]) - torch.abs(view_data["T_normal_F"])).mean()
            
            # Extract head rotation matrix from pose parameters
            view_pose_mat = rot6d_to_rotmat(view_data["body_pose"].view(-1, 6)).view(1, 21, 3, 3)
            head_rotmat = view_pose_mat[:, 14]  # Head joint is at index 14
            head_roll_loss = compute_head_roll_loss(head_rotmat, up_direction="view_y")
            
            # Ensure all tensors are on the same device
            ghum_smpl_pairs = SMPLX_object.ghum_smpl_pairs.to(device)
            landmark_data = view_data["landmark"].to(device)
            
            ghum_lmks = landmark_data[:, ghum_smpl_pairs[:, 0], :2]
            smpl_lmks = view_data["smpl_verts"][:, ghum_smpl_pairs[:, 1], :2]
            ghum_conf = landmark_data[:, ghum_smpl_pairs[:, 0], -1]
            
            valid_landmarks = ghum_conf > 0.5
            if valid_landmarks.any():
                landmark_loss = (torch.norm(ghum_lmks - smpl_lmks, dim=2) * ghum_conf * valid_landmarks.float()).sum() / valid_landmarks.float().sum()
            else:
                landmark_loss = torch.tensor(0.0, device=device)

            view_total_loss = sil_loss + 0.2 * normal_loss + 0.1 * landmark_loss + 0.3 * head_roll_loss
            view_data["total_loss"] = view_total_loss
            before_mean_losses[f'{view_data["name"]}_silhouette_iou'].append(sil_loss.item())
            before_mean_losses[f'{view_data["name"]}_landmark_error_px'].append(landmark_loss.item())
            before_mean_losses[f'{view_data["name"]}_sil_l1_loss'].append(sil_loss.item())
            before_mean_losses[f'{view_data["name"]}_normal_loss'].append(normal_loss.item())
            before_mean_losses[f'{view_data["name"]}_head_roll_loss'].append(head_roll_loss.item())
            before_mean_losses[f'{view_data["name"]}_total_loss'].append(view_total_loss.item())


        # === Compute mean of all corrected values ===
        mean_pose = torch.stack(pose_list, dim=0).mean(dim=0)

        head_indices = [14]
        body_indices = [i for i in range(21) if i not in head_indices]

        # Create separate optimizable parameters
        optimed_pose_all = mean_pose.clone().detach()
        optimed_pose_head = optimed_pose_all[:, head_indices, :].clone().detach().requires_grad_(True)
        optimed_pose_rest = optimed_pose_all[:, body_indices, :].clone().detach().requires_grad_(True)

        # Helper to combine for model input
        def combine_body_pose(rest, head, body_indices, head_indices):
            full_pose = torch.zeros((1, 21, 6), device=rest.device)
            full_pose[:, body_indices, :] = rest
            full_pose[:, head_indices, :] = head
            return full_pose
        
        # mean_jaw_pose = torch.stack(jaw_list, dim=0).mean(dim=0)

        closest_exp = exp_list[front_view].detach()
        closest_jaw = jaw_list[front_view].detach()
        #mean_jaw_pose = torch.stack(jaw_list, dim=0).mean(dim=0)

        mean_trans = torch.stack(trans_list, dim=0).mean(dim=0)
        mean_betas = torch.stack(betas_list, dim=0).mean(dim=0)
        mean_global_orient = torch.stack(rotated_global_orients, dim=0).mean(dim=0)

        # === Initialize optimization parameters ===
        optimed_pose = mean_pose
        optimed_trans = mean_trans.requires_grad_(True)
        optimed_betas = mean_betas.requires_grad_(True)
        #optimed_jaw_pose = mean_jaw_pose.clone().detach().requires_grad_(True)  
        optimed_orient = mean_global_orient.requires_grad_(True)

        # === Optimizer ===
        optimizer_smpl = torch.optim.Adam([
            optimed_pose_rest, optimed_pose_head, optimed_trans,
            optimed_betas, optimed_orient
        ], lr=1e-2, amsgrad=True)
        
        # === Scheduler ===
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl, mode="min", factor=0.5, verbose=0, min_lr=1e-5, patience=args.patience,
        )

        print(colored("✅ Canonical SMPL Initialization done (multi-view aware)", "cyan"))


        loop_smpl = tqdm(range(args.loop_smpl))
        save_vis_dir = os.path.join(args.out_dir, cfg.name, "png", "canonical_smpl_iters")
        os.makedirs(save_vis_dir, exist_ok=True)
        loss_values = []
        first_iter_losses = defaultdict(list)
        last_iter_losses = defaultdict(list)

        # Optimization loop
        for i in loop_smpl:
            optimizer_smpl.zero_grad()

            N_body, N_pose = optimed_pose.shape[:2]
            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).view(N_body, 1, 3, 3)
            # optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).view(N_body, N_pose, 3, 3)
            # Recombine full pose for the model
            optimed_pose_combined = combine_body_pose(optimed_pose_rest, optimed_pose_head, body_indices, head_indices)
            optimed_pose_mat = rot6d_to_rotmat(optimed_pose_combined.view(-1, 6)).view(N_body, N_pose, 3, 3)

            smpl_verts, smpl_landmarks, smpl_joints = dataset.smpl_model(
            shape_params=optimed_betas,
            expression_params=closest_exp,
            body_pose=optimed_pose_mat,
            global_pose=optimed_orient_mat,
            jaw_pose=closest_jaw,
            left_hand_pose=tensor2variable(first_data["left_hand_pose"], device),
            right_hand_pose=tensor2variable(first_data["right_hand_pose"], device),
            )

            smpl_verts = (smpl_verts + optimed_trans) * first_data["scale"]
            smpl_joints = (smpl_joints + optimed_trans) * first_data["scale"] * torch.tensor([
                1.0, 1.0, -1.0
            ]).to(device)
            
            frame_to_camera_dir = {}
            total_loss = 0.0

            for view_data in multi_view_data:
                frame_id = int(view_data["name"].split("_")[1])
                T_frame_to_target = transform_manager.get_transform_to_target(frame_id)
                T_frame_to_target[:3, 3] = 0.0

                view_data["smpl_verts"] = apply_homogeneous_transform(smpl_verts, (T_frame_to_target).T)
                view_data["smpl_joints"] = apply_homogeneous_transform(smpl_joints, (T_frame_to_target).T)

                # From SMPL-X rotmat output: [B, J, 3, 3]
                head_rotmat = optimed_pose_mat[:, 14] 

                # Compute roll loss
                head_roll_loss = compute_head_roll_loss(head_rotmat, up_direction="view_y")

                # Landmark processing
                smpl_joints_3d = (view_data["smpl_joints"][:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0) * 0.5
                smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]
                ghum_lmks = view_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                ghum_conf = view_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)

                # A. Apply vertical offset (before loss calculation)
                # This shifts SMPL landmarks down to better align with GT
                #smpl_lmks[:, :, 1] = smpl_lmks[:, :, 1] + 0.03  # Shift down by 2% of image height

                # Save landmark visualization
                """
                if frame_id == 0 or frame_id == 1 or frame_id == 6 or frame_id ==7:
                    os.makedirs(os.path.join(save_vis_dir, f"landmarks_{frame_id}"), exist_ok=True)
                    save_path = os.path.join(save_vis_dir, f"landmarks_{frame_id}/landmarks_view_{frame_id}_iter_{i:03d}.png")
                    visualize_landmarks_detailed(
                        view_data["img_icon"].to(device),
                        view_data["smpl_verts"],  # or smpl_verts if already in view space
                        None,           # faces (optional, not used in this scatter version)
                        smpl_lmks,
                        ghum_lmks,
                        ghum_conf,
                        save_path
                    )
                """
                
                # Compute landmark loss with confidence weighting
                valid_landmarks = ghum_conf > 0.5
                if valid_landmarks.any():
                    landmark_loss = (torch.norm(ghum_lmks - smpl_lmks, dim=2) * ghum_conf * valid_landmarks.float()).sum() / valid_landmarks.float().sum()
                else:
                    landmark_loss = torch.tensor(0.0, device=device)


                # Render and compute loss
                view_data["T_normal_F"], view_data["T_normal_B"] = dataset.render_normal(
                    view_data["smpl_verts"] * torch.tensor([-1.0, -1.0, 1.0]).to(device),
                    view_data["smpl_faces"],
                )
                view_data["T_mask_F"], view_data["T_mask_B"] = dataset.render.get_image(type="mask")

                view_data["normal_F"], view_data["normal_B"] = normal_net.netG(view_data)
                smpl_arr = torch.cat([view_data["T_mask_F"], view_data["T_mask_B"]], dim=-1)
                gt_arr = view_data["img_mask"].to(device).repeat(1, 1, 2)
                diff_S = torch.abs(smpl_arr - gt_arr)

                sil_loss = diff_S.mean()
                normal_loss = (torch.abs(view_data["normal_F"]) - torch.abs(view_data["T_normal_F"])).mean()
                #normal_loss = torch.abs(view_data["T_normal_F"]).mean()

                ### METRICS: Capture individual loss components for this view
                if i == 0:
                    # For Silhouette IoU (a better metric than L1 for tables)
                    rendered_mask = (view_data["T_mask_F"] > 0).bool()
                    gt_mask = (view_data["img_mask"].to(device) > 0.5).bool()
                    intersection = torch.sum(rendered_mask & gt_mask).float()
                    union = torch.sum(rendered_mask | gt_mask).float()
                    iou = intersection / (union + 1e-8)
                    first_iter_losses[f'{view_data["name"]}_silhouette_iou'].append(iou.item())
                    first_iter_losses[f'{view_data["name"]}_landmark_error_px'].append(landmark_loss.item())
                    first_iter_losses[f'{view_data["name"]}_sil_l1_loss'].append(sil_loss.item())
                    first_iter_losses[f'{view_data["name"]}_normal_loss'].append(normal_loss.item())
                    first_iter_losses[f'{view_data["name"]}_head_roll_loss'].append(head_roll_loss.item())
                    first_iter_losses[f'{view_data["name"]}_total_loss'].append(view_total_loss.item())

                if i == args.loop_smpl - 1:
                    # For Silhouette IoU (a better metric than L1 for tables)
                    rendered_mask = (view_data["T_mask_F"] > 0).bool()
                    gt_mask = (view_data["img_mask"].to(device) > 0.5).bool()
                    intersection = torch.sum(rendered_mask & gt_mask).float()
                    union = torch.sum(rendered_mask | gt_mask).float()
                    iou = intersection / (union + 1e-8)
                    last_iter_losses[f'{view_data["name"]}_silhouette_iou'].append(iou.item())
                    last_iter_losses[f'{view_data["name"]}_landmark_error_px'].append(landmark_loss.item())
                    last_iter_losses[f'{view_data["name"]}_sil_l1_loss'].append(sil_loss.item())
                    last_iter_losses[f'{view_data["name"]}_normal_loss'].append(normal_loss.item())
                    last_iter_losses[f'{view_data["name"]}_head_roll_loss'].append(head_roll_loss.item())
                    last_iter_losses[f'{view_data["name"]}_total_loss'].append(view_total_loss.item())

                # Combine all losses with appropriate weights
                view_total_loss = sil_loss + 0.2 * normal_loss + 0.1 * landmark_loss + 0.3 * head_roll_loss
                total_loss += view_total_loss

            total_loss /= len(multi_view_data)
            loss_values.append(total_loss.item())
            total_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(total_loss)
            
            # === Visualization (optional) ===
            with torch.no_grad():
                smpl_silhouette_images, gt_silhouette_images, diff_silhouette_images = [], [], []

                for view_data in multi_view_data:
                    frame_id = int(view_data["name"].split("_")[1])

                    smpl_mask_front = view_data["T_mask_F"]
                    gt_mask_front = view_data["img_mask"].to(device)

                    diff_S_front = torch.abs(smpl_mask_front - gt_mask_front)

                    smpl_silhouette_images.append(smpl_mask_front.repeat(1, 3, 1, 1))
                    gt_silhouette_images.append(gt_mask_front.unsqueeze(1).repeat(1, 3, 1, 1))
                    diff_silhouette_images.append(diff_S_front.repeat(1, 3, 1, 1))

                combined_panel = torch.cat([
                    torch.cat(smpl_silhouette_images, dim=3),
                    torch.cat(gt_silhouette_images, dim=3),
                    torch.cat(diff_silhouette_images, dim=3),
                ], dim=2)

                save_path_panel = os.path.join(save_vis_dir, f"panel_iter_{i:03d}.png")
                torchvision.utils.save_image(combined_panel, save_path_panel)

        # === After optimization ===
        # === Plot and save loss curve ===
        plt.figure(figsize=(6, 4))
        plt.plot(loss_values, color='blue', linewidth=2)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("SMPL Optimization Loss", fontsize=14)
        plt.grid(True)
        plt.tight_layout()

        # Save to file
        loss_curve_path = os.path.join(args.out_dir, cfg.name, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()

        print(colored(f"📉 Saved loss curve to {loss_curve_path}", "cyan"))
        

        # ==============================================================================
        # FINAL STEP: PROCESS COLLECTED LOSSES AND GENERATE THE COMPARISON TABLE
        # ==============================================================================
        print(colored("\n[3/3] Generating final quantitative comparison table...", "cyan"))

        # This list will hold the structured data for our DataFrame
        data_for_df = []

        # Helper function to parse the dictionaries and populate our list
        def process_metrics_dict(stage_name, metrics_dict, multi_view_data):
            """
            Parses a loss dictionary and adds structured rows to the data_for_df list.
            """
            for view_data in multi_view_data:
                view_name = view_data["name"]
                frame_id = int(view_name.split("_")[1])

                # Safely get values from the dictionary, defaulting to 0 if a key is missing
                # The [0] is because your values are stored in a list
                iou = metrics_dict.get(f'{view_name}_silhouette_iou', [0])[0]
                lmk_px = metrics_dict.get(f'{view_name}_landmark_error_px', [0])[0]
                normal_loss = metrics_dict.get(f'{view_name}_normal_loss', [0])[0]
                total_loss = metrics_dict.get(f'{view_name}_total_loss', [0])[0]

                data_for_df.append({
                    'Stage': stage_name,
                    'View': frame_id,
                    'Silhouette IoU ↑': iou,
                    'Landmark Error (px) ↓': lmk_px,
                    'Normal Loss ↓': normal_loss,
                    'Total Loss ↓': total_loss,
                })

        # Process each of the three stages
        process_metrics_dict('1. Initial Per-View', before_mean_losses, multi_view_data)
        process_metrics_dict('2. After Mean (Iter 0)', first_iter_losses, multi_view_data)
        process_metrics_dict('3. Final Optimized', last_iter_losses, multi_view_data)

        # Create a pandas DataFrame from our list of structured data
        df = pd.DataFrame(data_for_df)

        # Create a pivot table for a clean, multi-level indexed presentation
        # This is the format you wanted
        df_pivot = df.pivot_table(
            index=['Stage', 'View'],
            values=['Silhouette IoU ↑', 'Landmark Error (px) ↓', 'Normal Loss ↓', 'Total Loss ↓']
        )

        # Ensure the columns are in a logical order
        df_pivot = df_pivot[['Silhouette IoU ↑', 'Landmark Error (px) ↓', 'Normal Loss ↓', 'Total Loss ↓']]

        # Sort the table by Stage (as a categorical type to maintain order) and then by View
        df_pivot.index = df_pivot.index.set_levels(
            pd.Categorical(df_pivot.index.levels[0], categories=['1. Initial Per-View', '2. After Mean (Iter 0)', '3. Final Optimized'], ordered=True),
            level=0
        )
        df_pivot.sort_index(inplace=True)


        # Print the final, formatted table to the console
        print("\n" + "="*80)
        print(f" SMPL Optimization Quantitative Analysis: {multi_view_data[0]['name'].split('_')[0]}")
        print("="*80)
        print(df_pivot.to_string(float_format="%.5f"))
        print("="*80)

        # Save the table to a CSV file for easy use in your thesis
        csv_path = os.path.join(args.out_dir, cfg.name, "obj", f"{multi_view_data[0]['name'].split('_')[0]}_smpl_optimization_metrics.csv")
        df_pivot.to_csv(csv_path)
        print(colored(f"✅ Saved detailed metrics table to {csv_path}", "green"))


        save_obj_path = os.path.join(args.out_dir, cfg.name, "obj", "final_smpl.obj")
        final_smpl_obj = trimesh.Trimesh(
            smpl_verts[0].detach().cpu() * torch.tensor([1.0, -1.0, 1.0]),
            in_tensor["smpl_faces"][0].cpu()[:, [0, 2, 1]],
            process=False,
            maintains_order=True,
        )
        final_smpl_obj.export(save_obj_path)

        # Save SMPL parameters
        smpl_info = {
            "betas": optimed_betas.detach().cpu(),
            "body_pose": convert_rot_matrix_to_angle_axis(optimed_pose_mat.detach()).cpu(),
            "global_orient": convert_rot_matrix_to_angle_axis(optimed_orient_mat.detach()).cpu(),
            "transl": optimed_trans.detach().cpu(),
            "expression": closest_exp.detach().cpu(),
            "jaw_pose": convert_rot_matrix_to_angle_axis(closest_jaw.detach()).cpu(),
            "left_hand_pose": convert_rot_matrix_to_angle_axis(first_data["left_hand_pose"].detach()).cpu(),
            "right_hand_pose": convert_rot_matrix_to_angle_axis(first_data["right_hand_pose"].detach()).cpu(),
            "scale": first_data["scale"].detach().cpu(),
            "smpl_verts": smpl_verts.detach().cpu(),
            "smpl_joints": smpl_joints.detach().cpu(),
            "smpl_landmarks": smpl_landmarks.detach().cpu(),
            "smpl_lmks": smpl_lmks.detach().cpu(),
            "ghum_lmks": ghum_lmks.detach().cpu(),
            "ghum_conf": ghum_conf.detach().cpu()
        }
        np.save(save_obj_path.replace(".obj", ".npy"), smpl_info, allow_pickle=True)
        print(colored(f"✅ Saved SMPL parameters to {save_obj_path.replace('.obj', '.npy')}", "green"))

        # Save SMPL for all views
        smpl_obj_lst = []
        normal_list = []
        mask_list = []
        depth_list = []

        # Get canonical image dimensions from first view
        H, W = multi_view_data[0]["img_icon"].shape[-2:]

        for view_data in multi_view_data:
            frame_id = int(view_data["name"].split("_")[1])

            # Add view-specific data to in_tensor
            view_key = f"view_{frame_id}"

            img_norm_path_f = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_normalsmpl_{frame_id}.png")
            torchvision.utils.save_image((view_data["T_normal_F"].detach().cpu()), img_norm_path_f)

            # Get input RGB image
            rgb_image = view_data["img_icon"].squeeze(0).cpu().numpy()  # (H, W, 3)
            
            # Get segmentation mask
            mask = view_data["img_mask"].squeeze(0).cpu().numpy()  # (H, W)
            img_mask_path_f = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_mask_{frame_id}.png")
            # Convert mask to tensor and save
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
            torchvision.utils.save_image(mask_tensor, img_mask_path_f)

            img_norm_path_f = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_normal_{frame_id}.png")
            torchvision.utils.save_image((view_data["normal_F"].detach().cpu()), img_norm_path_f) 
            normal_map = view_data["normal_F"].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

            # Get SMPL depth as proxy
            view_data["depth_F"], view_data["depth_B"] = dataset.render_depth(
                view_data["smpl_verts"] * torch.tensor([-1.0, -1.0, 1.0]).to(device),
                view_data["smpl_faces"]
            )
            depth_map = view_data["depth_F"].squeeze(0).detach().cpu().numpy()  # (H, W)
            
            # Normalize depth map to [0, 1] range
            valid_mask = depth_map > -0.5
            valid_depth = depth_map[valid_mask]

            if valid_depth.size > 0:
                d_min = valid_depth.min()
                d_max = valid_depth.max()
                print(f"View {frame_id} depth range: {d_min:.4f} - {d_max:.4f}")

                # Normalize to [0, 1]
                depth_norm = np.zeros_like(depth_map)
                depth_norm[valid_mask] = (depth_map[valid_mask] - d_min) / (d_max - d_min + 1e-6)
                depth_map = depth_norm
            else:
                print(f"Warning: No valid depth pixels found for view {frame_id}!")
                depth_map = np.zeros_like(depth_map)

            # Save depth map as PNG
            depth_img_path = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_depth_{frame_id}.png")
            depth_vis = view_data["depth_F"].detach().cpu().squeeze(0)  # Remove batch dimension
            depth_vis = torch.tensor(depth_map).unsqueeze(0)  # Convert from cm to m and add channel dimension
            depth_np = depth_vis.detach().cpu().numpy()[0] 
            # Save normalized depth map
            if valid_depth.size > 0:
                d_min = valid_depth.min()
                d_max = valid_depth.max()
                print(f"Depth range: {d_min:.4f} - {d_max:.4f}")

                # Normalize to [0, 255]
                depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-6)
                depth_8bit = (depth_norm * 255).astype(np.uint8)
            else:
                print("Warning: No valid depth pixels found!")
                depth_8bit = np.zeros_like(depth_np, dtype=np.uint8)

            # Save
            cv2.imwrite(depth_img_path, depth_8bit)


            # Create and save mesh for this view
            view_obj = trimesh.Trimesh(
                view_data["smpl_verts"][0].detach().cpu().squeeze(0) * torch.tensor([1.0, -1.0, 1.0]),
                view_data["smpl_faces"][0].cpu()[:, [0, 2, 1]],
                process=False,
                maintains_order=True,
            )
            
            # Save individual view mesh
            view_obj_path = os.path.join(args.out_dir, cfg.name, "obj", f"canonical_smpl_view_{frame_id}.obj")
            view_obj.export(view_obj_path)
            
            # Add to list for downstream processing
            smpl_obj_lst.append(view_obj)
            
            # Store in lists - ensure mask is boolean and normal map is properly masked
            # Normalize normal map to [-1, 1] range
            normal_map = normal_map * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
            normal_list.append(normal_map)  # (H, W, 3)
            mask_list.append(mask.astype(bool))  # (H, W) - convert to boolean
            depth_list.append(depth_map)  # (H, W) - normalized to [0,1]
            
            in_tensor.update({
                f"view_{frame_id}": view_data
            })
        sp_out_dir = os.path.join(args.out_dir, cfg.name, "own_objects_normals")

        # Convert camera JSON
        # pose_all, intrinsic_all = convert_camera_json_to_npz(cam_param_path)
        # np.savez(os.path.join(sp_out_dir, "cameras_sphere.npz"), pose_all=pose_all, intrinsic_all=intrinsic_all)

        # Save lists as numpy arrays
        normal_array = np.stack(normal_list, axis=0)  # (N_views, H, W, 3)
        mask_array = np.stack(mask_list, axis=0)  # (N_views, H, W)
        depth_array = np.stack(depth_list, axis=0)  # (N_views, H, W)

        # Add arrays to in_tensor
        in_tensor.update({
            "normal_array": normal_array,
            "mask_array": mask_array,
            "depth_array": depth_array
        })

        print(colored(f"✅ Saved clothed image maps and SMPL proxy depth for all {len(multi_view_data)} views", "green"))

        
        # Clean up
        del optimizer_smpl
        del optimed_betas
        del optimed_orient
        del optimed_pose
        del optimed_trans
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------------------------------------------------------
        # clothing refinement        
        
        
        in_tensor["BNI_verts"] = []
        in_tensor["BNI_faces"] = []
        in_tensor["body_verts"] = []
        in_tensor["body_faces"] = []
      

        final_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_full.obj"

        side_mesh = smpl_obj_lst[back_view].copy()
        face_mesh = smpl_obj_lst[front_view].copy()
        hand_mesh = smpl_obj_lst[front_view].copy()
        smplx_mesh = smpl_obj_lst[back_view].copy()

        bni_mesh_list = []
        bni_object_list = []
        idx = 0
        for i in (front_view, back_view):
            frame_id = int(multi_view_data[i]["name"].split("_")[1])
            T_view_i = transform_manager.get_transform_to_target(frame_id)
            T_view_i[:3, 3] = 0.0
            
            view_data = in_tensor[f"view_{i}"]
            
            BNI_dict = save_normal_tensor(
                view_data,
                idx,
                osp.join(args.out_dir, cfg.name, f"BNI/{data['name']}"),
                cfg.bni.thickness,
            )
            
            # BNI process
            BNI_object = BNI(
                dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
                name=view_data["name"],
                BNI_dict=BNI_dict,
                cfg=cfg.bni,
                device=device
            )

            BNI_object.extract_surface()
            # Convert T_view_i from PyTorch tensor to NumPy array
            R = T_view_i[:3, :3].cpu().numpy()
            R_4x4 = np.eye(4)
            R_4x4[:3, :3] = R
            
            # Apply rotation to all BNI meshes
            BNI_object.F_trimesh.apply_transform(R_4x4)
            
            # Apply rotation to depth maps
            #BNI_object.F_depth = torch.matmul(torch.from_numpy(R).float(), BNI_object.F_depth.unsqueeze(-1)).squeeze(-1)
            #BNI_object.B_depth = torch.matmul(torch.from_numpy(R).float(), BNI_object.B_depth.unsqueeze(-1)).squeeze(-1)
            
            # Save front mesh for this view
            front_mesh_path = os.path.join(args.out_dir, cfg.name, "obj", f"bni_view_{i}.obj")
            BNI_object.F_trimesh.export(front_mesh_path)
            print(colored(f"✅ Saved front mesh for view {i} to {front_mesh_path}", "green"))
            bni_mesh_list.append(BNI_object.F_trimesh)
            bni_object_list.append(BNI_object)
        
        fused_bni_surface = trimesh.util.concatenate(bni_mesh_list)
        BNI_surface = trimesh2meshes(fused_bni_surface).to(device)
        fused_bni_surface.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_fused_BNI.obj")
        bni_object_list[1].B_depth.unsqueeze(0)

        T_view_side = transform_manager.get_transform_to_target(back_view)
        R_4x4_side = np.eye(4)
        R_4x4_side[:3, :3] = T_view_side[:3, :3].cpu().numpy()
        side_mesh.apply_transform(R_4x4_side)

        if cfg.bni.use_ifnet:

            side_mesh_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_IF.obj"
            side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)

            # mesh completion via IF-net
            in_tensor[f"view_{front_view}"].update(
                dataset.depth_to_voxel({
                    "depth_F": bni_object_list[0].F_depth.unsqueeze(0), 
                    "depth_B": bni_object_list[1].F_depth.unsqueeze(0)
                })
            )

            occupancies = VoxelGrid.from_mesh(side_mesh, cfg.vol_res, loc=[
                0,
            ] * 3, scale=2.0).data.transpose(2, 1, 0)
            occupancies = np.flip(occupancies, axis=1)

            in_tensor[f"view_{front_view}"]["body_voxels"] = torch.tensor(occupancies.copy()
                                                    ).float().unsqueeze(0).to(device)

            with torch.no_grad():
                sdf = ifnet.reconEngine(netG=ifnet.netG, batch=in_tensor[f"view_{front_view}"])
                verts_IF, faces_IF = ifnet.reconEngine.export_mesh(sdf)

            if ifnet.clean_mesh_flag:
                verts_IF, faces_IF = clean_mesh(verts_IF, faces_IF)

            side_mesh = trimesh.Trimesh(verts_IF, faces_IF)
            side_mesh = remesh_laplacian(side_mesh, side_mesh_path)

        else:
            side_mesh = apply_vertex_mask(
                side_mesh,
                (
                    SMPLX_object.front_flame_vertex_mask + SMPLX_object.smplx_mano_vertex_mask +
                    SMPLX_object.eyeball_vertex_mask
                ).eq(0).float(),
            )

            #register side_mesh to BNI surfaces
            side_mesh = Meshes(
                verts=[torch.tensor(side_mesh.vertices).float()],
                faces=[torch.tensor(side_mesh.faces).long()],
            ).to(device)
            sm = SubdivideMeshes(side_mesh)

            side_mesh = register(fused_bni_surface, sm(side_mesh), device)

        side_verts = torch.tensor(side_mesh.vertices).float().to(device)
        side_faces = torch.tensor(side_mesh.faces).long().to(device)
        side_mesh = trimesh.Trimesh(side_verts.cpu().numpy(), side_faces.cpu().numpy())
        side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_side_before.obj")

        # Possion Fusion between SMPLX and BNI
        # 1. keep the faces invisible to front+back cameras
        # 2. keep the front-FLAME+MANO faces
        # 3. remove eyeball faces

        # export intermediate meshes
        #BNI_object.F_B_trimesh.export(
        #    f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj"
        #)
        full_lst = []

        if "face" in cfg.bni.use_smpl:

            # only face
            face_mesh = apply_vertex_mask(face_mesh, SMPLX_object.front_flame_vertex_mask)

            if not face_mesh.is_empty:
                face_mesh.vertices = face_mesh.vertices - np.array([0, 0, cfg.bni.thickness])

                # remove face neighbor triangles
                fused_bni_surface = part_removal(
                    fused_bni_surface,
                    face_mesh,
                    cfg.bni.face_thres,
                    device,
                    smplx_mesh,
                    region="face"
                )
                side_mesh = part_removal(
                    side_mesh, face_mesh, cfg.bni.face_thres, device, smplx_mesh, region="face"
                )
                face_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_face.obj")
                full_lst += [face_mesh]

        if "hand" in cfg.bni.use_smpl:
            hand_mask = torch.zeros(SMPLX_object.smplx_verts.shape[0], )

            if data['hands_visibility'][idx][0]:

                mano_left_vid = np.unique(
                    np.concatenate([
                        SMPLX_object.smplx_vert_seg["leftHand"],
                        SMPLX_object.smplx_vert_seg["leftHandIndex1"],
                    ])
                )

                hand_mask.index_fill_(0, torch.tensor(mano_left_vid), 1.0)

            if data['hands_visibility'][idx][1]:

                mano_right_vid = np.unique(
                    np.concatenate([
                        SMPLX_object.smplx_vert_seg["rightHand"],
                        SMPLX_object.smplx_vert_seg["rightHandIndex1"],
                    ])
                )

                hand_mask.index_fill_(0, torch.tensor(mano_right_vid), 1.0)

            # only hands
            hand_mesh = apply_vertex_mask(hand_mesh, hand_mask)

            if not hand_mesh.is_empty:
                # remove hand neighbor triangles
                fused_bni_surface = part_removal(
                    fused_bni_surface,
                    hand_mesh,
                    cfg.bni.hand_thres,
                    device,
                    smplx_mesh,
                    region="hand"
                )
                side_mesh = part_removal(
                    side_mesh, hand_mesh, cfg.bni.hand_thres, device, smplx_mesh, region="hand"
                )
                hand_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_hand.obj")
                full_lst += [hand_mesh]

        full_lst += [fused_bni_surface]

        # initial side_mesh could be SMPLX or IF-net
        side_mesh = part_removal(
            side_mesh, sum(full_lst), 2e-2, device, smplx_mesh, region="", clean=False
        )

        full_lst += [side_mesh]

        # # export intermediate meshes
        #BNI_object.F_B_trimesh.export(
        #    f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj"
        #)
        side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_side.obj")

        if cfg.bni.use_poisson:
            recon_obj = poisson(
                sum(full_lst),
                final_path,
                cfg.bni.poisson_depth,
            )
            print(
                colored(
                    f"\n Poisson completion to {Format.start} {final_path} {Format.end}",
                    "yellow"
                )
            )
        else:
            recon_obj = sum(full_lst)
            recon_obj.export(final_path)

        
        mesh_pr = trimesh2meshes(recon_obj).to(device)

        initial_verts = mesh_pr.verts_padded().clone().detach()
        local_affine_model = LocalAffine(
            initial_verts.shape[1],
            initial_verts.shape[0],
            mesh_pr.edges_packed()
        ).to(device)

        # Setup optimizer and scheduler
        optimizer_cloth = torch.optim.Adam(local_affine_model.parameters(), lr=1e-4)
        scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cloth, mode="min", factor=0.5, patience=10, verbose=True)

        fixed_vertex_mask = get_fixed_vertex_mask(mesh_pr, front_view, back_view, transform_manager, device).squeeze()
        #print(f"Shape of fixed_vertex_mask after squeeze: {fixed_vertex_mask.shape}")


        # Base weights for our loss terms
        loss_weights = {
            "normal": 15.0,
            "sil": 7.5,
            "stiff": 15.0,
            "rigid": 15.0,
            "anchor": 1000.0,  # High weight to strongly enforce the anchor
            "laplacian": 750.0
        }

        batch_size, num_points, _, _ = local_affine_model.A.shape
        identity_matrix = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1)

        # --- 2. Improved Loss Setup: Define weights and annealing parameters ---
        num_cloth_iter = 1 # Total number of iterations

        # Annealing: We will start with a lower data-term weight and increase it over time.
        # This helps the mesh settle into a good general shape before fitting fine details.
        annealing_start_factor = 0.1 # Start with 10% of the data term weights

        # Optimization loop
        pbar = tqdm(range(num_cloth_iter), desc="Cloth Refinement (Anchored)")
        for i in pbar:
            optimizer_cloth.zero_grad()

            anneal_factor = min(1.0, annealing_start_factor + (1.0 - annealing_start_factor) * (i / (num_cloth_iter / 2.0)))
    
            current_weights = {
                "normal": loss_weights["normal"] * anneal_factor,
                "sil": loss_weights["sil"] * anneal_factor,
                "stiff": loss_weights["stiff"],
                "rigid": loss_weights["rigid"],
                "anchor": loss_weights["anchor"],
                "laplacian": loss_weights["laplacian"]
            }

            # Forward pass: Deform the initial vertices
            deformed_verts, stiffness, rigid = local_affine_model(initial_verts)
            deformed_mesh = Meshes(verts=deformed_verts, faces=mesh_pr.faces_padded())

            # Initialize lists for side-view losses
            side_normal_losses = []
            side_silhouette_losses = []
            # --- Compute DATA loss ONLY on side views ---
            for view_data in multi_view_data:
                frame_id = int(view_data['name'].split("_")[1])
                
                # Skip front and back views for data loss calculation
                if frame_id == front_view or frame_id == back_view:
                    continue
                
                # --- Calculate losses for the side view ---
                T_view = transform_manager.get_transform_to_target(frame_id).detach()
                verts_in_view = apply_homogeneous_transform(deformed_verts, T_view.T)
                P_norm, _ = dataset.render_normal(verts_in_view * torch.tensor([1.0, 1.0, 1.0]).to(device), deformed_mesh.faces_padded())
                P_mask, _ = dataset.render.get_image(type="mask")
                
                target_normal = view_data["normal_F"].detach()
                target_mask = view_data["img_mask"].to(device, dtype=torch.float32)

                normal_loss = (torch.abs(P_norm - target_normal) * target_mask).sum() / target_mask.sum().clamp(min=1.0)
                sil_loss = torch.abs(P_mask - target_mask).mean()

                side_normal_losses.append(normal_loss)
                side_silhouette_losses.append(sil_loss)

                # Save normal map comparisons for visualization
                if (i + 1) % 10 == 0 and i > 0:
                    # Ensure P_norm and target_normal are in the right format (1,3,H,W)
                    if len(P_norm.shape) == 3:
                        P_norm = P_norm.unsqueeze(0)
                    if len(target_normal.shape) == 3:
                        target_normal = target_normal.unsqueeze(0)
                    
                    # Save the normal comparison
                    normal_comparison_path = f"{args.out_dir}/{cfg.name}/normals/{data['name']}_iter_{i+1}_normals_{frame_id}.png"
                    os.makedirs(os.path.dirname(normal_comparison_path), exist_ok=True)
                    save_normal_comparison(P_norm, target_normal, normal_comparison_path)
                    print(colored(f"\nSaved normal comparison at iteration {i+1}", "cyan"))

            laplacian_loss = mesh_laplacian_smoothing(deformed_mesh, method="uniform")

            # Aggregate losses
            # Handle the case where there are no side views to prevent errors
            avg_normal_loss = torch.stack(side_normal_losses).mean() if side_normal_losses else torch.tensor(0.0, device=device)
            avg_sil_loss = torch.stack(side_silhouette_losses).mean() if side_silhouette_losses else torch.tensor(0.0, device=device)
            
            avg_stiffness_loss = stiffness.mean()
            avg_rigid_loss = rigid.mean()

            # --- NEW: Compute ANCHOR loss on fixed vertices ---
            # Get the affine transformations for the fixed vertices
            # Get the affine transformations for ALL vertices first
            A_all = local_affine_model.A[0].squeeze(0)  # Shape: (N, 3, 3) - Select the first batch item
            b_all = local_affine_model.b[0].squeeze(0)  # Shape: (N, 3, 1)

            # Now, apply the 1D boolean mask to get the fixed vertices
            A_fixed = A_all[fixed_vertex_mask]
            b_fixed = b_all[fixed_vertex_mask]

            # Also get the target identity matrix for the corresponding fixed vertices
            identity_matrix_all = identity_matrix[0].squeeze(0) # Shape (N, 3, 3)
            identity_fixed = identity_matrix_all[fixed_vertex_mask]

            # Penalize deviation from identity transformation (A=I, b=0)
            anchor_loss_A = torch.nn.functional.mse_loss(A_fixed, identity_fixed)
            anchor_loss_b = torch.mean(b_fixed**2)
            anchor_loss = anchor_loss_A + anchor_loss_b

            # Compute total weighted loss
            # Compute total weighted loss
            total_loss = (
                avg_normal_loss * current_weights["normal"] +
                avg_sil_loss * current_weights["sil"] +
                avg_stiffness_loss * current_weights["stiff"] +
                avg_rigid_loss * current_weights["rigid"] +
                anchor_loss * current_weights["anchor"] +
                laplacian_loss * current_weights["laplacian"]
            )

            # Backward pass and optimization
            total_loss.backward()
            optimizer_cloth.step()
            scheduler_cloth.step(total_loss)

            # Update progress bar
            pbar.set_description(
                f"Refining | Total: {total_loss.item():.4f} | "
                f"SideNormal: {avg_normal_loss.item():.4f} | SideSil: {avg_sil_loss.item():.4f} | "
                f"Anchor: {anchor_loss.item():.6f}"
            )

            # Save intermediate meshes. Note that we use `deformed_verts`, not `current_verts`.
            if (i + 1) % 10 == 0 and i > 0:
                intermediate_mesh = trimesh.Trimesh(
                    vertices=deformed_verts.detach().cpu().numpy()[0],
                    faces=deformed_mesh.faces_padded().detach().cpu().numpy()[0]
                )
                iteration_num = i + 1
                intermediate_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_iter_{iteration_num}.obj")
                print(colored(f"\nSaved intermediate mesh at iteration {iteration_num}", "cyan"))

            # Clear GPU memory
            torch.cuda.empty_cache()

        # Final output: Save the final deformed mesh
        print("Optimization complete.")
        final_mesh = trimesh.Trimesh(
            vertices=deformed_verts.detach().cpu().numpy()[0],
            faces=deformed_mesh.faces_padded().detach().cpu().numpy()[0]
        )
        final_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_final_refined_mesh.obj")
        print(colored(f"✅ Saved final refined mesh to {args.out_dir}/{cfg.name}/obj/{data['name']}_final_refined_mesh.obj", "green"))

        final_path = f"{args.out_dir}/{cfg.name}/obj/"

        # New class-based approach
        generator = TextureGenerator(texture_size=2048, debug=True)
        textured_mesh = generator(final_mesh, multi_view_data, transform_manager, final_path, "textured_mesh")


        """
        # 3. Texture Generation
        
        view_indices = {
            'front': front_view, # Corresponds to multi_view_data[2]
            'back': back_view,  # Corresponds to multi_view_data[6]
            'left': 0,  # Corresponds to multi_view_data[0]
            'right': 4  # Corresponds to multi_view_data[4]
        }

        if cfg.bni.texture_src == 'image':
            final_colors = semantic_texture_from_multi_view(
                final_mesh, 
                multi_view_data, 
                transform_manager, 
                view_indices, 
                device
            )
            final_mesh.visual.vertex_colors = final_colors

            # Export the final, textured mesh
            textured_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_final_semantic_textured.obj"
            final_mesh.export(textured_path)
            print(colored(f"✅ Saved final semantic textured mesh to {textured_path}", "green"))
        """
        canonical_verts = torch.tensor(final_mesh.vertices, dtype=torch.float32, device=device)
        canonical_faces = torch.tensor(final_mesh.faces, dtype=torch.long, device=device)
        transform_manager_final = CameraTransformManager(cam_param_path, target_frame=back_view, device=device, debug=False, use_blender_to_cv=False)
        T_canon_to_front_view = transform_manager_final.get_transform_to_target(0).detach()
        verts_in_front_view = apply_homogeneous_transform(canonical_verts, T_canon_to_front_view.T)

        (front_xy, front_z) = verts_in_front_view.split([2, 1], dim=-1)
        front_visibility = get_visibility(front_xy, front_z, canonical_faces[:, [0, 2, 1]]).flatten()

        transform_manager_final = CameraTransformManager(cam_param_path, target_frame=front_view, device=device, debug=False, use_blender_to_cv=False)
        T_canon_to_back_view = transform_manager_final.get_transform_to_target(0).detach()
        verts_in_back_view = apply_homogeneous_transform(canonical_verts, T_canon_to_back_view.T)

        (back_xy, back_z) = verts_in_back_view.split([2, 1], dim=-1)
        back_visibility = get_visibility(back_xy, back_z, canonical_faces[:, [0, 2, 1]]).flatten()

        T_canon_to_left_view = transform_manager_final.get_transform_to_target(0).detach()
        verts_in_left_view = apply_homogeneous_transform(canonical_verts, T_canon_to_left_view.T)

        (left_xy, left_z) = verts_in_left_view.split([2, 1], dim=-1)
        left_visibility = get_visibility(left_xy, left_z, canonical_faces[:, [0, 2, 1]]).flatten()

        T_canon_to_right_view = transform_manager_final.get_transform_to_target(0).detach()
        verts_in_right_view = apply_homogeneous_transform(canonical_verts, T_canon_to_right_view.T)

        (right_xy, right_z) = verts_in_right_view.split([2, 1], dim=-1)
        right_visibility = get_visibility(right_xy, right_z, canonical_faces[:, [0, 2, 1]]).flatten()

        if cfg.bni.texture_src == 'image':

            # coloring the final mesh (front: RGB pixels, back: normal colors)
            front_colors = query_color(
                verts=verts_in_front_view.squeeze(0), # Pass the TRANSFORMED vertices
                faces=canonical_faces,          # The mesh topology is the same
                image=in_tensor[f"view_{front_view}"]["image"],
                device=device,
                paint_normal=True # Set to False to see non-visible parts as gray
            )

            back_colors = query_color(
                verts=verts_in_back_view.squeeze(0), # Pass the TRANSFORMED vertices
                faces=canonical_faces,          # The mesh topology is the same
                image=in_tensor[f"view_{back_view}"]["image"],
                device=device,
                paint_normal=True # Set to False to see non-visible parts as gray
            )

            left_colors = query_color(
                verts=verts_in_left_view.squeeze(0), # Pass the TRANSFORMED vertices
                faces=canonical_faces,          # The mesh topology is the same
                image=in_tensor["view_0"]["image"],
                device=device,
                paint_normal=True # Set to False to see non-visible parts as gray
            )

            right_colors = query_color(
                verts=verts_in_right_view.squeeze(0), # Pass the TRANSFORMED vertices
                faces=canonical_faces,          # The mesh topology is the same
                image=in_tensor["view_4"]["image"],
                device=device,
                paint_normal=True # Set to False to see non-visible parts as gray
            )

            final_colors = front_colors.clone()
            update_mask = (back_visibility == 1) & (front_visibility == 0)
            final_colors[update_mask] = back_colors[update_mask]

            update_mask = (left_visibility == 1) & (front_visibility == 0) & (back_visibility == 0)
            final_colors[update_mask] = left_colors[update_mask]

            update_mask = (right_visibility == 1) & (front_visibility == 0) & (back_visibility == 0) & (left_visibility == 0)
            final_colors[update_mask] = right_colors[update_mask]

            final_mesh.visual.vertex_colors = final_colors
            final_mesh.export(final_path)
            print(colored(f"✅ Saved textured mesh to {final_path}", "green"))

        elif cfg.bni.texture_src == 'SD':

            # !TODO: add texture from Stable Diffusion
            pass

        exit()


# =========================================================================
# === HIERARCHICAL MULTI-VIEW TEXTURING (with Texture Diffusion) ==========
# =========================================================================
print("Starting hierarchical multi-view texturing...")

# --- 1. Initialization (Same as before) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
canonical_verts = torch.tensor(final_mesh.vertices, dtype=torch.float32, device=device)
canonical_faces = torch.tensor(final_mesh.faces, dtype=torch.long, device=device)
num_verts = canonical_verts.shape[0]

transform_manager_final = CameraTransformManager(cam_param_path, target_frame=0, device=device)
print("Applying base coat of normal colors...")
final_colors = query_normal_color(canonical_verts, canonical_faces, device)
is_textured_mask = torch.zeros(num_verts, dtype=torch.bool, device=device)

# --- 2. Define View Processing Order (Same as before) ---
priority_views = [args.front_view, args.back_view]
all_view_indices = list(range(len(multi_view_data)))
view_order = list(dict.fromkeys(priority_views + all_view_indices))
print(f"Processing {len(view_order)} views in this order: {view_order}")

# --- 3. Iterate Through All Views and Fill Texture (Same as before) ---
for view_idx in tqdm(view_order, desc="Texturing from Views"):
    # (The existing loop logic remains unchanged)
    current_view_data = in_tensor[f"view_{view_idx}"]
    T_canon_to_view = transform_manager_final.get_transform_to_target(view_idx).detach()
    verts_in_view = apply_homogeneous_transform(canonical_verts, T_canon_to_view.T).squeeze(0)
    verts_in_view = verts_in_view @ torch.tensor([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]).to(device)
    (view_xy, view_z) = verts_in_view.split([2, 1], dim=1)
    current_visibility = get_visibility(view_xy, view_z, canonical_faces[:, [0, 2, 1]]).flatten().to(device)
    update_mask = (current_visibility == 1) & (~is_textured_mask)
    if update_mask.sum().item() == 0:
        continue
    current_pass_colors = query_color(verts=verts_in_view, faces=canonical_faces, image=current_view_data["image"], device=device, paint_normal=False)
    final_colors[update_mask] = current_pass_colors[update_mask]
    is_textured_mask[update_mask] = True
    if is_textured_mask.all():
        break

# --- 4. NEW: Texture Diffusion to Fill Holes ---
uncolored_mask = ~is_textured_mask
if uncolored_mask.sum().item() > 0:
    print(f"\nFound {uncolored_mask.sum().item()} untextured vertices. Starting texture diffusion...")
    
    # Get the vertex adjacency list from the mesh
    # This tells us which vertices are neighbors
    adj = final_mesh.vertex_adjacency_graph
    
    # Perform a few iterations of diffusion
    for i in range(15): # 15 iterations is usually enough to fill large holes
        # Find the vertices at the boundary: uncolored vertices with colored neighbors
        boundary_verts_mask = torch.zeros_like(uncolored_mask)
        
        # Create a temporary copy to hold the new colors for this iteration
        iter_colors = final_colors.clone()
        
        # Get indices of all uncolored vertices
        uncolored_indices = torch.where(uncolored_mask)[0]

        # Use CPU for neighbor loops, as it's sparse and not easily parallelizable on GPU
        uncolored_indices_cpu = uncolored_indices.cpu().numpy()
        is_textured_mask_cpu = is_textured_mask.cpu().numpy()
        
        # This list will store which vertices we successfully colored in this iteration
        newly_colored_indices = []

        for v_idx in uncolored_indices_cpu:
            neighbors = adj.neighbors(v_idx)
            # Find which of the neighbors are already textured
            textured_neighbors = [n for n in neighbors if is_textured_mask_cpu[n]]
            
            if len(textured_neighbors) > 0:
                # If there are textured neighbors, this is a boundary vertex
                boundary_verts_mask[v_idx] = True
                # Calculate the average color of its textured neighbors
                neighbor_colors = final_colors[torch.tensor(textured_neighbors, device=device)]
                avg_color = torch.mean(neighbor_colors.float(), dim=0)
                iter_colors[v_idx] = avg_color
                newly_colored_indices.append(v_idx)
        
        # Update the final colors with the results from this iteration
        final_colors = iter_colors
        
        # Update the master textured mask
        if newly_colored_indices:
            newly_colored_tensor = torch.tensor(newly_colored_indices, dtype=torch.long, device=device)
            is_textured_mask[newly_colored_tensor] = True
            uncolored_mask = ~is_textured_mask # Recalculate the remaining uncolored
            print(f"Diffusion Iter {i+1}: Colored {len(newly_colored_indices)} vertices. Remaining: {uncolored_mask.sum().item()}")
        else:
            # If no new vertices were colored, the holes are filled.
            print("Diffusion complete. No more boundary vertices found.")
            break

# --- 5. Export the Final Textured Mesh ---
final_mesh.visual.vertex_colors = final_colors.cpu().numpy()
textured_path = final_path.replace('.obj', '_diffused_textured.obj')
final_mesh.export(textured_path)

print(colored(f"\n✅ Exported final diffused-texture mesh to {textured_path}", "green"))
