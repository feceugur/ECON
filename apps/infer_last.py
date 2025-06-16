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
from apps.SDFNetwork import Camera, extract_mesh_from_sdf, load_cameras_from_json, optimize_sdf, quaternion_to_rotation_matrix
from apps.prepare_supernormal_inputs import convert_camera_json_to_npz, rotate_normals_to_world, save_normal_map_exr_camera_to_world
import pyexr
import logging
import warnings
import os
import os.path as osp
from apps.BNIPipeline import BNIPipeline
from apps.FaceRigExporter import FaceRigExporter
from apps.mv_bini import extract_surface_multiview
from lib.common.ifnet_input_generator import IFNetsInputGenerator
from lib.common import BNI, BNI_multi
from pytorch3d.loss.chamfer import chamfer_distance

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
import cv2
import numpy as np
import torch
import torchvision
import trimesh
from pytorch3d.ops import SubdivideMeshes
from termcolor import colored
from tqdm.auto import tqdm
import numpy as np
import smplx
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from lib.common.train_util import update_mesh_shape_prior_losses
from pytorch3d.loss import mesh_laplacian_smoothing as laplacian_smoothing
import torchvision.transforms as T
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings
from pytorch3d.renderer.mesh import rasterize_meshes

from apps.IFGeo import IFGeo
from apps.Normal import Normal
from apps.sapiens import ImageProcessor
from apps.clean_mesh import MeshCleanProcess
from apps.SMPLXJointAligner import SMPLXJointAligner
from apps.CameraTransformManager import CameraTransformManager

from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor, save_normal_tensor_multi, save_normal_tensor_upt
from lib.common.config import cfg
from lib.common.imutils import blend_rgb_norm, load_img, transform_to_tensor, wrap
from lib.common.local_affine import LocalAffine, register, trimesh2meshes
from lib.common.render import query_color
from lib.common.train_util import Format, init_loss
from lib.common.voxelize import VoxelGrid
from lib.dataset.mesh_util import *
from lib.dataset.TestDataset import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis, rotmat_to_rot6d
from apps.ICON import ICON

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

# =================================================================================
# CORRECTED HELPER FUNCTION FOR CAMERA INTRINSICS
# =================================================================================
def get_intrinsics_matrix(camera_params_list, frame_id, final_image_size=(512, 512)):
    """
    Finds the correct camera entry from a list, and computes the 3x3 intrinsics matrix,
    adjusting for the final rendered image size.

    Args:
        camera_params_list (list): The list of camera data loaded from JSON.
        frame_id (int): The frame number to look for.
        final_image_size (tuple): The (width, height) of the processed image (e.g., 512x512).

    Returns:
        np.array: A 3x3 intrinsic matrix.
    """
    # Find the dictionary for the specified frame_id
    cam_info = next((item for item in camera_params_list if item["frame"] == frame_id), None)
    if cam_info is None:
        raise ValueError(f"Could not find camera parameters for frame {frame_id}")

    # The focal length was calculated for the original image size
    original_focal_length = cam_info["focal_length_px"]
    original_image_size = cam_info["image_size"] # [width, height]

    # Your `process_image` function crops and resizes to a square (512x512).
    # We must scale the focal length and principal point accordingly.
    # The crop is centered, so the scaling factor is based on the larger dimension of the original image.
    scale_factor = final_image_size[0] / max(original_image_size)
    
    fx = fy = original_focal_length * scale_factor
    
    # The principal point (cx, cy) is now the center of the new final image size.
    cx = final_image_size[0] / 2.0
    cy = final_image_size[1] / 2.0

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

def unproject_pixels_to_camera_space(depth, mask, K, device='cuda'):
    """
    Unprojects pixels to a 3D point cloud in camera space.
    depth: (H, W) tensor, depth in meters.
    mask: (H, W) boolean tensor, True for valid pixels.
    K: 3x3 numpy array, intrinsic matrix.
    """
    H, W = depth.shape
    K_inv = torch.from_numpy(np.linalg.inv(K)).float().to(device)
    
    # Create grid of pixel coordinates
    j, i = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    # Select coordinates and depth values for valid pixels only
    i_valid = i[mask]
    j_valid = j[mask]
    d_valid = depth[mask]

    # Create homogeneous pixel coordinates [u*z, v*z, z]
    pixels_homo = torch.stack([
        i_valid.float() * d_valid,
        j_valid.float() * d_valid,
        d_valid
    ], dim=0)

    # Unproject: (K_inv @ pixels_homo).T gives [X, Y, Z]
    points_cam = torch.matmul(K_inv, pixels_homo).T
    
    return points_cam

def sample_normals_at_pixels(normal_map, mask):
    """
    Samples normal vectors from a normal map at valid pixel locations.
    normal_map: (3, H, W) tensor in [-1, 1].
    mask: (H, W) boolean tensor.
    """
    # normal_map is (C, H, W), permute to (H, W, C) for easier indexing
    normals_permuted = normal_map.permute(1, 2, 0)
    # Get normals for valid pixels
    sampled_normals = normals_permuted[mask] # Shape: (N_valid, 3)
    
    # IMPORTANT: View-space normals are typically Y-down. Convert to standard 3D Y-up.
    sampled_normals[:, 1] *= -1.0
    
    return F.normalize(sampled_normals, dim=1)

def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion into a rotation matrix.
    q: torch.Tensor of shape (4,) or (N, 4) in (w, x, y, z) format.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = (w * w + x * x + y * y + z * z)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    
    R = torch.stack([
        1 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1 - (xx + yy)
    ], dim=-1).reshape(-1, 3, 3)
    
    if q.dim() == 1:
        return R.squeeze(0)
    return R
# =================================================================================

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
    #if True:
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
    #if True:
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

    canonical_smpl_verts = None
    canonical_smpl_joints = None
    canonical_smpl_landmarks = None
    canonical_saved = False
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
        in_tensor = {
            "smpl_faces": data["smpl_faces"], 
            "image": data["img_icon"].to(device), 
            "mask": data["img_mask"].to(device)
        }
        
        smpl_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_00.obj"

        # sapiens inference for current batch data

        if cfg.sapiens.use:
        #if True:
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
        cam_param_path = os.path.join(args.in_dir, "cam_params", "camera_parameters.json")
        transform_manager = CameraTransformManager(cam_param_path, target_frame=first_frame_id, device=device, debug=False)
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

        min_angle = float('inf')
        best_idx = 0

        for i, data in enumerate(multi_view_data):
            frame_id = int(data["name"].split("_")[1])
            T_view = transform_manager.get_transform_to_target(frame_id)
            R = T_view[:3, :3]

            cam_z = R[2, :]  # camera z-axis
            smpl_forward = torch.tensor([0.0, 0.0, 1.0], device=cam_z.device)
            cos_angle = torch.dot(-cam_z, smpl_forward) / (cam_z.norm() * smpl_forward.norm())
            angle = torch.acos(cos_angle.clamp(-1.0, 1.0))

            if angle.item() < min_angle:
                min_angle = angle.item()
                best_idx = i

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
        
        ############################################################
        #############CARLA/ERIC########################################
        ############################################################
        with torch.no_grad():
            N_body, N_pose = optimed_pose.shape[:2]

            # Convert 6D -> rotmat
            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).view(N_body, 1, 3, 3)

            # Define 180-degree flip around Y axis
            flip_y = torch.tensor([
                [-1, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, -1],
            ], dtype=torch.float32, device=optimed_orient.device)

            # Apply flip to orientation
            optimed_orient_mat = torch.matmul(flip_y.unsqueeze(0).unsqueeze(0), optimed_orient_mat)

            # Update optimed_orient
            optimed_orient.copy_(rotmat_to_rot6d(optimed_orient_mat.squeeze(1)).view_as(optimed_orient))

            # Flip translation too
            optimed_trans.mul_(torch.tensor([-1, 1, -1], device=optimed_trans.device))

        print(colored("✅ Applied 180° flip to global_orient and trans after initialization.", "yellow"))
        
        ############################################################
        loop_smpl = tqdm(range(args.loop_smpl))
        save_vis_dir = os.path.join(args.out_dir, cfg.name, "png", "canonical_smpl_iters")
        os.makedirs(save_vis_dir, exist_ok=True)
        loss_values = []

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
            
            frame_to_camera_dir = {}
            total_loss = 0.0

            for view_data in multi_view_data:
                frame_id = int(view_data["name"].split("_")[1])
                transform_manager = CameraTransformManager(cam_param_path, target_frame=int(multi_view_data[0]["name"].split("_")[1]), device=device, debug=False)
                T_frame_to_target = transform_manager.get_transform_to_target(frame_id)
                T_frame_to_target[:3, 3] = 0.0

                verts_in_view = apply_homogeneous_transform(smpl_verts, torch.inverse(T_frame_to_target))
                joints_in_view = apply_homogeneous_transform(smpl_joints, torch.inverse(T_frame_to_target))

                # From SMPL-X rotmat output: [B, J, 3, 3]
                head_rotmat = optimed_pose_mat[:, 14] 

                # Compute roll loss
                head_roll_loss = compute_head_roll_loss(head_rotmat, up_direction="view_y")

                # Landmark processing
                smpl_joints_3d = (joints_in_view[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0) * 0.5
                smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]
                ghum_lmks = view_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                ghum_conf = view_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)

                # A. Apply vertical offset (before loss calculation)
                # This shifts SMPL landmarks down to better align with GT
                #smpl_lmks[:, :, 1] = smpl_lmks[:, :, 1] + 0.03  # Shift down by 2% of image height

                # Save landmark visualization
                """
                if i % 10 == 0:  # Save every 10 iterations
                    save_path = os.path.join(save_vis_dir, f"landmarks_view_{frame_id}_iter_{i:03d}.png")
                    visualize_landmarks_detailed(
                        view_data["img_icon"].to(device),
                        verts_in_view,  # or smpl_verts if already in view space
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
                T_normal_F, T_normal_B = dataset.render_normal(
                    verts_in_view * torch.tensor([1.0, -1.0, -1.0]).to(device),
                    view_data["smpl_faces"],
                )
                T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

                smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
                gt_arr = view_data["img_mask"].to(device).repeat(1, 1, 2)
                diff_S = torch.abs(smpl_arr - gt_arr)

                sil_loss = diff_S.mean()
                # normal_loss = (torch.abs(T_normal_F) + torch.abs(T_normal_B)).mean() / 2.0
                normal_loss = torch.abs(T_normal_F).mean()

                # Combine all losses with appropriate weights
                view_total_loss = sil_loss + 0.5 * normal_loss + 0.1 * landmark_loss + 0.3 * head_roll_loss
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
                    T_frame_to_target = transform_manager.get_transform_to_target(frame_id)
                    T_frame_to_target[:3, 3] = 0.0

                    verts_in_view = apply_homogeneous_transform(smpl_verts, T_frame_to_target)

                    T_normal_F, _ = dataset.render_normal(
                        verts_in_view * torch.tensor([1.0, -1.0, -1.0]).to(device),
                        view_data["smpl_faces"],
                    )
                    T_mask_F, _ = dataset.render.get_image(type="mask")
                    view_data["T_mask_F"] = T_mask_F

                    smpl_mask_front = T_mask_F
                    gt_mask_front = view_data["img_mask"].to(device)

                    diff_S_front = torch.abs(smpl_mask_front - gt_mask_front)

                    smpl_silhouette_images.append(smpl_mask_front.unsqueeze(1).repeat(1, 3, 1, 1))
                    gt_silhouette_images.append(gt_mask_front.unsqueeze(1).repeat(1, 3, 1, 1))
                    diff_silhouette_images.append(diff_S_front.unsqueeze(1).repeat(1, 3, 1, 1))

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
        
        # Get final SMPL vertices and joints
        with torch.no_grad():
            smpl_verts, smpl_landmarks, smpl_joints = dataset.smpl_model(
                shape_params=optimed_betas,
                expression_params=closest_exp,
                body_pose=optimed_pose_mat,
                global_pose=optimed_orient_mat,
                jaw_pose=closest_jaw,
                left_hand_pose=tensor2variable(front_data["left_hand_pose"], device),
                right_hand_pose=tensor2variable(front_data["right_hand_pose"], device),
            )

            smpl_verts = (smpl_verts + optimed_trans) * front_data["scale"]
            smpl_joints = (smpl_joints + optimed_trans) * front_data["scale"] * torch.tensor([
                1.0, 1.0, -1.0
            ]).to(device)

            # Process landmarks
            smpl_joints_3d = (
                smpl_joints[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0
            ) * 0.5
            in_tensor["smpl_joint"] = smpl_joints[:,
                                                  dataset.smpl_data.smpl_joint_ids_24_pixie, :]

            # Get ground truth landmarks from last view
            
            ghum_lmks = front_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
            ghum_conf = front_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
            smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]

        # Save canonical SMPL
        #smpl_verts_save = smpl_verts * torch.tensor([1.0, 1.0, -1.0]).to(device)

        save_obj_path = os.path.join(args.out_dir, cfg.name, "obj", "canonical_smpl.obj")
        canonical_obj = trimesh.Trimesh(
            smpl_verts[0].detach().cpu() * torch.tensor([1.0, -1.0, 1.0]),
            in_tensor["smpl_faces"][0].cpu()[:, [0, 2, 1]],
            process=False,
            maintains_order=True,
        )
        canonical_obj.export(save_obj_path)

        # Save SMPL parameters
        smpl_info = {
            "betas": optimed_betas.detach().cpu(),
            "body_pose": convert_rot_matrix_to_angle_axis(optimed_pose_mat.detach()).cpu(),
            "global_orient": convert_rot_matrix_to_angle_axis(optimed_orient_mat.detach()).cpu(),
            "transl": optimed_trans.detach().cpu(),
            "expression": closest_exp.detach().cpu(),
            "jaw_pose": convert_rot_matrix_to_angle_axis(closest_jaw.detach()).cpu(),
            "left_hand_pose": convert_rot_matrix_to_angle_axis(front_data["left_hand_pose"].detach()).cpu(),
            "right_hand_pose": convert_rot_matrix_to_angle_axis(front_data["right_hand_pose"].detach()).cpu(),
            "scale": front_data["scale"].detach().cpu(),
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

        # Initialize in_tensor with all necessary data
        in_tensor = {
            "smpl_faces": front_data["smpl_faces"],
            "image": front_data["img_icon"].to(device),
            "mask": front_data["img_mask"].to(device),
            "smpl_verts": smpl_verts.detach(),
            "smpl_joints": smpl_joints.detach(),
            "smpl_landmarks": smpl_landmarks.detach(),
            "smpl_lmks": smpl_lmks.detach(),
            "ghum_lmks": ghum_lmks.detach(),
            "ghum_conf": ghum_conf.detach(),
            "multi_view_data": multi_view_data,
            "normal_list": normal_list,
            "mask_list": mask_list,
            "depth_list": depth_list,
            "smpl_obj_lst": smpl_obj_lst,
            "optimed_betas": optimed_betas.detach(),
            "optimed_pose_mat": optimed_pose_mat.detach(),
            "optimed_orient_mat": optimed_orient_mat.detach(),
            "optimed_trans": optimed_trans.detach(),
            "closest_exp": closest_exp.detach(),
            "closest_jaw": closest_jaw.detach(),
            "scale": front_data["scale"].detach(),
            "left_hand_pose": front_data["left_hand_pose"].detach(),
            "right_hand_pose": front_data["right_hand_pose"].detach(),
            "transform_manager": transform_manager,
            "cam_param_path": cam_param_path
        }

        for view_data in multi_view_data:
            frame_id = int(view_data["name"].split("_")[1])
            T_frame_to_target = transform_manager.get_transform_to_target(frame_id)
            T_frame_to_target[:3, 3] = 0.0

            # Transform vertices to this view
            smpl_verts_view = apply_homogeneous_transform(
                smpl_verts,
                T_frame_to_target
            )

            # Add view-specific data to in_tensor
            view_key = f"view_{frame_id}"
            in_tensor[view_key] = {
                "smpl_faces": view_data["smpl_faces"],
                "image": view_data["img_icon"].to(device),
                "mask": view_data["img_mask"].to(device),
                "smpl_verts": smpl_verts_view,
                "T_frame_to_target": T_frame_to_target,
                "frame_id": frame_id,
                "name": view_data["name"],
                #"smpl_joints": view_data["smpl_joints"],
                "normal_F": None,
                "normal_B": None,
                "M_crop": view_data["M_crop"],
                "M_square": view_data["M_square"],
                "T_mask_F": view_data["T_mask_F"],
            }

            # Render SMPL normals for this view
            in_tensor[view_key]["T_normal_F"], in_tensor[view_key]["T_normal_B"] = dataset.render_normal(
                smpl_verts_view * torch.tensor([1.0, -1.0, -1.0]).to(device),
                view_data["smpl_faces"],
            )
            img_norm_path_f = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_normalsmpl_{frame_id}.png")
            torchvision.utils.save_image((in_tensor[view_key]["T_normal_F"].detach().cpu()), img_norm_path_f)

            # Get input RGB image
            rgb_image = view_data["img_icon"].squeeze(0).cpu().numpy()  # (H, W, 3)
            
            # Get segmentation mask
            mask = view_data["img_mask"].squeeze(0).cpu().numpy()  # (H, W)
            img_mask_path_f = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_mask_{frame_id}.png")
            # Convert mask to tensor and save
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
            torchvision.utils.save_image(mask_tensor, img_mask_path_f)

            # Get normal map from sapiens if enabled
            if cfg.sapiens.use:
            #if True:
                in_tensor[view_key]["normal_F"] = sapiens_normal_net.process_image(
                    Image.fromarray(
                        view_data["img_raw"].squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                    ), "1b", cfg.sapiens.seg_model
                )
                img_norm_path_f = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_normal_{frame_id}.png")
                torchvision.utils.save_image((wrap(in_tensor["view_0"]["normal_F"], data["uncrop_param"], 0).detach().cpu()), img_norm_path_f)
                normal_map = in_tensor[view_key]["normal_F"].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                _, in_tensor[view_key]["normal_B"] = normal_net.netG(in_tensor[view_key])
            else:
                # If sapiens not enabled, use normal estimation from normal_net
                with torch.no_grad():
                    in_tensor[view_key]["normal_F"], in_tensor[view_key]["normal_B"] = normal_net.netG(in_tensor[view_key])
                    img_norm_path_f = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_normal_{frame_id}.png")
                    torchvision.utils.save_image((in_tensor[view_key]["normal_F"].detach().cpu()), img_norm_path_f)

                    normal_map = in_tensor[view_key]["normal_F"].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

                    in_tensor[view_key]["normal_FW"] = rotate_normals_to_world(in_tensor[view_key]["normal_F"], T_frame_to_target[:3, :3])
                    in_tensor[view_key]["normal_BW"] = rotate_normals_to_world(in_tensor[view_key]["normal_B"], T_frame_to_target[:3, :3])
                    img_norm_path_f = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_normal_FW_{frame_id}.png")
                    torchvision.utils.save_image((in_tensor[view_key]["normal_FW"].detach().cpu()), img_norm_path_f)


            # Get SMPL mask
            in_tensor[view_key]["T_mask_F"], _ = dataset.render.get_image(type="mask")
            img_mask_path_f = osp.join(args.out_dir, cfg.name, "png", f"{view_data['name']}_T_mask_F_{frame_id}.png")
            torchvision.utils.save_image((in_tensor[view_key]["T_mask_F"].detach().cpu()), img_mask_path_f)

            # Get SMPL depth as proxy
            in_tensor[view_key]["depth_F"], in_tensor[view_key]["depth_B"] = dataset.render_depth(
                smpl_verts_view * torch.tensor([1.0, -1.0, -1.0]).to(device),
                view_data["smpl_faces"]
            )
            depth_map = in_tensor[view_key]["depth_F"].squeeze(0).detach().cpu().numpy()  # (H, W)
            
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
            depth_vis = in_tensor[view_key]["depth_F"].detach().cpu().squeeze(0)  # Remove batch dimension
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
                smpl_verts_view.detach().cpu().squeeze(0) * torch.tensor([1.0, -1.0, 1.0]),
                in_tensor[view_key]["smpl_faces"][0].cpu()[:, [0, 2, 1]],
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

            # Add view-specific data to in_tensor
            in_tensor[view_key].update({
                "normal_map": normal_map,
                "mask": mask,
                "depth_map": depth_map,
                "view_obj": view_obj
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

        """
        print(colored("✅ Preparing multi-view tensors for ICON...", "cyan"))
        
        # We will create lists to hold the data for each view
        calib_list = []
        in_tensor_list = []
        
        with open(cam_param_path, 'r') as f:
            camera_params_list = json.load(f)
        
        # Create a dictionary for easy lookup by frame number
        camera_params_dict = {params['frame']: params for params in camera_params_list}
        
        # This assumes your 'multi_view_data' list and 'transform_manager' are still available
        for view_data in multi_view_data:
            frame_id = int(view_data['name'].split('_')[1])
            
            # Get the camera parameters for this view from the loaded JSON data
            if frame_id not in camera_params_dict:
                raise ValueError(f"No camera parameters found for frame {frame_id} in {cam_param_path}")
            
            camera_params = camera_params_dict[frame_id]
            
            # Extract quaternion and location from camera parameters
            quaternion = camera_params['quaternion']  # [w, x, y, z]
            location = camera_params['location']
            
            # Convert quaternion to rotation matrix
            # Note: quaternion is in [w, x, y, z] format
            qw, qx, qy, qz = quaternion
            R = torch.tensor([
                [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
            ], dtype=torch.float32).to(device)
            
            # Convert location to translation vector
            t = torch.tensor(location, dtype=torch.float32).to(device).view(3, 1)
            
            # For Blender cameras looking at origin:
            # 1. The rotation matrix needs to be inverted since we want world-to-camera transform
            # 2. The translation needs to be transformed by the rotation
            R = R.t()  # Transpose is inverse for rotation matrices
            t = -R @ t  # Transform translation to camera space
            
            # Create calibration matrix [R|t]
            calib = torch.cat([R, t], dim=1)  # [3, 4]
            calib = calib.unsqueeze(0)  # [1, 3, 4]
            calib_list.append(calib)
            
            # Create a dedicated input dictionary for this view
            view_tensor = {
                'image': view_data['img_icon'].to(device),
                'mask': view_data['img_mask'].to(device),
                'smpl_verts': in_tensor['smpl_verts'], # Canonical SMPL is the same for all
                'smpl_faces': in_tensor['smpl_faces'],
                'normal_F': in_tensor[f'view_{frame_id}']['normal_F'],
                'normal_B': in_tensor[f'view_{frame_id}']['normal_B'],
            }
            in_tensor_list.append(view_tensor)

        # Stack calibration matrices into a single tensor of shape [num_views, 1, 3, 4]
        calib_tensor = torch.stack(calib_list, dim=0)  # [num_views, 1, 3, 4]
        
        # Reshape to match the expected batch dimensions
        num_views = calib_tensor.shape[0]
        calib_tensor = calib_tensor.view(num_views, 3, 4)  # [num_views, 3, 4]
        
        # Debug print to check the final calibration tensor
        print("Final calibration tensor shape:", calib_tensor.shape)
        print("Final calibration tensor:", calib_tensor)

        in_tensor.update(
            dataset.compute_vis_cmap(in_tensor["smpl_verts"][0], in_tensor["smpl_faces"][0])
        )

        # Initialize ICON model
        icon_model = ICON(cfg)
        icon_model = icon_model.to(device)
        icon_model.eval()

        # Use the test_multiview method with stacked calibration tensor
        verts_pr, faces_pr, _ = icon_model.test_multiview(in_tensor_list, calib_tensor)

        recon_obj = trimesh.Trimesh(verts_pr, faces_pr, process=False, maintains_order=True)
        recon_obj.export(os.path.join(args.out_dir, cfg.name, f"obj/{data['name']}_recon_multiview.obj"))
        print(colored("✅ Generated multi-view reconstruction.", "green"))

        """




        """


        # (Keep all your helper functions as they are, they are correct)

        # --- Add these new, required imports from PyTorch3D ---


        # =================================================================================
        # FINAL CORRECTED PIPELINE (with PyTorch3D Coordinate System Fix)
        # This directly solves the occlusion and depth unit problems.
        # =================================================================================
        print(colored("🚀 Starting FINAL Multi-View Point Cloud Fusion...", "green"))

        all_points_world, all_normals_world = [], []
        canonical_smpl_verts = smpl_verts.detach().squeeze(0)

        with open(cam_param_path, "r") as f:
            cam_params_json_list = json.load(f)

        debug_dir = osp.join(args.out_dir, cfg.name, "obj", "point_cloud_debug")
        os.makedirs(debug_dir, exist_ok=True)

        # This coordinate system fix for the camera poses is still necessary and correct.
        T_coord_fix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)

        # Restore the full loop to process all cameras
        for cam_info in tqdm(cam_params_json_list, desc="Generating & Fusing Point Clouds"):
            frame_id = cam_info["frame"]
            data = next(d for d in multi_view_data if int(d['name'].split('_')[1]) == frame_id)

            mask = (data["img_mask"] > 0.5).squeeze(0)
            H, W = mask.shape
            gt_normal_F = in_tensor[f"view_{frame_id}"]["normal_F"]
            normal_map = gt_normal_F.squeeze(0)

            # --- Step 1: Construct Correct Camera Matrices (This logic is sound) ---
            location_z_up = torch.tensor(cam_info["location"], device=device, dtype=torch.float32)
            quaternion_z_up = torch.tensor(cam_info["quaternion"], device=device, dtype=torch.float32)
            R_z_up = quaternion_to_rotation_matrix(quaternion_z_up)
            T_cam_z_up = torch.eye(4, device=device); T_cam_z_up[:3, :3] = R_z_up; T_cam_z_up[:3, 3] = location_z_up
            T_cam_y_up = torch.matmul(T_coord_fix, T_cam_z_up)
            R_y_up, location_y_up = T_cam_y_up[:3, :3], T_cam_y_up[:3, 3]
            R_inv = R_y_up.T; t_inv = -torch.matmul(R_inv, location_y_up)
            T_world_to_view = torch.eye(4, device=device); T_world_to_view[:3, :3] = R_inv; T_world_to_view[:3, 3] = t_inv
            T_view_to_world = torch.inverse(T_world_to_view)

            # --- Step 2: Generate a Correct, World-Unit Depth Map ---
            # Get vertices in standard CV camera space (Y-up, camera looks along -Z)
            verts_in_cv_cam = apply_homogeneous_transform(canonical_smpl_verts.unsqueeze(0), T_world_to_view)
            
            # ** THE CRUCIAL FIX **
            # PyTorch3D's renderer uses a different camera convention (Y-down, camera looks along +Z).
            # We MUST convert our vertices to this space before rendering to get a correct Z-buffer.
            verts_for_render = verts_in_cv_cam.clone()
            verts_for_render[..., 1] *= -1.0  # Flip Y-axis (up -> down)
            verts_for_render[..., 2] *= -1.0  # Flip Z-axis (-Z forward -> +Z forward)

            # Create a Meshes object with these correctly-formatted vertices
            mesh_for_render = Meshes(verts=verts_for_render, faces=data["smpl_faces"])
            
            # Setup a PyTorch3D camera using our intrinsics. Note that P3D cameras
            # are defined at the origin looking along +Z, which matches our `verts_for_render`.
            K = get_intrinsics_matrix(cam_params_json_list, frame_id, final_image_size=(W, H))
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            cameras = PerspectiveCameras(focal_length=((fx, fy),), principal_point=((cx, cy),), image_size=((H, W),), device=device)
            raster_settings = RasterizationSettings(image_size=(H,W), blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True)
            
            # Rasterize the mesh. The resulting zbuf is a true depth map in world units.
                # Unpack the returned tuple directly into named variables.
            pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
                meshes=mesh_for_render, 
                image_size=(H, W),
                blur_radius=0.0,
                faces_per_pixel=1
            )
            
            # Now, use the `zbuf` tensor directly.
            z_buffer = zbuf.squeeze().squeeze(-1)

            # --- Step 3: Use the Correct Depth Map for Unprojection ---
            # The background is -1. We use only the valid pixels from our original mask.
            depth_map = z_buffer
            
            # The `unproject_pixels_to_camera_space` function is now guaranteed to work correctly.
            points_cam = unproject_pixels_to_camera_space(depth_map, mask, K, device=device)
            normals_cam = sample_normals_at_pixels(normal_map, mask)

            # Transform points to world space
            points_world = apply_homogeneous_transform(points_cam.unsqueeze(0), T_view_to_world).squeeze(0)
            normals_world = torch.matmul(normals_cam, T_view_to_world[:3, :3].T)

            all_points_world.append(points_world)
            all_normals_world.append(normals_world)

        # --- Final Combination (This should now work correctly) ---
        fused_points = torch.cat(all_points_world, dim=0).cpu().numpy()
        fused_normals = torch.cat(all_normals_world, dim=0).cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fused_points)
        pcd.normals = o3d.utility.Vector3dVector(fused_normals)

        o3d.io.write_point_cloud(osp.join(debug_dir, "final_raw_fused_point_cloud.ply"), pcd)
        pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        o3d.io.write_point_cloud(osp.join(debug_dir, "final_filtered_point_cloud.ply"), pcd_filtered)

        print(colored(f"Filtered to {len(pcd_filtered.points)} points. Running Poisson reconstruction...", "blue"))
        recon_obj_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_filtered, depth=10)

        recon_obj_o3d.remove_degenerate_triangles()
        recon_obj_o3d.remove_duplicated_triangles()
        recon_obj_o3d.remove_unreferenced_vertices()

        recon_obj = trimesh.Trimesh(
            vertices=np.asarray(recon_obj_o3d.vertices),
            faces=np.asarray(recon_obj_o3d.triangles),
            process=False
        )

        base_recon_path = osp.join(args.out_dir, cfg.name, "obj", "base_fused_recon.obj")
        recon_obj.export(base_recon_path)
        print(colored(f"✅ FINAL: Saved base fused mesh to {base_recon_path}", "green"))


        """










        # Store optimized SMPL verts and faces into in_tensor for downstream steps
        with torch.no_grad():
            in_tensor["smpl_verts"] = in_tensor[f"view_{front_view}"]["smpl_verts"].detach()
            in_tensor["smpl_faces"] = in_tensor[f"view_{front_view}"]["smpl_faces"].detach()

        per_data_lst = []

        batch_smpl_verts = in_tensor["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0],
                                                                           device=device)
        batch_smpl_faces = in_tensor["smpl_faces"].detach()[:, :, [0, 2, 1]]

        in_tensor["depth_F"], _ = dataset.render_depth(
            batch_smpl_verts, batch_smpl_faces
        )
        in_tensor["depth_B"], _ = dataset.render_depth(
            in_tensor[f"view_{back_view}"]["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0], device=device), 
            in_tensor[f"view_{back_view}"]["smpl_faces"].detach()[:, :, [0, 2, 1]]
        )

        for data in multi_view_data:
            frame_id = int(data["name"].split("_")[1])
            view_data = in_tensor[f"view_{frame_id}"]

            # Normalize depth map to [0, 1] range per image
            depth_map = view_data["depth_F"].detach().cpu()  # [B, H, W]
            depth_min = depth_map.amin(dim=[1, 2], keepdim=True)
            depth_max = depth_map.amax(dim=[1, 2], keepdim=True)
            depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
            depth_map_norm = depth_map_norm.unsqueeze(1)  # to [B, 1, H, W] for saving
            
            # If depth_map is a torch tensor of shape [1, H, W]
            depth_np = depth_map.detach().cpu().numpy()[0]  # [H, W]

            # Mask out background values (e.g., -1.0)
            valid_mask = depth_np > -0.5
            valid_depth = depth_np[valid_mask]

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
            cv2.imwrite(osp.join(args.out_dir, cfg.name, f"{view_data['name']}_smpl_front_depth.png"), depth_8bit)

        per_loop_lst = []

        in_tensor["BNI_verts"] = []
        in_tensor["BNI_faces"] = []
        in_tensor["body_verts"] = []
        in_tensor["body_faces"] = []
      

        final_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_full.obj"

        side_mesh = smpl_obj_lst[back_view].copy()
        face_mesh = smpl_obj_lst[back_view].copy()
        hand_mesh = smpl_obj_lst[back_view].copy()
        smplx_mesh = smpl_obj_lst[back_view].copy()

        #T_back_view = transform_manager.get_transform_to_target(back_view)
        
        #in_tensor[f"view_{back_view}"]["normal_F"] = (T_back_view @ in_tensor[f"view_{back_view}"]["normal_F"].squeeze(0).cpu().numpy()).to(device)
        """
        # save normals, depths and masks
        idx = 0
        BNI_dict = save_normal_tensor(
            in_tensor[f"view_{front_view}"],
            idx,
            osp.join(args.out_dir, cfg.name, f"BNI/{data['name']}"),
            cfg.bni.thickness,
        )

        # BNI process
        BNI_object = BNI(
            dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
            name=data["name"],
            BNI_dict=BNI_dict,
            cfg=cfg.bni,
            device=device
        )

        BNI_object.extract_surface(False)
        """
        bni_mesh_list = []
        bni_object_list = []
        extrinsics = []
        for i in (front_view, back_view):
            frame_id = int(multi_view_data[i]["name"].split("_")[1])
            T_view_i = transform_manager.get_transform_to_target(frame_id)
            extrinsics.append(T_view_i.cpu().numpy())
            
            view_data = in_tensor[f"view_{i}"]
            #depth_tensor = view_data["depth_F"]  # (H, W)
            #depth_list.append(depth_tensor.cpu().numpy())  # must be float32, in meters
            idx=0
            BNI_dict = save_normal_tensor(
                view_data,
                idx,
                osp.join(args.out_dir, cfg.name, f"BNI/{data['name']}_{idx}"),
                cfg.bni.thickness,
            )
            
            # BNI process
            BNI_object = BNI(
                dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
                name=data["name"],
                BNI_dict=BNI_dict,
                cfg=cfg.bni,
                device=device
            )

            BNI_object.extract_surface()
            # Convert T_view_i from PyTorch tensor to NumPy array
            R = T_view_i[:3, :3].cpu().numpy()
            R_4x4 = np.eye(4)
            R_4x4[:3, :3] = R
            # Apply to trimesh object
            BNI_object.F_trimesh.apply_transform(R_4x4)
            # Add 0.01m offset to vertices along their normal directions
            vertices = BNI_object.F_trimesh.vertices
            vertex_normals = BNI_object.F_trimesh.vertex_normals
            vertices -= 0.01 * vertex_normals
            BNI_object.F_trimesh.vertices = vertices
            # Save front mesh for this view
            front_mesh_path = os.path.join(args.out_dir, cfg.name, "obj", f"bni_view_{i}.obj")
            BNI_object.F_trimesh.export(front_mesh_path)
            print(colored(f"✅ Saved front mesh for view {i} to {front_mesh_path}", "green"))
            bni_mesh_list.append(BNI_object.F_trimesh)
            bni_object_list.append(BNI_object)

        # Add 0.01m offset in +z direction to first BNI mesh
        vertices = bni_mesh_list[0].vertices
        vertices[:, 2] += 0.02  # Add 0.01m in z direction
        bni_mesh_list[0].vertices = vertices
        #Add -2 degree rotation in +z direction to first BNI mesh
        vertices = bni_mesh_list[0].vertices
        # Convert to homogeneous coordinates by adding a column of ones
        vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        # Apply transformation
        transformed_vertices = vertices_homogeneous @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
        # Convert back from homogeneous coordinates
        bni_mesh_list[0].vertices = transformed_vertices[:, :3]

        fused_bni_surface = trimesh.util.concatenate(bni_mesh_list)
        BNI_surface = trimesh2meshes(fused_bni_surface).to(device)
        fused_bni_surface.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_fused_BNI.obj")
        

        in_tensor["body_verts"].append(torch.tensor(smpl_obj_lst[front_view].vertices).float())
        in_tensor["body_faces"].append(torch.tensor(smpl_obj_lst[front_view].faces).long())

        # requires shape completion when low overlap
        # replace SMPL by completed mesh as side_mesh   ----> TRY SIDE VIEW'S BNI MESHES

        if cfg.bni.use_ifnet:
        #if True:

            side_mesh_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_IF.obj"

            side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)

            # mesh completion via IF-net
            in_tensor.update(
                dataset.depth_to_voxel({
                    "depth_F": bni_object_list[0].F_depth.unsqueeze(0), 
                    "depth_B": bni_object_list[0].B_depth.unsqueeze(0)
                })
            )

            occupancies = VoxelGrid.from_mesh(side_mesh, cfg.vol_res, loc=[
                0,
            ] * 3, scale=2.0).data.transpose(2, 1, 0)
            occupancies = np.flip(occupancies, axis=1)

            in_tensor["body_voxels"] = torch.tensor(occupancies.copy()
                                                    ).float().unsqueeze(0).to(device)

            with torch.no_grad():
                sdf = ifnet.reconEngine(netG=ifnet.netG, batch=in_tensor)
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
                BNI_object.F_B_trimesh = part_removal(
                    BNI_object.F_B_trimesh,
                    face_mesh,
                    cfg.bni.face_thres,
                    device,
                    smplx_mesh,
                    region="face"
                )
                side_mesh = part_removal(
                    side_mesh, face_mesh, cfg.bni.face_thres, device, smplx_mesh, region="face"
                )
                face_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_face.obj")
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
                BNI_object.F_B_trimesh = part_removal(
                    BNI_object.F_B_trimesh,
                    hand_mesh,
                    cfg.bni.hand_thres,
                    device,
                    smplx_mesh,
                    region="hand"
                )
                side_mesh = part_removal(
                    side_mesh, hand_mesh, cfg.bni.hand_thres, device, smplx_mesh, region="hand"
                )
                hand_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_hand.obj")
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
        side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_side.obj")

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


        print("Remeshing Poisson output for clean topology...")
        verts_refine, faces_refine = remesh(recon_obj, osp.join(args.out_dir, cfg.name, "obj", "remesh.obj"), device)
        mesh_pr = Meshes(verts=verts_refine, faces=faces_refine).to(device)
        verts_ref = mesh_pr.verts_padded().clone().detach()

        local_affine_model = LocalAffine(mesh_pr.verts_padded().shape[1], 1, mesh_pr.edges_packed()).to(device)
        optimizer_cloth = torch.optim.Adam(local_affine_model.parameters(), lr=1e-5)
        scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cloth, mode="min", factor=0.5, patience=10)


        # --- Annealing and Loss Weights Setup ---
        num_cloth_loop = 100
        loop_cloth = tqdm(range(num_cloth_loop), desc="Cloth Refinement")
        
        # Start with high smoothing, low data term. End with low smoothing, high data term.
        initial_lapla_w, final_lapla_w = 20.0, 0.5
        initial_normal_w, final_normal_w = 0.5, 1.5
        blur_transform = T.GaussianBlur(kernel_size=5, sigma=1.0)
        for i in loop_cloth:
            optimizer_cloth.zero_grad()

            deformed_verts, stiffness, rigid = local_affine_model(mesh_pr.verts_padded())
            deformed_mesh = Meshes(verts=deformed_verts, faces=mesh_pr.faces_padded())

            # --- Anneal weights based on iteration progress ---
            progress = i / num_cloth_loop
            w_lapla = initial_lapla_w * (1 - progress)**2 + final_lapla_w * (1 - (1-progress)**2)
            w_normal = initial_normal_w * (1 - progress) + final_normal_w * progress

            # --- Regularization Losses ---
            loss_lapla = laplacian_smoothing(deformed_mesh)
            loss_stiff = torch.mean(stiffness)
            loss_rigid = torch.mean(rigid)
            loss_chamfer, _ = chamfer_distance(deformed_verts, verts_ref)
            
            # --- Multi-View Data-Driven Losses ---
            total_normal_loss = 0.0
            total_mask_loss = 0.0

            for view_data in multi_view_data:
                frame_id = int(view_data["name"].split("_")[1])
                T_view = transform_manager.get_transform_to_target(frame_id)
                #T_rot_x = torch.tensor([
                #    [-1.0, 0.0, 0.0, 0.0],
                #    [0.0, 1.0, 0.0, 0.0], 
                #    [0.0, 0.0, -1.0, 0.0],
                #    [0.0, 0.0, 0.0, 1.0]], dtype=torch.float, device=T_view.device)
                #T_combined = torch.matmul(T_view, T_rot_x)
                
                #T_rot_y = torch.tensor([
                #    [-1.0, 0.0, 0.0, 0.0],
                #    [0.0, 1.0, 0.0, 0.0],
                #    [0.0, 0.0, -1.0, 0.0],
                #    [0.0, 0.0, 0.0, 1.0]
                #], dtype=torch.float, device=T_view.device)
                
                # Combine with existing transformations
                #T_combined = torch.matmul(T_view, T_view)

                verts_view = apply_homogeneous_transform(mesh_pr.verts_padded(), T_view.T)
                
                # Render predicted normals and mask for the deformed mesh
                # The [-1,-1,1] scaling is crucial to match renderer conventions if needed.
                # Assuming your render_normal handles this.
                P_normal_F, _ = dataset.render_normal(verts_view.to(device), deformed_mesh.faces_padded())
                #P_normal_F, _ = dataset.render_normal(mesh_pr.verts_padded().to(device), deformed_mesh.faces_padded())
                P_mask_F, _ = dataset.render.get_image(type="mask")

                # Get ground truth data for this view
                gt_normal_F_raw = in_tensor[f"view_{frame_id}"]["normal_F"]
                gt_normal_F = blur_transform(gt_normal_F_raw)
                gt_mask_numpy = in_tensor[f"view_{frame_id}"]["mask"] # This is a (512, 512) numpy array
                gt_mask = torch.from_numpy(gt_mask_numpy).float().to(device) # Convert to tensor and move to GPU
                gt_mask = gt_mask.unsqueeze(0).unsqueeze(0) # Add Batch and Channel dims -> [1, 1, 512, 512]

                comparison_dir = os.path.join(args.out_dir, cfg.name, "debug_comparisons")
                os.makedirs(comparison_dir, exist_ok=True)

                comparison_path = os.path.join(comparison_dir, f"{data['name']}_{frame_id}.png")
                save_normal_comparison(P_normal_F, gt_normal_F, comparison_path)


                # Get SMPL body mask for this view
                smpl_mask = in_tensor[f"view_{frame_id}"]["T_mask_F"]
                smpl_mask = smpl_mask.unsqueeze(0).unsqueeze(0) # Add Batch and Channel dims -> [1, 1, 512, 512]

                # Calculate intersection mask between SMPL body and ground truth
                intersection_mask = torch.logical_and(gt_mask > 0.5, smpl_mask > 0.5)

                # --- Visibility Weighting ---
                with torch.no_grad():
                    view_direction_img = torch.tensor([0.0, 0.0, -1.0], device=device).view(1, 3, 1, 1)
                    visibility = torch.sum(P_normal_F * view_direction_img, dim=1, keepdim=True)
                    visibility_weights = torch.clamp(visibility, min=0.0).detach()

                    # Final weight mask combines intersection mask with visibility weight
                    combined_mask = intersection_mask * (visibility_weights > 0.1)

                # Only compute losses on intersection points
                if combined_mask.sum() > 0:
                    # 1. Weighted Normal Loss (Cosine Similarity)
                    normal_sim_loss = 1.0 - F.cosine_similarity(P_normal_F, gt_normal_F, dim=1)
                    view_normal_loss = (normal_sim_loss * combined_mask).sum() / combined_mask.sum().clamp(min=1.0)
                    
                    # 2. Silhouette Loss (IoU or L1) - only on intersection points
                    view_mask_loss = F.l1_loss(P_mask_F * combined_mask, gt_mask.squeeze(1) * combined_mask)

                    total_normal_loss += view_normal_loss
                    total_mask_loss += view_mask_loss

            # Average losses across all views
            avg_normal_loss = total_normal_loss / len(multi_view_data)
            avg_mask_loss = total_mask_loss / len(multi_view_data)

            # --- Combine, Backpropagate, and Optimize ---
            total_loss = (
                avg_normal_loss * w_normal +   # Annealed normal loss
                avg_mask_loss * 1.0 +          # Strong silhouette prior
                loss_lapla * w_lapla +         # Annealed smoothing
                loss_stiff * 0.1 +             # Weaker affine priors
                loss_rigid * 0.1 +
                loss_chamfer * 0.1             # Weakly keep original shape
            )

            loop_cloth.set_description(f"Refining | Total: {total_loss.item():.4f} | Normal: {avg_normal_loss.item():.4f} | Lap: {loss_lapla.item():.4f}")
            total_loss.backward()
            optimizer_cloth.step()
            scheduler_cloth.step(total_loss)

            mesh_pr = deformed_mesh.detach()

        final_verts = mesh_pr.verts_packed().detach().squeeze(0).cpu()
        final_faces = mesh_pr.faces_packed().detach().squeeze(0).cpu()
        final_obj = trimesh.Trimesh(final_verts, final_faces, process=False, maintains_order=True)
        final_colors = query_color(final_verts, final_faces, in_tensor["image"], device=device)
        final_obj.visual.vertex_colors = final_colors
        refine_path = osp.join(args.out_dir, cfg.name, "obj", f"{data['name']}_refine.obj")
        final_obj.export(refine_path)
        print(f"Saved refined mesh to {refine_path}")
        exit()