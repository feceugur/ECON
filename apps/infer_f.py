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

import logging
import warnings
import os
import os.path as osp
import json
from PIL import Image

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
import open3d as o3d
import smplx
import matplotlib.pyplot as plt
from termcolor import colored
from tqdm.auto import tqdm

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

# Assuming local imports from your project structure
from apps.IFGeo import IFGeo
from apps.Normal import Normal
from apps.sapiens import ImageProcessor
from lib.common.config import cfg
from lib.common.imutils import wrap
from lib.common.local_affine import LocalAffine
from lib.common.render import query_color
from lib.common.train_util import init_loss, Format, update_mesh_shape_prior_losses
from lib.dataset.mesh_util import remesh
from lib.dataset.TestDataset import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis, rotmat_to_rot6d
from apps.CameraTransformManager import CameraTransformManager

torch.backends.cudnn.benchmark = True

# --- New or Improved Helper Functions ---

def get_intrinsics_matrix(camera_info, image_size):
    """Creates a 3x3 intrinsics matrix from camera parameters."""
    fx = fy = camera_info["focal_length_px"]
    width, height = image_size
    # Use the provided image size, as cx/cy might differ from camera_info if cropped
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

def unproject_pixels_to_camera_space(depth, mask, K, device='cuda'):
    """
    Unprojects pixels to a 3D point cloud in camera space.
    depth: (H, W) tensor, depth in meters
    mask: (H, W) boolean tensor, True for valid pixels
    K: 3x3 numpy array, intrinsic matrix
    """
    H, W = depth.shape
    K_inv = torch.from_numpy(np.linalg.inv(K)).float().to(device)
    
    # Create grid of pixel coordinates
    j, i = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    # Select coordinates and depth values for valid pixels
    i_valid = i[mask]
    j_valid = j[mask]
    d_valid = depth[mask]

    # Create homogeneous pixel coordinates [u, v, 1]
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
    normal_map: (3, H, W) tensor in [-1, 1]
    mask: (H, W) boolean tensor
    """
    # normal_map is (C, H, W), we need to select based on mask (H, W)
    # Permute to (H, W, C) for easier indexing
    normals_permuted = normal_map.permute(1, 2, 0)
    # Get normals for valid pixels
    sampled_normals = normals_permuted[mask] # (N_valid, 3)
    
    # IMPORTANT: View-space normals are typically Y-down. Convert to standard 3D Y-up.
    sampled_normals[:, 1] *= -1.0
    
    return F.normalize(sampled_normals, dim=1)


def apply_homogeneous_transform(points, T):
    """
    Applies a 4x4 homogeneous transformation matrix `T` to a [N, 3] tensor `points`.
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError("Input points must be of shape [N, 3]")
    
    N, _ = points.shape
    # Add homogeneous coordinate
    points_homo = torch.cat([points, torch.ones(N, 1, device=points.device)], dim=-1) # [N, 4]
    
    # Apply transformation: (T @ points_homo.T).T
    transformed_points = torch.matmul(points_homo, T.T)
    
    return transformed_points[:, :3] # Return non-homogeneous [N, 3]

def convert_rot_matrix_to_angle_axis(rot_matrix):
    """ Helper to convert rotation matrices (B, J, 3, 3) or (B, 3, 3) to angle-axis. """
    shape = rot_matrix.shape
    is_batched_joints = len(shape) == 4

    # Flatten for pytorch3d function
    rot_flat = rot_matrix.reshape(-1, 3, 3)
    
    # pytorch3d expects a 4x4 matrix, but only uses the top-left 3x3 for rotation
    angle_axis = rotation_matrix_to_angle_axis(rot_flat)

    # Reshape back to original format (without the 3x3 dimensions)
    if is_batched_joints:
        return angle_axis.reshape(shape[0], shape[1], 3)
    else:
        return angle_axis.reshape(shape[0], 3)
        
if __name__ == "__main__":
    # --- Argument Parsing and Config Loading ---
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=50) # Increased for better convergence
    parser.add_argument("-patience", "--patience", type=int, default=7)
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ.yaml")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-multi", action="store_false")
    parser.add_argument("-novis", action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymafx/configs/pymafx_config.yaml")
    device = torch.device(f"cuda:{args.gpu_device}")
    cfg.merge_from_list(["test_gpus", [args.gpu_device], "mcube_res", 512, "clean_mesh", True, "test_mode", True, "batch_size", 1])
    cfg.freeze()

    out_obj_dir = osp.join(args.out_dir, cfg.name, "obj")
    out_png_dir = osp.join(args.out_dir, cfg.name, "png")
    os.makedirs(out_obj_dir, exist_ok=True)
    os.makedirs(out_png_dir, exist_ok=True)

    # --- Model Loading ---
    normal_net = Normal.load_from_checkpoint(cfg=cfg, checkpoint_path=cfg.normal_path, map_location=device, strict=False).to(device).eval()
    sapiens_normal_net = ImageProcessor(device=device)
    ifnet = IFGeo.load_from_checkpoint(cfg=cfg, checkpoint_path=cfg.ifnet_path, map_location=device, strict=False).to(device).eval()
    
    print(colored("✅ Models loaded successfully.", "green"))

    # --- Dataset Loading ---
    dataset_param = {
        "image_dir": args.in_dir,
        "seg_dir": args.seg_dir,
        "use_seg": True,    # w/ or w/o segmentation
        "hps_type": cfg.bni.hps_type,    # pymafx/pixie
        "vol_res": cfg.vol_res,
        "single": args.multi,
    }
    dataset = TestDataset(dataset_param, device)
    
    multi_view_data = [data for data in dataset]
    if not multi_view_data:
        raise ValueError("Dataset is empty. Check your -in_dir path.")
    print(colored(f"✅ Loaded {len(multi_view_data)} views from dataset.", "green"))

    # --- Multi-View Canonical SMPL Optimization ---
    print(colored("🚀 Starting Multi-View Canonical SMPL Optimization...", "cyan"))
    
    # Setup Camera Transformation Manager
    cam_param_path = os.path.join(args.in_dir, "cam_params", "camera_parameters.json")
    canonical_frame_id = int(multi_view_data[0]["name"].split("_")[1])
    transform_manager = CameraTransformManager(cam_param_path, target_frame=canonical_frame_id, device=device)

    # Collect initial parameters and align to canonical frame
    pose_list, trans_list, betas_list, exp_list, jaw_list, rotated_global_orients = [], [], [], [], [], []
    T_first_inv = torch.inverse(transform_manager.get_transform_to_target(canonical_frame_id))

    for data in multi_view_data:
        frame_id = int(data["name"].split("_")[1])
        T_view_to_world = transform_manager.get_transform_to_target(frame_id)
        
        # This aligns each view's orientation to the canonical world frame
        R_relative = T_view_to_world[:3, :3] @ T_first_inv[:3, :3].T
        global_orient_mat = rot6d_to_rotmat(data["global_orient"].view(-1, 6)).squeeze(0)
        corrected_orient_mat = R_relative @ global_orient_mat
        rotated_global_orients.append(rotmat_to_rot6d(corrected_orient_mat.unsqueeze(0)))
        
        pose_list.append(data["body_pose"])
        trans_list.append(data["trans"]) # We will average translations later in world space
        betas_list.append(data["betas"])
        exp_list.append(data["exp"])
        jaw_list.append(data["jaw_pose"])

    # Initialize optimizable parameters with the mean
    optimed_pose = torch.stack(pose_list, dim=0).mean(dim=0).requires_grad_(True)
    optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).view(1, 1, 3, 3)
    optimed_betas = torch.stack(betas_list).mean(dim=0).requires_grad_(True)
    optimed_orient = torch.cat(rotated_global_orients).mean(dim=0).requires_grad_(True)
    optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).view(1, 1, 3, 3)
    optimed_trans = torch.stack(trans_list).mean(dim=0).requires_grad_(True) # Initial guess
    
    # Use expression/jaw from the most frontal view (simple heuristic)
    front_view_idx = 0 # Assume first view is frontal, can be improved
    optimed_exp = exp_list[front_view_idx].detach()
    optimed_jaw_pose = jaw_list[front_view_idx].detach()

    optimizer_smpl = torch.optim.Adam([optimed_pose, optimed_betas, optimed_orient, optimed_trans], lr=1e-2, amsgrad=True)
    scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_smpl, 'min', factor=0.5, patience=args.patience)

    loop_smpl = tqdm(range(args.loop_smpl), desc="Optimizing SMPL")
    for i in loop_smpl:
        optimizer_smpl.zero_grad()
        
        total_loss = torch.tensor(0.0, device=device)
        
        # Get canonical SMPL mesh
        smpl_verts, _, smpl_joints = dataset.smpl_model(
            shape_params=optimed_betas, 
            expression_params=optimed_exp,
            body_pose=optimed_pose_mat, 
            global_pose=optimed_orient_mat,
            jaw_pose=optimed_jaw_pose.unsqueeze(1)
        )
        smpl_verts = (smpl_verts + optimed_trans) * multi_view_data[0]["scale"]

        # Loop through each view to compute loss
        for data in multi_view_data:
            frame_id = int(data["name"].split("_")[1])
            T_world_to_view = torch.inverse(transform_manager.get_transform_to_target(frame_id))

            # Project canonical SMPL into current view
            verts_in_view = apply_homogeneous_transform(smpl_verts.squeeze(0), T_world_to_view).unsqueeze(0)
            
            # Render silhouette
            T_normal_F, _ = dataset.render_normal(verts_in_view * torch.tensor([1.0, -1.0, -1.0], device=device), data["smpl_faces"])
            T_mask_F, _ = dataset.render.get_image(type="mask")

            # Silhouette Loss
            sil_loss = F.l1_loss(T_mask_F, data["img_mask"].to(device))
            
            total_loss += sil_loss
        
        total_loss /= len(multi_view_data)
        total_loss.backward()
        optimizer_smpl.step()
        scheduler_smpl.step(total_loss)
        loop_smpl.set_description(f"Optimizing SMPL (Loss: {total_loss.item():.4f})")

    print(colored("✅ Canonical SMPL optimization complete.", "green"))

    # --- Generate Final Canonical SMPL and Per-View Data ---
    with torch.no_grad():
        smpl_verts, _, _ = dataset.smpl_model(
            shape_params=optimed_betas, 
            expression_params=optimed_exp,
            body_pose=rot6d_to_rotmat(optimed_pose), 
            global_pose=rot6d_to_rotmat(optimed_orient.unsqueeze(0)).unsqueeze(1),
            jaw_pose=optimed_jaw_pose.unsqueeze(1)
        )
        canonical_smpl_verts = ((smpl_verts + optimed_trans) * multi_view_data[0]["scale"]).squeeze(0)
        canonical_smpl_faces = multi_view_data[0]["smpl_faces"].squeeze(0).cpu().numpy()

        canonical_smpl_mesh = trimesh.Trimesh(canonical_smpl_verts.cpu().numpy(), canonical_smpl_faces, process=False)
        canonical_smpl_mesh.export(osp.join(out_obj_dir, "canonical_smpl.obj"))
        print(colored("✅ Saved canonical SMPL mesh.", "green"))

    # --- Multi-View Point Cloud Fusion (NEW PIPELINE) ---
    print(colored("🚀 Starting Multi-View Point Cloud Fusion...", "cyan"))
    all_points_world, all_normals_world = [], []
    with open(cam_param_path, "r") as f:
        cam_params_json = json.load(f)

    for data in tqdm(multi_view_data, desc="Generating & Fusing Point Clouds"):
        frame_id = int(data["name"].split("_")[1])
        
        # 1. Get per-view data
        mask = (data["img_mask"] > 0.5).squeeze(0).squeeze(0) # (H, W) boolean
        _, _, H, W = data["img_icon"].shape
        
        # Use SAPIENS for high-quality normals
        normal_map = sapiens_normal_net.process_image(
            Image.fromarray((data["img_raw"][0].permute(1,2,0).numpy() * 255).astype(np.uint8)), "1b", "u2net"
        )
        normal_map = wrap(normal_map, data["uncrop_param"], 0).squeeze(0) # (3, H, W)

        # Get proxy depth from canonical SMPL rendered in this view
        T_world_to_view = torch.inverse(transform_manager.get_transform_to_target(frame_id))
        verts_in_view = apply_homogeneous_transform(canonical_smpl_verts, T_world_to_view).unsqueeze(0)
        depth_map, _ = dataset.render_depth(verts_in_view * torch.tensor([1.0, -1.0, -1.0], device=device))
        depth_map = depth_map.squeeze(0).squeeze(0) # (H, W)

        # 2. Unproject to get camera-space point cloud
        K = get_intrinsics_matrix(cam_params_json[f'frame_{frame_id}'], (W, H))
        points_cam = unproject_pixels_to_camera_space(depth_map, mask, K, device=device)
        normals_cam = sample_normals_at_pixels(normal_map, mask)

        # 3. Transform to canonical world space
        T_view_to_world = transform_manager.get_transform_to_target(frame_id)
        points_world = apply_homogeneous_transform(points_cam, T_view_to_world)
        normals_world = torch.matmul(normals_cam, T_view_to_world[:3, :3].T)
        
        all_points_world.append(points_world)
        all_normals_world.append(normals_world)

    # 4. Combine, filter, and reconstruct
    fused_points = torch.cat(all_points_world, dim=0).cpu().numpy()
    fused_normals = torch.cat(all_normals_world, dim=0).cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_points)
    pcd.normals = o3d.utility.Vector3dVector(fused_normals)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print(colored(f"Generated fused point cloud with {len(pcd.points)} points. Running Poisson reconstruction...", "blue"))
    base_recon_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    base_recon_mesh.remove_degenerate_triangles()
    base_recon_mesh.remove_duplicated_triangles()
    base_recon_mesh.remove_unreferenced_vertices()

    base_recon_path = osp.join(out_obj_dir, "base_fused_recon.obj")
    o3d.io.write_triangle_mesh(base_recon_path, base_recon_mesh)
    print(colored(f"✅ Saved base fused mesh to {base_recon_path}", "green"))
    
    # --- Intelligent Refinement Loop (NEW PIPELINE) ---
    print(colored("🚀 Starting Intelligent Refinement...", "cyan"))
    
    # 1. Prepare mesh for refinement
    verts_base = torch.tensor(np.asarray(base_recon_mesh.vertices), dtype=torch.float32).to(device)
    faces_base = torch.tensor(np.asarray(base_recon_mesh.triangles), dtype=torch.int64).to(device)
    
    # Remesh for better topology
    verts_remesh, faces_remesh = remesh(
        trimesh.Trimesh(verts_base.cpu().numpy(), faces_base.cpu().numpy()),
        osp.join(out_obj_dir, "remeshed_base.obj"),
        device
    )
    base_mesh = Meshes(verts=verts_remesh, faces=faces_remesh).to(device)
    
    # 2. Setup optimization
    local_affine_model = LocalAffine(base_mesh.verts_padded().shape[1], 1, base_mesh.edges_packed()).to(device)
    optimizer_cloth = torch.optim.Adam(local_affine_model.parameters(), lr=1e-4)
    scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cloth, 'min', factor=0.5, patience=20, verbose=False)

    # 3. Refinement loop with annealing weights
    num_iters = 300
    loop_cloth = tqdm(range(num_iters), desc="Refining Mesh")
    initial_lapla_w, final_lapla_w = 5.0, 0.5
    initial_normal_w, final_normal_w = 0.5, 2.0

    for i in loop_cloth:
        optimizer_cloth.zero_grad()
        
        progress = i / num_iters
        w_lapla = initial_lapla_w * (1 - progress) + final_lapla_w * progress
        w_normal = initial_normal_w * (1 - progress) + final_normal_w * progress

        deformed_verts, stiffness, rigid = local_affine_model(base_mesh.verts_padded())
        deformed_mesh = Meshes(verts=deformed_verts, faces=base_mesh.faces_padded())
        
        # --- Regularization Losses ---
        loss_lapla = laplacian_smoothing(deformed_mesh, method="uniform")
        loss_stiff = torch.mean(stiffness)
        loss_rigid = torch.mean(rigid)
        loss_chamfer, _ = chamfer_distance(deformed_verts, base_mesh.verts_padded())

        # --- Multi-View Data Losses ---
        total_normal_loss = torch.tensor(0.0, device=device)
        total_mask_loss = torch.tensor(0.0, device=device)

        for data in multi_view_data:
            frame_id = int(data["name"].split("_")[1])
            gt_mask = data["img_mask"].to(device)
            
            # Use pre-computed high-quality normals
            normal_map_raw = Image.fromarray((data["img_raw"][0].permute(1,2,0).numpy() * 255).astype(np.uint8))
            gt_normal_map = wrap(sapiens_normal_net.process_image(normal_map_raw, "1b", "u2net"), data["uncrop_param"], 0)

            # Project deforming mesh into view
            T_world_to_view = torch.inverse(transform_manager.get_transform_to_target(frame_id))
            verts_view = apply_homogeneous_transform(deformed_verts.squeeze(0), T_world_to_view).unsqueeze(0)
            
            # Render normals and mask
            pred_normal_map, _ = dataset.render_normal(verts_view * torch.tensor([1.0, -1.0, -1.0], device=device), deformed_mesh.faces_padded())
            pred_mask, _ = dataset.render.get_image(type="mask")

            # Calculate weighted losses for this view
            valid_mask = (gt_mask > 0.5)
            normal_diff = 1.0 - F.cosine_similarity(pred_normal_map, gt_normal_map, dim=1, eps=1e-6)
            total_normal_loss += (normal_diff.unsqueeze(1)[valid_mask]).mean()
            total_mask_loss += F.l1_loss(pred_mask, gt_mask)

        avg_normal_loss = total_normal_loss / len(multi_view_data)
        avg_mask_loss = total_mask_loss / len(multi_view_data)
        
        # --- Total Loss Combination ---
        total_loss = (
            avg_normal_loss * w_normal +
            avg_mask_loss * 1.0 +
            loss_lapla * w_lapla +
            loss_stiff * 0.5 +
            loss_rigid * 0.5 +
            loss_chamfer * 0.1
        )
        
        total_loss.backward()
        optimizer_cloth.step()
        scheduler_cloth.step(total_loss)
        
        loop_cloth.set_description(f"Refining (Total: {total_loss.item():.4f} | Normal: {avg_normal_loss.item():.4f})")
        
        # Update base mesh for next iteration's deformation
        base_mesh = deformed_mesh.detach()

    print(colored("✅ Intelligent refinement complete.", "green"))

    # --- Final Export ---
    final_verts_ts = base_mesh.verts_packed()
    final_faces_ts = base_mesh.faces_packed()

    # Query vertex colors from the first view's image
    final_colors = query_color(
        final_verts_ts, final_faces_ts,
        multi_view_data[0]["img_icon"].to(device),
        device=device
    )
    
    final_obj = trimesh.Trimesh(
        final_verts_ts.cpu().numpy(),
        final_faces_ts.cpu().numpy(),
        vertex_colors=final_colors,
        process=False
    )
    
    final_path = osp.join(out_obj_dir, f"{multi_view_data[0]['name']}_final_refined.obj")
    final_obj.export(final_path)
    print(colored(f"🎉🎉🎉 Final avatar saved to: {final_path} 🎉🎉🎉", "magenta"))