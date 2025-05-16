#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to find the best canonical SMPL body from a multi-view dataset.
This script only performs SMPL body estimation and silhouette evaluation,
saving the best canonical body for later use in infer_multi.py.
"""

import argparse
import os
import os.path as osp
import sys
import torch
import torch.nn as nn
import warnings
import numpy as np
from tqdm.auto import tqdm
from termcolor import colored
import torchvision
import trimesh
import json

# Add the root directory to the Python path to allow imports to work properly
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from apps.SilhouetteEvaluator import SilhouetteEvaluator
from apps.CameraTransformManager import CameraTransformManager
from lib.common.config import cfg
from lib.common.train_util import init_loss
from lib.common.train_util import Format
from lib.dataset.TestDataset import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis

warnings.filterwarnings("ignore")


def save_mesh_preview(dataset, verts, faces, output_path, device):
    """
    Render and save a preview of the mesh from multiple viewpoints.
    
    Args:
        dataset: Dataset with render function
        verts: Mesh vertices
        faces: Mesh faces
        output_path: Path to save the preview image
        device: Torch device
    """
    # Load the mesh
    dataset.render.load_meshes(
        verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
        faces
    )
    
    # Render from different camera angles
    renders = dataset.render.get_image(cam_type="four")
    
    # Save the visualization
    torchvision.utils.save_image(
        torch.cat(renders, dim=3),
        output_path
    )
    
    return renders


def validate_camera_params(data):
    """
    Validate the camera parameters JSON structure.
    
    Args:
        data: The loaded JSON data
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if data is a list
    if not isinstance(data, list):
        print(colored("Error: Camera parameters should be a list of frame entries", "red"))
        return False
    
    # Check for empty data
    if len(data) == 0:
        print(colored("Error: Camera parameters file contains no frames", "red"))
        return False
    
    # Check required fields in each entry
    required_fields = ["frame", "location", "quaternion"]
    for i, entry in enumerate(data):
        for field in required_fields:
            if field not in entry:
                print(colored(f"Error: Entry {i} is missing required field '{field}'", "red"))
                return False
        
        # Validate location (3D vector)
        if not isinstance(entry["location"], list) or len(entry["location"]) != 3:
            print(colored(f"Error: Entry {i} has invalid 'location' (should be a 3D vector)", "red"))
            return False
            
        # Validate quaternion (4D vector [w, x, y, z])
        if not isinstance(entry["quaternion"], list) or len(entry["quaternion"]) != 4:
            print(colored(f"Error: Entry {i} has invalid 'quaternion' (should be a 4D vector)", "red"))
            return False
    
    return True


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Find the best canonical SMPL body')
    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=50)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ.yaml")
    parser.add_argument("-start_frame", type=int, default=None, help="Start frame index")
    parser.add_argument("-end_frame", type=int, default=None, help="End frame index")
    parser.add_argument("-temp_target", type=int, default=0, help="Temporary target frame for initial transformations")
    parser.add_argument("-save_steps", action="store_true", help="Save intermediate optimization steps")
    parser.add_argument("-save_viz", action="store_true", help="Save visualizations of all candidate SMPL bodies")
    parser.add_argument("-check_transforms", action="store_true", help="Check and print available frame IDs in camera_parameters.json")
    args = parser.parse_args()

    # Load configuration
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymafx/configs/pymafx_config.yaml")
    
    # GPU device setup
    device = torch.device(f"cuda:{args.gpu_device}")
    
    # Configure test settings
    cfg_show_list = [
        "test_gpus", [args.gpu_device],
        "mcube_res", 512,
        "clean_mesh", True,
        "test_mode", True,
        "batch_size", 1
    ]
    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()
    
    # Check camera transforms if requested
    cam_param_path = os.path.join(args.in_dir, "cam_params", "camera_parameters.json")
    if args.check_transforms:
        if not os.path.exists(cam_param_path):
            print(colored(f"Error: Camera parameters file not found at {cam_param_path}", "red"))
            return
            
        try:
            with open(cam_param_path, "r") as f:
                data = json.load(f)
                
            print(colored(f"Camera transforms file found at {cam_param_path}", "green"))
            print(colored(f"Contains {len(data)} frame entries", "green"))
            
            # Validate the JSON structure
            if not validate_camera_params(data):
                print(colored("Camera parameters file has an invalid structure", "red"))
                return
            
            # Print available frame IDs
            frame_ids = sorted([entry.get("frame") for entry in data])
            print(colored(f"Available frame IDs: {frame_ids}", "green"))
            
            # Print sample data structure
            if len(data) > 0:
                print(colored("Sample data structure for first frame:", "green"))
                print(data[0])
                
            return
        except Exception as e:
            print(colored(f"Error parsing camera parameters file: {e}", "red"))
            return
    
    # Set up dataset
    dataset_param = {
        "image_dir": args.in_dir,
        "seg_dir": args.seg_dir,
        "use_seg": True,    # w/ or w/o segmentation
        "hps_type": cfg.bni.hps_type,    # pymafx/pixie
        "vol_res": cfg.vol_res,
        "single": False,  # Process multiple frames
    }
    dataset = TestDataset(dataset_param, device)
    
    # Initialize the camera transformation manager
    try:
        transform_manager = CameraTransformManager(
            cam_param_path, target_frame=args.temp_target, device=device
        )
    except Exception as e:
        print(colored(f"Error initializing CameraTransformManager: {e}", "red"))
        print(colored("This might be due to missing camera parameters or incorrect frame IDs.", "red"))
        print(colored("Please check your camera_parameters.json file.", "red"))
        return
    
    # Initialize SilhouetteEvaluator
    silhouette_evaluator = SilhouetteEvaluator(dataset, device, transform_manager, args.out_dir, cfg)
    
    # Create output directories
    os.makedirs(osp.join(args.out_dir, cfg.name, "canonical"), exist_ok=True)
    if args.save_steps:
        os.makedirs(osp.join(args.out_dir, cfg.name, "smpl_steps"), exist_ok=True)
    if args.save_viz:
        os.makedirs(osp.join(args.out_dir, cfg.name, "viz"), exist_ok=True)
    
    # Determine frame range
    if args.start_frame is not None and args.end_frame is not None:
        frame_range = range(args.start_frame, args.end_frame + 1)
    else:
        frame_range = range(len(dataset))
    
    # Dictionary to store ground truth masks
    gt_masks = {}
    
    # Process each frame to estimate SMPL and register candidates
    print(colored("Processing frames to estimate SMPL bodies and collect ground truth masks...", "green"))
    
    for frame_idx in tqdm(frame_range):
        if frame_idx >= len(dataset):
            print(f"Skipping frame {frame_idx} as it exceeds dataset length {len(dataset)}")
            continue
        
        # Get the data for this frame
        data = dataset[frame_idx]
        try:
            frame_id = int(data['name'].split('_')[1]) if '_' in data['name'] else int(data['name'])
        except (ValueError, IndexError):
            print(colored(f"Warning: Could not parse frame ID from {data['name']}. Using index {frame_idx} instead.", "yellow"))
            frame_id = frame_idx
        
        # Check if this frame exists in the transforms
        try:
            # Temporarily set this frame as target to verify it exists
            original_target = transform_manager.target_frame
            transform_manager.target_frame = frame_id
            # Try to get a transform (even if it's identity) to verify frame exists
            transform_manager.get_transform_to_target(frame_id)
            # Restore original target
            transform_manager.target_frame = original_target
        except KeyError:
            print(colored(f"Warning: Frame {frame_id} not found in camera transforms. Skipping.", "yellow"))
            continue
        
        print(colored(f"\nProcessing frame {frame_id}", "blue"))
        
        # Store the ground truth segmentation mask
        gt_mask = data["img_mask"].to(device)
        gt_masks[frame_id] = {'front': gt_mask, 'back': gt_mask.clone()}
        
        # Initialize losses
        losses = init_loss()
        
        # Set up optimizers for SMPL parameters
        optimed_pose = data["body_pose"].requires_grad_(True)
        optimed_trans = data["trans"].requires_grad_(True)
        optimed_betas = data["betas"].requires_grad_(True)
        optimed_orient = data["global_orient"].requires_grad_(True)
        
        optimizer_smpl = torch.optim.Adam([
            optimed_pose, optimed_trans, optimed_betas, optimed_orient
        ], lr=1e-2, amsgrad=True)
        
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=args.patience,
        )
        
        # Tensor for input to the model
        in_tensor = {
            "smpl_faces": data["smpl_faces"],
            "image": data["img_icon"].to(device),
            "mask": data["img_mask"].to(device)
        }
        
        # SMPL optimization loop
        loop_smpl = tqdm(range(args.loop_smpl))
        
        for i in loop_smpl:
            optimizer_smpl.zero_grad()
            
            N_body, N_pose = optimed_pose.shape[:2]
            
            # 6d_rot to rot_mat
            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).view(N_body, 1, 3, 3)
            optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).view(N_body, N_pose, 3, 3)
            
            # Get SMPL vertices, landmarks, and joints
            smpl_verts, smpl_landmarks, smpl_joints = dataset.smpl_model(
                shape_params=optimed_betas,
                expression_params=data["exp"].to(device),
                body_pose=optimed_pose_mat,
                global_pose=optimed_orient_mat,
                jaw_pose=data["jaw_pose"].to(device),
                left_hand_pose=data["left_hand_pose"].to(device),
                right_hand_pose=data["right_hand_pose"].to(device),
            )
            
            smpl_verts = (smpl_verts + optimed_trans) * data["scale"]
            smpl_joints = (smpl_joints + optimed_trans) * data["scale"] * torch.tensor([
                1.0, 1.0, -1.0
            ]).to(device)
            
            # 3D joint errors
            smpl_joints_3d = (smpl_joints[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0) * 0.5
            in_tensor["smpl_joint"] = smpl_joints[:, dataset.smpl_data.smpl_joint_ids_24_pixie, :]
            
            # Landmark errors
            ghum_lmks = data["landmark"][:, dataset.smpl_data.ghum_smpl_pairs[:, 0], :2].to(device)
            ghum_conf = data["landmark"][:, dataset.smpl_data.ghum_smpl_pairs[:, 0], -1].to(device)
            smpl_lmks = smpl_joints_3d[:, dataset.smpl_data.ghum_smpl_pairs[:, 1], :2]
            
            # Render optimized mesh as normal [-1,1]
            in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
                in_tensor["smpl_faces"],
            )
            
            # Get silhouette masks
            T_mask_F, T_mask_B = dataset.render.get_image(type="mask")
            
            # Silhouette loss
            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
            gt_arr = in_tensor["mask"].repeat(1, 1, 2)
            diff_S = torch.abs(smpl_arr - gt_arr)
            losses["silhouette"]["value"] = diff_S.mean()
            
            # Large cloth overlap detection
            cloth_overlap = diff_S.sum(dim=[1, 2]) / gt_arr.sum(dim=[1, 2])
            cloth_overlap_flag = cloth_overlap > cfg.cloth_overlap_thres
            losses["joint"]["weight"] = [10.0 if flag else 1.0 for flag in cloth_overlap_flag]
            
            # Small body overlap detection (occlusion)
            bg_value = in_tensor["T_normal_F"][0, 0, 0, 0]
            smpl_arr_fake = torch.cat([
                in_tensor["T_normal_F"][:, 0].ne(bg_value).float(),
                in_tensor["T_normal_B"][:, 0].ne(bg_value).float()
            ], dim=-1)
            
            body_overlap = (gt_arr * smpl_arr_fake.gt(0.0)
                          ).sum(dim=[1, 2]) / smpl_arr_fake.gt(0.0).sum(dim=[1, 2])
            body_overlap_mask = (gt_arr * smpl_arr_fake).unsqueeze(1)
            body_overlap_flag = body_overlap < cfg.body_overlap_thres
            
            # Normal loss
            diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["mask"].repeat(1, 3, 1, 1))
            diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["mask"].repeat(1, 3, 1, 1))
            losses["normal"]["value"] = (
                diff_F_smpl * body_overlap_mask[..., :512] +
                diff_B_smpl * body_overlap_mask[..., 512:]
            ).mean() / 2.0
            
            # Adjust weights for occluded frames
            losses["silhouette"]["weight"] = [0 if flag else 1.0 for flag in body_overlap_flag]
            occluded_idx = torch.where(body_overlap_flag)[0]
            ghum_conf[occluded_idx] *= ghum_conf[occluded_idx] > 0.95
            losses["joint"]["value"] = (torch.norm(ghum_lmks - smpl_lmks, dim=2) * ghum_conf).mean(dim=1)
            
            # Weighted sum of the losses
            smpl_loss = 0.0
            pbar_desc = f"Frame {frame_id} Body Fitting -- "
            for k in ["normal", "silhouette", "joint"]:
                per_loop_loss = (
                    losses[k]["value"] * torch.tensor(losses[k]["weight"]).to(device)
                ).mean()
                pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                smpl_loss += per_loop_loss
            
            pbar_desc += f"Total: {smpl_loss:.3f}"
            loop_smpl.set_description(pbar_desc)
            
            # Update parameters
            smpl_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(smpl_loss)
        
        # After optimization is complete, register this candidate
        silhouette_evaluator.register_candidate(
            frame_id,
            smpl_verts.detach().clone(),
            data["smpl_faces"].detach().clone(),
            data["scale"].item()
        )
        
        # Save the SMPL parameters if requested
        if args.save_steps:
            smpl_info = {
                "betas": optimed_betas.detach().cpu(),
                "body_pose": rotation_matrix_to_angle_axis(optimed_pose_mat.detach()).cpu(),
                "global_orient": rotation_matrix_to_angle_axis(optimed_orient_mat.detach()).cpu(),
                "transl": optimed_trans.detach().cpu(),
                "expression": data["exp"].cpu(),
                "jaw_pose": rotation_matrix_to_angle_axis(data["jaw_pose"]).cpu(),
                "left_hand_pose": rotation_matrix_to_angle_axis(data["left_hand_pose"]).cpu(),
                "right_hand_pose": rotation_matrix_to_angle_axis(data["right_hand_pose"]).cpu(),
                "scale": data["scale"].cpu(),
                "vertices": smpl_verts.detach().cpu(),
                "faces": data["smpl_faces"].detach().cpu()
            }
            
            smpl_path = osp.join(args.out_dir, cfg.name, "smpl_steps", f"frame_{frame_id:04d}_smpl.pt")
            torch.save(smpl_info, smpl_path)
            print(f"Saved SMPL parameters for frame {frame_id} to {smpl_path}")
        
        # Save visualization if requested
        if args.save_viz:
            viz_path = osp.join(args.out_dir, cfg.name, "viz", f"frame_{frame_id:04d}_smpl.png")
            save_mesh_preview(
                dataset, 
                smpl_verts.detach() * torch.tensor([1.0, 1.0, 1.0]).to(device), 
                data["smpl_faces"].detach(), 
                viz_path,
                device
            )
            print(f"Saved visualization for frame {frame_id} to {viz_path}")
            
            # Also save obj file
            mesh_path = osp.join(args.out_dir, cfg.name, "viz", f"frame_{frame_id:04d}_smpl.obj")
            mesh = trimesh.Trimesh(
                smpl_verts.detach().cpu()[0] * torch.tensor([1.0, -1.0, 1.0]),
                data["smpl_faces"].detach().cpu()[0][:, [0, 2, 1]],
                process=False,
                maintains_order=True,
            )
            mesh.export(mesh_path)
    
    # Evaluate all candidates and select the best canonical SMPL body
    print(colored("\nEvaluating all candidate SMPL bodies to select the best canonical body...", "green"))
    best_frame_id, best_verts, best_faces, avg_diff = silhouette_evaluator(gt_masks)
    
    # Save the canonical SMPL body
    canonical_path = osp.join(args.out_dir, cfg.name, "canonical", "canonical_smpl.pt")
    torch.save({
        "verts": best_verts,
        "faces": best_faces,
        "frame_id": best_frame_id,
        "avg_diff": avg_diff
    }, canonical_path)
    
    # Save visualization of the best canonical SMPL body
    if args.save_viz:
        canonical_viz_path = osp.join(args.out_dir, cfg.name, "canonical", "canonical_smpl_viz.png")
        save_mesh_preview(
            dataset, 
            best_verts * torch.tensor([1.0, 1.0, 1.0]).to(device), 
            best_faces, 
            canonical_viz_path,
            device
        )
        print(f"Saved visualization of canonical SMPL body to {canonical_viz_path}")
        
        # Also save as obj
        canonical_mesh_path = osp.join(args.out_dir, cfg.name, "canonical", "canonical_smpl.obj")
        canonical_mesh = trimesh.Trimesh(
            best_verts.cpu()[0] * torch.tensor([1.0, -1.0, 1.0]),
            best_faces.cpu()[0][:, [0, 2, 1]],
            process=False,
            maintains_order=True,
        )
        canonical_mesh.export(canonical_mesh_path)
    
    # Create a summary visualization with all candidates
    if args.save_viz and len(silhouette_evaluator.candidates) > 1:
        candidate_frames = list(silhouette_evaluator.candidates.keys())
        summary_rows = []
        n_cols = 4  # Show 4 angles for each candidate
        
        # Heading row with frame numbers and difference percentages
        heading_img = torch.ones(1, 3, 64, n_cols * 128).to(device) * 0.5
        for i, frame_id in enumerate(candidate_frames):
            avg_diff = silhouette_evaluator.avg_diffs.get(frame_id, 0.0)
            # Highlight the best frame
            color = torch.tensor([0.0, 1.0, 0.0]).to(device) if frame_id == best_frame_id else torch.tensor([1.0, 1.0, 1.0]).to(device)
            
            # Create text image with frame info
            text = f"Frame {frame_id} (Diff: {avg_diff:.2f}%)"
            summary_rows.append(heading_img.clone())
            
        # For each candidate, render from 4 angles
        for frame_id in candidate_frames:
            verts = silhouette_evaluator.candidates[frame_id]['verts']
            faces = silhouette_evaluator.candidates[frame_id]['faces']
            
            dataset.render.load_meshes(
                verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
                faces
            )
            renders = dataset.render.get_image(cam_type="four")
            row_img = torch.cat(renders, dim=3)
            
            # Highlight the best frame with green border
            if frame_id == best_frame_id:
                border_size = 3
                row_img[:, :, :border_size, :] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).to(device)
                row_img[:, :, -border_size:, :] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).to(device)
                row_img[:, :, :, :border_size] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).to(device)
                row_img[:, :, :, -border_size:] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).to(device)
            
            summary_rows.append(row_img)
        
        # Concatenate all rows
        summary_img = torch.cat(summary_rows, dim=0)
        summary_path = osp.join(args.out_dir, cfg.name, "canonical", "all_candidates_summary.png")
        torchvision.utils.save_image(summary_img, summary_path)
        print(f"Saved summary visualization of all candidates to {summary_path}")
    
    print(colored(f"Selected frame {best_frame_id} as the canonical SMPL body with average silhouette difference: {avg_diff:.2f}%", "green"))
    print(colored(f"Saved canonical SMPL body to {canonical_path}", "green"))
    print(colored(f"You can now run infer_multi.py with -target_frame {best_frame_id}", "green"))


if __name__ == "__main__":
    main() 