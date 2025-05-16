#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
from termcolor import colored
from tqdm.auto import tqdm
import os.path as osp

from apps.SilhouetteEvaluator import SilhouetteEvaluator
from apps.CameraTransformManager import CameraTransformManager
from lib.common.config import cfg
from lib.dataset.TestDataset import TestDataset

def get_frame_id(name):
    """Extract frame ID from the input name (e.g., 'frame_0001' -> 1)"""
    if '_' in name:
        return int(name.split('_')[1])
    return int(name)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Select best canonical SMPL body based on silhouette evaluation')
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples", help="Input directory with images")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None, help="Directory with segmentation masks")
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ.yaml", help="Config file")
    parser.add_argument("-gpu", "--gpu_device", type=int, default=0, help="GPU device ID")
    parser.add_argument("-start", "--start_frame", type=int, default=None, help="Start frame ID for processing")
    parser.add_argument("-end", "--end_frame", type=int, default=None, help="End frame ID for processing")
    parser.add_argument("-target", "--target_frame", type=int, default=None, help="Target frame ID as initial guess")
    args = parser.parse_args()

    # Load configuration
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymafx/configs/pymafx_config.yaml")
    device = torch.device(f"cuda:{args.gpu_device}")

    # Settings for testing
    cfg_show_list = [
        "test_gpus", [args.gpu_device], 
        "mcube_res", 512, 
        "clean_mesh", True, 
        "test_mode", True,
        "batch_size", 1
    ]
    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    # Create dataset
    dataset_param = {
        "image_dir": args.in_dir,
        "seg_dir": args.seg_dir,
        "use_seg": True,
        "hps_type": cfg.bni.hps_type,
        "vol_res": cfg.vol_res,
        "single": False,  # Process multiple frames
    }
    dataset = TestDataset(dataset_param, device)
    
    # Load camera parameters
    cam_param_path = os.path.join(args.in_dir, "cam_params", "camera_parameters.json")
    target_frame_id = args.target_frame if args.target_frame is not None else 0
    transform_manager = CameraTransformManager(cam_param_path, target_frame=target_frame_id, device=device)
    
    # Create SilhouetteEvaluator
    evaluator = SilhouetteEvaluator(dataset, device, transform_manager, args.out_dir, cfg)
    
    # Process frames and collect ground truth masks and SMPL predictions
    gt_masks = {}
    
    print(colored(f"Processing frames to collect silhouette data...", "green"))
    
    # Determine range of frames to process
    if args.start_frame is not None and args.end_frame is not None:
        frame_range = range(args.start_frame, args.end_frame + 1)
    else:
        frame_range = range(len(dataset))
    
    for i in tqdm(frame_range):
        if i >= len(dataset):
            print(f"Skipping frame {i} as it exceeds dataset length {len(dataset)}")
            continue
            
        data = dataset[i]
        frame_id = get_frame_id(data["name"])
        
        # Extract ground truth mask from the data
        gt_mask = data["img_mask"].to(device)
        gt_masks[frame_id] = {'front': gt_mask, 'back': gt_mask.clone()}
        
        # Run SMPL optimization (simplified version from infer_multi.py)
        # In a real application, you would run your SMPL optimization here
        # For this demo, we're using a simplified approach
        
        optimed_pose = data["body_pose"]
        optimed_trans = data["trans"]
        optimed_betas = data["betas"]
        optimed_orient = data["global_orient"]
        
        # Get SMPL vertices and faces
        N_body, N_pose = optimed_pose.shape[:2]
        optimed_orient_mat = torch.eye(3, 3).unsqueeze(0).unsqueeze(0).repeat(N_body, 1, 1, 1).to(device)
        optimed_pose_mat = torch.eye(3, 3).unsqueeze(0).unsqueeze(0).repeat(N_body, N_pose, 1, 1).to(device)
        
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
        
        # Register this SMPL body as a candidate
        evaluator.register_candidate(
            frame_id,
            smpl_verts, 
            data["smpl_faces"],
            data["scale"]
        )
        
        # Render and save mask for visualization
        dataset.render.load_meshes(
            smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
            data["smpl_faces"]
        )
        front_mask, back_mask = dataset.render.get_image(type="mask")
        
        # Save masks for each frame
        os.makedirs(osp.join(args.out_dir, cfg.name, "masks"), exist_ok=True)
        torch.save(
            {"front": front_mask, "back": back_mask},
            osp.join(args.out_dir, cfg.name, "masks", f"frame_{frame_id:04d}_masks.pt")
        )
    
    # Run silhouette evaluation to find the best canonical SMPL body
    best_frame_id, best_verts, best_faces, avg_diff = evaluator(gt_masks)
    
    # Save the best canonical SMPL body
    canonical_dir = osp.join(args.out_dir, cfg.name, "canonical")
    os.makedirs(canonical_dir, exist_ok=True)
    
    torch.save(
        {
            "verts": best_verts,
            "faces": best_faces,
            "frame_id": best_frame_id,
            "avg_diff": avg_diff
        },
        osp.join(canonical_dir, "canonical_smpl.pt")
    )
    
    print(colored(f"Saved best canonical SMPL body from frame {best_frame_id} to {canonical_dir}", "green"))

if __name__ == "__main__":
    main() 