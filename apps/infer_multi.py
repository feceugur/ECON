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

from apps.FaceRigExporter import FaceRigExporter

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
import os
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

from apps.IFGeo import IFGeo
from apps.Normal import Normal
from apps.sapiens import ImageProcessor
from apps.clean_mesh import MeshCleanProcess
from apps.SMPLXJointAligner import SMPLXJointAligner
from apps.CameraTransformManager import CameraTransformManager

from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.common.config import cfg
from lib.common.imutils import blend_rgb_norm, load_img, transform_to_tensor, wrap
from lib.common.local_affine import register
from lib.common.render import query_color
from lib.common.train_util import Format, init_loss
from lib.common.voxelize import VoxelGrid
from lib.dataset.mesh_util import *
from lib.dataset.TestDataset import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis, rotmat_to_rot6d

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Add this function at the beginning of your script
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
    kmat[0, 1], kmat[0, 2] = -v[0, 2], v[0, 1]
    kmat[1, 0], kmat[1, 2] = v[0, 2], -v[0, 0]
    kmat[2, 0], kmat[2, 1] = -v[0, 1], v[0, 0]

    rotation = torch.eye(3, device=a.device) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-8))
    return rotation

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=40)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ.yaml")
    parser.add_argument("-multi", action="store_false")
    parser.add_argument("-novis", action="store_true")

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

    canonical_smpl_verts = None
    canonical_smpl_joints = None
    canonical_smpl_landmarks = None
    canonical_saved = False

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

        head_indices = [12,15]  # Neck and head
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

        closest_exp = exp_list[6].detach()
        closest_jaw = jaw_list[6].detach()
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
        """
        optimizer_smpl = torch.optim.Adam([
            optimed_trans, optimed_betas, optimed_orient,
            optimed_pose_rest, optimed_jaw_pose
        ], lr=1e-2, amsgrad=True)
        """
        optimizer_smpl = torch.optim.Adam([
            {'params': optimed_pose_rest, 'lr': 1e-2},
            {'params': optimed_pose_head, 'lr': 1e-2},  # much smaller
            {'params': optimed_trans, 'lr': 1e-2},
            {'params': optimed_betas, 'lr': 1e-2},
            {'params': optimed_orient, 'lr': 1e-2},
            #{'params': optimed_jaw_pose, 'lr': 1e-2},
        ], amsgrad=True)
        
        # === Scheduler ===
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl, mode="min", factor=0.5, verbose=0, min_lr=1e-5, patience=args.patience,
        )

        print(colored("✅ Canonical SMPL Initialization done (multi-view aware)", "cyan"))

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

                """
                # === Compute head direction from joints ===
                head_joint_idx = 15
                neck_joint_idx = 12
                head_pos = joints_in_view[:, head_joint_idx]
                neck_pos = joints_in_view[:, neck_joint_idx]
                head_direction = head_pos - neck_pos  # [1, 3]

                # === Compute camera-based target direction ===
                cam_entry = next(cam for cam in cam_params if cam["frame"] == frame_id)
                R_canon2view = T_frame_to_target[:3, :3]
                # 2. vector (0,0,0) → camera in canonical/world space
                cam_pos_world = torch.as_tensor(cam_entry["location"],
                                                device=device, dtype=torch.float32)  # [3]
                direction_world = F.normalize(-cam_pos_world, dim=0)                   # unit

                # 3. bring that vector into *view* space
                direction_view = F.normalize(R_canon2view @ direction_world, dim=0)   # [3]
                target_dir = direction_view.unsqueeze(0)  

                head_dir_unit = F.normalize(head_direction, dim=1)
                head_align_loss = (1 - torch.sum(head_dir_unit * target_dir, dim=1)).mean()

                cos_sim = torch.sum(head_dir_unit * target_dir, dim=1)
                print("iteration", i, "view", frame_id, "cos_sim", cos_sim.mean())
                """

                # Landmark processing
                smpl_joints_3d = (joints_in_view[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0) * 0.5
                smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]
                ghum_lmks = view_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                ghum_conf = view_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)

                # A. Apply vertical offset (before loss calculation)
                # This shifts SMPL landmarks down to better align with GT
                smpl_lmks[:, :, 1] = smpl_lmks[:, :, 1] + 0.03  # Shift down by 2% of image height

                # Save landmark visualization
                
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
                view_total_loss = sil_loss + 0.5 * normal_loss + 0.1 * landmark_loss
                total_loss += view_total_loss

            total_loss /= len(multi_view_data)

            # === Light regularization to keep face close to canonical ===
            # reg_exp = F.smooth_l1_loss(optimed_exp, closest_exp)
            # reg_jaw = F.smooth_l1_loss(optimed_jaw_pose, closest_jaw)
            # reg_loss = 0.1 * (reg_exp + reg_jaw) 

            loss_values.append(total_loss.item())
            total_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(total_loss)
            # print(f"Iter {i} - Grad norm optimed_pose_rest: {optimed_pose_rest.grad.norm().item() if optimed_pose_rest.grad is not None else 'None'}")
            # print(f"Iter {i} - Grad norm optimed_pose_head: {optimed_pose_head.grad.norm().item() if optimed_pose_head.grad is not None else 'None'}")
            # print(f"Iter {i} - Grad norm optimed_jaw_pose: {optimed_jaw_pose.grad.norm().item() if optimed_jaw_pose.grad is not None else 'None'}")
            # print(f"Iter {i} - Grad norm optimed_orient: {optimed_orient.grad.norm().item() if optimed_orient.grad is not None else 'None'}")
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

        
        def rot_from_a_to_b(a, b):                       # both shape [3]
            a = F.normalize(a, dim=0);  b = F.normalize(b, dim=0)
            v = torch.cross(a, b)
            c = torch.dot(a, b)
            s = v.norm()
            vx = torch.tensor([[   0, -v[2],  v[1]],
                            [ v[2],    0, -v[0]],
                            [-v[1],  v[0],    0]], device=a.device)
            R = torch.eye(3, device=a.device) + vx + vx @ vx * ((1 - c)/(s**2 + 1e-8))
            return R
        
        def remove_roll(R: torch.Tensor,
                world_up=torch.tensor([0., 0., 1.])) -> torch.Tensor:
            """
            R : 3×3 rotation matrix (columns = right, up, forward)
            Returns a new 3×3 whose forward axis == R[:,2] but whose up axis
            is re-aligned with the global Z-up direction (no roll).
            Blender uses Z-up coordinate system.
            """
            fwd = F.normalize(R[:, 2], dim=0)                       # keep same forward (z)
            up_target = world_up.to(R.device)  # [0, 0, 1] for Z-up

            # if forward ~ parallel to world_up, switch to X-up fallback
            if torch.abs(torch.dot(fwd, up_target)) > 0.999:
                up_target = torch.tensor([1., 0., 0.], device=R.device)

            right = F.normalize(torch.cross(up_target, fwd), dim=0) # new x
            up    = torch.cross(fwd, right)                         # new z (vertical)
            return torch.stack([right, up, fwd], dim=1)             # 3×3

        last_frame_id = int(multi_view_data[-1]["name"].split("_")[1])   # <= moved up

        # 1. current head→neck direction
        with torch.no_grad():
            neck_pos = smpl_joints[0, 123]
            head_pos = smpl_joints[0, 114]
            curr_dir = F.normalize(head_pos - neck_pos, dim=0)

        # 2. desired direction (towards last camera)
        last_cam = next(c for c in cam_params if c["frame"] == last_frame_id)
        cam_pos  = torch.as_tensor(last_cam["location"], device=device, dtype=torch.float32)
        target_dir = F.normalize(-cam_pos, dim=0)          # body → camera

        # 3. rotation that aligns the two
        R_fix = rot_from_a_to_b(curr_dir, target_dir)      # function you already defined

        # 4. apply only to neck and head
        pose_mat = optimed_pose_mat.clone().squeeze(0)     # [21,3,3]  **already rot-mat**
        
        # Set head rotation to identity matrix (zero rotation)
        pose_mat[14] = torch.eye(3, device=pose_mat.device)  # head joint (index 15)
        
        # 5. convert back to 6-D for the SMPL layer
        pose_6d_corr = rotmat_to_rot6d(pose_mat).view(1,21,6)

        # 6. regenerate final vertices & joints
        with torch.no_grad():
            smpl_verts, _, smpl_joints = dataset.smpl_model(
                shape_params      = optimed_betas,
                expression_params = closest_exp,
                body_pose         = rot6d_to_rotmat(pose_6d_corr.view(-1,6)).view(1,21,3,3),
                global_pose       = optimed_orient_mat,
                jaw_pose          = closest_jaw,
            )

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
        # Save canonical SMPL
        smpl_verts_save = smpl_verts * torch.tensor([1.0, 1.0, -1.0]).to(device)

        save_obj_path = os.path.join(args.out_dir, cfg.name, "obj", "canonical_smpl.obj")
        canonical_obj = trimesh.Trimesh(
            smpl_verts_save[0].detach().cpu() * torch.tensor([1.0, -1.0, 1.0]),
            in_tensor["smpl_faces"][0].cpu()[:, [0, 2, 1]],
            process=False,
            maintains_order=True,
        )
        canonical_obj.export(save_obj_path)

        # Rotate canonical SMPL to last view
        last_frame_id = int(multi_view_data[-1]["name"].split("_")[1])
        T_first_to_last = transform_manager.get_transform_to_target(last_frame_id)
        T_first_to_last[:3, 3] = 0.0

        smpl_verts_aligned = apply_homogeneous_transform(
            smpl_verts_save,
            T_first_to_last
        )

        # === Now re-render SMPL normals after rotation ===
        in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
            smpl_verts_aligned * torch.tensor([1.0, -1.0, -1.0]).to(device),
            in_tensor["smpl_faces"],
        )

        T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

        # === Prepare cloth normal estimation input ===
        in_tensor["image"] = multi_view_data[-1]["img_icon"].to(device)  # use last input view
        in_tensor["mask"] = multi_view_data[-1]["img_mask"].to(device)

        with torch.no_grad():
            in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)

        # Save canonical SMPL (First View)
        canonical_obj_path = os.path.join(args.out_dir, cfg.name, "obj", "canonical_smpl_first_view.obj")
        canonical_obj.export(canonical_obj_path)

        # Save rotated SMPL (Last View)
        final_obj_path = os.path.join(args.out_dir, cfg.name, "obj", "canonical_smpl_last_view.obj")
        final_obj = trimesh.Trimesh(
            smpl_verts_aligned.detach().cpu().squeeze(0) * torch.tensor([1.0, -1.0, 1.0]),
            in_tensor["smpl_faces"][0].cpu()[:, [0, 2, 1]],
            process=False,
            maintains_order=True,
        )
        final_obj.export(final_obj_path)

        # Save params
        smpl_info = {
            "betas": optimed_betas.detach().cpu(),
            "body_pose": convert_rot_matrix_to_angle_axis(optimed_pose_mat.detach()).cpu(),
            "global_orient": convert_rot_matrix_to_angle_axis(optimed_orient_mat.detach()).cpu(),
            "transl": optimed_trans.detach().cpu(),
        }
        np.save(final_obj_path.replace(".obj", ".npy"), smpl_info)

        print(colored(f"✅ Saved Final Canonical SMPL aligned to last view at {final_obj_path}", "green"))
        smpl_obj_lst = [final_obj] 
        
        # Clean up
        del optimizer_smpl
        del optimed_betas
        del optimed_orient
        del optimed_pose
        del optimed_trans
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------------------------------------------------------
        # clothing refinement

        # Store optimized SMPL verts and faces into in_tensor for downstream steps
        with torch.no_grad():
            in_tensor["smpl_verts"] = smpl_verts_save.detach()
            in_tensor["smpl_faces"] = in_tensor["smpl_faces"].detach()

        per_data_lst = []

        batch_smpl_verts = in_tensor["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0],
                                                                           device=device)
        batch_smpl_faces = in_tensor["smpl_faces"].detach()[:, :, [0, 2, 1]]

        in_tensor["depth_F"], in_tensor["depth_B"] = dataset.render_depth(
            batch_smpl_verts, batch_smpl_faces
        )

        # Normalize depth map to [0, 1] range per image
        depth_map = in_tensor["depth_F"].detach().cpu()  # [B, H, W]
        depth_min = depth_map.amin(dim=[1, 2], keepdim=True)
        depth_max = depth_map.amax(dim=[1, 2], keepdim=True)
        depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
        depth_map_norm = depth_map_norm.unsqueeze(1)  # to [B, 1, H, W] for saving
        """
        torchvision.utils.save_image(
            depth_map_norm,
            osp.join(output_dir, f"{data['name']}_smpl_front_depth.png")
        )
        """
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
        cv2.imwrite(osp.join(args.out_dir, cfg.name, f"{data['name']}_smpl_front_depth.png"), depth_8bit)

        per_loop_lst = []

        in_tensor["BNI_verts"] = []
        in_tensor["BNI_faces"] = []
        in_tensor["body_verts"] = []
        in_tensor["body_faces"] = []
      
        for idx in range(N_body):

            final_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full.obj"

            side_mesh = smpl_obj_lst[idx].copy()
            face_mesh = smpl_obj_lst[idx].copy()
            hand_mesh = smpl_obj_lst[idx].copy()
            smplx_mesh = smpl_obj_lst[idx].copy()

            # save normals, depths and masks
            BNI_dict = save_normal_tensor(
                in_tensor,
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

            BNI_object.extract_surface(False)

            in_tensor["body_verts"].append(torch.tensor(smpl_obj_lst[idx].vertices).float())
            in_tensor["body_faces"].append(torch.tensor(smpl_obj_lst[idx].faces).long())

            # requires shape completion when low overlap
            # replace SMPL by completed mesh as side_mesh
            """
            front_data_list = []

            # Inside the loop
            front_data_list.append({
                "F_depth": BNI_object.F_depth.detach().clone(),
                "F_verts": BNI_object.F_verts.detach().clone(),
                "F_faces": BNI_object.F_faces.detach().clone(),
                "F_trimesh": BNI_object.F_trimesh.copy()
            })
            """
            if cfg.bni.use_ifnet:

                side_mesh_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_IF.obj"

                side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)

                # mesh completion via IF-net
                in_tensor.update(
                    dataset.depth_to_voxel({
                        "depth_F": BNI_object.F_depth.unsqueeze(0), 
                        "depth_B": BNI_object.B_depth.unsqueeze(0)
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
                side_mesh = register(BNI_object.F_B_trimesh, sm(side_mesh), device)

            side_verts = torch.tensor(side_mesh.vertices).float().to(device)
            side_faces = torch.tensor(side_mesh.faces).long().to(device)

            # Possion Fusion between SMPLX and BNI
            # 1. keep the faces invisible to front+back cameras
            # 2. keep the front-FLAME+MANO faces
            # 3. remove eyeball faces

            # export intermediate meshes
            BNI_object.F_B_trimesh.export(
                f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj"
            )
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
                    hand_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_hand.obj")
                    full_lst += [hand_mesh]

            full_lst += [BNI_object.F_B_trimesh]

            # initial side_mesh could be SMPLX or IF-net
            side_mesh = part_removal(
                side_mesh, sum(full_lst), 2e-2, device, smplx_mesh, region="", clean=False
            )

            full_lst += [side_mesh]

            # # export intermediate meshes
            BNI_object.F_B_trimesh.export(
                f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj"
            )
            side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_side.obj")

            if cfg.bni.use_poisson:
                final_mesh = poisson(
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
                final_mesh = sum(full_lst)
                final_mesh.export(final_path)

            
            if not args.novis:
                dataset.render.load_meshes(final_mesh.vertices, final_mesh.faces)
                rotate_recon_lst = dataset.render.get_image(cam_type="four")
                per_loop_lst.extend([in_tensor['image'][idx:idx + 1]] + rotate_recon_lst)

            if cfg.bni.texture_src == 'image':

                # coloring the final mesh (front: RGB pixels, back: normal colors)
                final_colors = query_color(
                    torch.tensor(final_mesh.vertices).float(),
                    torch.tensor(final_mesh.faces).long(),
                    in_tensor["image"][idx:idx + 1],
                    device=device,
                )
                final_mesh.visual.vertex_colors = final_colors
                final_mesh.export(final_path)

            elif cfg.bni.texture_src == 'SD':

                # !TODO: add texture from Stable Diffusion
                pass

        if len(per_loop_lst) > 0 and (not args.novis):

            per_data_lst.append(get_optim_grid_image(per_loop_lst, None, nrow=5, type="cloth"))
            per_data_lst[-1].save(osp.join(args.out_dir, cfg.name, f"png/{data['name']}_cloth.png"))

            # for video rendering
            in_tensor["BNI_verts"].append(torch.tensor(final_mesh.vertices).float())
            in_tensor["BNI_faces"].append(torch.tensor(final_mesh.faces).long())

            os.makedirs(osp.join(args.out_dir, cfg.name, "vid"), exist_ok=True)
            in_tensor["uncrop_param"] = data["uncrop_param"]
            in_tensor["img_raw"] = data["img_raw"]
            torch.save(
                in_tensor, osp.join(args.out_dir, cfg.name, f"vid/{data['name']}_in_tensor.pt")
            )
        
        def save_obj(vertices, faces, out_path):
            with open(out_path, 'w') as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces + 1:  # OBJ is 1-indexed
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
                    
        def generate_expression_blendshapes(model_path, gender="neutral", num_expr=10, device='cpu'):
            smplx_model = smplx.create(
                model_path=model_path,
                model_type='smplx',
                gender=gender,
                model_filename='SMPLX_NEUTRAL_2020.npz',
                num_expression_coeffs=num_expr,
                use_face_contour=True,
                create_expression=True,
                create_betas=False,
                create_global_orient=False,
                create_body_pose=False,
                create_jaw_pose=False,
                create_left_hand_pose=False,
                create_right_hand_pose=False,
                create_transl=False
            ).to(device)

            expression_meshes = []
            faces = smplx_model.faces

            os.makedirs(args.out_dir, exist_ok=True)

            with torch.no_grad():
                for i in range(num_expr):
                    expr_vector = torch.zeros(1, num_expr).to(device)
                    expr_vector[0, i] = 1.0
                    output = smplx_model(expression=expr_vector, return_verts=True)
                    expr_verts = output.vertices[0].cpu().numpy()
                    expression_meshes.append(expr_verts)  # <- full expression mesh

                    # Save .obj file
                    os.makedirs(os.path.join(args.out_dir, f"expressions"), exist_ok=True)
                    out_path = os.path.join(args.out_dir, f"expressions/expression_{i:02d}.obj")
                    save_obj(expr_verts, faces, out_path)

            return expression_meshes, smplx_model


        #expression_meshes, smplx_model = generate_expression_blendshapes(
        #    model_path="/home/ubuntu/Data/Fulden/HPS/pixie_data",
        #    gender="neutral",
        #    num_expr=100,
        #    device="cuda"
        #)
        
        # exporter = FaceRigExporter(smplx_object=SMPLX_object, final_mesh=final_mesh, align_mode='smplx')

        # exporter.export(
        #    data=data,
        #    smpl_verts=smpl_verts,  # shape: [1, N, 3]
        #    out_dir=args.out_dir,
        #    cfg_name=cfg.name,
        #    expression_meshes=expression_meshes  # shape: List[np.ndarray of (N, 3)]
        # )
        
        final_watertight_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full_wt.obj"
        watertightifier = MeshCleanProcess(final_path, final_watertight_path)
        result = watertightifier.process(reconstruction_method='poisson', depth=10)

        if result:
            print("The mesh is watertight and has been saved successfully!")
        else:
            print("The mesh is not watertight. Further inspection may be needed.")

        final_mesh = MeshCleanProcess.process_watertight_mesh(
            final_watertight_path=f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full_wt.obj",
            output_path=f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_final.obj",
            face_vertex_mask=SMPLX_object.front_flame_vertex_mask,
            target_faces=15000
        )
    
        break