from collections import defaultdict
import logging
import warnings
import os
import os.path as osp
from apps.Helper import apply_homogeneous_transform, compute_head_roll_loss, load_cameras_from_json
from apps.sapiens import ImageProcessor
from lib.common import BNI

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
import matplotlib.pyplot as plt
import pandas as pd

from apps.IFGeo import IFGeo
from apps.Normal import Normal
from apps.CameraTransformManager import CameraTransformManager

from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.common.config import cfg
from lib.common.imutils import wrap
from lib.common.local_affine import LocalAffine, register, trimesh2meshes
from lib.common.render import Render, query_color
from lib.common.train_util import Format, init_loss
from lib.common.voxelize import VoxelGrid
from lib.dataset.mesh_util import *
from lib.dataset.TestDataset import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis, rotmat_to_rot6d

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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


def copy_view_data_to_device(view_data, device):
    for k in view_data:
        if torch.is_tensor(view_data[k]):
            view_data[k] = view_data[k].detach().cpu() if device == "cpu" else view_data[k].to(device)
    return view_data

def print_gpu_memory_usage():
    free, total = torch.cuda.mem_get_info()   # bytes
    print(f"  GPU memory usage      : {(total - free)/1e9:.2f} GB")

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
    normal_path = "/var/locally-mounted/myshareddir/Fulden/ckpt/normal.ckpt"
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

    #if cfg.sapiens.use:
    #    sapiens_normal_net = ImageProcessor(device=device)

    # SMPLX object
    SMPLX_object = SMPLX()

    lmk_ids = np.load("/var/locally-mounted/myshareddir/Fulden/smpl_related/smplx_vertex_lmkid.npy")  # shape: [N]

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

    import time
    start = time.perf_counter()
    DEBUG = False

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

        #if cfg.sapiens.use:
        #    sapiens_normal = sapiens_normal_net.process_image(
        #        Image.fromarray(
        #            data["img_raw"].squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        #        ), "1b", cfg.sapiens.seg_model
        #    )
        #    print(colored("Estimating normal maps from input image, using Sapiens-normal", "green"))

        #    sapiens_normal_square_lst = []
        #    for idx in range(len(data["img_icon"])):
        #        sapiens_normal_square_lst.append(wrap(sapiens_normal, data["uncrop_param"], idx))
        #    sapiens_normal_square = torch.cat(sapiens_normal_square_lst)
        
        # smpl optimization
        loop_smpl = tqdm(range(args.loop_smpl))
        
        multi_view_data = []
        cam_param_path = os.path.join(args.in_dir, "cam_params", "camera_parameters.json")
        
        for data in pbar:
            # Collect multiple views
            frame_id = int(data["name"].split("_")[1])
            in_tensor[f"view_{frame_id}"] = {
                "smpl_faces": data["smpl_faces"], 
                "image": data["img_icon"], #.to(device), 
                "mask": data["img_mask"] #.to(device)
            }
            data["image"] = data["img_icon"] #.to(device)
            multi_view_data.append(data)

            if len(multi_view_data) < len(dataset):  
                continue

            print(colored(f"Optimizing Canonical SMPL Body using {len(multi_view_data)} views...", "blue"))

        losses = init_loss()
        #front_data = multi_view_data[front_view]
        #back_data = multi_view_data[back_view]
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
            #print(f"Camera {i} center reconstructed = {C}")
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
            #pose_list.append(data["body_pose"])
            #trans_list.append(data["trans"])
            #betas_list.append(data["betas"])
            #exp_list.append(data["exp"])
            #jaw_list.append(data["jaw_pose"])
            torch.cuda.empty_cache() # clean gpu memory

        # compute losses for each view before mean computation
        if DEBUG:
            before_mean_losses = defaultdict(list)
            for view_data in multi_view_data:

                # move to gpu
                view_data = copy_view_data_to_device(view_data, device)

                frame_id = int(view_data["name"].split("_")[1])
                
                # Get transformation matrix for 180 degree rotation around z-axis
                rot_180_z = torch.tensor([
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0], 
                    [0.0, 0.0, -1.0]
                ], device=device)

                # Apply rotation to vertices
                view_data["smpl_verts"] = torch.matmul(view_data["smpl_verts"], rot_180_z.T)
                view_data["smpl_verts"] = view_data["smpl_verts"] * torch.tensor([1.0, 1.0, -1.0]).to(device)


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

                view_total_loss = sil_loss + 0.2 * normal_loss + 0.1 * landmark_loss 
                view_data["total_loss"] = view_total_loss
                before_mean_losses[f'{view_data["name"]}_silhouette_iou'].append(sil_loss.item())
                before_mean_losses[f'{view_data["name"]}_landmark_error_px'].append(landmark_loss.item())
                before_mean_losses[f'{view_data["name"]}_sil_l1_loss'].append(sil_loss.item())
                before_mean_losses[f'{view_data["name"]}_normal_loss'].append(normal_loss.item())
                before_mean_losses[f'{view_data["name"]}_head_roll_loss'].append(head_roll_loss.item())
                before_mean_losses[f'{view_data["name"]}_total_loss'].append(view_total_loss.item())

                view_data = copy_view_data_to_device(view_data, "cpu")

        # === Compute mean of all corrected values ===
        mean_pose = torch.stack([data["body_pose"] for data in multi_view_data], dim=0).mean(dim=0)
        mean_pose = mean_pose.to(device)
        #mean_pose = torch.stack(pose_list, dim=0).mean(dim=0)
        #mean_pose = pose_list[front_view]


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
        
        closest_exp = multi_view_data[front_view]["exp"].detach().to(device)
        closest_jaw = multi_view_data[front_view]["jaw_pose"].detach().to(device)
        #closest_exp = exp_list[front_view].detach()
        #closest_jaw = jaw_list[front_view].detach()
        #mean_jaw_pose = torch.stack(jaw_list, dim=0).mean(dim=0)

        mean_trans = torch.stack([data["trans"] for data in multi_view_data], dim=0).mean(dim=0).to(device)
        mean_betas = torch.stack([data["betas"] for data in multi_view_data], dim=0).mean(dim=0).to(device)
        #mean_trans = torch.stack(trans_list, dim=0).mean(dim=0)
        #mean_betas = torch.stack(betas_list, dim=0).mean(dim=0)
        mean_global_orient = torch.stack(rotated_global_orients, dim=0).mean(dim=0).to(device)

        # clean up
        del pose_list, trans_list, betas_list, exp_list, jaw_list
        torch.cuda.empty_cache()

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

       

        
        save_vis_dir = os.path.join(args.out_dir, cfg.name, "png", "canonical_smpl_iters")
        os.makedirs(save_vis_dir, exist_ok=True)
        loss_values = []
        first_iter_losses = defaultdict(list)
        last_iter_losses = defaultdict(list)
        
        if DEBUG:
            print_gpu_memory_usage()
            args.loop_smpl = 35

        loop_smpl = tqdm(range(args.loop_smpl))
        # Optimization loop
        for i in loop_smpl:
            #optimizer_smpl.zero_grad()
            optimizer_smpl.zero_grad(set_to_none=True) 

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

            smpl_verts = (smpl_verts + optimed_trans) * first_data["scale"].to(device)
            smpl_joints = (smpl_joints + optimed_trans) * first_data["scale"].to(device) * torch.tensor([
                1.0, 1.0, -1.0
            ]).to(device)
            #smpl_verts = smpl_verts * torch.tensor([1.0, 1.0, -1.0]).to(device)
            
            frame_to_camera_dir = {}
            total_loss = 0.0

            for j, view_data in enumerate(multi_view_data):
                view_data = copy_view_data_to_device(view_data, device)
                frame_id = int(view_data["name"].split("_")[1])
                T_frame_to_target = transform_manager.get_transform_to_target(frame_id)
                T_frame_to_target[:3, 3] = 0.0

                view_data["smpl_verts"] = apply_homogeneous_transform(smpl_verts, (T_frame_to_target))
                view_data["smpl_joints"] = apply_homogeneous_transform(smpl_joints, (T_frame_to_target))

                if len(multi_view_data) == 2:
                    rot_180_z = torch.tensor([
                        [-1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0], 
                        [0.0, 0.0, -1.0]
                    ], device=device)

                    # Apply rotation to vertices
                    view_data["smpl_verts"] = torch.matmul(view_data["smpl_verts"], rot_180_z.T)

                head_rotmat = optimed_pose_mat[:, 14] 

                # Compute roll loss
                head_roll_loss = compute_head_roll_loss(head_rotmat, up_direction="view_y")

                # Landmark processing
                smpl_joints_3d = (view_data["smpl_joints"][:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0) * 0.5
                smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]
                ghum_lmks = view_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                ghum_conf = view_data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)

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
                normal_loss = (torch.abs(view_data["T_normal_F"]) - torch.abs(view_data["normal_F"])).mean()
                    
                # Combine all losses with appropriate weights
                if len(multi_view_data) == 2:
                    view_total_loss = sil_loss + 0.2 * normal_loss + 0.1 * landmark_loss + 0.01 * head_roll_loss
                else:
                    view_total_loss = sil_loss + 0.2 * normal_loss + 0.1 * landmark_loss + 0.1 * head_roll_loss
                total_loss += view_total_loss

                del ghum_lmks, ghum_conf, gt_arr
                del view_total_loss, sil_loss, normal_loss, landmark_loss, head_roll_loss
                torch.cuda.empty_cache()

                view_data = copy_view_data_to_device(view_data, "cpu")
                
                if DEBUG:
                    print_gpu_memory_usage()

            total_loss = total_loss/len(multi_view_data)
            total_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(total_loss.item())
            loss_values.append(total_loss.detach().cpu())

            del total_loss
            torch.cuda.empty_cache()


            
            # === Visualization (optional) ===
            if DEBUG:
                with torch.no_grad():
                    smpl_silhouette_images, gt_silhouette_images, diff_silhouette_images = [], [], []

                    for view_data in multi_view_data:
                        view_data = copy_view_data_to_device(view_data, device)
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
                    view_data = copy_view_data_to_device(view_data, "cpu")

        del scheduler_smpl

        # === After optimization ===
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
            #"ghum_lmks": ghum_lmks.detach().cpu(),
            #"ghum_conf": ghum_conf.detach().cpu()
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
            view_data = copy_view_data_to_device(view_data, device)

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

                # Normalize to [0, 1]
                depth_norm = np.zeros_like(depth_map)
                depth_norm[valid_mask] = (depth_map[valid_mask] - d_min) / (d_max - d_min + 1e-6)
                depth_map = depth_norm
            else:
                depth_map = np.zeros_like(depth_map)

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

            view_data = copy_view_data_to_device(view_data, "cpu")

        sp_out_dir = os.path.join(args.out_dir, cfg.name, "own_objects_normals")

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
        
        # Clean up
        del optimizer_smpl
        del optimed_betas
        del optimed_orient
        del optimed_pose
        del optimed_trans
        torch.cuda.empty_cache()

        if DEBUG:
            print_gpu_memory_usage()



        # ------------------------------------------------------------------------------------------------------------------
        # clothing refinement        
        
        
        in_tensor["BNI_verts"] = []
        in_tensor["BNI_faces"] = []
        in_tensor["body_verts"] = []
        in_tensor["body_faces"] = []
      

        final_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_full.obj"

        side_mesh = smpl_obj_lst[back_view].copy()
        face_mesh = smpl_obj_lst[back_view].copy()
        hand_mesh = smpl_obj_lst[back_view].copy()
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
            if i == back_view:
                #add 0.2mm up to the back mesh
                BNI_object.F_trimesh.vertices[:, 0] -= 0.02
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

        # Save the final mesh to the output directory
        final_mesh_path = os.path.join(args.out_dir, cfg.name, "obj", f"{data['name']}_final.obj")
        recon_obj.export(final_mesh_path)
        print(colored(f"✅ Saved final mesh to {final_path}", "green"))
        
        
        end = time.perf_counter()
        elapsed = end - start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"Running time: {minutes} min {seconds} sec")
        exit()
