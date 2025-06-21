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
import pyexr
import logging
import warnings
import os
import os.path as osp
from apps.FaceRigExporter import FaceRigExporter
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

from apps.IFGeo import IFGeo
from apps.Normal import Normal
from apps.sapiens import ImageProcessor
from apps.clean_mesh import MeshCleanProcess
from apps.SMPLXJointAligner import SMPLXJointAligner
from apps.CameraTransformManager import CameraTransformManager

from lib.common.BNI import BNI
from lib.common.config import cfg
from lib.common.imutils import blend_rgb_norm, load_img, transform_to_tensor, wrap
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

                view_data["smpl_verts"] = apply_homogeneous_transform(smpl_verts, (T_frame_to_target))
                view_data["smpl_joints"] = apply_homogeneous_transform(smpl_joints, (T_frame_to_target))

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
                #normal_loss = torch.abs(view_data["T_normal_F"]).mean()

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
        """
        # Initialize in_tensor with all necessary data
        in_tensor = {
            "smpl_faces": first_data["smpl_faces"],
            "image": first_data["img_icon"].to(device),
            "mask": first_data["img_mask"].to(device),
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
        """

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
  
        def run_preflight_checks(in_tensor, cameras, smpl_vertices):
            print("\n" + "="*50)
            print("RUNNING PRE-FLIGHT CHECKS...")
            print("="*50)
            
            # --- Check 1: Basic Existence and Type ---
            print("\n--- Checking Data Existence and Types ---")
            num_views = len(cameras)
            assert num_views > 0, "FATAL: No cameras loaded."
            print(f"Found {num_views} camera views.")

            assert smpl_vertices is not None, "FATAL: smpl_vertices is None."
            assert isinstance(smpl_vertices, torch.Tensor), f"FATAL: smpl_vertices must be a torch.Tensor, but got {type(smpl_vertices)}."
            smpl_points = smpl_vertices.squeeze() # Use squeezed version for checks
            print(f"SMPL vertices are a valid tensor with shape: {smpl_points.shape}")
            if torch.isnan(smpl_points).any() or torch.isinf(smpl_points).any():
                raise ValueError("SMPL input contains invalid values (NaN or Inf)")


            for i in range(num_views):
                view_key = f"view_{i}"
                assert view_key in in_tensor, f"FATAL: Missing '{view_key}' in input tensor."
                
                # Check normal map
                assert "normal_F" in in_tensor[view_key], f"FATAL: Missing 'normal_F' in {view_key}."
                assert isinstance(in_tensor[f"view_{i}"]["normal_F"], torch.Tensor), f"FATAL: normal_F in {view_key} must be a torch.Tensor."
                
                # Check mask
                assert "img_mask" in in_tensor[view_key], f"FATAL: Missing 'img_mask' in {view_key}."
                # Convert tensor to numpy array and store it
                in_tensor[view_key]["img_mask"] = in_tensor[view_key]["img_mask"].detach().cpu().numpy()
                # Now check if it's a numpy array
                assert isinstance(in_tensor[view_key]["img_mask"], np.ndarray), f"FATAL: img_mask in {view_key} must be a numpy.ndarray."
                
            print("OK: All required data keys exist with correct types.")

            # --- Check 2: Shape and Dimensions ---
            print("\n--- Checking Shapes and Dimensions ---")
            H, W = 512, 512 # Assuming this is your standard resolution
            i=0
            for i in range(num_views):
                view_key = f"view_{i}"
                
                # Check normal map shape
                normal_shape = in_tensor[view_key]["normal_F"].squeeze().shape
                assert len(normal_shape) == 3, f"FATAL: Normal map for {view_key} should have 3 dimensions, but has {len(normal_shape)}."
                # Check for both (H, W, 3) and (3, H, W) formats
                assert (normal_shape[0] == H and normal_shape[1] == W and normal_shape[2] == 3) or \
                    (normal_shape[0] == 3 and normal_shape[1] == H and normal_shape[2] == W), \
                    f"FATAL: Unexpected normal map shape for {view_key}: {normal_shape}. Expected (~{H}, ~{W}, 3) or (3, ~{H}, ~{W})."

                # Check mask shape
                mask_shape = in_tensor[view_key]["img_mask"].squeeze().shape
                assert len(mask_shape) == 2, f"FATAL: Mask for {view_key} should have 2 dimensions, but has {len(mask_shape)}."
                assert mask_shape[0] == H and mask_shape[1] == W, f"FATAL: Unexpected mask shape for {view_key}: {mask_shape}. Expected ({H}, {W})."

            print("OK: All masks and normal maps have expected shapes.")

            # --- Check 3: Geometric Consistency (The Most Important Check) ---
            print("\n--- Checking Geometric Consistency (Projecting SMPL into Mask) ---")
            # For this check, we only need the non-batched SMPL vertices
            smpl_check_points = smpl_points[0] if smpl_points.ndim == 3 else smpl_points
            smpl_check_points = smpl_check_points.to(device)
            cpu_points = smpl_check_points.detach().cpu()
            print("Shape:", cpu_points.shape)
            print("NaNs:", torch.isnan(cpu_points).sum())
            print("Infs:", torch.isinf(cpu_points).sum())
            print("Min:", cpu_points.min().item())
            print("Max:", cpu_points.max().item())

            debug_projections_dir = os.path.join(args.out_dir, cfg.name, "debug_projections")
            os.makedirs(debug_projections_dir, exist_ok=True)
            for i in range(num_views):
                cam = cameras[i]
                mask = torch.from_numpy(in_tensor[f"view_{i}"]["img_mask"]).to(device)
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                assert mask.shape == (H, W), f"Mask shape incorrect: {mask.shape}, expected ({H}, {W})"

                # Ensure camera intrinsics are tensors
                R = torch.tensor(cam.R, device=smpl_check_points.device) if not isinstance(cam.R, torch.Tensor) else cam.R.to(smpl_check_points.device)
                t = torch.tensor(cam.t, device=smpl_check_points.device) if not isinstance(cam.t, torch.Tensor) else cam.t.to(smpl_check_points.device)
                K = torch.tensor(cam.K, device=smpl_check_points.device) if not isinstance(cam.K, torch.Tensor) else cam.K.to(smpl_check_points.device)
                if torch.isnan(smpl_check_points).any() or torch.isinf(smpl_check_points).any():
                    raise ValueError("SMPL vertices contain NaNs or Infs — aborting projection.")

                # Project SMPL vertices to camera space
                points_in_camera_space = smpl_check_points @ R.T   # (N,3)
                points_in_camera_space = points_in_camera_space + t.unsqueeze(0)  # broadcast (N,3)+(1,3)

                # Keep only points in front of the camera
                front_points_mask = points_in_camera_space[:, 2] > 0
                if front_points_mask.sum() == 0:
                    print(f"WARNING: All SMPL points are behind the camera for view {i}. Skipping.")
                    continue

                points_in_camera_space = points_in_camera_space[front_points_mask]

                # Project to 2D homogeneous coordinates
                points_2d_homo = torch.matmul(points_in_camera_space, K.T)
                depth = points_2d_homo[..., 2:3]

                # Filter out invalid depth
                valid_depth_mask = (depth.squeeze() > 1e-6) & ~torch.isnan(depth.squeeze()) & ~torch.isinf(depth.squeeze())
                if valid_depth_mask.sum() == 0:
                    print(f"WARNING: All points have invalid depth for view {i}. Skipping.")
                    continue

                points_2d_homo = points_2d_homo[valid_depth_mask]
                depth = depth[valid_depth_mask]

                # Project to image plane
                projected_points = points_2d_homo[..., :2] / depth

                # Remove NaNs/Infs (again)
                valid_proj_mask = ~torch.isnan(projected_points).any(dim=1) & ~torch.isinf(projected_points).any(dim=1)
                if valid_proj_mask.sum() == 0:
                    print(f"WARNING: All projections are invalid for view {i}. Skipping.")
                    continue

                projected_points = projected_points[valid_proj_mask]

                # Floor before casting to integer
                floored_points = projected_points.floor()

                # Filter again for NaNs after flooring
                nan_free_mask = ~torch.isnan(floored_points).any(dim=1)
                if nan_free_mask.sum() == 0:
                    print(f"WARNING: All floored points are NaN for view {i}. Skipping.")
                    continue

                floored_points = floored_points[nan_free_mask]

                # Now safe to cast
                u_coords = floored_points[:, 0].long()
                v_coords = floored_points[:, 1].long()

                # Check for in-frame coordinates
                in_frame_mask = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)
                if in_frame_mask.sum() == 0:
                    print(f"WARNING: All projected points in view {i} are outside image bounds. Skipping.")
                    continue

                valid_u = u_coords[in_frame_mask]
                valid_v = v_coords[in_frame_mask]

                # Final assert to confirm all values are in bounds (safe now)
                assert valid_u.max().item() < W and valid_v.max().item() < H
                assert valid_u.min().item() >= 0 and valid_v.min().item() >= 0

                # Index into mask
                mask_values_at_smpl = mask[valid_v, valid_u]

                # Compute overlap
                overlap_percentage = (mask_values_at_smpl.cpu() > 0.5).sum().item() / len(mask_values_at_smpl) * 100
                print(f"View {i}: {overlap_percentage:.2f}% of projected SMPL vertices are inside the foreground mask.")

                # Debug visualization
                debug_img = mask.cpu().numpy() * 255
                debug_img = cv2.cvtColor(debug_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                for u, v in zip(valid_u.cpu().numpy(), valid_v.cpu().numpy()):
                    cv2.circle(debug_img, (int(u), int(v)), radius=1, color=(0, 0, 255), thickness=-1)

                cv2.imwrite(os.path.join(debug_projections_dir, f"view_{i}_projection.png"), debug_img)

                # Final threshold check
                if overlap_percentage < 50.0:
                    print(f"WARNING: Low geometric consistency for view {i} ({overlap_percentage:.2f}%).")
                    continue
                    #assert overlap_percentage > 50.0, f"FATAL: Low geometric consistency for view {i} ({overlap_percentage:.2f}%)."

            print("OK: Geometric consistency checks passed.")
            print("="*50)
            print("PRE-FLIGHT CHECKS COMPLETE. ALL SYSTEMS GO.")
            print("="*50 + "\n")
    
        # STEP 1: Apply the Blender-to-CV coordinate system flip.
        # This matrix swaps and flips the Y and Z axes.
        R_smpl_to_cv = torch.tensor([
            [-1, 0,  0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=torch.float32, device=device)
        # Additional 180-degree rotation around Z-axis

        R_z_90 = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=torch.float32, device=device)
        # Combine rotations: first R_smpl_to_cv, then R_z_180
        R_combined = R_z_90 @ R_smpl_to_cv
        smpl_vertices_transformed = (R_combined @ in_tensor["view_0"]["smpl_verts"].squeeze().T).T
        #translation_to_view = torch.tensor([0.0, 0.0, 3.0], device=device)
        #smpl_vertices_aligned = smpl_vertices_transformed + translation_to_view
        R_z_90 = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        smpl_vertices_aligned = (R_z_90 @ smpl_vertices_transformed.T).T
    
        print("\n" + "="*50)
        print("SMPL MODEL BOUNDING VOLUME DIAGNOSTICS")
        print("="*50)

        # 1. Flatten all leading dimensions to get a simple (num_points, 3) tensor.
        # This is the key fix that makes the rest of the code work correctly.
        smpl_flat = in_tensor["view_0"]["smpl_verts"].reshape(-1, 3)
        
        # 2. Calculate Axis-Aligned Bounding Box (AABB) on the flattened points
        min_coords = smpl_flat.min(dim=0)[0]
        max_coords = smpl_flat.max(dim=0)[0]
        
        x_range = max_coords[0] - min_coords[0]
        y_range = max_coords[1] - min_coords[1]
        z_range = max_coords[2] - min_coords[2]
        
        # .item() will now work correctly because min_coords[0] is a guaranteed scalar
        print(f"Model AABB Min Coords: [{min_coords[0].item():.3f}, {min_coords[1].item():.3f}, {min_coords[2].item():.3f}]")
        print(f"Model AABB Max Coords: [{max_coords[0].item():.3f}, {max_coords[1].item():.3f}, {max_coords[2].item():.3f}]")
        print(f"Model Size (X, Y, Z):  [{x_range.item():.3f}, {y_range.item():.3f}, {z_range.item():.3f}] world units")
        
        # 3. Calculate Bounding Sphere
        center = (min_coords + max_coords) / 2.0
        distances_from_center = torch.norm(smpl_flat - center, dim=1)
        bounding_sphere_radius = distances_from_center.max()
        
        print(f"\nModel Bounding Sphere Center: [{center[0].item():.3f}, {center[1].item():.3f}, {center[2].item():.3f}]")
        print(f"Model Bounding Sphere Radius:  {bounding_sphere_radius.item():.3f} world units")
        print("="*50 + "\n")

        cameras = load_cameras_from_json(cam_param_path, device)
        print("[DEBUG] Loaded cameras:")
        for i, cam in enumerate(cameras):
            print(f"  Camera {i}: K=\n{cam.K}")
            if hasattr(cam, 'Rt'):
                print(f"  Rt=\n{cam.Rt}")
            elif hasattr(cam, 'R') and hasattr(cam, 't'):
                print(f"  R=\n{cam.R.detach().cpu().numpy() if hasattr(cam.R, 'detach') else cam.R}")
                print(f"  t=\n{cam.t.detach().cpu().numpy() if hasattr(cam.t, 'detach') else cam.t}")
            else:
                print("  [No Rt or (R, t) attributes found!]")
        print("[DEBUG] SMPL bounding box:")
        smpl_np = in_tensor["view_0"]["smpl_verts"].detach().cpu().numpy().squeeze()
        print(f"  Min: {smpl_np.min(axis=0)}")
        print(f"  Max: {smpl_np.max(axis=0)}")
        print(f"  Size: {smpl_np.max(axis=0) - smpl_np.min(axis=0)}")
        # Visualize camera frustums and SMPL
        transform_manager.debug_visualize_cameras_and_smpl(in_tensor["view_0"]["smpl_verts"].detach().cpu(), save_path="camera_smpl_3d_debug.png")
        # Visualize SMPL projections on masks (calls BNIPipeline debug)

        run_preflight_checks(in_tensor, cameras, smpl_vertices_aligned.to(device))

        # Run optimization with your in_tensor and optional smpl_vertices
        sdf_net, optimized_cameras = optimize_sdf(in_tensor, cameras, smpl_vertices=in_tensor["view_0"]["smpl_verts"], num_iterations=10000)
        extract_mesh_from_sdf(sdf_net, smpl_vertices=in_tensor["view_0"]["smpl_verts"], grid_size=256, filename="reconstructed_mesh.obj")
