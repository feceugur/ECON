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
    if True:
        sapiens_normal_net = ImageProcessor(device=device)

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

    # if cfg.bni.use_ifnet:
    if True:
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

        #if cfg.sapiens.use:
        if True:
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

        # Normalize depth map to [0, 1] range per image
        depth_map = in_tensor["depth_F"].detach().cpu()  # [B, H, W]
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
        cv2.imwrite(osp.join(args.out_dir, cfg.name, f"{data['name']}_smpl_front_depth.png"), depth_8bit)

        per_loop_lst = []

        in_tensor["BNI_verts"] = []
        in_tensor["BNI_faces"] = []
        in_tensor["body_verts"] = []
        in_tensor["body_faces"] = []
      

        final_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full.obj"

        side_mesh = smpl_obj_lst[back_view].copy()
        face_mesh = smpl_obj_lst[back_view].copy()
        hand_mesh = smpl_obj_lst[back_view].copy()
        smplx_mesh = smpl_obj_lst[back_view].copy()

        
        #T_back_view = transform_manager.get_transform_to_target(back_view)
        
        #in_tensor[f"view_{back_view}"]["normal_F"] = (T_back_view @ in_tensor[f"view_{back_view}"]["normal_F"].squeeze(0).cpu().numpy()).to(device)
        """

        """
        # save normals, depths and masks
        BNI_dict = save_normal_tensor(
            in_tensor[f"view_{front_view}"],
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
        """

        """
        bni_mesh_list = []
        extrinsics = []
        for i in (front_view, back_view):
            frame_id = int(multi_view_data[i]["name"].split("_")[1])
            T_view_i = transform_manager.get_transform_to_target(frame_id)
            extrinsics.append(T_view_i.cpu().numpy())
            
            view_data = in_tensor[f"view_{i}"]
            #depth_tensor = view_data["depth_F"]  # (H, W)
            #depth_list.append(depth_tensor.cpu().numpy())  # must be float32, in meters
            
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
            # Save front mesh for this view
            front_mesh_path = os.path.join(args.out_dir, cfg.name, "obj", f"bni_view_{i}.obj")
            BNI_object.F_trimesh.export(front_mesh_path)
            print(colored(f"✅ Saved front mesh for view {i} to {front_mesh_path}", "green"))
            bni_mesh_list.append(BNI_object.F_trimesh)
        
        fused_bni_surface = trimesh.util.concatenate(bni_mesh_list)
        BNI_surface = trimesh2meshes(fused_bni_surface).to(device)
        fused_bni_surface.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_fused_BNI.obj")
        

        in_tensor["body_verts"].append(torch.tensor(smpl_obj_lst[front_view].vertices).float())
        in_tensor["body_faces"].append(torch.tensor(smpl_obj_lst[front_view].faces).long())

        # requires shape completion when low overlap
        # replace SMPL by completed mesh as side_mesh   ----> TRY SIDE VIEW'S BNI MESHES
  


        if cfg.bni.use_ifnet:
        #if True:

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


        print("Remeshing Poisson output for clean topology...")
        verts_refine, faces_refine = remesh(
            recon_obj,
            os.path.join(args.out_dir, cfg.name, "obj", "remesh.obj"),
            device
        )
        mesh_pr = Meshes(verts=verts_refine, faces=faces_refine).to(device)
        verts_ref = mesh_pr.verts_padded().clone().detach()
        """"""

        local_affine_model = LocalAffine(
            mesh_pr.verts_padded().shape[1],
            mesh_pr.verts_padded().shape[0],
            mesh_pr.edges_packed()
        ).to(device)

        optimizer_cloth = torch.optim.Adam([{'params': local_affine_model.parameters()}], lr=1e-4)
        scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cloth, mode="min", factor=0.5, patience=10)

        loop_cloth = tqdm(range(200), desc="Cloth Refinement")
    
        losses = init_loss() 
        losses["stiff"]["weight"] = 1.0  # Increased from default
        losses["rigid"]["weight"] = 1.0  # Increased from default
        losses["cloth"]["weight"] = 1.0   # Keep the main loss weight at 1.0
        
        
        for i in loop_cloth:
            optimizer_cloth.zero_grad()

            deformed_verts, stiffness, rigid = local_affine_model(
                mesh_pr.verts_padded().to(device)
            )

            #print(f"[{i}] Stiffness mean:", stiffness.mean().item(), "| Rigid mean:", rigid.mean().item())
            deformed_mesh = Meshes(verts=deformed_verts, faces=mesh_pr.faces_padded())

            #mesh_pr = mesh_pr.update_padded(deformed_verts)

            update_mesh_shape_prior_losses(deformed_mesh, losses)

            cloth_losses = []
            for i in range(len(in_tensor["multi_view_data"])):
                view_key = f"view_{i}"
                view_data = in_tensor[view_key]
                T_view = transform_manager.get_transform_to_target(int(view_data["name"].split("_")[1]))
                gt_normal_F = view_data["normal_F"]
                # Add 180 degree rotation around y-axis
                #T_rot_x = torch.tensor([[-1.0, 0.0, 0.0, 0.0],
                #                         [0.0, 1.0, 0.0, 0.0], 
                #                         [0.0, 0.0, -1.0, 0.0],
                #                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float, device=T_view.device)
                #T_combined = torch.matmul(T_view, T_rot_x)

                ############################################################
                #############CARLA/ERIC########################################
                ############################################################
                # Add additional 90 degree rotation around y-axis
                T_rot_y = torch.tensor([
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ], dtype=torch.float, device=T_view.device)
                
                # Combine with existing transformations
                T_combined = torch.matmul(T_view, T_rot_y)
                ############################################################

                verts_view = apply_homogeneous_transform(mesh_pr.verts_padded(), T_combined.T)
                #verts_view_rendered = (verts_view * torch.tensor([1.0, -1.0, -1.0], device=device))

                P_normal_F, _ = dataset.render_normal(verts_view,deformed_mesh.faces_padded())

                #print(f"[{i}] view_{i} - Pred F normal range:", P_normal_F.min().item(), P_normal_F.max().item())
                #print(f"[{i}] view_{i} - GT F normal range:", gt_normal_F.min().item(), gt_normal_F.max().item())
                # Save visuals to folder
                comparison_dir = os.path.join(args.out_dir, cfg.name, "debug_comparisons")
                os.makedirs(comparison_dir, exist_ok=True)

                comparison_path = os.path.join(comparison_dir, f"{data['name']}_{view_key}.png")
                save_normal_comparison(P_normal_F, gt_normal_F, comparison_path)

                if "mask" in view_data:
                    mask = torch.tensor(view_data["mask"]).to(device).float().unsqueeze(0)
                    diff = torch.abs(P_normal_F - gt_normal_F) * mask
                    cloth_loss_view = diff.sum() / mask.sum().clamp(min=1.0)
                else:
                    cloth_loss_view = torch.abs(P_normal_F - gt_normal_F).mean()

                cloth_losses.append(cloth_loss_view)
            
            # Average over views
            losses["cloth"]["value"] = torch.stack(cloth_losses).mean()
            losses["stiff"]["value"] = torch.mean(stiffness)
            losses["rigid"]["value"] = torch.mean(rigid)

            # === Total Loss ===
            total_loss = torch.zeros(1, device=device) 
            loss_msg = "Cloth Refinement --- "

            # no requires_grad=True needed

            for k, v in losses.items():
                if v["weight"] > 0.0:
                    total_loss += v["value"] * v["weight"] 
                    loss_msg += f"{k}:{float(v['value'] * v['weight']):.5f} | "

            loop_cloth.set_description(loss_msg + f"Total: {total_loss.item():.5f}")

            total_loss.backward()
            optimizer_cloth.step()
            scheduler_cloth.step(total_loss)

            mesh_pr = deformed_mesh.detach()

        final_verts = mesh_pr.verts_packed().detach().squeeze(0).cpu()
        final_faces = mesh_pr.faces_packed().detach().squeeze(0).cpu()

        final_obj = trimesh.Trimesh(final_verts, final_faces, process=False, maintains_order=True)

        final_colors = query_color(
            final_verts, final_faces,
            in_tensor["image"],
            device=device
        )
        final_obj.visual.vertex_colors = final_colors

        refine_path = os.path.join(args.out_dir, cfg.name, f"obj/{in_tensor['multi_view_data'][0]['name']}_refine.obj")
        final_obj.export(refine_path)
        print(f"Saved refined mesh to {refine_path}")
            
        """
        """
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
        """
        """
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
                in_tensor["view_0"],
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

            # Save BNI mesh
            BNI_object.F_B_trimesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj")
        """
        """
        idx = 0

        side_mesh = smpl_obj_lst[idx].copy()
        face_mesh = smpl_obj_lst[idx].copy()
        hand_mesh = smpl_obj_lst[idx].copy()
        smplx_mesh = smpl_obj_lst[idx].copy()

        
        bni_mesh_list = []
        extrinsics = []
            
        #for i in range(len(multi_view_data)):
        for i in (0,2,4,6):
            frame_id = int(multi_view_data[i]["name"].split("_")[1])
            T_view_i = transform_manager.get_transform_to_target(frame_id)
            extrinsics.append(T_view_i.cpu().numpy())
            
            view_data = in_tensor[f"view_{i}"]
            depth_tensor = view_data["depth_F"]  # (H, W)
            depth_list.append(depth_tensor.cpu().numpy())  # must be float32, in meters
            
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
            BNI_object.F_B_trimesh.apply_transform(R_4x4)
            # Save front mesh for this view
            front_mesh_path = os.path.join(args.out_dir, cfg.name, "obj", f"bni_view_{i}.obj")
            BNI_object.F_B_trimesh.export(front_mesh_path)
            print(colored(f"✅ Saved front mesh for view {i} to {front_mesh_path}", "green"))
            bni_mesh_list.append(BNI_object.F_B_trimesh)
        """
        """
        idx = 0

        BNI_dict = save_normal_tensor_multi(
                in_tensor,
                idx,
                osp.join(args.out_dir, cfg.name, f"BNI/{data['name']}_{idx}"),
                cfg.bni.thickness
            )
       
        BNI_object = BNI_f(
            dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
            name=data["name"],
            BNI_dict=BNI_dict,
            cfg=cfg.bni,
            device=device
        )

        BNI_object.extract_surface()
        """
        """
        for view_data in multi_view_data:
            frame_id  = int(view_data["name"].split("_")[1])
            view_key  = f"view_{frame_id}"

            # -- full transform including translation
            T_view2canon  = transform_manager.get_transform_to_target(frame_id)  # view → canon
            T_canon2view  = torch.inverse(T_view2canon)                          # canon → view

            smpl_full = apply_homogeneous_transform(smpl_verts, T_canon2view)
            in_tensor[view_key]["smpl_verts"] = smpl_full        # used by BNI

            # -- rotate normals ----------------------------------------------------
            R_view2canon = T_view2canon[:3, :3]
            for key in ["T_normal_F", "T_normal_B"]:
                n_view = in_tensor[view_key][key]
                in_tensor[view_key][key] = rotate_normal_map(n_view, R_view2canon)

            for key in ["normal_F", "normal_B"]:
                n_view = in_tensor[view_key][key]
                if n_view is not None:
                    in_tensor[view_key][key] = rotate_normal_map(n_view, R_view2canon)

        
        result = extract_surface_multiview(in_tensor["normal_list"], in_tensor["mask_list"], in_tensor["depth_list"], cam_params)
        result['mesh'].export(os.path.join(args.out_dir, cfg.name, "obj", f"{data['name']}_fused_BNI.obj"))
        """

        """

        def run_preflight_checks(in_tensor, cameras, smpl_vertices):
            
            #Performs a series of checks on the input data to ensure validity and consistency
            #before starting the main optimization loop.
          
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

            for i in range(num_views):
                view_key = f"view_{i}"
                assert view_key in in_tensor, f"FATAL: Missing '{view_key}' in input tensor."
                
                # Check normal map
                assert "normal_F" in in_tensor[view_key], f"FATAL: Missing 'normal_F' in {view_key}."
                assert isinstance(in_tensor[f"view_{i}"]["normal_F"], torch.Tensor), f"FATAL: normal_F in {view_key} must be a torch.Tensor."
                
                # Check mask
                assert "mask" in in_tensor[view_key], f"FATAL: Missing 'mask' in {view_key}."
                assert isinstance(in_tensor[view_key]["mask"], np.ndarray), f"FATAL: mask in {view_key} must be a numpy.ndarray."
                
            print("OK: All required data keys exist with correct types.")

            # --- Check 2: Shape and Dimensions ---
            print("\n--- Checking Shapes and Dimensions ---")
            H, W = 512, 512 # Assuming this is your standard resolution
            
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
                mask_shape = in_tensor[view_key]["mask"].squeeze().shape
                assert len(mask_shape) == 2, f"FATAL: Mask for {view_key} should have 2 dimensions, but has {len(mask_shape)}."
                assert mask_shape[0] == H and mask_shape[1] == W, f"FATAL: Unexpected mask shape for {view_key}: {mask_shape}. Expected ({H}, {W})."

            print("OK: All masks and normal maps have expected shapes.")

            # --- Check 3: Geometric Consistency (The Most Important Check) ---
            print("\n--- Checking Geometric Consistency (Projecting SMPL into Mask) ---")
            # For this check, we only need the non-batched SMPL vertices
            smpl_check_points = smpl_points[0] if smpl_points.ndim == 3 else smpl_points
            smpl_check_points = smpl_check_points.to(device)

            for i in range(num_views):
                cam = cameras[i]
                mask = torch.from_numpy(in_tensor[f"view_{i}"]["mask"]).to(device)
                
                # Project SMPL vertices using this camera's parameters
                points_in_camera_space = torch.matmul(smpl_check_points, cam.R.T) + cam.t
                
                # Filter points that are behind the camera
                front_points_mask = points_in_camera_space[:, 2] > 0
                if front_points_mask.sum() == 0:
                    print(f"WARNING: For view {i}, all SMPL points are behind the camera. Check camera position/orientation.")
                    continue
                    
                points_in_camera_space = points_in_camera_space[front_points_mask]
                
                # Perform projection to 2D image coordinates
                points_2d_homo = torch.matmul(points_in_camera_space, cam.K.T)
                depth = points_2d_homo[..., 2:3]
                projected_points = points_2d_homo[..., :2] / (depth + 1e-8)
                
                # Check how many projected SMPL points fall inside the foreground mask
                u_coords = projected_points[:, 0].long()
                v_coords = projected_points[:, 1].long()
                
                # Create a validity mask for points that project within the image boundaries
                valid_projection_mask = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)
                
                if valid_projection_mask.sum() == 0:
                    print(f"WARNING: For view {i}, all SMPL points project outside the image frame. Check camera parameters.")
                    continue

                valid_u = u_coords[valid_projection_mask]
                valid_v = v_coords[valid_projection_mask]
                
                # Sample the mask at the projected locations
                mask_values_at_smpl = mask[valid_v, valid_u]
                
                # Calculate the percentage of projected vertices that are inside the mask
                overlap_percentage = (mask_values_at_smpl > 0.5).sum() / len(mask_values_at_smpl) * 100
                
                print(f"View {i}: {overlap_percentage:.2f}% of projected SMPL vertices are inside the foreground mask.")
                
                assert overlap_percentage > 50.0, f"FATAL: Low geometric consistency for view {i} ({overlap_percentage:.2f}%). The SMPL model does not align with the mask. Check camera parameters, SMPL pose, or coordinate systems."

            print("OK: Geometric consistency checks passed.")
            print("="*50)
            print("PRE-FLIGHT CHECKS COMPLETE. ALL SYSTEMS GO.")
            print("="*50 + "\n")
        """
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
        smpl_vertices_transformed = (R_combined @ in_tensor["smpl_verts"].squeeze().T).T
        translation_to_view = torch.tensor([0.0, 0.0, 3.0], device=device)
        smpl_vertices_aligned = smpl_vertices_transformed + translation_to_view
        R_z_90 = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        smpl_vertices_aligned = (R_z_90 @ smpl_vertices_aligned.T).T
        print("\n" + "="*50)
        print("SMPL MODEL BOUNDING VOLUME DIAGNOSTICS")
        print("="*50)

        # 1. Flatten all leading dimensions to get a simple (num_points, 3) tensor.
        # This is the key fix that makes the rest of the code work correctly.
        smpl_flat = smpl_vertices_aligned.reshape(-1, 3)
        
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

        """

        cameras = load_cameras_from_json(cam_param_path, device)

        for i in range(8):
            print(f"\n--- Checking View {i} ---")
            cam = cameras[i]
            # Calculate camera's world position: C = -R' * t
            cam_pos = -cam.R.T @ cam.t
            print(f"DEBUG: Camera {i} World Position: {cam_pos.cpu().detach().numpy()}")
        try:
        # The call is now cleaner, as the function gets everything from in_tensor
            debug_and_visualize_alignment(
                in_tensor=in_tensor,
                cameras=cameras,
                smpl_vertices=smpl_vertices_aligned
            )
        except AssertionError as e:
            print(f"\nPRE-FLIGHT CHECK FAILED: {e}")
            # Exit gracefully so you can inspect the image
            exit()

        #run_preflight_checks(in_tensor, cameras, in_tensor["smpl_verts"].detach())
        # Run optimization with your in_tensor and optional smpl_vertices
        #sdf_net, optimized_cameras = optimize_sdf(in_tensor, cameras, smpl_vertices=smpl_vertices_aligned, num_iterations=10000)
        #extract_mesh_from_sdf(sdf_net, smpl_vertices=smpl_vertices_aligned, grid_size=256, filename="reconstructed_mesh.obj")

        """
        
        pipeline = BNIPipeline(cam_param_path, lambda_c=1e-4, lambda_s = 1e-3)
        mesh = pipeline.run(in_tensor, smpl_vertices_aligned.cpu().numpy())
        mesh.export(os.path.join(args.out_dir, cfg.name, "obj", f"{data['name']}_fused_BNI.obj"))

        
        
        fused_bni_mesh = trimesh.util.concatenate(bni_mesh_list)
        fused_bni_mesh.export(os.path.join(args.out_dir, cfg.name, "obj", f"{data['name']}_{idx}_fused_BNI.obj"))
        occupancies = VoxelGrid.from_mesh(fused_bni_mesh, resolution=cfg.vol_res, loc=[0, 0, 0], scale=2.0, method='fill').data
        occupancies = np.flip(occupancies.transpose(2, 1, 0), axis=1)

        # Make a copy of the array to avoid negative strides
        occupancies = occupancies.copy()
        in_tensor["body_voxels"] = torch.tensor(occupancies).float().unsqueeze(0).to(device)
        # ---- Step 1: IF-Nets mesh completion ----
        with torch.no_grad():
            # Initialize depth voxels from body voxels if not present
            if "depth_voxels" not in in_tensor:
                in_tensor["depth_voxels"] = in_tensor["body_voxels"]
            
            sdf = ifnet.reconEngine(netG=ifnet.netG, batch=in_tensor)
            verts_IF, faces_IF = ifnet.reconEngine.export_mesh(sdf)

        # Optional mesh cleaning
        if ifnet.clean_mesh_flag:
            verts_IF, faces_IF = clean_mesh(verts_IF, faces_IF)

        side_mesh = trimesh.Trimesh(verts_IF, faces_IF)

        # Optional remeshing (Laplacian smoothing)
        side_mesh_path = os.path.join(args.out_dir, cfg.name, "obj", f"{data['name']}_{idx}_IF.obj")
        side_mesh = remesh_laplacian(side_mesh, side_mesh_path)

        # ---- Step 2: Remove invisible or undesirable regions using masks ----
        # Remove eyeballs, keep front face and hands if configured

        full_lst = []

        if "face" in cfg.bni.use_smpl:
            face_mesh = apply_vertex_mask(face_mesh, SMPLX_object.front_flame_vertex_mask)

            if not face_mesh.is_empty:
                face_mesh.vertices -= np.array([0, 0, cfg.bni.thickness])  # offset slightly back

                # Remove face from other meshes
                side_mesh = part_removal(side_mesh, face_mesh, cfg.bni.face_thres, device, smplx_mesh, region="face")
                face_mesh.export(os.path.join(args.out_dir, cfg.name, "obj", f"{data['name']}_{idx}_face.obj"))
                full_lst.append(face_mesh)

        if "hand" in cfg.bni.use_smpl:
            hand_mask = torch.zeros(SMPLX_object.smplx_verts.shape[0], dtype=torch.float32)

            if data['hands_visibility'][idx][0]:
                mano_left_vid = np.unique(np.concatenate([
                    SMPLX_object.smplx_vert_seg["leftHand"],
                    SMPLX_object.smplx_vert_seg["leftHandIndex1"],
                ]))
                hand_mask.index_fill_(0, torch.tensor(mano_left_vid), 1.0)

            if data['hands_visibility'][idx][1]:
                mano_right_vid = np.unique(np.concatenate([
                    SMPLX_object.smplx_vert_seg["rightHand"],
                    SMPLX_object.smplx_vert_seg["rightHandIndex1"],
                ]))
                hand_mask.index_fill_(0, torch.tensor(mano_right_vid), 1.0)

            hand_mesh = apply_vertex_mask(hand_mesh, hand_mask)

            if not hand_mesh.is_empty:
                side_mesh = part_removal(side_mesh, hand_mesh, cfg.bni.hand_thres, device, smplx_mesh, region="hand")
                hand_mesh.export(os.path.join(args.out_dir, cfg.name, "obj", f"{data['name']}_{idx}_hand.obj"))
                full_lst.append(hand_mesh)

        # ---- Step 3: Add cleaned IF-Nets result ----
        side_mesh = part_removal(side_mesh, sum(full_lst), 2e-2, device, smplx_mesh, region="", clean=False)
        full_lst.append(side_mesh)

        # ---- Step 4: Export combined mesh ----
        final_path = os.path.join(args.out_dir, cfg.name, "obj", f"{data['name']}_{idx}_full.obj")

        if cfg.bni.use_poisson:
            final_mesh = poisson(sum(full_lst), final_path, cfg.bni.poisson_depth)
            print(colored(f"\n🧪 Poisson completion to {final_path}", "yellow"))
        else:
            final_mesh = sum(full_lst)
            final_mesh.export(final_path)

        # ---- Step 5: Visualization ----
        if not args.novis:
            dataset.render.load_meshes(final_mesh.vertices, final_mesh.faces)
            rotate_recon_lst = dataset.render.get_image(cam_type="four")
            per_loop_lst.extend([in_tensor['image'][idx:idx + 1]] + rotate_recon_lst)

        # ---- Step 6: Add color from input image ----
        if cfg.bni.texture_src == 'image':
            final_colors = query_color(
                torch.tensor(final_mesh.vertices).float(),
                torch.tensor(final_mesh.faces).long(),
                in_tensor["image"][idx:idx + 1],
                device=device,
            )
            final_mesh.visual.vertex_colors = final_colors
            final_mesh.export(final_path)

        # ---- Step 7: Watertight conversion ----
        final_watertight_path = final_path.replace("_full.obj", "_full_wt.obj")
        watertightifier = MeshCleanProcess(final_path, final_watertight_path)
        result = watertightifier.process(reconstruction_method='poisson', depth=10)

        if result:
            print("✅ The mesh is watertight and has been saved successfully!")
        else:
            print("❗The mesh is not watertight. Further inspection may be needed.")

        # ---- Step 8: Final mesh post-processing ----
        final_output_path = final_path.replace("_full.obj", "_final.obj")
        final_mesh = MeshCleanProcess.process_watertight_mesh(
            final_watertight_path=final_watertight_path,
            output_path=final_output_path,
            face_vertex_mask=SMPLX_object.front_flame_vertex_mask,
            target_faces=15000
        )

        # Save in_tensor for video rendering if needed
        if len(per_loop_lst) > 0 and not args.novis:
            per_data_lst.append(get_optim_grid_image(per_loop_lst, None, nrow=5, type="cloth"))
            per_data_lst[-1].save(os.path.join(args.out_dir, cfg.name, f"png/{data['name']}_cloth.png"))

            os.makedirs(os.path.join(args.out_dir, cfg.name, "vid"), exist_ok=True)
            in_tensor["BNI_verts"] = [torch.tensor(final_mesh.vertices).float()]
            in_tensor["BNI_faces"] = [torch.tensor(final_mesh.faces).long()]
            in_tensor["uncrop_param"] = data["uncrop_param"]
            in_tensor["img_raw"] = data["img_raw"]
            torch.save(in_tensor, os.path.join(args.out_dir, cfg.name, f"vid/{data['name']}_in_tensor.pt"))

        """
        #in_tensor["body_verts"].append(torch.tensor(smpl_obj_lst[idx].vertices).float())
        #in_tensor["body_faces"].append(torch.tensor(smpl_obj_lst[idx].faces).long())

        # requires shape completion when low overlap
        # replace SMPL by completed mesh as side_mesh
        """
        """
        front_data_list = []

        # Inside the loop
        front_data_list.append({
            "F_depth": BNI_object.F_depth.detach().clone(),
            "F_verts": BNI_object.F_verts.detach().clone(),
            "F_faces": BNI_object.F_faces.detach().clone(),
            "F_trimesh": BNI_object.F_trimesh.copy()
        })
    
        # if cfg.bni.use_ifnet:
        if True:

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
        #    model_path="/var/locally-mounted/myshareddir/Fulden/HPS/pixie_data",
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
    
        break"""