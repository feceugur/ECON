import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
import os

import numpy as np
import torch
import torchvision
import trimesh
from pytorch3d.ops import SubdivideMeshes
from termcolor import colored
from tqdm.auto import tqdm

from apps.IFGeo import IFGeo
from apps.Normal_f import Normal
from apps.sapiens import ImageProcessor
from apps.clean_mesh import MeshCleanProcess
from apps.face_rig_exporter import FaceRigExporter

from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.common.config import cfg
from lib.common.imutils import blend_rgb_norm, load_img, transform_to_tensor, wrap
from lib.common.local_affine import register
from lib.common.render import query_color
from lib.common.train_util import Format, init_loss
from lib.common.voxelize import VoxelGrid
from lib.dataset.mesh_util import *
from lib.dataset.TestDataset_f import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis
import argparse
import os
import os.path as osp

import numpy as np
import torch
import trimesh
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree
from termcolor import colored

import lib.smplx as smplx
from lib.common.local_affine import register
from lib.dataset.mesh_util import (
    SMPLX,
    export_obj,
    keep_largest,
    poisson,
)
from lib.smplx.lbs import general_lbs
from trimesh import util

from rembg import remove

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=50)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-in_b_dir", "--in_b_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ_f.yaml")
    parser.add_argument("-multi", action="store_false")
    parser.add_argument("-novis", action="store_true")
    parser.add_argument("-n", "--name", type=str, default="fulden_tpose_f1")
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-uv", action="store_true")
    parser.add_argument("-dress", action="store_true")
    return parser.parse_args()

def make_foreground_mask_from_color(
    img: torch.Tensor,
    bg_color_str: str,
    tol: float = 0.1
) -> torch.Tensor:
    """
    Builds a foreground mask by finding pixels different from a known background color.
    This is more robust than border sampling.

    Args:
        img (torch.Tensor): Input image tensor in [-1, 1] range.
        bg_color_str (str): The known background color, e.g., "red" or "blue".
        tol (float): Distance tolerance for masking.

    Returns:
        torch.Tensor: The calculated foreground mask.
    """
    B, C, H, W = img.shape
    device = img.device

    if bg_color_str.lower() == "red":
        # RGB [255, 0, 0] in [0,255] -> [1.0, -1.0, -1.0] in [-1,1]
        bg_tensor_val = torch.tensor([1.0, -1.0, -1.0], device=device)
    elif bg_color_str.lower() == "blue":
        # RGB [0, 0, 255] in [0,255] -> [-1.0, -1.0, 1.0] in [-1,1]
        bg_tensor_val = torch.tensor([-1.0, -1.0, 1.0], device=device)
    else:
        # Default to black for any other case
        bg_tensor_val = torch.tensor([-1.0, -1.0, -1.0], device=device)

    # Reshape for broadcasting against the image tensor
    bg_colour = bg_tensor_val.view(1, C, 1, 1)

    # Calculate Euclidean distance from the background color and create mask
    dist = torch.sqrt(torch.sum((img - bg_colour)**2, dim=1, keepdim=True))
    mask = (dist > tol).float()
    return mask
def erase_bg(norm_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Sets background texels to zero so their colour cannot
    be confused with a valid surface normal.

    Args
    ----
    norm_tensor : (B, 3, H, W)  normal map   in [-1, 1]
    mask        : (B, H, W)     1 = foreground, 0 = background
    """
    return norm_tensor * mask.unsqueeze(1)          # keep xyz

def setup_configuration(args):
    # Load and merge configuration files.
    cfg.bni.graft_v6_stitch_method = "trimesh_boundary_loft"
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymafx/configs/pymafx_config.yaml")
    device = torch.device(f"cuda:{args.gpu_device}")
    cfg_show_list = [
        "test_gpus", [args.gpu_device],
        "mcube_res", 512,
        "clean_mesh", True,
        "test_mode", True,
        "batch_size", 1
    ]
    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()
    return cfg, device


def load_models(cfg, args, device):
    # Load the normal estimation model.
    normal_path = "/var/locally-mounted/myshareddir/Fulden/ckpt/normal.ckpt"
    normal_net = Normal.load_from_checkpoint(
        cfg=cfg, checkpoint_path=normal_path, map_location=device, strict=False
    )
    normal_net = normal_net.to(device)
    normal_net.netG.eval()
    print(colored(f"Resume Normal Estimator from {Format.start} {cfg.normal_path} {Format.end}", "green"))

    # Optionally load Sapiens if needed.
    sapiens_normal_net = None
    if cfg.sapiens.use:
        sapiens_normal_net = ImageProcessor(device=device)

    # Load IFNet if needed.
    ifnet = None
    if cfg.bni.use_ifnet:
        ifnet = IFGeo.load_from_checkpoint(
            cfg=cfg, checkpoint_path=cfg.ifnet_path, map_location=device, strict=False
        )
        ifnet = ifnet.to(device)
        ifnet.netG.eval()
        print(colored(f"Resume IF-Net+ from {Format.start} {cfg.ifnet_path} {Format.end}", "green"))
        print(colored(f"Complete with {Format.start} IF-Nets+ (Implicit) {Format.end}", "green"))
    else:
        print(colored(f"Complete with {Format.start} SMPL-X (Explicit) {Format.end}", "green"))

    SMPLX_object = SMPLX()
    [attr for attr in dir(SMPLX_object) if not attr.startswith('_')]

    return normal_net, sapiens_normal_net, ifnet, SMPLX_object


def setup_dataset(args, cfg, device, bg_color):
    dataset_param = {
        "image_dir": args.in_dir,
        "image_b_dir": args.in_b_dir,
        "seg_dir": args.seg_dir,
        "use_seg": True,  # with or without segmentation
        "hps_type": cfg.bni.hps_type,  # pymafx/pixie
        "vol_res": cfg.vol_res,
        "single": args.multi,
    }
    dataset = TestDataset(dataset_param, device, bg_color=bg_color)
    print(colored(f"Dataset Size: {len(dataset)}", "green"))
    return dataset

def save_obj(vertices, faces, out_path):
            with open(out_path, 'w') as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces + 1:  # OBJ is 1-indexed
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
                    
def generate_expression_blendshapes(model_path, args, gender="neutral", num_expr=10, device='cpu'):
    """
    Generates expression blendshape meshes for the first num_expr expressions,
    applying weights [-2.0, -1.0, 0.0, 1.0, 2.0] to each.

    Args:
        model_path (str): Path to the directory containing SMPL-X model files.
        args (object): Arguments object containing args.out_dir.
        gender (str): Model gender ('neutral', 'male', 'female').
        num_expr (int): The number of expression coefficients/dimensions to use
                        and generate blendshapes for.
        device (str): PyTorch device ('cpu' or 'cuda').

    Returns:
        tuple: (list, smplx.SMPLX)
            - expression_meshes: A list containing NumPy arrays of vertex positions.
                                 The list will contain num_expr * 5 meshes.
                                 Order: [expr0_w-2, expr0_w-1, ..., expr0_w2, expr1_w-2, ...]
            - smplx_model: The loaded SMPL-X model instance.
    """
    smplx_model = smplx.create(
        model_path=model_path,
        model_type='smplx',
        gender=gender,
        # Assuming the neutral model filename is standard, adjust if needed
        model_filename='SMPLX_NEUTRAL_2020.npz', # Or specific like SMPLX_NEUTRAL_2020.npz
        num_expression_coeffs=num_expr, # Use the passed num_expr
        use_face_contour=True,
        create_expression=False,
        create_betas=False, # Keep other params as potentially needed by the model
        create_global_orient=True,
        create_body_pose=False,
        create_jaw_pose=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_transl=False
    ).to(device)

    expression_meshes = []
    # It's safer to get faces directly from the loaded model instance
    faces = smplx_model.faces_tensor.cpu().numpy() # Get faces as numpy array

    # Define the weights to iterate over
    weights_to_permute = [-5.0, -2.5, 0.0, 2.5, 5.0]

    # Prepare output directory for expression objs
    expression_obj_dir = os.path.join(args.out_dir, "expressions")
    os.makedirs(expression_obj_dir, exist_ok=True)

    with torch.no_grad():
        # Loop through the first num_expr expression indices
        for i in range(num_expr):
            # Loop through the desired weights for the current expression i
            for weight in weights_to_permute:
                # Create a zero vector for expression parameters for each weight iteration
                expr_vector = torch.zeros(1, num_expr).to(device)
                # Set the i-th expression coefficient to the current weight
                expr_vector[0, i] = weight

                # Run the SMPL-X model with the specific expression vector
                output = smplx_model(expression=expr_vector, return_verts=True)
                expr_verts = output.vertices[0].cpu().numpy()

                # Append the resulting mesh vertices to the list
                expression_meshes.append(expr_verts)

                # Save .obj file with a name indicating expression index and weight
                # Create a safe string label for the weight (e.g., neg2p0, 0p0, 1p0)
                weight_label = f"{weight}".replace(".", "p").replace("-", "neg")
                out_path = os.path.join(expression_obj_dir, f"expression_{i:02d}_w{weight_label}.obj")
                save_obj(expr_verts, faces, out_path)
                # print(f"Saved: {out_path}") # Optional: print progress

    return expression_meshes, smplx_model

def process_sample_combined(data, data_b, args, cfg, device, normal_net, sapiens_normal_net, ifnet, SMPLX_object, dataset, bg_color):
    """
    Combined function to process a sample, with corrected rendering and mask generation.
    """
    losses = init_loss()
    sample_name = data['name']

    # Create output directories.
    png_dir = osp.join(args.out_dir, cfg.name, "png")
    obj_dir = osp.join(args.out_dir, cfg.name, "obj")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)

    # Build input tensor.
    in_tensor = {
        "smpl_faces": data["smpl_faces"],
        "image": data["img_icon"].to(device),
        "image_back": data_b["img_icon"].to(device),
        # NOTE: We will overwrite these masks if they are invalid.
        "mask": data["img_mask"].to(device),
        "mask_back": data_b["img_mask"].to(device)
    }

    # --- FIX 1: Regenerate masks from known background color ---
    # This is robust and corrects the all-white mask issue.
    if in_tensor["mask"].mean() > 0.99:
        print(colored(f"Warning: Front mask is invalid. Regenerating from '{bg_color}' background.", "yellow"))
        in_tensor["mask"] = make_foreground_mask_from_color(in_tensor["image"], bg_color, tol=0.15).detach().squeeze(1)

    if in_tensor["mask_back"].mean() > 0.99:
        print(colored(f"Warning: Back mask is invalid. Regenerating from '{bg_color}' background.", "yellow"))
        in_tensor["mask_back"] = make_foreground_mask_from_color(in_tensor["image_back"], bg_color, tol=0.15).detach().squeeze(1)

    # Set optimizable parameters.
    optimed_pose = data["body_pose"].requires_grad_(True)
    optimed_trans = data["trans"].requires_grad_(True)
    optimed_betas = data["betas"].requires_grad_(True)
    optimed_orient = data["global_orient"].requires_grad_(True)

    optimizer_smpl = torch.optim.Adam(
        [optimed_pose, optimed_trans, optimed_betas, optimed_orient], lr=1e-2, amsgrad=True
    )
    scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_smpl, mode="min", factor=0.5, verbose=0, min_lr=1e-5, patience=args.patience
    )

    per_data_lst = []
    N_body, N_pose = optimed_pose.shape[:2]
    smpl_path = f"{args.out_dir}/{cfg.name}/obj/{sample_name}_smpl_00.obj"

    if cfg.sapiens.use:
        sapiens_normal = sapiens_normal_net.process_image(
            Image.fromarray(data["img_raw"].squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)),
            "1b", cfg.sapiens.seg_model
        )
        sapiens_normal_square_lst = [wrap(sapiens_normal, data["uncrop_param"], idx) for idx in range(len(data["img_icon"]))]
        sapiens_normal_square = torch.cat(sapiens_normal_square_lst)

    if osp.exists(smpl_path) and (not cfg.force_smpl_optim):
        smpl_verts_lst, smpl_faces_lst = [], []
        for idx in range(N_body):
            smpl_obj = f"{args.out_dir}/{cfg.name}/obj/{sample_name}_smpl_{idx:02d}.obj"
            smpl_mesh = trimesh.load(smpl_obj)
            smpl_verts_lst.append(torch.tensor(smpl_mesh.vertices, device=device).float())
            smpl_faces_lst.append(torch.tensor(smpl_mesh.faces, device=device).long())
        
        batch_smpl_verts = torch.stack(smpl_verts_lst)
        batch_smpl_faces = torch.stack(smpl_faces_lst)
        
        in_tensor["T_normal_F"], in_tensor["T_normal_B"], T_mask_F, T_mask_B = dataset.render_normal(
            batch_smpl_verts, batch_smpl_faces
        )
        
        with torch.no_grad():
            in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)
            
        in_tensor["smpl_verts"] = batch_smpl_verts * torch.tensor([1., -1., 1.], device=device)
        in_tensor["smpl_faces"] = batch_smpl_faces[:, :, [0, 2, 1]]
    else:
        loop_smpl = tqdm(range(args.loop_smpl))
        for i in loop_smpl:
            optimizer_smpl.zero_grad()

            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).view(N_body, 1, 3, 3)
            optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).view(N_body, N_pose, 3, 3)
            
            smpl_verts, smpl_landmarks, smpl_joints = dataset.smpl_model(
                shape_params=optimed_betas, expression_params=tensor2variable(data["exp"], device),
                body_pose=optimed_pose_mat, global_pose=optimed_orient_mat,
                jaw_pose=tensor2variable(data["jaw_pose"], device),
                left_hand_pose=tensor2variable(data["left_hand_pose"], device),
                right_hand_pose=tensor2variable(data["right_hand_pose"], device),
            )
            smpl_verts = (smpl_verts + optimed_trans) * data["scale"]
            
            in_tensor["T_normal_F"], in_tensor["T_normal_B"], T_mask_F, T_mask_B = dataset.render_normal(
                smpl_verts * torch.tensor([1.0, -1.0, -1.0], device=device),
                in_tensor["smpl_faces"]
            )
            
            with torch.no_grad():
                in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor, args.out_dir)

            smpl_joints = (smpl_joints + optimed_trans) * data["scale"] * torch.tensor([1.0, 1.0, -1.0], device=device)
            smpl_joints_3d = (smpl_joints[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0) * 0.5
            ghum_lmks = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
            ghum_conf = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
            smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]
            diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
            diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])
            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
            gt_arr = in_tensor["mask"].repeat(1, 1, 2)
            diff_S = torch.abs(smpl_arr - gt_arr)
            losses["silhouette"]["value"] = diff_S.mean()
            cloth_overlap = diff_S.sum(dim=[1, 2]) / gt_arr.sum(dim=[1, 2])
            cloth_overlap_flag = cloth_overlap > cfg.cloth_overlap_thres
            losses["joint"]["weight"] = [10.0 if flag else 1.0 for flag in cloth_overlap_flag]
            bg_value = in_tensor["T_normal_F"][0, 0, 0, 0]
            smpl_arr_fake = torch.cat([in_tensor["T_normal_F"][:, 0].ne(bg_value).float(), in_tensor["T_normal_B"][:, 0].ne(bg_value).float()], dim=-1)
            body_overlap = (gt_arr * smpl_arr_fake.gt(0.0)).sum(dim=[1, 2]) / smpl_arr_fake.gt(0.0).sum(dim=[1, 2])
            body_overlap_mask = (gt_arr * smpl_arr_fake).unsqueeze(1)
            body_overlap_flag = body_overlap < cfg.body_overlap_thres
            if not cfg.sapiens.use:
                losses["normal"]["value"] = (diff_F_smpl * body_overlap_mask[..., :512] + diff_B_smpl * body_overlap_mask[..., 512:]).mean() / 2.0
            else:
                losses["normal"]["value"] = diff_F_smpl * body_overlap_mask[..., :512]
            losses["silhouette"]["weight"] = [0 if flag else 1.0 for flag in body_overlap_flag]
            occluded_idx = torch.where(body_overlap_flag)[0]
            ghum_conf[occluded_idx] *= ghum_conf[occluded_idx] > 0.95
            losses["joint"]["value"] = (torch.norm(ghum_lmks - smpl_lmks, dim=2) * ghum_conf).mean(dim=1)
            smpl_loss = sum((losses[k]["value"] * torch.tensor(losses[k]["weight"], device=device)).mean() for k in ["normal", "silhouette", "joint"])
            
            smpl_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(smpl_loss)
            
            pbar_desc = f"Body Fitting -- Total: {smpl_loss:.3f}"
            loop_smpl.set_description(pbar_desc)

        in_tensor["smpl_verts"] = smpl_verts * torch.tensor([1.0, 1.0, -1.0], device=device)
        in_tensor["smpl_faces"] = in_tensor["smpl_faces"][:, :, [0, 2, 1]]

    if not args.novis:
        # --- FIX: Resize tensors to match before concatenation ---
        normal_F_masked = erase_bg(in_tensor["normal_F"], in_tensor["mask"])
        normal_B_masked = erase_bg(in_tensor["normal_B"], in_tensor["mask_back"])
        
        rgb_norm_F = blend_rgb_norm(normal_F_masked, data)
        rgb_norm_B = blend_rgb_norm(normal_B_masked, data_b)
        
        # Get target shape from the front-view image
        target_h, target_w = data["img_raw"].shape[2], data["img_raw"].shape[3]
        target_size = (target_h, target_w)

        # Resize the back-view blended normal to match the front-view's dimensions
        # This prevents the torch.cat error if front/back images have different resolutions.
        if rgb_norm_B.shape[2:] != target_size:
            print(colored(f"Warning: Resizing back overlap image from {rgb_norm_B.shape[2:]} to {target_size} to match front.", "yellow"))
            rgb_norm_B = F.interpolate(rgb_norm_B, size=target_size, mode='bilinear', align_corners=False)

        img_overlap_path = osp.join(args.out_dir, cfg.name, f"png/{sample_name}_overlap.png")
        torchvision.utils.save_image(
            torch.cat([data["img_raw"], rgb_norm_F, rgb_norm_B], dim=-1) / 255.,
            img_overlap_path
        )

    smpl_obj_lst = []
    for idx in range(N_body):
        smpl_obj = trimesh.Trimesh(
            in_tensor["smpl_verts"].detach().cpu()[idx] * torch.tensor([1.0, -1.0, 1.0]),
            in_tensor["smpl_faces"].detach().cpu()[0][:, [0, 2, 1]],
            process=False,
            maintains_order=True,
        )
        smpl_obj_path = f"{args.out_dir}/{cfg.name}/obj/{sample_name}_smpl_{idx:02d}.obj"
        if not osp.exists(smpl_obj_path) or cfg.force_smpl_optim:
            smpl_obj.export(smpl_obj_path)
            smpl_info = {
                "betas": optimed_betas[idx].detach().cpu().unsqueeze(0),
                "body_pose": rotation_matrix_to_angle_axis(optimed_pose_mat[idx].detach()).cpu().unsqueeze(0),
                "global_orient": rotation_matrix_to_angle_axis(optimed_orient_mat[idx].detach()).cpu().unsqueeze(0),
                "transl": optimed_trans[idx].detach().cpu(),
                "expression": data["exp"][idx].cpu().unsqueeze(0),
                "jaw_pose": rotation_matrix_to_angle_axis(data["jaw_pose"][idx]).cpu().unsqueeze(0),
                "left_hand_pose": rotation_matrix_to_angle_axis(data["left_hand_pose"][idx]).cpu().unsqueeze(0),
                "right_hand_pose": rotation_matrix_to_angle_axis(data["right_hand_pose"][idx]).cpu().unsqueeze(0),
                "scale": data["scale"][idx].cpu(),
            }
            np.save(
                smpl_obj_path.replace(".obj", ".npy"),
                smpl_info,
                allow_pickle=True,
            )

        smpl_obj_lst.append(smpl_obj)
    
    del optimizer_smpl, optimed_betas, optimed_orient, optimed_pose, optimed_trans
    torch.cuda.empty_cache()

    per_data_lst = []
    batch_smpl_verts = in_tensor["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0], device=device)
    batch_smpl_faces = in_tensor["smpl_faces"].detach()[:, :, [0, 2, 1]]
    in_tensor["depth_F"], in_tensor["depth_B"] = dataset.render_depth(batch_smpl_verts, batch_smpl_faces)
    per_loop_lst = []
    in_tensor["BNI_verts"] = []
    in_tensor["BNI_faces"] = []
    in_tensor["body_verts"] = []
    in_tensor["body_faces"] = []

    for idx in range(N_body):
        final_path = f"{args.out_dir}/{cfg.name}/obj/{sample_name}_{idx}_full.obj"
        side_mesh = smpl_obj_lst[idx].copy()
        face_mesh = smpl_obj_lst[idx].copy()
        hand_mesh = smpl_obj_lst[idx].copy()
        smplx_mesh = smpl_obj_lst[idx].copy()

        in_tensor["normal_F"] = erase_bg(in_tensor["normal_F"], in_tensor["mask"])
        in_tensor["normal_B"] = erase_bg(in_tensor["normal_B"], in_tensor["mask_back"])
        
        BNI_dict = save_normal_tensor(
            in_tensor, idx, osp.join(args.out_dir, cfg.name, f"BNI/{sample_name}_{idx}"), cfg.bni.thickness,
        )
        BNI_object = BNI(
            dir_path=osp.join(args.out_dir, cfg.name, "BNI"), name=sample_name, BNI_dict=BNI_dict, cfg=cfg.bni, device=device
        )
        BNI_object.extract_surface(False)
        in_tensor["body_verts"].append(torch.tensor(smpl_obj_lst[idx].vertices).float())
        in_tensor["body_faces"].append(torch.tensor(smpl_obj_lst[idx].faces).long())        
        if cfg.bni.use_ifnet:
            side_mesh_path = f"{args.out_dir}/{cfg.name}/obj/{sample_name}_{idx}_IF.obj"
            side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)
            in_tensor.update(
                dataset.depth_to_voxel({
                    "depth_F": BNI_object.F_depth.unsqueeze(0),
                    "depth_B": BNI_object.B_depth.unsqueeze(0)
                })
            )
            occupancies = VoxelGrid.from_mesh(side_mesh, cfg.vol_res, loc=[0, 0, 0], scale=2.0).data.transpose(2, 1, 0)
            occupancies = np.flip(occupancies, axis=1)
            in_tensor["body_voxels"] = torch.tensor(occupancies.copy()).float().unsqueeze(0).to(device)
            with torch.no_grad():
                sdf = ifnet.reconEngine(netG=ifnet.netG, batch=in_tensor)
                verts_IF, faces_IF = ifnet.reconEngine.export_mesh(sdf)
            if ifnet.clean_mesh_flag:
                verts_IF, faces_IF = clean_mesh(verts_IF, faces_IF)
            side_mesh = trimesh.Trimesh(verts_IF, faces_IF)
            # Remesh laplacian (from original process_sample)
            side_mesh = remesh_laplacian(side_mesh, side_mesh_path)
        else:
            
            side_mesh = apply_vertex_mask(
                side_mesh,
                (SMPLX_object.front_flame_vertex_mask + SMPLX_object.smplx_mano_vertex_mask + SMPLX_object.eyeball_vertex_mask).eq(0).float(),
            )
            side_mesh = Meshes(
                verts=[torch.tensor(side_mesh.vertices).float()],
                faces=[torch.tensor(side_mesh.faces).long()],
            ).to(device)
            sm = SubdivideMeshes(side_mesh)
            side_mesh = register(BNI_object.F_B_trimesh, sm(side_mesh), device)

        side_verts = torch.tensor(side_mesh.vertices).float().to(device)
        side_faces = torch.tensor(side_mesh.faces).long().to(device)
        full_lst = []

        if "face" in cfg.bni.use_smpl:
            # NEW: keep both the skin (front-flame) **and** the eyeball spheres
            face_plus_eye_mask = (
                    SMPLX_object.front_flame_vertex_mask
                + SMPLX_object.eyeball_vertex_mask        # =1 on the two spheres
            ).clamp_max(1.0)                                # ensure mask ∈ {0,1}

            face_mesh = apply_vertex_mask(face_mesh, face_plus_eye_mask)

            # face_mesh = apply_vertex_mask(face_mesh, SMPLX_object.front_flame_vertex_mask)
            if not face_mesh.is_empty:
                face_mesh.vertices = face_mesh.vertices - np.array([0, 0, cfg.bni.thickness])
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
                # Export face mesh (from original process_sample)
                face_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{sample_name}_{idx}_face.obj")
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
            hand_mesh = apply_vertex_mask(hand_mesh, hand_mask)
            
            if not hand_mesh.is_empty:
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

                # Export hand mesh (from original process_sample)
                hand_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{sample_name}_{idx}_hand.obj")
                
                full_lst += [hand_mesh]

        full_lst += [BNI_object.F_B_trimesh]
        side_mesh = part_removal(
            side_mesh, sum(full_lst), 2e-2, device, smplx_mesh, region="", clean=False
        )
        full_lst += [side_mesh]

        # Export intermediate meshes (from original process_sample)
        BNI_object.F_B_trimesh.export(
            f"{args.out_dir}/{cfg.name}/obj/{sample_name}_{idx}_BNI.obj"
        )
        side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{sample_name}_{idx}_side.obj")

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
            # Export final mesh if not using Poisson (from original process_sample)
            final_mesh.export(final_path)

        if not args.novis:
            dataset.render.load_meshes(final_mesh.vertices, final_mesh.faces)
            # Use the bg_color parameter here
            rotate_recon_lst = dataset.render.get_image(cam_type="four", bg=bg_color)
            per_loop_lst_front = [in_tensor['image'][idx:idx + 1]] + rotate_recon_lst
            per_loop_lst_back  = [in_tensor['image_back'][idx:idx + 1]] + rotate_recon_lst

            # Export final mesh if texturing from image (from original process_sample)
            if cfg.bni.texture_src == 'image':
                 final_mesh.export(final_path)
            elif cfg.bni.texture_src == 'SD':
                pass # No export needed based on original logic

            if len(per_loop_lst_front) > 0 and len(per_loop_lst_back) > 0:
                cloth_front = get_optim_grid_image(per_loop_lst_front, None, nrow=5, type="cloth_front")
                # Use bg_color in filename
                clean_bg_color = bg_color.strip('"').strip("'") if isinstance(bg_color, str) else str(bg_color) # Handle potential tuple/list
                cloth_front_path = osp.join(args.out_dir, cfg.name, f"png/{sample_name}_cloth_front_{clean_bg_color}.png")
                cloth_front.save(cloth_front_path)
                cloth_back = get_optim_grid_image(per_loop_lst_back, None, nrow=5, type="cloth_back")
                cloth_back_path = osp.join(args.out_dir, cfg.name, f"png/{sample_name}_cloth_back_{clean_bg_color}.png")
                cloth_back.save(cloth_back_path)

            in_tensor["BNI_verts"].append(torch.tensor(final_mesh.vertices).float())
            in_tensor["BNI_faces"].append(torch.tensor(final_mesh.faces).long())
            os.makedirs(osp.join(args.out_dir, cfg.name, "vid"), exist_ok=True)
            in_tensor["uncrop_param"] = data["uncrop_param"]
            in_tensor["img_raw"] = data["img_raw"]
            torch.save(
                in_tensor, osp.join(args.out_dir, cfg.name, f"vid/{sample_name}_in_tensor.pt")
            )

        # Expression Blendshapes (from original process_sample)
        # expression_meshes, smplx_model = generate_expression_blendshapes(
        #     "/var/locally-mounted/myshareddir/Fulden/HPS/pixie_data", # Assuming this path is correct or configured elsewhere
        #     args,
        #     gender="neutral",
        #     num_expr=10,
        #     device="cuda" # Assuming device is cuda, adjust if needed
        # )

        # Face Rig Export (from original process_sample)
        # exporter = FaceRigExporter(smplx_object=SMPLX_object, final_mesh=final_mesh, align_mode='smplx')

        # exporter.export(
        #     data=data,
        #     smpl_verts=smpl_verts,  # shape: [1, N, 3] - Make sure smpl_verts is accessible here
        #     out_dir=args.out_dir,
        #     cfg_name=cfg.name,
        #     expression_meshes=expression_meshes  # shape: List[np.ndarray of (N, 3)]
        # )

        # Watertightening (from original process_sample)
        final_watertight_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full_wt.obj"
        watertightifier = MeshCleanProcess(final_path, final_watertight_path)
        # Pass poisson depth from config if available, e.g., cfg.bni.poisson_depth
        # Ensure cfg.bni.poisson_depth exists or use a default like 10
        poisson_depth_wt = getattr(cfg.bni, 'poisson_depth', 9)
        result = watertightifier.process(reconstruction_method='poisson', depth=poisson_depth_wt)

        if result: # result from watertightifier.process (e.g., True if _full_wt.obj was created successfully)
            print(f"Watertight base mesh for grafting saved successfully to {final_watertight_path}")

            # Define output path for this grafting attempt
            original_smplx_face_path = f"{args.out_dir}/{cfg.name}/obj/{sample_name}_{idx}_face.obj" # Your SMPLX face mask

            if not osp.exists(original_smplx_face_path):
                print(f"ERROR: SMPLX face mesh {original_smplx_face_path} not found. Cannot perform grafting.")
            elif not osp.exists(final_watertight_path): # This is the input body mesh (_full_wt.obj)
                print(f"ERROR: Base body mesh {final_watertight_path} not found. Cannot perform grafting.")
            else: # SMPLX face and body mesh exist
                
                grafted_output_path_v6 = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_final_grafted_v6.obj"

                debug_graft_dir_sample_v6 = None
                if not args.novis:
                    debug_graft_dir_sample_v6 = osp.join(args.out_dir, cfg.name, "debug_grafting_v6", f"{data['name']}_{idx}")
                
                print(f"\nCalling MeshCleanProcess.process_mesh_graft_smplx_face_v6:")

                final_grafted_mesh_obj_v6 = MeshCleanProcess.run_face_grafting_pipeline(
                    full_body_mesh_path=final_watertight_path,
                    output_path=grafted_output_path_v6,
                    smplx_face_mesh_path=original_smplx_face_path,
                    body_simplification_target_faces=12000,
                    hole_boundary_smoothing_iterations = 25,
                    # hole_boundary_smoothing_factor = 0.1,
                    # close_smplx_face_holes = True,
                    # save_debug_body_with_hole = False,
                    # pre_hole_cut_smoothing_iterations = 0,
                    # pre_hole_cut_smoothing_lambda = 0.5,
                    # use_open3d_cleaning = False,
                    # regularize_hole_loop_iterations = 5,
                    # regularize_hole_loop_relaxation = 0.35,
                    # thin_triangle_quality_threshold = 0.005, # <-- ADD THIS LINE
                    # debug_smply_integrity_trace = True # <-- ADD THIS LINE                    
                )

                if final_grafted_mesh_obj_v6 and not final_grafted_mesh_obj_v6.is_empty:
                    print(f"Final grafted mesh (v6) saved to {grafted_output_path_v6}")
                    prefix_for_ply = f"{data['name']}" # Using data['name'] as the base prefix
                    
                    # Construct the full path for the .ply file
                    # It will be in the same directory as the .obj ({args.out_dir}/{cfg.name}/obj/)
                    ply_output_filename = f"{prefix_for_ply}_{idx}_full_soups.ply" # Using idx as per typical naming
                    # If you strictly need "_0_" regardless of idx:
                    # ply_output_filename = f"{prefix_for_ply}_0_full_soups.ply"

                    ply_output_path = osp.join(args.out_dir, cfg.name, "obj", ply_output_filename)

                    try:
                        # final_grafted_mesh_obj_v6 is the trimesh.Trimesh object
                        final_grafted_mesh_obj_v6.export(ply_output_path)
                        print(f"Additionally saved final grafted mesh as .ply to {ply_output_path}")
                    except Exception as e_ply_export:
                        print(f"Error exporting final grafted mesh to .ply ({ply_output_path}): {e_ply_export}")

                else:
                    print(f"Failed to create valid grafted mesh (v6) for {grafted_output_path_v6}")
        
        else: # This 'else' corresponds to 'if result:' from watertightifier.process
            print(f"Failed to create or save watertight base mesh from {final_path}. Skipping grafting process.")

#  tools/pose2tpose.py
import numpy as np, torch, smplx, trimesh
from sklearn.neighbors import NearestNeighbors

def cloth_to_tpose(
        cloth_obj      : str,           # ECON output, *posed* .obj
        smpl_params_npy: str,           # the .npy you already save per subject
        smpl_model_dir : str,           # directory that has the SMPLX_*.npz files
        gender         : str = "neutral",
        device         : str = "cpu"):

    # 0) load stuff -----------------------------------------------------------
    cloth = trimesh.load(cloth_obj, process=False)
    P     = np.load(smpl_params_npy, allow_pickle=True).item()

    smpl  = smplx.create(smpl_model_dir, "smplx",
                         gender=gender, use_pca=False).to(device)

    # 1) recreate the *posed* SMPL-X body ECON fitted
    smpl_out = smpl(
        betas         = torch.tensor(P["betas"][None],        device=device),
        body_pose     = torch.tensor(P["body_pose"][None],    device=device),
        global_orient = torch.tensor(P["global_orient"][None],device=device),
        expression    = torch.tensor(P["expression"][None],   device=device),
        transl        = torch.tensor(P["transl"][None],       device=device),
        pose2rot=True, return_full_pose=True)

    v_posed = smpl_out.vertices[0]                 # (10475,3)
    G       = smpl_out.global_trnsfm[0]            # 24×4×4
    Ginv    = torch.inverse(G)                     # pre-invert once

    # 2) barycentric map: for every cloth vert find its 4 nearest SMPL verts
    nbrs = NearestNeighbors(4).fit(v_posed.cpu())
    d, idx = nbrs.kneighbors(cloth.vertices)
    w = torch.from_numpy(1.0/np.maximum(d,1e-4)); w /= w.sum(1, keepdims=True)

    # 3) inverse LBS: world → T-pose
    v_h   = torch.hstack([torch.tensor(cloth.vertices), torch.ones(len(cloth.vertices),1)])
    v0    = torch.sum((w[:,:,None,None] * Ginv[idx]) @ v_h[:,None,:,None], dim=1)[:,:3,0]

    cloth.vertices[:] = v0.numpy()
    out_path = cloth_obj.replace(".obj", "_tpose.obj")
    cloth.export(out_path)
    print("Neutral mesh written to", out_path)

def main():
    args = parse_args()
    cfg, device = setup_configuration(args)

    # Update output directory based on people list and IFNet usage.
    people = ['Fulden', 'Carla', 'Eric','Rafa', 'Jon', 'Jon2', 'Roger', 'Albert', 'Stefan']
    if cfg.bni.use_ifnet:
        args.out_dir = f'{args.out_dir}/{people[0]}/IFN+_face_thresh_{cfg.bni.face_thres:.2f}'
    else:
        args.out_dir = f'{args.out_dir}/{people[0]}/face_thresh_{cfg.bni.face_thres:.2f}'
    os.makedirs(args.out_dir, exist_ok=True)

    # --- First Run (Red Background) ---
    print(colored("Starting first run with RED background...", "cyan"))
    normal_net, sapiens_normal_net, ifnet, SMPLX_object = load_models(cfg, args, device)
    dataset_red = setup_dataset(args, cfg, device, [255, 0, 0, 255])
    
    pbar_red = tqdm(dataset_red, desc="Red Background Run")
    for data, data_b in pbar_red:
        pbar_red.set_description(f"{data['name']}")
        process_sample_combined(
            data, data_b, args, cfg, device, normal_net, sapiens_normal_net, ifnet,
            SMPLX_object, dataset_red, bg_color="red"
        )
    
    # --- Clean up before second run ---
    del normal_net, sapiens_normal_net, ifnet, SMPLX_object, dataset_red, pbar_red
    torch.cuda.empty_cache()
    
    # --- Second Run (Blue Background) ---
    print(colored("\nStarting second run with BLUE background...", "cyan"))
    # Reload models to ensure a fresh, stateless start, especially for normal_net
    normal_net, sapiens_normal_net, ifnet, SMPLX_object = load_models(cfg, args, device)
    dataset_blue = setup_dataset(args, cfg, device, [0, 0, 255, 255])

    pbar_blue = tqdm(dataset_blue, desc="Blue Background Run")
    for data, data_b in pbar_blue:
        pbar_blue.set_description(f"{data['name']}")
        process_sample_combined(
            data, data_b, args, cfg, device, normal_net, sapiens_normal_net, ifnet,
            SMPLX_object, dataset_blue, bg_color="blue"
        )
        # call it

    # cloth_to_tpose(
    #     cloth_obj       = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_final_grafted_v6.obj",
    #     smpl_params_npy = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_00.npy",
    #     smpl_model_dir  = "/home/ubuntu/models/smplx")

if __name__ == "__main__":
    main()
