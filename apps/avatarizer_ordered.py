import argparse # Keep for imports even if not parsing
import os
import os.path as osp
import cv2
import re
import subprocess, glob

import numpy as np
import torch
import trimesh
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree
# from termcolor import colored # Not used in the provided snippet

import lib.smplx as smplx
from lib.common.local_affine import register
from lib.dataset.mesh_util import (
    SMPLX,
    export_obj,
    keep_largest,
    # poisson, # Not used in the always-generate path for econ_da
)
# from lib.smplx.lbs import general_lbs # Not directly used

from PIL import Image
from torchvision import transforms
from lib.common.render import query_color, query_normal_color
from lib.common.render_utils import Pytorch3dRasterizer
import xatlas # Ensure xatlas is imported for UV generation

# --- Hardcoded Configurations ---
CONFIG_NAME = "carla_Apose"
CONFIG_GPU_ID = 0
CONFIG_PROCESS_UV_EXPORT = True
CONFIG_USE_DRESS_LOGIC = False


def setup_environment_and_load_inputs(name, gpu_id):
    print(f"Setting up for sample: {name}")
    device = torch.device(f"cuda:{gpu_id}")
    smplx_container = SMPLX()

    prefix = f"./results/Carla/IFN+_face_thresh_0.01/econ/obj/{name}"
    cache_path = f"{prefix.replace('obj','cache')}"

    os.makedirs(prefix, exist_ok=True)
    os.makedirs(cache_path, exist_ok=True)

    smpl_path = f"{prefix}_smpl_00.npy"
    smplx_param = np.load(smpl_path, allow_pickle=True).item()

    econ_path = f"{prefix}_0_full_soups.ply"
    econ_obj_orig = trimesh.load(econ_path)
    assert econ_obj_orig.vertex_normals.shape[1] == 3, "ECON obj must have vertex normals"

    # Align econ with SMPL-X (initial alignment)
    econ_obj_aligned = econ_obj_orig.copy()
    econ_obj_aligned.vertices *= np.array([1.0, -1.0, -1.0])
    econ_obj_aligned.vertices /= smplx_param["scale"].cpu().numpy()
    econ_obj_aligned.vertices -= smplx_param["transl"].cpu().numpy()

    smplx_param_cpu = {}
    for key in smplx_param.keys():
        if isinstance(smplx_param[key], torch.Tensor):
            smplx_param_cpu[key] = smplx_param[key].cpu().view(1, -1)
        else:
            smplx_param_cpu[key] = smplx_param[key]


    return device, smplx_container, prefix, cache_path, smplx_param_cpu, econ_obj_aligned

def prepare_smplx_model_and_poses(smplx_container, smplx_param_cpu_dict):
    print("Preparing SMPL-X model and poses...")
    smpl_model = smplx.create(
        smplx_container.model_dir, model_type="smplx", gender="neutral", age="adult",
        use_face_contour=True, use_pca=False, create_expression=True, create_betas=False,
        create_global_orient=True, create_body_pose=True, create_jaw_pose=True,
        create_left_hand_pose=False, create_right_hand_pose=False, create_transl=False,
        num_betas=smplx_param_cpu_dict["betas"].shape[1],
        num_expression_coeffs=smplx_param_cpu_dict["expression"].shape[1], ext="pkl"
    )
    smpl_out_lst = []
    for pose_type in ["a-pose", "t-pose", "da-pose", "pose"]: # pose_type indices: 0, 1, 2, 3
        smpl_out = smpl_model(
            body_pose=smplx_param_cpu_dict["body_pose"], global_orient=smplx_param_cpu_dict["global_orient"],
            betas=smplx_param_cpu_dict["betas"], expression=smplx_param_cpu_dict["expression"],
            jaw_pose=smplx_param_cpu_dict["jaw_pose"], left_hand_pose=smplx_param_cpu_dict["left_hand_pose"],
            right_hand_pose=smplx_param_cpu_dict["right_hand_pose"], return_verts=True,
            return_full_pose=True, return_joint_transformation=True,
            return_vertex_transformation=True, pose_type=pose_type
        )
        smpl_out_lst.append(smpl_out)
    return smpl_model, smpl_out_lst

def generate_da_space_meshes(econ_obj_aligned, smpl_out_lst, smpl_model, smplx_container, prefix_path, device_obj, use_dress_logic_flag):
    print("Generating DA-space meshes (econ_da, smpl_da)...")
    # Initial nearest neighbor search from original ECON to original posed SMPL-X (smpl_out_lst[3])
    smpl_verts_posed = smpl_out_lst[3].vertices.detach()[0].cpu().numpy()
    smpl_tree_posed = cKDTree(smpl_verts_posed)
    _dist_orig, idx_econ_to_smpl_posed = smpl_tree_posed.query(econ_obj_aligned.vertices, k=3)

    # t-pose for ECON (based on original econ_obj_aligned)
    econ_verts_tensor = torch.tensor(econ_obj_aligned.vertices, dtype=torch.float32)
    # Transformation from original posed SMPL-X to T-pose
    rot_mat_t = smpl_out_lst[3].vertex_transformation.detach()[0][idx_econ_to_smpl_posed[:, 0]]
    homo_coord = torch.ones_like(econ_verts_tensor)[..., :1]
    econ_cano_verts_calc = torch.inverse(rot_mat_t) @ torch.cat([econ_verts_tensor, homo_coord], dim=1).unsqueeze(-1)
    econ_cano_verts_calc = econ_cano_verts_calc[:, :3, 0].cpu()
    econ_cano_trimesh = trimesh.Trimesh(econ_cano_verts_calc.numpy(), econ_obj_aligned.faces)

    # da-pose for ECON (based on original econ_obj_aligned)
    # Transformation from T-pose to DA-pose (using SMPL-X DA-pose transformations)
    rot_mat_da_smplx = smpl_out_lst[2].vertex_transformation.detach()[0][idx_econ_to_smpl_posed[:, 0]]
    econ_da_verts_calc = rot_mat_da_smplx @ torch.cat([econ_cano_verts_calc, homo_coord], dim=1).unsqueeze(-1)
    econ_da_initial = trimesh.Trimesh(econ_da_verts_calc[:, :3, 0].cpu().numpy(), econ_obj_aligned.faces)

    # da-pose for SMPL-X
    smpl_da_trimesh = trimesh.Trimesh(
        smpl_out_lst[2].vertices.detach()[0].cpu().numpy(),
        smpl_model.faces, maintain_orders=True, process=True
    )
    smpl_da_trimesh.export(f"{prefix_path}/smpl_da.obj")

    # --- Stitching and Refinement in DA-space ---
    # ignore parts: hands, front_flame, eyeball
    ignore_vid = np.concatenate([
        smplx_container.smplx_mano_vid,
        smplx_container.smplx_front_flame_vid,
        smplx_container.smplx_eyeball_vid,
    ])
    if use_dress_logic_flag:
        ignore_vid = np.concatenate([ignore_vid, smplx_container.smplx_leg_vid])

    # remove ignore parts from ECON_da_initial (based on mapping to original SMPL-X)
    econ_da_body = econ_da_initial.copy()
    mano_mask_on_econ_orig_verts = ~np.isin(idx_econ_to_smpl_posed[:, 0], smplx_container.smplx_mano_vid) # Mask on original econ vertices
    
    # Remove over-stretched faces from ECON_da_initial (streching compared to original econ_obj_aligned)
    edge_before = np.sqrt(
        ((econ_obj_aligned.vertices[econ_cano_trimesh.edges[:, 0]] -
          econ_obj_aligned.vertices[econ_cano_trimesh.edges[:, 1]])**2).sum(axis=1)
    )
    edge_after = np.sqrt(
        ((econ_da_initial.vertices[econ_cano_trimesh.edges[:, 0]] -
          econ_da_initial.vertices[econ_cano_trimesh.edges[:, 1]])**2).sum(axis=1)
    )
    edge_diff = edge_after / edge_before.clip(1e-2)
    streched_vid_on_cano = np.unique(econ_cano_trimesh.edges[edge_diff > 6])
    
    # Map streched_vid_on_cano (which are indices on econ_cano_trimesh.vertices) back to original econ_obj_aligned vertex indices
    # Assuming econ_cano_trimesh.vertices and econ_obj_aligned.vertices have same order and count
    mano_mask_final_on_econ_verts = mano_mask_on_econ_orig_verts.copy()
    if streched_vid_on_cano.size > 0 : # Check if not empty
         mano_mask_final_on_econ_verts[streched_vid_on_cano] = False # streched_vid_on_cano are effectively indices for econ_da_initial.vertices

    econ_da_body.update_faces(mano_mask_final_on_econ_verts[econ_da_initial.faces].all(axis=1))
    econ_da_body.remove_unreferenced_vertices()
    econ_da_body = keep_largest(econ_da_body)


    # Prepare SMPL-X DA body for registration
    smpl_da_body_trimesh = smpl_da_trimesh.copy() # Start with full SMPL-X DA
    # No specific removal from SMPL-X body before registration in original code, it used register_mask = all True
    # smpl_da_body_trimesh.update_faces(register_mask[smpl_da_body_trimesh.faces].all(axis=1)) # register_mask was all True
    # smpl_da_body_trimesh.remove_unreferenced_vertices() # Not strictly needed if no faces removed
    # smpl_da_body_trimesh = keep_largest(smpl_da_body_trimesh) # Not strictly needed

    smpl_da_body_pytorch3d = Meshes(
        verts=[torch.tensor(smpl_da_body_trimesh.vertices, dtype=torch.float32)],
        faces=[torch.tensor(smpl_da_body_trimesh.faces, dtype=torch.int64)],
    ).to(device_obj)
    subdivider = SubdivideMeshes()
    smpl_da_body_subdivided = subdivider(smpl_da_body_pytorch3d)
    
    # Register subdivided SMPL-X DA body to econ_da_body
    smpl_da_body_registered = register(econ_da_body, smpl_da_body_subdivided, device_obj) # Returns Trimesh

    # Stitch: econ_da_body (cleaned), smpl_da_body_registered (inpainting parts), smpl_hand
    econ_da_tree = cKDTree(econ_da_initial.vertices) # Query against original econ_da_initial for distances
    dist_smpl_reg_to_econ_da, _ = econ_da_tree.query(smpl_da_body_registered.vertices, k=1)
    
    smpl_da_body_to_stitch = smpl_da_body_registered.copy()
    smpl_da_body_to_stitch.update_faces((dist_smpl_reg_to_econ_da > 0.02)[smpl_da_body_to_stitch.faces].all(axis=1))
    smpl_da_body_to_stitch.remove_unreferenced_vertices()

    smpl_hand_trimesh = smpl_da_trimesh.copy()
    smpl_hand_trimesh.update_faces(
        smplx_container.smplx_mano_vertex_mask.numpy()[smpl_hand_trimesh.faces].all(axis=1)
    )
    smpl_hand_trimesh.remove_unreferenced_vertices()

    econ_da_final = sum([smpl_hand_trimesh, smpl_da_body_to_stitch, econ_da_body])
    # econ_da_final.export(f"{prefix_path}/econ_da.obj") # Optional: save the final stitched econ_da
    
    return econ_da_final, smpl_da_trimesh


def build_smplx_compatible_econ(econ_da_final_trimesh, smpl_da_trimesh, smpl_model_obj, smpl_outputs):
    print("Building SMPL-X compatible ECON model...")
    smpl_da_tree = cKDTree(smpl_da_trimesh.vertices)
    dist_econ_to_smpl_da, idx_econ_da_to_smpl_da = smpl_da_tree.query(econ_da_final_trimesh.vertices, k=3)
    knn_weights = np.exp(-(dist_econ_to_smpl_da**2))
    knn_weights /= knn_weights.sum(axis=1, keepdims=True).clip(min=1e-10) # Add clip for stability

    econ_J_regressor = (smpl_model_obj.J_regressor[:, idx_econ_da_to_smpl_da] * knn_weights[None]).sum(dim=-1)
    econ_lbs_weights = (smpl_model_obj.lbs_weights.T[:, idx_econ_da_to_smpl_da] * knn_weights[None]).sum(dim=-1).T

    num_posedirs = smpl_model_obj.posedirs.shape[0]
    econ_posedirs = ((
        smpl_model_obj.posedirs.view(num_posedirs, -1, 3)[:, idx_econ_da_to_smpl_da, :] * knn_weights[None, ..., None]
    ).sum(dim=-2).view(num_posedirs, -1).float())

    econ_J_regressor /= econ_J_regressor.sum(dim=1, keepdims=True).clip(min=1e-10)
    econ_lbs_weights /= econ_lbs_weights.sum(dim=1, keepdims=True).clip(min=1e-10) # Add clip

    # Canonicalize the NEW ECON (econ_da_final_trimesh)
    rot_mat_da_transform = smpl_outputs[2].vertex_transformation.detach()[0][idx_econ_da_to_smpl_da[:, 0]]
    econ_da_verts_tensor = torch.tensor(econ_da_final_trimesh.vertices, dtype=torch.float32)
    homo_coord_econ = torch.ones_like(econ_da_verts_tensor)[..., :1]
    
    econ_cano_verts_final = torch.inverse(rot_mat_da_transform) @ torch.cat(
        [econ_da_verts_tensor, homo_coord_econ], dim=1
    ).unsqueeze(-1)
    econ_cano_verts_final = econ_cano_verts_final[:, :3, 0].double()

    return econ_J_regressor, econ_lbs_weights, econ_posedirs, econ_cano_verts_final, idx_econ_da_to_smpl_da


def animate_econ_to_pose(econ_cano_verts_final_tensor, econ_da_faces_np, smpl_outputs,
                         idx_econ_to_smpl_da_knn, smplx_params_dict, prefix_path):
    print("Animating ECON to original pose...")
    # Use transformation from original pose (smpl_outputs[3]) based on NEW knn mapping (idx_econ_to_smpl_da_knn)
    rot_mat_pose_transform = smpl_outputs[3].vertex_transformation.detach()[0][idx_econ_to_smpl_da_knn[:, 0]]
    homo_coord_cano = torch.ones_like(econ_cano_verts_final_tensor.float())[..., :1]

    posed_econ_verts_tensor = rot_mat_pose_transform @ torch.cat(
        [econ_cano_verts_final_tensor.float(), homo_coord_cano], dim=1
    ).unsqueeze(-1)
    posed_econ_verts_tensor = posed_econ_verts_tensor[:, :3, 0].double()

    aligned_econ_verts_np = posed_econ_verts_tensor.detach().cpu().numpy()
    aligned_econ_verts_np += smplx_params_dict["transl"].cpu().numpy() # smplx_params_dict should be the cpu version
    aligned_econ_verts_np *= smplx_params_dict["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
    
    econ_pose_trimesh = trimesh.Trimesh(aligned_econ_verts_np, econ_da_faces_np)
    assert econ_pose_trimesh.vertex_normals.shape[1] == 3, "Posed ECON must have vertex normals"
    econ_pose_trimesh.export(f"{prefix_path}/econ_pose.ply")
    return econ_pose_trimesh

def generate_uv_coordinates(econ_pose_mesh, cache_path_base, device_obj):
    print("Generating UV coordinates with xatlas...")
    v_np_uv = econ_pose_mesh.vertices
    f_np_uv = econ_pose_mesh.faces
    
    # Define cache paths for UVs, but we will always regenerate and overwrite
    vt_cache_file = osp.join(cache_path_base, "vt.pt")
    ft_cache_file = osp.join(cache_path_base, "ft.pt")

    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np_uv, f_np_uv)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    chart_options.max_iterations = 4
    pack_options.resolution = 8192 # Texture map resolution
    pack_options.bruteForce = True # Ensures good packing
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    
    vmapping_np, ft_np, vt_np = atlas[0] 

    vt_tensor = torch.from_numpy(vt_np.astype(np.float32)).float().to(device_obj)
    ft_tensor = torch.from_numpy(ft_np.astype(np.int64)).int().to(device_obj)

    torch.save(vt_tensor.cpu(), vt_cache_file) # Save the newly generated UVs
    torch.save(ft_tensor.cpu(), ft_cache_file) # Save the newly generated UVs
    print(f"UV coordinates generated and saved to {vt_cache_file}, {ft_cache_file}")
    
    return vt_tensor, ft_tensor # vmapping_np is also available if needed elsewhere

def perform_color_mapping(econ_pose_mesh, vt_coords_tensor, ft_coords_tensor, device_obj,
                          uv_rasterizer, name_identifier, prefix_path, cache_path_base):
    print("Performing color mapping (Pass 1 and 2)...")
    verts_tensor = torch.tensor(econ_pose_mesh.vertices, dtype=torch.float32).to(device_obj)
    faces_tensor = torch.tensor(econ_pose_mesh.faces, dtype=torch.int64).to(device_obj)

    # --- Pass 1: Red ---
    cloth_front_red_path = f"./results/Carla/IFN+_face_thresh_0.01/econ/png/{name_identifier}_cloth_front_red.png"
    cloth_back_red_path  = f"./results/Carla/IFN+_face_thresh_0.01/econ/png/{name_identifier}_cloth_back_red.png"
    
    tensor_front_1 = transforms.ToTensor()(Image.open(cloth_front_red_path))[:, :, :512] # Assuming 512 width
    tensor_back_1  = transforms.ToTensor()(Image.open(cloth_back_red_path))[:, :, :512]  # Assuming 512 width

    front_image_1 = ((tensor_front_1 - 0.5) * 2.0).unsqueeze(0).to(device_obj)
    back_image_1  = ((tensor_back_1 - 0.5) * 2.0).unsqueeze(0).to(device_obj)

    final_rgb_pass1_np = query_color(
        verts_tensor, faces_tensor, front_image_1, back_image_1, device=device_obj
    ).cpu().numpy()

    texture_map1_np = uv_rasterizer.get_texture(
        torch.cat([(vt_coords_tensor - 0.5) * 2.0, torch.ones_like(vt_coords_tensor[:, :1])], dim=1),
        ft_coords_tensor,
        verts_tensor.unsqueeze(0), # Batch dim
        faces_tensor.unsqueeze(0), # Batch dim
        torch.tensor(final_rgb_pass1_np, dtype=torch.float32).unsqueeze(0).to(device_obj) / 255.0, # Batch dim
    ) # Returns numpy array
    texture_map1_8bit_np = (texture_map1_np * 255.0).astype(np.uint8)
    os.makedirs(f"{cache_path_base}/red", exist_ok=True)
    Image.fromarray(texture_map1_8bit_np).save(f"{cache_path_base}/red/texture_red.png")
    print("First-pass (red) texture map saved.")

    # --- Pass 2: Blue ---
    cloth_front_blue_path = f"./results/Carla/IFN+_face_thresh_0.01/econ/png/{name_identifier}_cloth_front_blue.png"
    cloth_back_blue_path  = f"./results/Carla/IFN+_face_thresh_0.01/econ/png/{name_identifier}_cloth_back_blue.png"

    tensor_front_2 = transforms.ToTensor()(Image.open(cloth_front_blue_path))[:, :, :512]
    tensor_back_2  = transforms.ToTensor()(Image.open(cloth_back_blue_path))[:, :, :512]

    front_image_2 = ((tensor_front_2 - 0.5) * 2.0).unsqueeze(0).to(device_obj)
    back_image_2  = ((tensor_back_2 - 0.5) * 2.0).unsqueeze(0).to(device_obj)

    final_rgb_pass2_np = query_color(
        verts_tensor, faces_tensor, front_image_2, back_image_2, device=device_obj
    ).cpu().numpy()
    
    texture_map2_np = uv_rasterizer.get_texture(
        torch.cat([(vt_coords_tensor - 0.5) * 2.0, torch.ones_like(vt_coords_tensor[:, :1])], dim=1),
        ft_coords_tensor,
        verts_tensor.unsqueeze(0),
        faces_tensor.unsqueeze(0),
        torch.tensor(final_rgb_pass2_np, dtype=torch.float32).unsqueeze(0).to(device_obj) / 255.0,
    )
    texture_map2_8bit_np = (texture_map2_np * 255.0).astype(np.uint8)
    os.makedirs(f"{cache_path_base}/blue", exist_ok=True)
    Image.fromarray(texture_map2_8bit_np).save(f"{cache_path_base}/blue/texture_blue.png")
    print("Second-pass (blue) texture map saved.")

    # --- Final RGB determination and export ---
    # Original script uses pass2 as final if both are computed
    final_rgb_colors_np = final_rgb_pass2_np 
    econ_pose_mesh_copy = econ_pose_mesh.copy() # Avoid modifying the input mesh directly
    econ_pose_mesh_copy.visual.vertex_colors = final_rgb_colors_np
    econ_pose_mesh_copy.export(f"{prefix_path}/econ_icp_rgb.ply")
    print("ICP RGB PLY exported.")
    
    return final_rgb_colors_np, texture_map1_8bit_np, texture_map2_8bit_np


def perform_normal_based_coloring(econ_pose_mesh, device_obj, prefix_path):
    print("Performing normal-based coloring...")
    verts_tensor = torch.tensor(econ_pose_mesh.vertices, dtype=torch.float32).to(device_obj)
    faces_tensor = torch.tensor(econ_pose_mesh.faces, dtype=torch.int64).to(device_obj)

    # Always generate normal colors anew
    normal_colors_np = query_normal_color(
        verts_tensor, faces_tensor, device=device_obj,
    ).cpu().numpy()
    
    econ_pose_mesh_copy = econ_pose_mesh.copy()
    econ_pose_mesh_copy.visual.vertex_colors = normal_colors_np
    econ_pose_mesh_copy.export(f"{prefix_path}/econ_icp_normal.ply")
    print("ICP Normal PLY exported.")
    return normal_colors_np

def save_econ_data_pt(econ_cano_verts_template, posedirs_data, j_regressor_data, smpl_parents,
                      lbs_weights_data, rgb_colors, normal_colors, econ_faces, cache_path_base):
    print("Saving ECON data (.pt)...")
    econ_dict = {
        "v_template": econ_cano_verts_template.unsqueeze(0), # Add batch dim
        "posedirs": posedirs_data,
        "J_regressor": j_regressor_data,
        "parents": smpl_parents,
        "lbs_weights": lbs_weights_data,
        "final_rgb": rgb_colors,
        "final_normal": normal_colors,
        "faces": econ_faces,
    }
    torch.save(econ_dict, f"{cache_path_base}/econ.pt")
    print(f"ECON data saved to {cache_path_base}/econ.pt")

def export_uv_textures_and_defrag(econ_pose_mesh, vt_coords_tensor, ft_coords_tensor, final_rgb_colors_np,
                                  cache_path_base, uv_rasterizer, name_identifier,
                                  texture_map_red_8bit, texture_map_blue_8bit): # Pass pre-computed textures
    print("Starting UV texture export and defragmentation process...")
    v_np = econ_pose_mesh.vertices
    f_np = econ_pose_mesh.faces

    # texture_npy for final_rgb (used if red/blue specific defrag isn't the goal for final output)
    # This was the texture_npy from the original if args.uv block.
    # It uses the `final_rgb_colors_np` which is typically from pass2 (blue).
    _texture_map_final_rgb_np = uv_rasterizer.get_texture(
        torch.cat([(vt_coords_tensor - 0.5) * 2.0, torch.ones_like(vt_coords_tensor[:, :1])], dim=1),
        ft_coords_tensor,
        torch.tensor(v_np, dtype=torch.float32).unsqueeze(0).to(vt_coords_tensor.device),
        torch.tensor(f_np, dtype=torch.int64).unsqueeze(0).to(ft_coords_tensor.device),
        torch.tensor(final_rgb_colors_np, dtype=torch.float32).unsqueeze(0).to(vt_coords_tensor.device) / 255.0,
    )
    # texture_map_final_rgb_8bit = (texture_map_final_rgb_np * 255.0).astype(np.uint8)
    # Image.fromarray(texture_map_final_rgb_8bit).save(f"{cache_path_base}/texture_final_rgb_defrag_input.png")


    # --- Export meshes for defrag (using pre-computed red/blue texture maps) ---
    red_dir = osp.join(cache_path_base, "red") # Already created in color_mapping
    blue_dir = osp.join(cache_path_base, "blue") # Already created
    
    # Red mesh uses texture_map_red_8bit (which is texture_red.png)
    export_obj(v_np, f_np, vt_coords_tensor.cpu().numpy(), ft_coords_tensor.cpu().numpy(), osp.join(red_dir, "mesh_red.obj"))
    with open(osp.join(red_dir, "material.mtl"), "w") as fp:
        fp.write("newmtl mat0\n")
        fp.write("Ka 1.000000 1.000000 1.000000\nKd 1.000000 1.000000 1.000000\nKs 0.000000 0.000000 0.000000\n")
        fp.write("Tr 1.000000\nillum 1\nNs 0.000000\nmap_Kd texture_red.png\n")

    # Blue mesh uses texture_map_blue_8bit (which is texture_blue.png)
    export_obj(v_np, f_np, vt_coords_tensor.cpu().numpy(), ft_coords_tensor.cpu().numpy(), osp.join(blue_dir, "mesh_blue.obj"))
    with open(osp.join(blue_dir, "material.mtl"), "w") as fp:
        fp.write("newmtl mat0\n")
        fp.write("Ka 1.000000 1.000000 1.000000\nKd 1.000000 1.000000 1.000000\nKs 0.000000 0.000000 0.000000\n")
        fp.write("Tr 1.000000\nillum 1\nNs 0.000000\nmap_Kd texture_blue.png\n") # References texture_blue.png

    defrag_path = osp.join(cache_path_base, "defrag_assets")
    os.makedirs(defrag_path, exist_ok=True)
    final_dir = osp.join(cache_path_base, "final_files")
    os.makedirs(final_dir, exist_ok=True)

    texture_defrag_exe = os.path.abspath("./texture-defrag/build/texture-defrag")
    if not osp.exists(texture_defrag_exe):
        print(f"Warning: texture-defrag executable not found at {texture_defrag_exe}. Skipping defrag.")
        return

    # --- Run texture-defrag ---
    mesh_file_red = os.path.abspath(osp.join(red_dir, "mesh_red.obj"))
    output_file_red = os.path.abspath(osp.join(defrag_path, "defrag_red.obj"))
    cmd_red = ["xvfb-run", "-a", texture_defrag_exe, mesh_file_red, "-l", "0", "-o", output_file_red]
    print(f"Running texture-defrag for red mesh: {' '.join(cmd_red)}")
    try:
        subprocess.run(cmd_red, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running texture-defrag for red mesh: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return # Stop if defrag fails

    mesh_file_blue = os.path.abspath(osp.join(blue_dir, "mesh_blue.obj"))
    output_file_blue = os.path.abspath(osp.join(defrag_path, "defrag_blue.obj"))
    cmd_blue = ["xvfb-run", "-a", texture_defrag_exe, mesh_file_blue, "-l", "0", "-o", output_file_blue]
    print(f"Running texture-defrag for blue mesh: {' '.join(cmd_blue)}")
    try:
        subprocess.run(cmd_blue, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running texture-defrag for blue mesh: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return

    # --- Process defrag outputs (inpainting) ---
    red_pattern = re.compile(r"defrag_red_texture_(.+)\.png")
    blue_pattern = re.compile(r"defrag_blue_texture_(.+)\.png")
    red_images_map = {}
    blue_images_map = {}

    for file_item in os.listdir(defrag_path):
        if "defrag" in file_item and file_item.endswith(".png"):
            match_red = red_pattern.search(file_item)
            if match_red:
                img_id = match_red.group(1)
                red_images_map[img_id] = osp.join(defrag_path, file_item)
            match_blue = blue_pattern.search(file_item)
            if match_blue:
                img_id = match_blue.group(1)
                blue_images_map[img_id] = osp.join(defrag_path, file_item)
    
    inpainted_texture_paths = {} # Stores {img_id: path_to_inpainted_texture}

    for img_id in red_images_map:
        if img_id in blue_images_map:
            red_img_path = red_images_map[img_id]
            blue_img_path = blue_images_map[img_id]
            print(f"Processing defragged textures for ID: {img_id}")
            
            red_img_pil = Image.open(red_img_path).convert("RGB")
            blue_img_pil = Image.open(blue_img_path).convert("RGB")

            if red_img_pil.size != blue_img_pil.size:
                print(f"Warning: Texture size mismatch for ID {img_id}. Red: {red_img_pil.size}, Blue: {blue_img_pil.size}. Skipping inpainting for this ID.")
                # Use blue as fallback if sizes differ, or skip
                inpainted_texture_filename = f"texture_map_inpainted_{img_id}.png"
                final_texture_path = osp.join(final_dir, inpainted_texture_filename)
                blue_img_pil.save(final_texture_path) # Save blue directly
                inpainted_texture_paths[img_id] = final_texture_path
                continue


            red_img_np = np.array(red_img_pil, dtype=np.float32)
            blue_img_np = np.array(blue_img_pil, dtype=np.float32)
            
            diff = np.abs(red_img_np - blue_img_np)
            diff_sum = np.sum(diff, axis=2)
            # Increased threshold slightly to be less sensitive to minor aliasing diffs from defrag
            diff_mask = (diff_sum > 1.0).astype(np.uint8) * 255 # Threshold for difference
            missing_mask_red = (red_img_np.sum(axis=2) == 0) # Pure black in red
            missing_mask_blue = (blue_img_np.sum(axis=2) == 0) # Pure black in blue
            # Consider a pixel for inpainting if it's different OR missing in EITHER red or blue
            # (assuming black means untextured after defrag)
            final_mask_for_inpaint = np.maximum(diff_mask, (missing_mask_red | missing_mask_blue).astype(np.uint8) * 255)
            
            # Inpainting: Use RED image as base, inpaint areas defined by final_mask_for_inpaint
            texture_to_inpaint_8bit = red_img_np.astype(np.uint8) # Use red as base
            
            # Dilate the mask to ensure edges of holes are covered
            small_kernel = np.ones((10,10), np.uint8) # Kernel for erosion before dilation
            eroded_mask = cv2.erode(final_mask_for_inpaint, small_kernel, iterations=3)
            dilation_kernel = np.ones((45, 45), np.uint8) # Larger kernel for dilation
            dilated_mask_for_inpaint = cv2.dilate(eroded_mask, dilation_kernel, iterations=2)
            
            inpainted_texture_cv = cv2.inpaint(texture_to_inpaint_8bit, dilated_mask_for_inpaint, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            # cv2 functions often use BGR, ensure it's RGB if needed, or save as BGR if cv2.imwrite expects that
            # inpainted_texture_cv_rgb = cv2.cvtColor(inpainted_texture_cv, cv2.COLOR_BGR2RGB) # If input was RGB

            inpainted_texture_filename = f"texture_map_inpainted_{img_id}.png"
            final_texture_path = osp.join(final_dir, inpainted_texture_filename)
            Image.fromarray(inpainted_texture_cv).save(final_texture_path) # Save with PIL assuming RGB
            # cv2.imwrite(final_texture_path, inpainted_texture_cv) # If saving with OpenCV (usually BGR)
            print(f"Saved inpainted texture for ID {img_id} at {final_texture_path}")
            inpainted_texture_paths[img_id] = final_texture_path
        else: # Only red or blue exists for this ID after defrag
            img_path_to_copy = red_images_map.get(img_id) or blue_images_map.get(img_id)
            if img_path_to_copy:
                print(f"Only one defragged texture found for ID {img_id}. Copying: {osp.basename(img_path_to_copy)}")
                img_to_copy = Image.open(img_path_to_copy)
                copied_texture_filename = f"texture_map_inpainted_{img_id}.png" # Use same naming
                final_texture_path = osp.join(final_dir, copied_texture_filename)
                img_to_copy.save(final_texture_path)
                inpainted_texture_paths[img_id] = final_texture_path


    # --- Create final MTL and OBJ ---
    # Use defrag_blue.obj.mtl to guide the creation of the final material
    defrag_blue_mtl_path = os.path.abspath(osp.join(defrag_path, "defrag_blue.obj.mtl"))
    if not osp.exists(defrag_blue_mtl_path):
        print(f"Error: {defrag_blue_mtl_path} not found. Cannot create final MTL.")
        return

    with open(defrag_blue_mtl_path, "r") as f:
        mtl_lines = f.readlines()

    material_to_id = {}
    current_mat = None
    for line in mtl_lines:
        line = line.strip()
        if line.startswith("newmtl"):
            parts = line.split()
            if len(parts) >= 2:
                current_mat = parts[1]
        elif line.startswith("map_Kd") and current_mat:
            m = re.search(r"defrag_blue_texture_(.+)\.png", line) # Match against blue texture name
            if m:
                mat_id = m.group(1)
                material_to_id[current_mat] = mat_id
            current_mat = None # Reset for next material

    final_mtl_path = osp.join(final_dir, "final_material.mtl")
    with open(final_mtl_path, "w") as fp:
        for mat_name, mat_id in material_to_id.items():
            # Check if we have an inpainted texture for this mat_id
            if mat_id in inpainted_texture_paths:
                final_texture_name = osp.basename(inpainted_texture_paths[mat_id])
                fp.write(f"newmtl {mat_name}\n")
                fp.write("Ka 1.000000 1.000000 1.000000\nKd 1.000000 1.000000 1.000000\nKs 0.000000 0.000000 0.000000\n")
                fp.write("Tr 1.000000\nillum 1\nNs 0.000000\n")
                fp.write(f"map_Kd {final_texture_name}\n\n")
            else:
                print(f"Warning: No inpainted texture found for material ID {mat_id} (material {mat_name}). Skipping in final MTL.")
    print(f"Final material file saved at: {final_mtl_path}")

    # Update the defrag_blue.obj to reference the new final_material.mtl
    final_obj_path = osp.join(final_dir, "final_model.obj")
    if not osp.exists(output_file_blue): # output_file_blue is defrag_blue.obj
         print(f"Error: {output_file_blue} (defrag_blue.obj) not found. Cannot create final OBJ.")
         return
         
    with open(output_file_blue, "r") as f_obj_in, open(final_obj_path, "w") as f_obj_out:
        for line in f_obj_in:
            if line.startswith("mtllib"):
                f_obj_out.write("mtllib final_material.mtl\n")
            else:
                f_obj_out.write(line)
    print(f"Final OBJ file saved at: {final_obj_path}")
    print("UV texture export and defragmentation complete.")


def main():
    # Use hardcoded configurations
    name = CONFIG_NAME
    gpu_id = CONFIG_GPU_ID
    process_uv = CONFIG_PROCESS_UV_EXPORT
    use_dress = CONFIG_USE_DRESS_LOGIC

    if name == "your_sample_name":
        print("Please update 'CONFIG_NAME' at the top of the script with your actual sample name.")
        return

    device, smplx_container, prefix, cache_path, \
    smplx_param_cpu, econ_obj_aligned = setup_environment_and_load_inputs(name, gpu_id)

    smpl_model, smpl_out_lst = prepare_smplx_model_and_poses(smplx_container, smplx_param_cpu)

    econ_da_final, smpl_da_trimesh = generate_da_space_meshes(
        econ_obj_aligned, smpl_out_lst, smpl_model, smplx_container, prefix, device, use_dress
    )

    econ_J_reg, econ_lbs_w, econ_pdirs, \
    econ_cano_verts, idx_econ_da_to_smpl_da_knn = build_smplx_compatible_econ(
        econ_da_final, smpl_da_trimesh, smpl_model, smpl_out_lst
    )

    econ_pose_trimesh = animate_econ_to_pose(
        econ_cano_verts, econ_da_final.faces, smpl_out_lst,
        idx_econ_da_to_smpl_da_knn, smplx_param_cpu, prefix
    )
    
    # Initialize UV Rasterizer (used by color mapping and UV export)
    # The image_size for Pytorch3dRasterizer should match the PackOptions.resolution for xatlas
    uv_rasterizer_main = Pytorch3dRasterizer(image_size=8192, device=device)

    # Generate UV coordinates for econ_pose_trimesh (always done)
    vt_tensor, ft_tensor = generate_uv_coordinates(econ_pose_trimesh, cache_path, device)

    final_rgb_np, tex_map1_np, tex_map2_np = perform_color_mapping(
        econ_pose_trimesh, vt_tensor, ft_tensor, device, uv_rasterizer_main, name, prefix, cache_path
    )

    normal_colors_np = perform_normal_based_coloring(econ_pose_trimesh, device, prefix)

    save_econ_data_pt(
        econ_cano_verts, econ_pdirs, econ_J_reg, smpl_model.parents,
        econ_lbs_w, final_rgb_np, normal_colors_np, econ_pose_trimesh.faces, cache_path
    )

    if process_uv:
        export_uv_textures_and_defrag(
            econ_pose_trimesh, vt_tensor, ft_tensor, final_rgb_np, cache_path,
            uv_rasterizer_main, name, tex_map1_np, tex_map2_np
        )
    
    print("Processing complete.")
    print(
     "If the dress/skirt is torn in the DA pose (check <prefix>/smpl_da.obj or intermediate outputs if saved),"
     " consider re-running with CONFIG_USE_DRESS_LOGIC = True."
    )

if __name__ == "__main__":
    main()