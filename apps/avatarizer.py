#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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
from termcolor import colored

from pytorch3d.transforms import axis_angle_to_matrix     # <<< NEW

import lib.smplx as smplx
from lib.common.local_affine import register
from lib.dataset.mesh_util import (
    SMPLX,
    export_obj,
    keep_largest,
    poisson,
)
from lib.smplx.lbs import general_lbs

# loading cfg file
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-uv", action="store_true")
parser.add_argument("-dress", action="store_true")
args = parser.parse_args() 

from typing import List, Optional, Tuple
from collections import Counter

from typing import List, Optional, Tuple
from collections import Counter
import pymeshlab
import tempfile
class MeshHoleUtils:
    """A utility class containing static methods for mesh hole manipulation."""

    @staticmethod
    def _is_mesh_valid(mesh: Optional[trimesh.Trimesh]) -> bool:
        """Checks if a trimesh object is valid for processing."""
        return not (mesh is None or mesh.is_empty or \
                    not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces') or \
                    mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3 or \
                    mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3)

    @staticmethod
    def _get_boundary_edges_manually(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        """Robustly calculates boundary edges from faces."""
        if not MeshHoleUtils._is_mesh_valid(mesh): return None
        try:
            edges = mesh.edges_sorted.copy()
            unique, counts = np.unique(edges, axis=0, return_counts=True)
            boundary_edges = unique[counts == 1]
            return boundary_edges
        except Exception:
            return np.array([])

    @staticmethod
    def _order_loop_vertices(loop_vidx_unique: np.ndarray, all_edges_loop: np.ndarray) -> Optional[np.ndarray]:
        """Orders a set of vertices into a continuous loop based on their edges."""
        if loop_vidx_unique is None or len(loop_vidx_unique) < 3 or all_edges_loop is None or len(all_edges_loop) < 2:
            return None
        
        adj = {v: [] for v in loop_vidx_unique}
        for u, v in all_edges_loop:
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)
        
        start_node = loop_vidx_unique[0]
        path = [start_node]
        visited_edges = set()
        curr = start_node
        
        for _ in range(len(all_edges_loop) + 2):
            found_next = False
            for neighbor in adj.get(curr, []):
                edge = tuple(sorted((curr, neighbor)))
                if edge not in visited_edges:
                    path.append(neighbor)
                    visited_edges.add(edge)
                    curr = neighbor
                    found_next = True
                    break
            if not found_next:
                break
        
        if len(path) > 1 and path[0] == path[-1]:
            path = path[:-1]
            
        if len(np.unique(path)) == len(loop_vidx_unique):
            return np.array(path, dtype=int)
        return None

    @staticmethod
    def get_all_boundary_loops(mesh: trimesh.Trimesh, min_loop_len: int = 3) -> List[np.ndarray]:
        """Finds all distinct, ordered boundary loops in a mesh."""
        all_loops = []
        if not MeshHoleUtils._is_mesh_valid(mesh): return all_loops
        
        boundary_edges = MeshHoleUtils._get_boundary_edges_manually(mesh)
        if boundary_edges is None or len(boundary_edges) < min_loop_len: return all_loops
        
        components = trimesh.graph.connected_components(boundary_edges, min_len=min_loop_len)
        
        for comp_unord in components:
            comp_set = set(comp_unord)
            edges_comp = [e for e in boundary_edges if e[0] in comp_set and e[1] in comp_set]
            ordered = MeshHoleUtils._order_loop_vertices(comp_unord, np.array(edges_comp))
            if ordered is not None:
                all_loops.append(ordered)
        return all_loops

    @staticmethod
    def smooth_all_boundary_loops(mesh: trimesh.Trimesh, iterations: int = 35, factor: float = 0.1) -> trimesh.Trimesh:
        """Applies Laplacian smoothing to all boundary loops of a mesh."""
        if not MeshHoleUtils._is_mesh_valid(mesh) or iterations <= 0:
            return mesh

        all_loops = MeshHoleUtils.get_all_boundary_loops(mesh)
        if not all_loops:
            print("Smoothing: No boundary loops found.")
            return mesh
        
        print(f"Smoothing {len(all_loops)} boundary loop(s) for {iterations} iterations...")
        vertices_copy = mesh.vertices.copy()
        
        for loop_vidx in all_loops:
            num_loop_verts = len(loop_vidx)
            if num_loop_verts < 3: continue
            
            for _ in range(iterations):
                new_positions = vertices_copy[loop_vidx].copy()
                for i in range(num_loop_verts):
                    prev_v_idx = loop_vidx[(i - 1 + num_loop_verts) % num_loop_verts]
                    next_v_idx = loop_vidx[(i + 1) % num_loop_verts]
                    laplacian_target = (vertices_copy[prev_v_idx] + vertices_copy[next_v_idx]) / 2.0
                    new_positions[i] += factor * (laplacian_target - new_positions[i])
                vertices_copy[loop_vidx] = new_positions

        smoothed_mesh = mesh.copy()
        smoothed_mesh.vertices = vertices_copy
        return smoothed_mesh

    @staticmethod
    def fill_holes_pymeshlab(mesh: trimesh.Trimesh, debug_save_prefix: str) -> Optional[trimesh.Trimesh]:
        """
        Repairs a mesh and fills all holes using a robust PyMeshLab pipeline.
        This version uses manual, backwards-compatible methods for diagnostics.
        """
        if not MeshHoleUtils._is_mesh_valid(mesh): return None
        
        print("Attempting to repair and fill holes using robust PyMeshLab pipeline...")
        ms = pymeshlab.MeshSet()
        
        # --- Create a visible debug directory ---
        debug_dir = osp.join(osp.dirname(debug_save_prefix), "debug_wrist_closure")
        os.makedirs(debug_dir, exist_ok=True)
        
        fd_in, temp_in_path = tempfile.mkstemp(suffix=".ply")
        os.close(fd_in)
        
        output_path_from_pml = osp.join(debug_dir, "pml_processed_mesh.obj")
        
        try:
            mesh.export(temp_in_path)
            ms.load_new_mesh(temp_in_path)
            
            # --- ROBUST REPAIR PIPELINE ---
            print("  - Step 1/3: Repairing mesh topology (removing duplicates, fixing non-manifolds)...")
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_duplicate_vertices()
            try:
                ms.meshing_repair_non_manifold_edges(method='Split Vertices')
            except Exception as e_nm:
                print(f"    - Note: Could not repair non-manifold edges: {e_nm}")
            
            print("  - Step 2/3: Closing holes on the repaired mesh...")
            ms.meshing_close_holes(maxholesize=500)
            
            print("  - Step 3/3: Performing final cleanup...")
            ms.meshing_remove_unreferenced_vertices()
            
            ms.save_current_mesh(output_path_from_pml)
            
            # --- COMPATIBLE DIAGNOSTIC CHECK ---
            filled_mesh = trimesh.load(output_path_from_pml, process=True)
            
            if not MeshHoleUtils._is_mesh_valid(filled_mesh):
                print(colored("Critical Failure: The mesh is empty or invalid after PyMeshLab processing.", "red"))
                print(f"  - The processed (but failed) mesh has been saved for inspection at: {output_path_from_pml}")
                return None

            # Manual watertightness check using functions known to work
            boundary_edges = MeshHoleUtils._get_boundary_edges_manually(filled_mesh)
            
            if boundary_edges is not None and len(boundary_edges) == 0:
                print(colored("PyMeshLab repair-and-fill successful. Mesh is watertight.", "green"))
                return filled_mesh
            else:
                num_boundary_edges = len(boundary_edges) if boundary_edges is not None else "an unknown number of"
                print(colored(f"Critical Failure: PyMeshLab processing did not produce a watertight mesh.", "red"))
                print(f"  - DIAGNOSIS: The final mesh still has {num_boundary_edges} boundary edges (open holes).")
                print(f"  - The processed (but failed) mesh has been saved for inspection at: {output_path_from_pml}")
                return None
                
        except Exception as e:
            print(colored(f"An unhandled error occurred during PyMeshLab processing: {e}", "red"))
            return None
        finally:
            if os.path.exists(temp_in_path): os.remove(temp_in_path)
            del ms

smplx_container = SMPLX()
device = torch.device(f"cuda:{args.gpu}")

# loading SMPL-X and econ objs inferred with ECON
# prefix = f"./results_Fulden/econ/obj/{args.name}"
prefix = f"./results/Fulden/face_thresh_0.01/econ/obj/{args.name}"

smpl_path = f"{prefix}_smpl_00.npy"
smplx_param = np.load(smpl_path, allow_pickle=True).item()

# export econ obj with pre-computed normals
econ_path = f"{prefix}_0_full_soups.ply"
econ_obj = trimesh.load(econ_path)
assert econ_obj.vertex_normals.shape[1] == 3
os.makedirs(f"{prefix}/", exist_ok=True)

# align econ with SMPL-X
econ_obj.vertices *= np.array([1.0, -1.0, -1.0])
econ_obj.vertices /= smplx_param["scale"].cpu().numpy()
econ_obj.vertices -= smplx_param["transl"].cpu().numpy()

for key in smplx_param.keys():
    smplx_param[key] = smplx_param[key].cpu().view(1, -1)

smpl_model = smplx.create(
    smplx_container.model_dir,
    model_type="smplx",
    gender="neutral",
    age="adult",
    use_face_contour=True,
    use_pca=False,
    create_expression=True,
    create_betas=False,
    create_global_orient=True,
    create_body_pose=True,
    create_jaw_pose=True,
    create_left_hand_pose=False,
    create_right_hand_pose=False,
    create_transl=False,
    num_betas=smplx_param["betas"].shape[1],
    num_expression_coeffs=smplx_param["expression"].shape[1],
    ext="pkl",
)

smpl_out_lst = []

# obtain the pose params of T-pose, DA-pose, and the original pose
for pose_type in ["a-pose", "t-pose", "da-pose", "pose"]:
    smpl_out_lst.append(
        smpl_model(
            body_pose=smplx_param["body_pose"],
            global_orient=smplx_param["global_orient"],
            betas=smplx_param["betas"],
            expression=smplx_param["expression"],
            jaw_pose=smplx_param["jaw_pose"],
            left_hand_pose=smplx_param["left_hand_pose"],
            right_hand_pose=smplx_param["right_hand_pose"],
            return_verts=True,
            return_full_pose=True,
            return_joint_transformation=True,
            return_vertex_transformation=True,
            pose_type=pose_type,
        )
    )

# -------------------------- align econ and SMPL-X in DA-pose space ------------------------- #
# 1. find the vertex-correspondence between SMPL-X and econ
# 2. ECON + SMPL-X: posed space --> T-pose space --> DA-pose space
# 3. ECON (w/o hands & over-streched faces) + SMPL-X (w/ hands & registered inpainting parts)
# ------------------------------------------------------------------------------------------- #

smpl_verts = smpl_out_lst[3].vertices.detach()[0]
smpl_tree = cKDTree(smpl_verts.cpu().numpy())
dist, idx = smpl_tree.query(econ_obj.vertices, k=3)

if not osp.exists(f"{prefix}/econ_da.obj") or not osp.exists(f"{prefix}/smpl_da.obj"):
    # t-pose for ECON
    econ_verts = torch.tensor(econ_obj.vertices).float()
    rot_mat_t = smpl_out_lst[3].vertex_transformation.detach()[0][idx[:, 0]]
    homo_coord = torch.ones_like(econ_verts)[..., :1]
    econ_cano_verts = torch.inverse(rot_mat_t) @ torch.cat([econ_verts, homo_coord],
                                                        dim=1).unsqueeze(-1)
    econ_cano_verts = econ_cano_verts[:, :3, 0].cpu()
    econ_cano = trimesh.Trimesh(econ_cano_verts, econ_obj.faces)

    # da-pose for ECON
    rot_mat_da = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
    econ_da_verts = rot_mat_da @ torch.cat([econ_cano_verts, homo_coord], dim=1).unsqueeze(-1)
    econ_da = trimesh.Trimesh(econ_da_verts[:, :3, 0].cpu(), econ_obj.faces)

    # da-pose for SMPL-X (needed for reference, but we won't merge its geometry)
    smpl_da = trimesh.Trimesh(
        smpl_out_lst[2].vertices.detach()[0],
        smpl_model.faces,
        maintain_orders=True,
        process=False,
    )
    smpl_da.export(f"{prefix}/smpl_da.obj")

    # --- START: SURGICAL HAND REMOVAL AND WRIST CLOSING (with DEBUGGING) ---
    
    # 1. Identify ONLY hand vertices based on SMPL-X template correspondence.
    print("Step 1: Identifying hand vertices for surgical removal...")
    hand_vids_on_smplx_template = np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
    
    # 2. Create a copy of the mesh and remove faces connected to hand vertices.
    econ_da_body = econ_da.copy()
    face_mask_to_keep = ~hand_vids_on_smplx_template[econ_da_body.faces].any(axis=1)
    econ_da_body.update_faces(face_mask_to_keep)
    econ_da_body.remove_unreferenced_vertices()
    
    # 3. Clean up any small, disconnected parts that might result from the removal.
    econ_da_body = keep_largest(econ_da_body)
    print(f"Step 2: Removed hands. Mesh now has {len(econ_da_body.vertices)} vertices.")
    econ_da_body.export(f"{prefix}/debug_01_after_hand_removal.obj") # DEBUG SAVE
    print(colored("Saved intermediate mesh to debug_01_after_hand_removal.obj", "cyan"))

    # 4. Smooth the raw wrist holes to create a better shape for filling.
    print("Step 3: Smoothing wrist hole boundaries...")
    smoothed_body = MeshHoleUtils.smooth_all_boundary_loops(econ_da_body, iterations=30, factor=0.1)
    smoothed_body.export(f"{prefix}/debug_02_after_smoothing.obj") # DEBUG SAVE
    print(colored("Saved intermediate mesh to debug_02_after_smoothing.obj", "cyan"))

    # 5. Fill the smoothed holes with the robust PyMeshLab algorithm.
    print("Step 4: Filling smoothed wrist holes with PyMeshLab...")
    closed_body = MeshHoleUtils.fill_holes_pymeshlab(smoothed_body, debug_save_prefix=prefix)    
    if closed_body:
        econ_da = closed_body # This is now our final geometry.
        econ_da.export(f"{prefix}/econ_da.obj") # Save the final, closed DA-pose mesh.
        print(colored("Final DA-pose mesh with closed wrists saved to econ_da.obj", "green"))
    else:
        print(colored("Critical Failure: Could not close wrist holes. The original econ_da will be used as a fallback.", "red"))
        econ_da.export(f"{prefix}/econ_da.obj") # Save the original as a fallback.

    # --- END: SURGICAL HAND REMOVAL AND WRIST CLOSING ---
else:
    econ_da = trimesh.load(f"{prefix}/econ_da.obj")
    smpl_da = trimesh.load(f"{prefix}/smpl_da.obj", maintain_orders=True, process=False)

# ---------------------- SMPL-X compatible ECON ---------------------- #
# 1. Find the new vertex-correspondence between NEW ECON and SMPL-X
# 2. Build the new J_regressor, lbs_weights, posedirs
# 3. canonicalize the NEW ECON
# ------------------------------------------------------------------- #

print("Start building the SMPL-X compatible ECON model...")

smpl_tree = cKDTree(smpl_da.vertices)
dist, idx = smpl_tree.query(econ_da.vertices, k=3)
knn_weights = np.exp(-(dist**2))
knn_weights /= knn_weights.sum(axis=1, keepdims=True)

econ_J_regressor = (smpl_model.J_regressor[:, idx] * knn_weights[None]).sum(dim=-1)
econ_lbs_weights = (smpl_model.lbs_weights.T[:, idx] * knn_weights[None]).sum(dim=-1).T

num_posedirs = smpl_model.posedirs.shape[0]
econ_posedirs = ((
    smpl_model.posedirs.view(num_posedirs, -1, 3)[:, idx, :] * knn_weights[None, ..., None]
).sum(dim=-2).view(num_posedirs, -1).float())

econ_J_regressor /= econ_J_regressor.sum(dim=1, keepdims=True).clip(min=1e-10)
econ_lbs_weights /= econ_lbs_weights.sum(dim=1, keepdims=True)

rot_mat_da = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
econ_da_verts = torch.tensor(econ_da.vertices).float()
econ_cano_verts = torch.inverse(rot_mat_da) @ torch.cat([
    econ_da_verts, torch.ones_like(econ_da_verts)[..., :1]
],
                                                        dim=1).unsqueeze(-1)
econ_cano_verts = econ_cano_verts[:, :3, 0].double()

# ----------------------------------------------------
# use original pose to animate ECON reconstruction
# ----------------------------------------------------

rot_mat_pose = smpl_out_lst[3].vertex_transformation.detach()[0][idx[:, 0]]
posed_econ_verts = rot_mat_pose @ torch.cat([
    econ_cano_verts.float(),
    torch.ones_like(econ_cano_verts.float())[..., :1]
],
                                            dim=1).unsqueeze(-1)
posed_econ_verts = posed_econ_verts[:, :3, 0].double()

aligned_econ_verts = posed_econ_verts.detach().cpu().numpy()
aligned_econ_verts += smplx_param["transl"].cpu().numpy()
aligned_econ_verts *= smplx_param["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
econ_pose = trimesh.Trimesh(aligned_econ_verts, econ_da.faces)
assert econ_pose.vertex_normals.shape[1] == 3
econ_pose.export(f"{prefix}/econ_pose.ply")

# ----------------------------------------------------------------
# ► MOD 1  Build T-pose mesh with the same root orientation as the posed mesh
# ----------------------------------------------------------------
root_R = axis_angle_to_matrix(smplx_param["global_orient"]).cpu().numpy()[0]   # 3×3
aligned_tpose_verts = (root_R @ econ_cano_verts.cpu().numpy().T).T
aligned_tpose_verts += smplx_param["transl"].cpu().numpy()   # then translate
aligned_tpose_verts *= smplx_param["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
econ_tpose = trimesh.Trimesh(aligned_tpose_verts, econ_da.faces)
assert econ_tpose.vertex_normals.shape[1] == 3
econ_tpose.export(f"{prefix}/econ_tpose.ply")


cache_path = f"{prefix.replace('obj','cache')}"
os.makedirs(cache_path, exist_ok=True)

# -----------------------------------------------------------------
# create UV texture (.obj .mtl .png) from posed ECON reconstruction
# -----------------------------------------------------------------
print("Start Color mapping...")

from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import trimesh
import os.path as osp
from lib.common.render import query_color, query_normal_color
from lib.common.render_utils import Pytorch3dRasterizer
from lib.dataset.mesh_util import export_obj

##########################################
# First Pass: Using _cloth_front and _cloth_back_red images
##########################################

cloth_front_red_path = f"./results/Fulden/face_thresh_0.01/econ/png/{args.name}_cloth_front_red.png"
cloth_back_red_path  = f"./results/Fulden/face_thresh_0.01/econ/png/{args.name}_cloth_back_red.png"

tensor_front_1 = transforms.ToTensor()(Image.open(cloth_front_red_path))[:, :, :512]
tensor_back_1  = transforms.ToTensor()(Image.open(cloth_back_red_path))[:, :, :512]
H, W = tensor_front_1.shape[1], tensor_front_1.shape[2]

front_image_1 = ((tensor_front_1 - 0.5) * 2.0).unsqueeze(0).to(device)
back_image_1  = ((tensor_back_1  - 0.5) * 2.0).unsqueeze(0).to(device)

verts_tensor = torch.tensor(econ_pose.vertices).float().to(device)
faces_tensor = torch.tensor(econ_pose.faces).long().to(device)

final_rgb_pass1 = query_color(
    verts_tensor,
    faces_tensor,
    front_image_1,
    back_image_1,
    device=device,
).numpy()

uv_rasterizer = Pytorch3dRasterizer(image_size=8192, device=device)
if not ('vt' in globals() and 'ft' in globals() and 'vmapping' in globals()):
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(econ_tpose.vertices, econ_tpose.faces)   # <<< MOD A
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    chart_options.max_iterations = 4
    pack_options.resolution = 8192
    pack_options.bruteForce = True
    atlas.generate(chart_options=chart_options)
    vmapping, ft_np, vt_np = atlas[0]
    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
    vt_cache = osp.join(cache_path, "vt_tpose.pt")         # <<< MOD B
    ft_cache = osp.join(cache_path, "ft_tpose.pt")         # <<< MOD B
    torch.save(vt.cpu(), vt_cache)
    torch.save(ft.cpu(), ft_cache)

v_np = econ_tpose.vertices       # <<< MOD C
f_np = econ_tpose.faces          # <<< MOD C
texture_map1 = uv_rasterizer.get_texture(
    torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
    ft,
    torch.tensor(v_np).unsqueeze(0).float(),
    torch.tensor(f_np).unsqueeze(0).long(),
    torch.tensor(final_rgb_pass1).unsqueeze(0).float() / 255.0,
)
texture_map1_8bit = (texture_map1 * 255.0).astype(np.uint8)
os.makedirs(f"{cache_path}/red", exist_ok=True)
Image.fromarray(texture_map1_8bit).save(f"{cache_path}/red/texture_red.png")
print("First-pass texture map saved as texture_red.png.")

##########################################
# Second Pass: Using _cloth_front_red_blue and _cloth_back_blue images
##########################################

cloth_front_path_blue = f"./results/Fulden/face_thresh_0.01/econ/png/{args.name}_cloth_front_blue.png"
cloth_back_path_blue  = f"./results/Fulden/face_thresh_0.01/econ/png/{args.name}_cloth_back_blue.png"

tensor_front_2 = transforms.ToTensor()(Image.open(cloth_front_path_blue))[:, :, :512]
tensor_back_2  = transforms.ToTensor()(Image.open(cloth_back_path_blue))[:, :, :512]

front_image_2 = ((tensor_front_2 - 0.5) * 2.0).unsqueeze(0).to(device)
back_image_2  = ((tensor_back_2  - 0.5) * 2.0).unsqueeze(0).to(device)

final_rgb_pass2 = query_color(
    verts_tensor,
    faces_tensor,
    front_image_2,
    back_image_2,
    device=device,
).numpy()

texture_map2 = uv_rasterizer.get_texture(
    torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
    ft,
    torch.tensor(v_np).unsqueeze(0).float(),
    torch.tensor(f_np).unsqueeze(0).long(),
    torch.tensor(final_rgb_pass2).unsqueeze(0).float() / 255.0,
)
texture_map2_8bit = (texture_map2 * 255.0).astype(np.uint8)
os.makedirs(f"{cache_path}/blue", exist_ok=True)
Image.fromarray(texture_map2_8bit).save(f"{cache_path}/blue/texture_blue.png")
print("Second-pass texture map saved as texture_blue.png.")

##########################################
# Compare Texture Maps and Create Difference Mask
##########################################

# diff_map = np.abs(texture_map1_8bit.astype(np.float32) - texture_map2_8bit.astype(np.float32)) / 255.0
# # threshold = 0.1
# mask_diff = np.any(diff_map > 0.01, axis=2)
# Image.fromarray((mask_diff.astype(np.uint8) * 255)).save(f"{cache_path}/diff_mask.png")
# print("Difference mask saved as diff_mask.png.")

##########################################
# Assign Final Vertex Colors and Export Mesh
##########################################

final_rgb = final_rgb_pass2
econ_pose.visual.vertex_colors = final_rgb
econ_pose.export(f"{prefix}/econ_icp_rgb.ply")

# ----------------------------------------------------------------
# >>> MOD 2 : apply same colors to the new T-pose mesh
# ----------------------------------------------------------------
econ_tpose.visual.vertex_colors = final_rgb            # identical vertex order
econ_tpose.export(f"{prefix}/econ_tpose_rgb.ply")

##########################################
# Normal-based Color Mapping (Choice 2, unchanged)
##########################################

if not osp.exists(f"{prefix}/econ_icp_normal.ply"):
    file_normal = query_normal_color(
        verts_tensor,
        faces_tensor,
        device=device,
    ).numpy()
    econ_pose.visual.vertex_colors = file_normal
    econ_pose.export(f"{prefix}/econ_icp_normal.ply")
else:
    mesh = trimesh.load(f"{prefix}/econ_icp_normal.ply")
    file_normal = mesh.visual.vertex_colors[:, :3]

##########################################
# Save econ Data for Further Processing
##########################################

econ_dict = {
    "v_template": econ_cano_verts.unsqueeze(0),
    "posedirs": econ_posedirs,
    "J_regressor": econ_J_regressor,
    "parents": smpl_model.parents,
    "lbs_weights": econ_lbs_weights,
    "final_rgb": final_rgb,
    "final_normal": file_normal,
    "faces": econ_pose.faces,
}

torch.save(econ_dict, f"{cache_path}/econ.pt")

# print(
#     "If the dress/skirt is torn in `<file_name>/econ_da.obj`, please delete ./file_name and regenerate them with `-dress`\n"
#     "python -m apps.avatarizer -n <file_name> -dress"
# )

##########################################
# UV Texture Generation (if enabled)
##########################################

args.uv = True
args.dress = False
if args.uv:
    print("Start UV texture generation...")

    # ----------------------------------------------------------------
    # >>> MOD 3 : unwrap the T-pose instead of the posed mesh
    # ----------------------------------------------------------------
    v_np = econ_tpose.vertices
    f_np = econ_tpose.faces

    vt_cache = osp.join(cache_path, "vt_tpose.pt")   # <<< MOD 2
    ft_cache = osp.join(cache_path, "ft_tpose.pt")   # <<< MOD 2


    if osp.exists(vt_cache) and osp.exists(ft_cache):
        vt = torch.load(vt_cache).to(device)
        ft = torch.load(ft_cache).to(device)
    else:
        import xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        pack_options = xatlas.PackOptions()
        chart_options.max_iterations = 4
        pack_options.resolution = 8192
        pack_options.bruteForce = True
        atlas.generate(chart_options=chart_options)
        vmapping, ft_np, vt_np = atlas[0]
        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
        torch.save(vt.cpu(), vt_cache)
        torch.save(ft.cpu(), ft_cache)

    uv_rasterizer = Pytorch3dRasterizer(image_size=8192, device=device)
    texture_npy = uv_rasterizer.get_texture(
        torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
        ft,
        torch.tensor(v_np).unsqueeze(0).float(),
        torch.tensor(f_np).unsqueeze(0).long(),
        torch.tensor(final_rgb).unsqueeze(0).float() / 255.0,
    )
    print("UV texture generation complete.")

    # Export meshes and temporary MTL files.
    red_dir = osp.join(cache_path, "red")
    blue_dir = osp.join(cache_path, "blue")
    os.makedirs(red_dir, exist_ok=True)
    os.makedirs(blue_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # >>> MOD 4 : write OBJs using T-pose vertices so the rest pipeline
    #             (defrag, final OBJ/MTL) works on the T-pose geometry
    # ----------------------------------------------------------------
    export_obj(np.array(econ_tpose.vertices), f_np, vt, ft, osp.join(red_dir, "mesh_red.obj"))
    with open(osp.join(red_dir, "material.mtl"), "w") as fp:
        fp.write("newmtl mat0 \n")
        fp.write("Ka 1.000000 1.000000 1.000000 \n")
        fp.write("Kd 1.000000 1.000000 1.000000 \n")
        fp.write("Ks 0.000000 0.000000 0.000000 \n")
        fp.write("Tr 1.000000 \n")
        fp.write("illum 1 \n")
        fp.write("Ns 0.000000 \n")
        fp.write("map_Kd texture_red.png \n")

    export_obj(np.array(econ_tpose.vertices), f_np, vt, ft, osp.join(blue_dir, "mesh_blue.obj"))
    with open(osp.join(blue_dir, "material.mtl"), "w") as fp:
        fp.write("newmtl mat0 \n")
        fp.write("Ka 1.000000 1.000000 1.000000 \n")
        fp.write("Kd 1.000000 1.000000 1.000000 \n")
        fp.write("Ks 0.000000 0.000000 0.000000 \n")
        fp.write("Tr 1.000000 \n")
        fp.write("illum 1 \n")
        fp.write("Ns 0.000000 \n")
        fp.write("map_Kd texture_blue.png \n")
    
    # Create a folder for intermediate defrag outputs.
    defrag_path = osp.join(cache_path, "defrag_assets")
    os.makedirs(defrag_path, exist_ok=True)
    
    # Create a separate folder for final outputs.
    final_dir = osp.join(cache_path, "final_files")
    os.makedirs(final_dir, exist_ok=True)

    # Run texture-defrag on both meshes.
    texture_defrag_exe = os.path.abspath("./texture-defrag/build/texture-defrag")
    mesh_file_red = os.path.abspath(osp.join(red_dir, "mesh_red.obj"))
    output_file_red = os.path.abspath(osp.join(defrag_path, "defrag_red.obj"))
    cmd_red = ["xvfb-run", "-a", texture_defrag_exe, mesh_file_red, "-l", "0", "-o", output_file_red]
    print("Running texture-defrag for red mesh:", " ".join(cmd_red))
    subprocess.run(cmd_red, check=True)

    mesh_file_blue = os.path.abspath(osp.join(blue_dir, "mesh_blue.obj"))
    output_file_blue = os.path.abspath(osp.join(defrag_path, "defrag_blue.obj"))
    cmd_blue = ["xvfb-run", "-a", texture_defrag_exe, mesh_file_blue, "-l", "0", "-o", output_file_blue]
    print("Running texture-defrag for blue mesh:", " ".join(cmd_blue))
    subprocess.run(cmd_blue, check=True)

    # Process texture pairs in the defrag_assets folder.
    red_pattern = re.compile(r"defrag_red_texture_(.+)\.png")
    blue_pattern = re.compile(r"defrag_blue_texture_(.+)\.png")
    red_images = {}
    blue_images = {}

    for file in os.listdir(defrag_path):
        if "defrag" in file and file.endswith(".png"):
            match_red = red_pattern.search(file)
            if match_red:
                img_id = match_red.group(1)
                red_images[img_id] = osp.join(defrag_path, file)
            match_blue = blue_pattern.search(file)
            if match_blue:
                img_id = match_blue.group(1)
                blue_images[img_id] = osp.join(defrag_path, file)

    # For each matching texture pair, compute and save a difference mask,
    # then run the inpainting process and save a final texture per ID in final_dir.
    for img_id in red_images:
        if img_id in blue_images:
            red_path = red_images[img_id]
            blue_path = blue_images[img_id]
            print(f"Processing textures for ID: {img_id}")
            
            red_img = np.array(Image.open(red_path).convert("RGB"), dtype=np.float32)
            blue_img = np.array(Image.open(blue_path).convert("RGB"), dtype=np.float32)
            
            diff = np.abs(red_img - blue_img)
            diff_sum = np.sum(diff, axis=2)
            diff_mask = (diff_sum > 0.01).astype(np.uint8) * 255
            missing_mask = ((red_img.sum(axis=2) == 0) | (blue_img.sum(axis=2) == 0)).astype(np.uint8) * 255
            final_mask = np.maximum(diff_mask, missing_mask)
            
            output_mask_path = osp.join(defrag_path, f"difference_mask_{img_id}.png")
            Image.fromarray(final_mask).save(output_mask_path)
            print(f"Saved difference mask for ID {img_id} at {output_mask_path}")
            
            # Inpainting process:
            texture_8bit = red_img.astype(np.uint8)
            small_kernel = np.ones((10,10), np.uint8)
            eroded_mask = cv2.erode(final_mask, small_kernel, iterations=3)
            dilation_kernel = np.ones((45, 45), np.uint8)
            dilated_mask = cv2.dilate(eroded_mask, dilation_kernel, iterations=2)
            
            diffmask_dilated_path = osp.join(defrag_path, f"diffmask_dilated_{img_id}.png")
            cv2.imwrite(diffmask_dilated_path, dilated_mask)
            print("Saved dilated difference mask at:", diffmask_dilated_path)
            
            if dilated_mask.shape != texture_8bit.shape[:2]:
                dilated_mask = cv2.resize(dilated_mask, (texture_8bit.shape[1], texture_8bit.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            # Use the dilated mask for inpainting.
            inpainted_texture = cv2.inpaint(texture_8bit, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            inpainted_texture = cv2.cvtColor(inpainted_texture, cv2.COLOR_RGB2BGR)
            final_texture_filename = f"texture_map_inpainted_{img_id}.png"
            final_texture_path = osp.join(final_dir, final_texture_filename)
            cv2.imwrite(final_texture_path, inpainted_texture)
            print("Saved inpainted texture at:", final_texture_path)

    texture_pattern = osp.join(final_dir, "texture_map_inpainted*.png")
    texture_paths = sorted(glob.glob(texture_pattern))
    print("Found final texture files:", texture_paths)

    # === 2. Parse defrag_blue.obj.mtl to follow texture IDs.
    # Read the defrag_blue MTL file from the defrag_assets folder.
    defrag_blue_mtl_path = os.path.abspath(osp.join(defrag_path, "defrag_blue.obj.mtl"))
    with open(defrag_blue_mtl_path, "r") as f:
        mtl_lines = f.readlines()

    # Build a mapping from material name to its associated texture ID,
    # by parsing the "map_Kd" line which is assumed to be in the format:
    # map_Kd defrag_blue_texture_<id>.png
    material_to_id = {}
    current_mat = None
    for line in mtl_lines:
        line = line.strip()
        if line.startswith("newmtl"):
            parts = line.split()
            if len(parts) >= 2:
                current_mat = parts[1]
        elif line.startswith("map_Kd") and current_mat:
            m = re.search(r"defrag_blue_texture_(.+)\.png", line)
            if m:
                mat_id = m.group(1)
                material_to_id[current_mat] = mat_id
            current_mat = None

    print("Extracted material-to-ID mapping from defrag_blue MTL:", material_to_id)

    # === 3. Create the final MTL file using the IDs from defrag_blue.obj.mtl.
    final_mtl_path = osp.join(final_dir, "final_material.mtl")
    with open(final_mtl_path, "w") as fp:
        for mat_name, mat_id in material_to_id.items():
            # Use the corresponding final inpainted texture based on the ID.
            final_texture = f"texture_map_inpainted_{mat_id}.png"
            fp.write(f"newmtl {mat_name}\n")
            fp.write("Ka 1.000000 1.000000 1.000000\n")
            fp.write("Kd 1.000000 1.000000 1.000000\n")
            fp.write("Ks 0.000000 0.000000 0.000000\n")
            fp.write("Tr 1.000000\n")
            fp.write("illum 1\n")
            fp.write("Ns 0.000000\n")
            fp.write(f"map_Kd {final_texture}\n\n")
    print("Final material file saved at:", final_mtl_path)

    # === 4. Update the OBJ file to reference the new MTL file and save it in final_dir.
    final_obj_path = osp.join(final_dir, "final_model.obj")
    new_obj_lines = []
    mtllib_updated = False
    # Use the blue defrag OBJ as the base.
    with open(output_file_blue, "r") as f:
        obj_lines = f.readlines()

    for line in obj_lines:
        if line.startswith("mtllib"):
            new_obj_lines.append("mtllib final_material.mtl")
            mtllib_updated = True
        else:
            new_obj_lines.append(line.rstrip())
    if not mtllib_updated:
        new_obj_lines.insert(0, "mtllib final_material.mtl")

    final_obj_contents = "\n".join(new_obj_lines)
    with open(final_obj_path, "w") as f:
        f.write(final_obj_contents)
    print("Final OBJ file saved at:", final_obj_path)
