#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard library imports
import argparse
import glob
import os
import os.path as osp
import re
import subprocess
import tempfile
from typing import List, Optional, Tuple

# Third-party imports
import cv2
import numpy as np
import pymeshlab
import torch
import trimesh
from PIL import Image
from pytorch3d.transforms import axis_angle_to_matrix
from scipy.spatial import cKDTree
from termcolor import colored
from torchvision import transforms
from trimesh.transformations import rotation_matrix

# Local application/library specific imports
import lib.smplx as smplx
from lib.common.render import query_color
from lib.common.render_utils import Pytorch3dRasterizer
from lib.dataset.mesh_util import SMPLX, export_obj, keep_largest

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-uv", action="store_true")
parser.add_argument(
    "-ct", "--clean_texture", action="store_true",
    help="Apply final artifact cleaning to texture maps."
)
args = parser.parse_args()


def get_facial_landmarks_on_tpose(
    final_tpose_mesh: trimesh.Trimesh,
    smpl_model: smplx.SMPLX,
    smpl_params: dict,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify the 68 SMPL-X facial landmarks on ``final_tpose_mesh``.
    """
    print("Extracting facial landmarks from the final T-pose mesh...")

    # 1. Get canonical SMPL-X face landmarks (T-pose, world origin)
    with torch.no_grad():
        betas = smpl_params["betas"].to(device)
        expression = smpl_params["expression"].to(device)
        smpl_out = smpl_model(betas=betas, expression=expression, return_verts=True)
        
        verts = smpl_out.vertices.squeeze(0).to(device)
        faces = smpl_model.faces_tensor.to(device).long()
        lmk_faces_idx = smpl_model.lmk_faces_idx.to(device).long()
        lmk_bary_coords = smpl_model.lmk_bary_coords.to(device)

        face_vertices = verts[faces[lmk_faces_idx]]
        canonical_landmarks = (lmk_bary_coords.unsqueeze(-1) * face_vertices).sum(dim=1)

    # 2. Replicate ECON's transformation pipeline to align landmarks
    root_R = axis_angle_to_matrix(smpl_params["global_orient"].to(device)).squeeze(0)
    landmarks_posed = (root_R @ canonical_landmarks.T).T
    landmarks_translated = landmarks_posed + smpl_params["transl"].to(device)
    scale = smpl_params["scale"].to(device)
    coord_flip = torch.tensor([1.0, -1.0, -1.0], device=device)
    final_landmarks = landmarks_translated * scale * coord_flip
    final_landmarks_np = final_landmarks.cpu().numpy()

    # 3. Find nearest vertices on the target T-pose mesh
    kdtree = cKDTree(np.asarray(final_tpose_mesh.vertices))
    _, landmark_indices = kdtree.query(final_landmarks_np)
    landmark_coords = final_tpose_mesh.vertices[landmark_indices]

    print(f"Successfully extracted {len(landmark_indices)} landmark indices.")
    return landmark_indices, landmark_coords


def load_image(path):
    """Load an image in color mode and return as numpy array."""
    return cv2.imread(path)


def detect_pixels(image):
    """Apply a 3x3 edge-detection kernel to highlight and average edge pixels."""
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result = image.copy()
    kernels = [
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]),
        np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]]),
    ]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = []
    for kernel in kernels:
        convolved = cv2.filter2D(gray, -1, kernel)
        _, mask = cv2.threshold(np.abs(convolved), 30, 255, cv2.THRESH_BINARY)
        masks.append(mask)
        positive_mask = kernel > 0
        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                if mask[y, x] > 0:
                    patch = padded[y : y + 3, x : x + 3]
                    selected_pixels = patch[positive_mask]
                    if selected_pixels.size > 0:
                        avg_color = selected_pixels.mean(axis=0).astype(np.uint8)
                        result[y, x] = avg_color
    mask = np.maximum.reduce(masks).astype(np.uint8)
    return mask, result


def save_image(path, image):
    """Save image to the given path."""
    cv2.imwrite(path, image)


class MeshHoleUtils:
    """A utility class containing static methods for mesh hole manipulation."""

    @staticmethod
    def _is_mesh_valid(mesh: Optional[trimesh.Trimesh]) -> bool:
        return not (
            mesh is None or mesh.is_empty or not hasattr(mesh, "vertices") or
            not hasattr(mesh, "faces") or mesh.vertices.ndim != 2 or
            mesh.vertices.shape[1] != 3 or mesh.faces.ndim != 2 or
            mesh.faces.shape[1] != 3
        )

    @staticmethod
    def _get_boundary_edges_manually(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        if not MeshHoleUtils._is_mesh_valid(mesh): return None
        try:
            edges = mesh.edges_sorted.copy()
            unique, counts = np.unique(edges, axis=0, return_counts=True)
            return unique[counts == 1]
        except Exception:
            return np.array([])

    @staticmethod
    def _order_loop_vertices(
        loop_vidx_unique: np.ndarray, all_edges_loop: np.ndarray
    ) -> Optional[np.ndarray]:
        if (loop_vidx_unique is None or len(loop_vidx_unique) < 3 or
            all_edges_loop is None or len(all_edges_loop) < 2):
            return None
        adj = {v: [] for v in loop_vidx_unique}
        for u, v in all_edges_loop:
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)
        start_node, path, visited_edges, curr = loop_vidx_unique[0], [loop_vidx_unique[0]], set(), loop_vidx_unique[0]
        for _ in range(len(all_edges_loop) + 2):
            found_next = False
            for neighbor in adj.get(curr, []):
                edge = tuple(sorted((curr, neighbor)))
                if edge not in visited_edges:
                    path.append(neighbor)
                    visited_edges.add(edge)
                    curr, found_next = neighbor, True
                    break
            if not found_next: break
        if len(path) > 1 and path[0] == path[-1]: path = path[:-1]
        if len(np.unique(path)) == len(loop_vidx_unique):
            return np.array(path, dtype=int)
        return None

    @staticmethod
    def get_all_boundary_loops(
        mesh: trimesh.Trimesh, min_loop_len: int = 3
    ) -> List[np.ndarray]:
        all_loops = []
        if not MeshHoleUtils._is_mesh_valid(mesh): return all_loops
        boundary_edges = MeshHoleUtils._get_boundary_edges_manually(mesh)
        if boundary_edges is None or len(boundary_edges) < min_loop_len: return all_loops
        components = trimesh.graph.connected_components(boundary_edges, min_len=min_loop_len)
        for comp_unord in components:
            comp_set = set(comp_unord)
            edges_comp = [e for e in boundary_edges if e[0] in comp_set and e[1] in comp_set]
            ordered = MeshHoleUtils._order_loop_vertices(comp_unord, np.array(edges_comp))
            if ordered is not None: all_loops.append(ordered)
        return all_loops

    @staticmethod
    def smooth_all_boundary_loops(
        mesh: trimesh.Trimesh, iterations: int = 35, factor: float = 0.1
    ) -> trimesh.Trimesh:
        if not MeshHoleUtils._is_mesh_valid(mesh) or iterations <= 0: return mesh
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
    def fill_holes_pymeshlab(mesh: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
        if not MeshHoleUtils._is_mesh_valid(mesh): return None
        print("Attempting to repair and fill holes using PyMeshLab...")
        ms = pymeshlab.MeshSet()
        fd_in, temp_in_path = tempfile.mkstemp(suffix=".ply")
        os.close(fd_in)
        try:
            mesh.export(temp_in_path)
            ms.load_new_mesh(temp_in_path)
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_duplicate_vertices()
            try:
                ms.meshing_repair_non_manifold_edges(method='Split Vertices')
            except Exception:
                pass  # Non-critical if it fails
            ms.meshing_close_holes(maxholesize=500)
            ms.meshing_remove_unreferenced_vertices()
            with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as temp_out:
                output_path_from_pml = temp_out.name
            ms.save_current_mesh(output_path_from_pml)
            filled_mesh = trimesh.load(output_path_from_pml, process=True)
            os.remove(output_path_from_pml)
            boundary_edges = MeshHoleUtils._get_boundary_edges_manually(filled_mesh)
            if boundary_edges is not None and len(boundary_edges) == 0:
                print(colored("PyMeshLab repair-and-fill successful. Mesh is watertight.", "green"))
                return filled_mesh
            else:
                print(colored("Critical Failure: PyMeshLab did not produce a watertight mesh.", "red"))
                return None
        except Exception as e:
            print(colored(f"An unhandled error occurred during PyMeshLab processing: {e}", "red"))
            return None
        finally:
            if os.path.exists(temp_in_path):
                os.remove(temp_in_path)
            del ms


# --- Main Execution ---
smplx_container = SMPLX()
device = torch.device(f"cuda:{args.gpu}")

prefix = f"./results/Carla/face_thresh_0.01/econ/obj/{args.name}"
smpl_path = f"{prefix}_smpl_00.npy"
smplx_param = np.load(smpl_path, allow_pickle=True).item()

econ_path = f"{prefix}_0_full_soups.ply"
econ_obj = trimesh.load(econ_path)
assert econ_obj.vertex_normals.shape[1] == 3
os.makedirs(f"{prefix}/", exist_ok=True)

# Align ECON with SMPL-X
econ_obj.vertices *= np.array([1.0, -1.0, -1.0])
econ_obj.vertices /= smplx_param["scale"].cpu().numpy()
econ_obj.vertices -= smplx_param["transl"].cpu().numpy()

for key in smplx_param.keys():
    smplx_param[key] = smplx_param[key].cpu().view(1, -1)

smpl_model = smplx.create(
    smplx_container.model_dir, model_type="smplx", gender="neutral", age="adult",
    use_face_contour=True, use_pca=False, create_expression=True, create_betas=False,
    create_global_orient=True, create_body_pose=True, create_jaw_pose=True,
    create_left_hand_pose=False, create_right_hand_pose=False, create_transl=False,
    num_betas=smplx_param["betas"].shape[1],
    num_expression_coeffs=smplx_param["expression"].shape[1], ext="pkl",
)

smpl_out_lst = []
for pose_type in ["a-pose", "t-pose", "da-pose", "pose"]:
    smpl_out_lst.append(
        smpl_model(
            body_pose=smplx_param["body_pose"], global_orient=smplx_param["global_orient"],
            betas=smplx_param["betas"], expression=smplx_param["expression"],
            jaw_pose=smplx_param["jaw_pose"], left_hand_pose=smplx_param["left_hand_pose"],
            right_hand_pose=smplx_param["right_hand_pose"], return_verts=True,
            return_full_pose=True, return_joint_transformation=True,
            return_vertex_transformation=True, pose_type=pose_type,
        )
    )

smpl_verts = smpl_out_lst[3].vertices.detach()[0]
smpl_tree = cKDTree(smpl_verts.cpu().numpy())
dist, idx = smpl_tree.query(econ_obj.vertices, k=3)

if not osp.exists(f"{prefix}/econ_da.obj") or not osp.exists(f"{prefix}/smpl_da.obj"):
    # Create T-pose and DA-pose for ECON
    econ_verts = torch.tensor(econ_obj.vertices).float()
    rot_mat_t = smpl_out_lst[3].vertex_transformation.detach()[0][idx[:, 0]]
    homo_coord = torch.ones_like(econ_verts)[..., :1]
    econ_cano_verts = torch.inverse(rot_mat_t) @ torch.cat([econ_verts, homo_coord], dim=1).unsqueeze(-1)
    econ_cano_verts = econ_cano_verts[:, :3, 0].cpu()
    
    rot_mat_da = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
    econ_da_verts = rot_mat_da @ torch.cat([econ_cano_verts, homo_coord], dim=1).unsqueeze(-1)
    econ_da = trimesh.Trimesh(econ_da_verts[:, :3, 0].cpu(), econ_obj.faces)

    smpl_da = trimesh.Trimesh(smpl_out_lst[2].vertices.detach()[0], smpl_model.faces, maintain_orders=True, process=False)
    smpl_da.export(f"{prefix}/smpl_da.obj")

    # Surgically remove hands and close wrists
    print("Removing hands and closing wrist holes...")
    hand_vids_on_smplx_template = np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
    econ_da_body = econ_da.copy()
    face_mask_to_keep = ~hand_vids_on_smplx_template[econ_da_body.faces].any(axis=1)
    econ_da_body.update_faces(face_mask_to_keep)
    econ_da_body.remove_unreferenced_vertices()
    econ_da_body = keep_largest(econ_da_body)
    
    smoothed_body = MeshHoleUtils.smooth_all_boundary_loops(econ_da_body, iterations=30, factor=0.1)
    closed_body = MeshHoleUtils.fill_holes_pymeshlab(smoothed_body)
    
    if closed_body:
        econ_da = closed_body
        econ_da.export(f"{prefix}/econ_da.obj")
        print(colored("Final DA-pose mesh with closed wrists saved.", "green"))
    else:
        print(colored("Critical Failure: Could not close wrist holes. Using original as fallback.", "red"))
        econ_da.export(f"{prefix}/econ_da.obj")
else:
    econ_da = trimesh.load(f"{prefix}/econ_da.obj")
    smpl_da = trimesh.load(f"{prefix}/smpl_da.obj", maintain_orders=True, process=False)

print("Building SMPL-X compatible ECON model...")
smpl_tree = cKDTree(smpl_da.vertices)
dist, idx = smpl_tree.query(econ_da.vertices, k=3)
knn_weights = np.exp(-(dist**2))
knn_weights /= knn_weights.sum(axis=1, keepdims=True)

econ_J_regressor = (smpl_model.J_regressor[:, idx] * knn_weights[None]).sum(dim=-1)
econ_lbs_weights = (smpl_model.lbs_weights.T[:, idx] * knn_weights[None]).sum(dim=-1).T
num_posedirs = smpl_model.posedirs.shape[0]
econ_posedirs = ((smpl_model.posedirs.view(num_posedirs, -1, 3)[:, idx, :] * knn_weights[None, ..., None]).sum(dim=-2).view(num_posedirs, -1).float())
econ_J_regressor /= econ_J_regressor.sum(dim=1, keepdims=True).clip(min=1e-10)
econ_lbs_weights /= econ_lbs_weights.sum(dim=1, keepdims=True)

rot_mat_da = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
econ_da_verts = torch.tensor(econ_da.vertices).float()
econ_cano_verts = torch.inverse(rot_mat_da) @ torch.cat([econ_da_verts, torch.ones_like(econ_da_verts)[..., :1]], dim=1).unsqueeze(-1)
econ_cano_verts = econ_cano_verts[:, :3, 0].double()

# Animate ECON reconstruction to original pose
rot_mat_pose = smpl_out_lst[3].vertex_transformation.detach()[0][idx[:, 0]]
posed_econ_verts = rot_mat_pose @ torch.cat([econ_cano_verts.float(), torch.ones_like(econ_cano_verts.float())[..., :1]], dim=1).unsqueeze(-1)
posed_econ_verts = posed_econ_verts[:, :3, 0].double()
aligned_econ_verts = posed_econ_verts.detach().cpu().numpy()
aligned_econ_verts += smplx_param["transl"].cpu().numpy()
aligned_econ_verts *= smplx_param["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
econ_pose = trimesh.Trimesh(aligned_econ_verts, econ_da.faces)
econ_pose.export(f"{prefix}/econ_pose.ply")

# Build T-pose mesh with the same root orientation as the posed mesh
root_R = axis_angle_to_matrix(smplx_param["global_orient"]).cpu().numpy()[0]
aligned_tpose_verts = (root_R @ econ_cano_verts.cpu().numpy().T).T
aligned_tpose_verts += smplx_param["transl"].cpu().numpy()
aligned_tpose_verts *= smplx_param["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
econ_tpose = trimesh.Trimesh(aligned_tpose_verts, econ_da.faces)
econ_tpose.export(f"{prefix}/econ_tpose.ply")

cache_path = f"{prefix.replace('obj','cache')}"
os.makedirs(cache_path, exist_ok=True)

print("Mapping vertex colors from images...")
# First pass (red images)
cloth_front_red_path = f"./results/Carla/face_thresh_0.01/econ/png/{args.name}_cloth_front_red.png"
cloth_back_red_path = f"./results/Carla/face_thresh_0.01/econ/png/{args.name}_cloth_back_red.png"
tensor_front_1 = transforms.ToTensor()(Image.open(cloth_front_red_path))[:, :, :512]
tensor_back_1 = transforms.ToTensor()(Image.open(cloth_back_red_path))[:, :, :512]
front_image_1 = ((tensor_front_1 - 0.5) * 2.0).unsqueeze(0).to(device)
back_image_1 = ((tensor_back_1 - 0.5) * 2.0).unsqueeze(0).to(device)
verts_tensor = torch.tensor(econ_pose.vertices).float().to(device)
faces_tensor = torch.tensor(econ_pose.faces).long().to(device)
final_rgb_pass1 = query_color(verts_tensor, faces_tensor, front_image_1, back_image_1, device=device).numpy()

# UV Unwrapping
uv_rasterizer = Pytorch3dRasterizer(image_size=8192, device=device)
if not ("vt" in globals() and "ft" in globals() and "vmapping" in globals()):
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(econ_tpose.vertices, econ_tpose.faces)
    chart_options, pack_options = xatlas.ChartOptions(), xatlas.PackOptions()
    chart_options.max_iterations, pack_options.resolution, pack_options.bruteForce = 4, 8192, True
    atlas.generate(chart_options=chart_options)
    vmapping, ft_np, vt_np = atlas[0]
    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
    torch.save(vt.cpu(), osp.join(cache_path, "vt_tpose.pt"))
    torch.save(ft.cpu(), osp.join(cache_path, "ft_tpose.pt"))

# Generate first texture map
v_np, f_np = econ_tpose.vertices, econ_tpose.faces
texture_map1 = uv_rasterizer.get_texture(
    torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1), ft,
    torch.tensor(v_np).unsqueeze(0).float(), torch.tensor(f_np).unsqueeze(0).long(),
    torch.tensor(final_rgb_pass1).unsqueeze(0).float() / 255.0,
)
os.makedirs(f"{cache_path}/red", exist_ok=True)
Image.fromarray((texture_map1 * 255.0).astype(np.uint8)).save(f"{cache_path}/red/texture_red.png")

# Second pass (blue images)
cloth_front_path_blue = f"./results/Carla/face_thresh_0.01/econ/png/{args.name}_cloth_front_blue.png"
cloth_back_path_blue = f"./results/Carla/face_thresh_0.01/econ/png/{args.name}_cloth_back_blue.png"
tensor_front_2 = transforms.ToTensor()(Image.open(cloth_front_path_blue))[:, :, :512]
tensor_back_2 = transforms.ToTensor()(Image.open(cloth_back_path_blue))[:, :, :512]
front_image_2 = ((tensor_front_2 - 0.5) * 2.0).unsqueeze(0).to(device)
back_image_2 = ((tensor_back_2 - 0.5) * 2.0).unsqueeze(0).to(device)
final_rgb_pass2 = query_color(verts_tensor, faces_tensor, front_image_2, back_image_2, device=device).numpy()

texture_map2 = uv_rasterizer.get_texture(
    torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1), ft,
    torch.tensor(v_np).unsqueeze(0).float(), torch.tensor(f_np).unsqueeze(0).long(),
    torch.tensor(final_rgb_pass2).unsqueeze(0).float() / 255.0,
)
os.makedirs(f"{cache_path}/blue", exist_ok=True)
Image.fromarray((texture_map2 * 255.0).astype(np.uint8)).save(f"{cache_path}/blue/texture_blue.png")

# Assign vertex colors and save colored meshes
final_rgb = final_rgb_pass2
econ_pose.visual.vertex_colors = final_rgb
econ_pose.export(f"{prefix}/econ_icp_rgb.ply")
econ_tpose.visual.vertex_colors = final_rgb
econ_tpose.export(f"{prefix}/econ_tpose_rgb.ply")

# Save final ECON data for rigging
econ_dict = {
    "v_template": econ_cano_verts.unsqueeze(0), "posedirs": econ_posedirs,
    "J_regressor": econ_J_regressor, "parents": smpl_model.parents,
    "lbs_weights": econ_lbs_weights, "final_rgb": final_rgb,
    "faces": econ_pose.faces,
}
torch.save(econ_dict, f"{cache_path}/econ.pt")

if args.uv:
    print("Generating UV textures...")
    # Get UV coordinates
    v_np, f_np = econ_tpose.vertices, econ_tpose.faces
    vt_cache, ft_cache = osp.join(cache_path, "vt_tpose.pt"), osp.join(cache_path, "ft_tpose.pt")
    if osp.exists(vt_cache) and osp.exists(ft_cache):
        vt, ft = torch.load(vt_cache).to(device), torch.load(ft_cache).to(device)
    else:
        import xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options, pack_options = xatlas.ChartOptions(), xatlas.PackOptions()
        chart_options.max_iterations, pack_options.resolution, pack_options.bruteForce = 4, 8192, True
        atlas.generate(chart_options=chart_options)
        vmapping, ft_np, vt_np = atlas[0]
        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
        torch.save(vt.cpu(), vt_cache)
        torch.save(ft.cpu(), ft_cache)

    # Export intermediate assets for texture-defrag
    red_dir, blue_dir = osp.join(cache_path, "red"), osp.join(cache_path, "blue")
    os.makedirs(red_dir, exist_ok=True); os.makedirs(blue_dir, exist_ok=True)
    export_obj(np.array(econ_tpose.vertices), f_np, vt, ft, osp.join(red_dir, "mesh_red.obj"))
    with open(osp.join(red_dir, "material.mtl"), "w") as fp:
        fp.writelines(["newmtl mat0 \n", "map_Kd texture_red.png \n"])
    export_obj(np.array(econ_tpose.vertices), f_np, vt, ft, osp.join(blue_dir, "mesh_blue.obj"))
    with open(osp.join(blue_dir, "material.mtl"), "w") as fp:
        fp.writelines(["newmtl mat0 \n", "map_Kd texture_blue.png \n"])
    
    defrag_path = osp.join(cache_path, "defrag_assets")
    final_dir = osp.join(cache_path, "final_files")
    os.makedirs(defrag_path, exist_ok=True); os.makedirs(final_dir, exist_ok=True)
    texture_defrag_exe = os.path.abspath("./texture-defrag/build/texture-defrag")
    
    # Run texture-defrag
    mesh_file_red = os.path.abspath(osp.join(red_dir, "mesh_red.obj"))
    output_file_red = os.path.abspath(osp.join(defrag_path, "defrag_red.obj"))
    cmd_red = ["xvfb-run", "-a", texture_defrag_exe, mesh_file_red, "-l", "0", "-o", output_file_red]
    print("Running texture-defrag for red mesh...")
    subprocess.run(cmd_red, check=True)

    mesh_file_blue = os.path.abspath(osp.join(blue_dir, "mesh_blue.obj"))
    output_file_blue = os.path.abspath(osp.join(defrag_path, "defrag_blue.obj"))
    cmd_blue = ["xvfb-run", "-a", texture_defrag_exe, mesh_file_blue, "-l", "0", "-o", output_file_blue]
    print("Running texture-defrag for blue mesh...")
    subprocess.run(cmd_blue, check=True)

    # Inpaint and clean textures
    red_pattern, blue_pattern = re.compile(r"defrag_red_texture_(.+)\.png"), re.compile(r"defrag_blue_texture_(.+)\.png")
    red_images, blue_images = {}, {}
    for file in os.listdir(defrag_path):
        if "defrag" in file and file.endswith(".png"):
            if match := red_pattern.search(file): red_images[match.group(1)] = osp.join(defrag_path, file)
            if match := blue_pattern.search(file): blue_images[match.group(1)] = osp.join(defrag_path, file)

    for img_id in red_images:
        if img_id in blue_images:
            print(f"Processing textures for ID: {img_id}")
            red_img = np.array(Image.open(red_images[img_id]).convert("RGB"), dtype=np.float32)
            blue_img = np.array(Image.open(blue_images[img_id]).convert("RGB"), dtype=np.float32)
            diff_mask = (np.sum(np.abs(red_img - blue_img), axis=2) > 0.01).astype(np.uint8) * 255
            missing_mask = ((red_img.sum(axis=2) == 0) | (blue_img.sum(axis=2) == 0)).astype(np.uint8) * 255
            final_mask = np.maximum(diff_mask, missing_mask)
            eroded_mask = cv2.erode(final_mask, np.ones((10,10), np.uint8), iterations=3)
            dilated_mask = cv2.dilate(eroded_mask, np.ones((45, 45), np.uint8), iterations=2)
            if dilated_mask.shape != red_img.shape[:2]:
                dilated_mask = cv2.resize(dilated_mask, (red_img.shape[1], red_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            inpainted_texture = cv2.inpaint(red_img.astype(np.uint8), dilated_mask, 3, cv2.INPAINT_TELEA)
            final_texture_path = osp.join(final_dir, f"texture_map_inpainted_{img_id}.png")
            cv2.imwrite(final_texture_path, cv2.cvtColor(inpainted_texture, cv2.COLOR_RGB2BGR))
            
            if args.clean_texture:
                print(f"Applying post-process artifact cleaning to texture {img_id}...")
                clean_mask, cleaned_image = detect_pixels(load_image(final_texture_path))
                save_image(final_texture_path, cleaned_image)
                save_image(osp.join(final_dir, f"texture_map_cleaned_mask_{img_id}.png"), clean_mask)

    # Create final MTL and OBJ files
    with open(os.path.abspath(osp.join(defrag_path, "defrag_blue.obj.mtl")), "r") as f:
        mtl_lines = f.readlines()
    material_to_id, current_mat = {}, None
    for line in mtl_lines:
        if line.startswith("newmtl"): current_mat = line.strip().split()[1]
        elif line.startswith("map_Kd") and current_mat:
            if m := re.search(r"defrag_blue_texture_(.+)\.png", line):
                material_to_id[current_mat] = m.group(1)
            current_mat = None
    
    final_mtl_path = osp.join(final_dir, "final_material.mtl")
    with open(final_mtl_path, "w") as fp:
        for mat_name, mat_id in material_to_id.items():
            fp.write(f"newmtl {mat_name}\n")
            fp.write("Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
            fp.write(f"Tr 1.0\nillum 1\nNs 0.0\nmap_Kd texture_map_inpainted_{mat_id}.png\n\n")
    print("Final material file saved at:", final_mtl_path)

    final_obj_path = osp.join(final_dir, "final_model.obj")
    with open(output_file_blue, "r") as f_in, open(final_obj_path, "w") as f_out:
        f_out.write("mtllib final_material.mtl\n")
        for line in f_in:
            if not line.startswith("mtllib"): f_out.write(line)
    print("Final OBJ file saved at:", final_obj_path)

print("🔎  Extracting landmarks on the defragmented mesh…")
final_mesh = trimesh.load(final_obj_path, process=False, force='mesh')
landmark_indices, landmark_coords = get_facial_landmarks_on_tpose(
    final_tpose_mesh=final_mesh, smpl_model=smpl_model,
    smpl_params={
        "betas": smplx_param["betas"], "expression": smplx_param["expression"],
        "global_orient": smplx_param["global_orient"], "transl": smplx_param["transl"],
        "scale": smplx_param["scale"],
    },
    device=device,
)

np.save(osp.join(final_dir, "landmark_indices.npy"), landmark_indices)
landmark_cloud = trimesh.Trimesh(landmark_coords, process=False)
landmark_cloud.apply_transform(rotation_matrix(np.pi / 2.0, [1, 0, 0]))
landmark_cloud.export(osp.join(final_dir, "landmarks_only.ply"))

print("✅  Landmarks saved:\n   • landmark_indices.npy\n   • landmarks_only.ply")