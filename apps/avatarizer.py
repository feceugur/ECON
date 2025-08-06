# Standard library imports
import argparse
import os
import os.path as osp
import re
import subprocess
import tempfile
from typing import Tuple, Optional, List

# Third-party imports
import cv2
import numpy as np
import pymeshlab
import torch
import trimesh
from PIL import Image
from pytorch3d.transforms import axis_angle_to_matrix
from scipy.spatial import cKDTree
from torchvision import transforms
from trimesh.transformations import rotation_matrix
from termcolor import colored

# Local application/library specific imports
import lib.smplx as smplx
from lib.common.render import query_color
from lib.common.render_utils import Pytorch3dRasterizer
from lib.dataset.mesh_util import SMPLX, export_obj, keep_largest
from smplx.lbs import (
    vertices2landmarks,
    find_dynamic_lmk_idx_and_bcoords,
)

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, required=True, help="Name of the subject/scan.")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device ID to use.")
parser.add_argument("-uv", action="store_true", help="Generate UV maps and textured .obj file.")
parser.add_argument(
    "-ct", "--clean_texture", action="store_true",
    help="Apply final artifact cleaning to texture maps (requires -uv)."
)
args = parser.parse_args()

def extract_landmarks_from_smplx_params(
    smpl_params: dict,
    smplx_model: smplx.SMPLX,
    device: torch.device
) -> np.ndarray:
    """
    Calculates 68 3D facial landmarks by generating them on a canonical
    subject-specific T-pose and then applying the exact same world
    transformation used to create the 'econ_tpose' mesh, ensuring a
    perfect coordinate system match.
    """
    print("⇢ Extracting 68 landmarks via Canonical Transformation...")
    
    # 1. Generate landmarks on a subject-specific CANONICAL T-POSE
    smplx_model.to(device).eval()
    with torch.no_grad():
        # Use the subject's shape (betas) and expression, but force a neutral pose
        # to get the landmarks in the correct T-pose coordinate space.
        canonical_output = smplx_model(
            betas=smpl_params["betas"].to(device),
            expression=smpl_params["expression"].to(device),
            global_orient=torch.zeros_like(smpl_params["global_orient"]).to(device),
            body_pose=torch.zeros_like(smpl_params["body_pose"]).to(device),
            jaw_pose=torch.zeros_like(smpl_params["jaw_pose"]).to(device),
            return_verts=True,
            return_full_pose=True
        )

    # Calculate the 68 landmark coordinates from the vertices of this canonical mesh
    posed_vertices = canonical_output.vertices
    full_pose = canonical_output.full_pose
    faces_tensor = smplx_model.faces_tensor.to(device)
    
    static_lmks = vertices2landmarks(posed_vertices, faces_tensor, smplx_model.lmk_faces_idx.to(device).unsqueeze(0), smplx_model.lmk_bary_coords.to(device).unsqueeze(0))
    dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(posed_vertices, full_pose, smplx_model.dynamic_lmk_faces_idx.to(device), smplx_model.dynamic_lmk_bary_coords.to(device), torch.as_tensor(smplx_model.neck_kin_chain, dtype=torch.long, device=device))
    dynamic_lmks = vertices2landmarks(posed_vertices, faces_tensor, dyn_lmk_faces_idx, dyn_lmk_bary_coords)
    
    canonical_landmarks = torch.cat([static_lmks, dynamic_lmks], dim=1)

    # 2. Apply the SAME world transformation used for `econ_tpose`
    R = axis_angle_to_matrix(smpl_params["global_orient"].to(device))
    t = smpl_params["transl"].to(device)
    s = smpl_params["scale"].to(device)
    flip = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], device=device)

    # Apply transformations: Rotate, Translate, Scale, Flip.
    posed_landmarks = torch.bmm(canonical_landmarks, R.transpose(1, 2))
    posed_landmarks = (posed_landmarks + t.unsqueeze(1)) * s.view(-1, 1, 1)
    posed_landmarks = torch.matmul(posed_landmarks, flip.T)
    
    # 3. Squeeze batch dimension and return as numpy array
    final_landmark_coords = posed_landmarks.squeeze(0).cpu().numpy()
    
    print(f"✓ Directly extracted and aligned {final_landmark_coords.shape[0]} facial landmark coordinates.")
    return final_landmark_coords

def load_image(path: str) -> np.ndarray:
    """Load an image in color mode and return as numpy array."""
    return cv2.imread(path)


def save_image(path: str, image: np.ndarray):
    """Save image to the given path."""
    cv2.imwrite(path, image)


def detect_pixels(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a 3x3 edge-detection kernel to highlight and average edge pixels.
    This version is vectorized to be significantly faster than the original
    by replacing slow Python loops with optimized OpenCV/NumPy operations.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = image.copy()
    
    # Each tuple contains: (edge detection kernel, corresponding neighbor averaging kernel)
    kernel_pairs = [
        (np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]), np.array([[0, 0.5, 0], [0, 0, 0], [0, 0.5, 0]])), # Vertical
        (np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]), np.array([[0, 0, 0], [0.5, 0, 0.5], [0, 0, 0]])), # Horizontal
        (np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]]), np.array([[0.5, 0, 0], [0, 0, 0], [0, 0, 0.5]])), # Diagonal /
        (np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]]), np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]])), # Diagonal \
    ]
    
    all_detection_masks = []

    for detect_kernel, avg_kernel in kernel_pairs:
        # 1. Detect edges for this direction on the grayscale image
        convolved = cv2.filter2D(gray, -1, detect_kernel)
        _, detection_mask = cv2.threshold(np.abs(convolved), 30, 255, cv2.THRESH_BINARY)
        all_detection_masks.append(detection_mask)

        # 2. Pre-calculate the averaged neighbor colors for the ENTIRE image
        # This is the core optimization: one fast operation instead of millions of slow ones.
        averaged_colors = cv2.filter2D(image, -1, avg_kernel)

        # 3. Apply the averaged colors only where the edge was detected
        # This uses boolean array indexing, which is extremely fast.
        result[detection_mask > 0] = averaged_colors[detection_mask > 0]

    # Combine all masks to get the final mask of all changed pixels
    combined_mask = np.maximum.reduce(all_detection_masks).astype(np.uint8)
    
    return combined_mask, result

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
    def _get_boundary_edges(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        if not MeshHoleUtils._is_mesh_valid(mesh): return None
        try:
            edges = mesh.edges_sorted.copy()
            unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
            return unique_edges[counts == 1]
        except Exception:
            return np.array([])

    @staticmethod
    def _order_loop_vertices(loop_vertices: np.ndarray, loop_edges: np.ndarray) -> Optional[np.ndarray]:
        if loop_vertices is None or len(loop_vertices) < 3 or loop_edges is None or len(loop_edges) < 2:
            return None
        
        adj = {v: [] for v in loop_vertices}
        for u, v in loop_edges:
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)
        
        start_node = loop_vertices[0]
        path = [start_node]
        visited_edges = set()
        current_node = start_node

        for _ in range(len(loop_edges) + 2):
            found_next = False
            for neighbor in adj.get(current_node, []):
                edge = tuple(sorted((current_node, neighbor)))
                if edge not in visited_edges:
                    path.append(neighbor)
                    visited_edges.add(edge)
                    current_node = neighbor
                    found_next = True
                    break
            if not found_next: break
        
        if len(path) > 1 and path[0] == path[-1]:
            path = path[:-1]
        
        if len(np.unique(path)) == len(loop_vertices):
            return np.array(path, dtype=int)
        return None

    @staticmethod
    def get_all_boundary_loops(mesh: trimesh.Trimesh, min_loop_len: int = 3) -> list:
        if not MeshHoleUtils._is_mesh_valid(mesh): return []
        
        boundary_edges = MeshHoleUtils._get_boundary_edges(mesh)
        if boundary_edges is None or len(boundary_edges) < min_loop_len: return []
        
        all_loops = []
        components = trimesh.graph.connected_components(boundary_edges, min_len=min_loop_len)
        for component_verts in components:
            comp_set = set(component_verts)
            edges_in_comp = [edge for edge in boundary_edges if edge[0] in comp_set and edge[1] in comp_set]
            ordered_loop = MeshHoleUtils._order_loop_vertices(component_verts, np.array(edges_in_comp))
            if ordered_loop is not None:
                all_loops.append(ordered_loop)
        return all_loops

    @staticmethod
    def smooth_all_boundary_loops(mesh: trimesh.Trimesh, iterations: int = 35, factor: float = 0.1) -> trimesh.Trimesh:
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
            
            boundary_edges = MeshHoleUtils._get_boundary_edges(filled_mesh)
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
if __name__ == "__main__":
    
    # --- 1. Initialization and Data Loading ---
    print(f"--- Starting Avatarizer for subject: {args.name} ---")
    device = torch.device(f"cuda:{args.gpu}")
    smplx_container = SMPLX()

    prefix = f"./results/Carla/face_thresh_0.01/econ/obj/{args.name}"
    smpl_path = f"{prefix}_smpl_00.npy"
    econ_path = f"{prefix}_0_full_soups.ply"
    
    smpl_params = np.load(smpl_path, allow_pickle=True).item()
    initial_econ_mesh = trimesh.load(econ_path)
    assert initial_econ_mesh.vertex_normals.shape[1] == 3
    os.makedirs(prefix, exist_ok=True)

    # --- 2. Initial Alignment and SMPL-X Model Creation ---
    print("Aligning initial ECON mesh with SMPL-X parameters...")
    initial_econ_mesh.vertices *= np.array([1.0, -1.0, -1.0])
    initial_econ_mesh.vertices /= smpl_params["scale"].cpu().numpy()
    initial_econ_mesh.vertices -= smpl_params["transl"].cpu().numpy()

    for key in smpl_params.keys():
        smpl_params[key] = smpl_params[key].cpu().view(1, -1)

    smpl_model = smplx.create(
        smplx_container.model_dir, model_type="smplx", gender="neutral", age="adult",
        use_face_contour=True, use_pca=False, create_expression=True, create_betas=True,
        create_global_orient=True, create_body_pose=True, create_jaw_pose=True,
        create_left_hand_pose=True, create_right_hand_pose=True, create_transl=False,
        num_betas=smpl_params["betas"].shape[1],
        num_expression_coeffs=smpl_params["expression"].shape[1], ext="pkl",
    )

    smpl_outputs = {
        pose_type: smpl_model(
            body_pose=smpl_params["body_pose"], global_orient=smpl_params["global_orient"],
            betas=smpl_params["betas"], expression=smpl_params["expression"],
            jaw_pose=smpl_params["jaw_pose"], left_hand_pose=smpl_params["left_hand_pose"],
            right_hand_pose=smpl_params["right_hand_pose"], return_verts=True,
            return_full_pose=True, return_joint_transformation=True,
            return_vertex_transformation=True, pose_type=pose_type,
        ) for pose_type in ["a-pose", "t-pose", "da-pose", "pose"]
    }

    # --- 3. Create and Cache DA-Pose Mesh ---
    econ_da_path = f"{prefix}/econ_da.obj"
    smpl_da_path = f"{prefix}/smpl_da.obj"
    
    if not osp.exists(econ_da_path) or not osp.exists(smpl_da_path):
        print("Generating and caching DA-pose meshes...")
        smpl_posed_verts = smpl_outputs["pose"].vertices.detach()[0]
        smpl_tree = cKDTree(smpl_posed_verts.cpu().numpy())
        dist, idx = smpl_tree.query(initial_econ_mesh.vertices, k=3)

        econ_verts_tensor = torch.tensor(initial_econ_mesh.vertices).float()
        rot_mat_t = smpl_outputs["pose"].vertex_transformation.detach()[0][idx[:, 0]]
        homo_coord = torch.ones_like(econ_verts_tensor)[..., :1]
        econ_cano_verts = torch.inverse(rot_mat_t) @ torch.cat([econ_verts_tensor, homo_coord], dim=1).unsqueeze(-1)
        econ_cano_verts = econ_cano_verts[:, :3, 0].cpu()
        
        rot_mat_da = smpl_outputs["da-pose"].vertex_transformation.detach()[0][idx[:, 0]]
        econ_da_verts = rot_mat_da @ torch.cat([econ_cano_verts, homo_coord], dim=1).unsqueeze(-1)
        econ_da = trimesh.Trimesh(econ_da_verts[:, :3, 0].cpu(), initial_econ_mesh.faces)

        smpl_da = trimesh.Trimesh(smpl_outputs["da-pose"].vertices.detach()[0], smpl_model.faces, maintain_orders=True, process=False)
        smpl_da.export(smpl_da_path)

        print("Removing hands and closing wrist holes...")
        hand_vids_on_smplx = np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
        econ_da_body = econ_da.copy()
        face_mask_to_keep = ~hand_vids_on_smplx[econ_da_body.faces].any(axis=1)
        econ_da_body.update_faces(face_mask_to_keep)
        econ_da_body.remove_unreferenced_vertices()
        econ_da_body = keep_largest(econ_da_body)
        
        smoothed_body = MeshHoleUtils.smooth_all_boundary_loops(econ_da_body, iterations=30, factor=0.1)
        closed_body = MeshHoleUtils.fill_holes_pymeshlab(smoothed_body)
        
        if closed_body:
            econ_da = closed_body
            econ_da.export(econ_da_path)
            print(colored("Final DA-pose mesh with closed wrists saved.", "green"))
        else:
            print(colored("Critical Failure: Could not close wrist holes. Using original as fallback.", "red"))
            econ_da.export(econ_da_path)
    else:
        print("Loading cached DA-pose meshes...")
        econ_da = trimesh.load(econ_da_path)
        smpl_da = trimesh.load(smpl_da_path, maintain_orders=True, process=False)

    # --- 4. Build ECON-Specific SMPL-X Model Components ---
    print("Building ECON-specific SMPL-X model components (skinning weights, etc.)...")
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

    rot_mat_da = smpl_outputs["da-pose"].vertex_transformation.detach()[0][idx[:, 0]]
    econ_da_verts_tensor = torch.tensor(econ_da.vertices).float()
    econ_cano_verts = torch.inverse(rot_mat_da) @ torch.cat([econ_da_verts_tensor, torch.ones_like(econ_da_verts_tensor)[..., :1]], dim=1).unsqueeze(-1)
    econ_cano_verts = econ_cano_verts[:, :3, 0].double()

    # --- 5. Generate Final Posed and T-Pose ECON Meshes ---
    print("Generating final posed and T-pose ECON meshes...")
    rot_mat_pose = smpl_outputs["pose"].vertex_transformation.detach()[0][idx[:, 0]]
    posed_econ_verts = rot_mat_pose @ torch.cat([econ_cano_verts.float(), torch.ones_like(econ_cano_verts.float())[..., :1]], dim=1).unsqueeze(-1)
    posed_econ_verts = posed_econ_verts[:, :3, 0].double()
    
    aligned_econ_verts = posed_econ_verts.detach().cpu().numpy()
    aligned_econ_verts += smpl_params["transl"].cpu().numpy()
    aligned_econ_verts *= smpl_params["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
    econ_pose = trimesh.Trimesh(aligned_econ_verts, econ_da.faces)
    econ_pose.export(f"{prefix}/econ_pose.ply")

    root_R = axis_angle_to_matrix(smpl_params["global_orient"]).cpu().numpy()[0]
    aligned_tpose_verts = (root_R @ econ_cano_verts.cpu().numpy().T).T
    aligned_tpose_verts += smpl_params["transl"].cpu().numpy()
    aligned_tpose_verts *= smpl_params["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
    econ_tpose = trimesh.Trimesh(aligned_tpose_verts, econ_da.faces)
    econ_tpose.export(f"{prefix}/econ_tpose.ply")

    # --- 6. Texture Mapping from Rendered Images ---
    cache_path = f"{prefix.replace('obj','cache')}"
    os.makedirs(cache_path, exist_ok=True)
    print("Mapping vertex colors from rendered images...")
    
    # Pass 1 (Red)
    cloth_front_red_path = f"./results/Carla/face_thresh_0.01/econ/png/{args.name}_cloth_front_red.png"
    cloth_back_red_path = f"./results/Carla/face_thresh_0.01/econ/png/{args.name}_cloth_back_red.png"
    tensor_front_1 = transforms.ToTensor()(Image.open(cloth_front_red_path))[:, :, :512]
    tensor_back_1 = transforms.ToTensor()(Image.open(cloth_back_red_path))[:, :, :512]
    front_image_1 = ((tensor_front_1 - 0.5) * 2.0).unsqueeze(0).to(device)
    back_image_1 = ((tensor_back_1 - 0.5) * 2.0).unsqueeze(0).to(device)
    verts_tensor = torch.tensor(econ_pose.vertices).float().to(device)
    faces_tensor = torch.tensor(econ_pose.faces).long().to(device)
    final_rgb_pass1 = query_color(verts_tensor, faces_tensor, front_image_1, back_image_1, device=device).numpy()

    # Pass 2 (Blue)
    cloth_front_blue_path = f"./results/Carla/face_thresh_0.01/econ/png/{args.name}_cloth_front_blue.png"
    cloth_back_blue_path = f"./results/Carla/face_thresh_0.01/econ/png/{args.name}_cloth_back_blue.png"
    tensor_front_2 = transforms.ToTensor()(Image.open(cloth_front_blue_path))[:, :, :512]
    tensor_back_2 = transforms.ToTensor()(Image.open(cloth_back_blue_path))[:, :, :512]
    front_image_2 = ((tensor_front_2 - 0.5) * 2.0).unsqueeze(0).to(device)
    back_image_2 = ((tensor_back_2 - 0.5) * 2.0).unsqueeze(0).to(device)
    final_rgb_pass2 = query_color(verts_tensor, faces_tensor, front_image_2, back_image_2, device=device).numpy()

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

    # --- 7. UV Unwrapping and Texture Generation (Optional) ---
    if args.uv:
        print("--- Starting UV Generation and Texturing Pipeline ---")
        final_dir = osp.join(cache_path, "final_files")
        os.makedirs(final_dir, exist_ok=True)
        
        # Get UV coordinates
        vt_cache, ft_cache = osp.join(cache_path, "vt_tpose.pt"), osp.join(cache_path, "ft_tpose.pt")
        if osp.exists(vt_cache) and osp.exists(ft_cache):
            vt, ft = torch.load(vt_cache).to(device), torch.load(ft_cache).to(device)
        else:
            import xatlas
            atlas = xatlas.Atlas()
            atlas.add_mesh(econ_tpose.vertices, econ_tpose.faces)
            chart_options, pack_options = xatlas.ChartOptions(), xatlas.PackOptions()
            chart_options.max_iterations, pack_options.resolution, pack_options.bruteForce = 4, 8192, True
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]
            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
            torch.save(vt.cpu(), vt_cache)
            torch.save(ft.cpu(), ft_cache)

        # Generate texture maps using the two color passes
        uv_rasterizer = Pytorch3dRasterizer(image_size=8192, device=device)
        texture_map1 = uv_rasterizer.get_texture(torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1), ft, torch.tensor(econ_tpose.vertices).unsqueeze(0).float(), torch.tensor(econ_tpose.faces).unsqueeze(0).long(), torch.tensor(final_rgb_pass1).unsqueeze(0).float() / 255.0)
        texture_map2 = uv_rasterizer.get_texture(torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1), ft, torch.tensor(econ_tpose.vertices).unsqueeze(0).float(), torch.tensor(econ_tpose.faces).unsqueeze(0).long(), torch.tensor(final_rgb_pass2).unsqueeze(0).float() / 255.0)
        
        red_dir, blue_dir = osp.join(cache_path, "red"), osp.join(cache_path, "blue")
        os.makedirs(red_dir, exist_ok=True); os.makedirs(blue_dir, exist_ok=True)
        Image.fromarray((texture_map1 * 255.0).astype(np.uint8)).save(osp.join(red_dir, "texture_red.png"))
        Image.fromarray((texture_map2 * 255.0).astype(np.uint8)).save(osp.join(blue_dir, "texture_blue.png"))

        # Export intermediate assets for texture-defrag
        export_obj(np.array(econ_tpose.vertices), econ_tpose.faces, vt, ft, osp.join(red_dir, "mesh_red.obj"))
        with open(osp.join(red_dir, "material.mtl"), "w") as fp: fp.writelines(["newmtl mat0 \n", "map_Kd texture_red.png \n"])
        export_obj(np.array(econ_tpose.vertices), econ_tpose.faces, vt, ft, osp.join(blue_dir, "mesh_blue.obj"))
        with open(osp.join(blue_dir, "material.mtl"), "w") as fp: fp.writelines(["newmtl mat0 \n", "map_Kd texture_blue.png \n"])
        
        # Run texture-defrag
        defrag_path = osp.join(cache_path, "defrag_assets")
        os.makedirs(defrag_path, exist_ok=True)
        texture_defrag_exe = os.path.abspath("./texture-defrag/build/texture-defrag")
        
        for color in ["red", "blue"]:
            mesh_file = os.path.abspath(osp.join(cache_path, color, f"mesh_{color}.obj"))
            output_file = os.path.abspath(osp.join(defrag_path, f"defrag_{color}.obj"))
            cmd = ["xvfb-run", "-a", texture_defrag_exe, mesh_file, "-l", "0", "-o", output_file]
            print(f"Running texture-defrag for {color} mesh...")
            subprocess.run(cmd, check=True)

        # Inpaint and clean textures
        red_pattern, blue_pattern = re.compile(r"defrag_red_texture_(.+)\.png"), re.compile(r"defrag_blue_texture_(.+)\.png")
        red_images, blue_images = {}, {}
        for file in os.listdir(defrag_path):
            if "defrag" in file and file.endswith(".png"):
                if match := red_pattern.search(file): red_images[match.group(1)] = osp.join(defrag_path, file)
                if match := blue_pattern.search(file): blue_images[match.group(1)] = osp.join(defrag_path, file)

        for img_id in red_images:
            if img_id in blue_images:
                print(f"Processing and inpainting textures for ID: {img_id}")
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
                
                # args.clean_texture = True
                if args.clean_texture:
                    print(f"Applying post-process artifact cleaning to texture {img_id}...")
                    clean_mask, cleaned_image = detect_pixels(load_image(final_texture_path))
                    save_image(final_texture_path, cleaned_image)
                    save_image(osp.join(final_dir, f"texture_map_cleaned_mask_{img_id}.png"), clean_mask)

        # Create final MTL and OBJ files
        with open(os.path.abspath(osp.join(defrag_path, "defrag_blue.obj.mtl")), "r") as f: mtl_lines = f.readlines()
        material_to_id, current_mat = {}, None
        for line in mtl_lines:
            if line.startswith("newmtl"): current_mat = line.strip().split()[1]
            elif line.startswith("map_Kd") and current_mat:
                if m := re.search(r"defrag_blue_texture_(.+)\.png", line): material_to_id[current_mat] = m.group(1)
                current_mat = None
        
        final_mtl_path = osp.join(final_dir, "final_material.mtl")
        with open(final_mtl_path, "w") as fp:
            for mat_name, mat_id in material_to_id.items():
                fp.write(f"newmtl {mat_name}\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
                fp.write(f"Tr 1.0\nillum 1\nNs 0.0\nmap_Kd texture_map_inpainted_{mat_id}.png\n\n")
        print("Final material file saved at:", final_mtl_path)

        final_obj_path = osp.join(final_dir, "final_model.obj")
        with open(os.path.abspath(osp.join(defrag_path, "defrag_blue.obj")), "r") as f_in, open(final_obj_path, "w") as f_out:
            f_out.write(f"mtllib {osp.basename(final_mtl_path)}\n")
            for line in f_in:
                if not line.startswith("mtllib"): f_out.write(line)
        print("Final OBJ file saved at:", final_obj_path)

    # --- 8. Final Landmark Extraction ---
    print("--- Extracting Final Landmarks ---")
    final_landmark_coords = extract_landmarks_from_smplx_params(
        smpl_params=smpl_params,
        smplx_model=smpl_model,
        device=device
    )

    # Apply final rotation for visualization consistency
    R90x = np.array([[1, 0, 0], [0, 0,-1], [0, 1, 0]], dtype=np.float64)
    final_landmark_coords_rotated = final_landmark_coords @ R90x.T

    # Save final landmark files
    output_dir = final_dir if args.uv else prefix
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(osp.join(output_dir, "landmarks_coords.npy"), final_landmark_coords_rotated)
    landmark_cloud = trimesh.Trimesh(final_landmark_coords_rotated, process=False)
    landmark_cloud.export(osp.join(output_dir, "landmarks_only.ply"))

    print(colored("✅ Avatarizer process completed successfully!", "green"))
    print(f"   Landmarks saved to: {osp.join(output_dir, 'landmarks_only.ply')}")