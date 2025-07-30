import sys
import os
import argparse
import numpy as np
import trimesh
import torch
from typing import Tuple

# --- START: CRITICAL PATH FIX ---
# Add the project's root directory ('ECON') to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_root)
# --- END: CRITICAL PATH FIX ---

import lib.smplx as smplx
from lib.dataset.mesh_util import SMPLX
from scipy.spatial import cKDTree
from termcolor import colored
from pytorch3d.transforms import axis_angle_to_matrix

# This is the FRAGILE landmark detection function from your avatarizer.py.
# We use it because we need to replicate that script's behavior exactly.
def get_facial_landmarks_on_tpose(
    final_tpose_mesh: trimesh.Trimesh, smpl_model: smplx.SMPLX, smpl_params: dict, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        betas = smpl_params["betas"].to(device)
        expression = smpl_params["expression"].to(device)
        smpl_out = smpl_model(betas=betas, expression=expression, return_verts=True)
        verts = smpl_out.vertices.squeeze(0)
        faces = smpl_model.faces_tensor.to(device).long()
        lmk_faces_idx = smpl_model.lmk_faces_idx.to(device).long()
        lmk_bary_coords = smpl_model.lmk_bary_coords.to(device)
        face_vertices = verts[faces[lmk_faces_idx]]
        canonical_landmarks = (lmk_bary_coords.unsqueeze(-1) * face_vertices).sum(dim=1)
    root_R = axis_angle_to_matrix(smpl_params["global_orient"].to(device)).squeeze(0)
    landmarks_posed = (root_R @ canonical_landmarks.T).T
    landmarks_translated = landmarks_posed + smpl_params["transl"].to(device)
    scale = smpl_params["scale"].to(device)
    coord_flip = torch.tensor([1.0, -1.0, -1.0], device=device)
    final_landmarks = landmarks_translated * scale * coord_flip
    final_landmarks_np = final_landmarks.cpu().numpy()
    kdtree = cKDTree(np.asarray(final_tpose_mesh.vertices))
    _, landmark_indices = kdtree.query(final_landmarks_np)
    return landmark_indices, final_tpose_mesh.vertices[landmark_indices]


def load_as_single_mesh(file_path: str) -> trimesh.Trimesh:
    loaded_data = trimesh.load(file_path, process=False)
    if isinstance(loaded_data, trimesh.Scene):
        print(f"  - Note: '{os.path.basename(file_path)}' loaded as a Scene. Concatenating geometry...")
        mesh_list = [geom for geom in loaded_data.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        return trimesh.util.concatenate(mesh_list)
    return loaded_data


def main(args):
    print(f"--- Landmark Integrity Check for '{args.name}' ---")
    device = torch.device(f"cuda:{args.gpu}")
    smplx_container = SMPLX()

    # --- 1. Load All Required Assets ---
    print("\n[Step 1/4] Loading models and data...")
    grafting_dir = f"./results/Carla/face_thresh_0.01/econ/obj/"
    avatarizer_cache_dir = f"./results/Carla/face_thresh_0.01/econ/cache/{args.name}"

    econ_obj_path = os.path.join(grafting_dir, f"{args.name}_0_full_soups.ply")
    final_mesh_path = os.path.join(avatarizer_cache_dir, "final_files", "final_model.obj")
    smpl_param_path = os.path.join(grafting_dir, f"{args.name}_smpl_00.npy")

    for f_path in [econ_obj_path, final_mesh_path, smpl_param_path]:
        if not os.path.exists(f_path):
            print(colored(f"Error: Required file not found at: {f_path}", "red")); return

    econ_obj = trimesh.load(econ_obj_path)
    final_mesh = load_as_single_mesh(final_mesh_path)
    
    smpl_params = np.load(smpl_param_path, allow_pickle=True).item()
    for key in smpl_params.keys():
        smpl_params[key] = smpl_params[key].cpu().view(1, -1)
            
    smpl_model = smplx.create(
        smplx_container.model_dir, model_type="smplx", ext="pkl",
        num_betas=smpl_params["betas"].shape[1],
        num_expression_coeffs=smpl_params["expression"].shape[1],
    ).to(device)

    # --- 2. Replicate the `econ_tpose` creation from `avatarizer.py` ---
    print("\n[Step 2/4] Recreating the 'econ_tpose' mesh (the 'Before' state)...")
    econ_obj.vertices *= np.array([1.0, -1.0, -1.0])
    econ_obj.vertices /= smpl_params["scale"].cpu().numpy()
    econ_obj.vertices -= smpl_params["transl"].cpu().numpy()

    smpl_out_posed = smpl_model(
        body_pose=smpl_params["body_pose"], global_orient=smpl_params["global_orient"],
        betas=smpl_params["betas"], expression=smpl_params["expression"], jaw_pose=smpl_params["jaw_pose"],
        return_vertex_transformation=True, pose_type="pose"
    )
    smpl_tree = cKDTree(smpl_out_posed.vertices.detach().squeeze(0).cpu().numpy())
    _, idx = smpl_tree.query(econ_obj.vertices, k=1)
    
    # Calculate canonical vertices
    rot_mat_t = smpl_out_posed.vertex_transformation.detach().squeeze(0)[idx]
    
    # --- START OF FIX: Move tensors to the correct device ---
    econ_verts = torch.tensor(econ_obj.vertices, device=device).float()
    homo_coord = torch.ones_like(econ_verts)[..., :1]
    # --- END OF FIX ---

    econ_cano_verts = torch.inverse(rot_mat_t) @ torch.cat([econ_verts, homo_coord], dim=1).unsqueeze(-1)
    econ_cano_verts = econ_cano_verts[:, :3, 0].cpu().double()

    root_R = axis_angle_to_matrix(smpl_params["global_orient"]).cpu().numpy()[0]
    aligned_tpose_verts = (root_R @ econ_cano_verts.numpy().T).T
    aligned_tpose_verts += smpl_params["transl"].cpu().numpy()
    aligned_tpose_verts *= smpl_params["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
    
    mesh_before = trimesh.Trimesh(aligned_tpose_verts, econ_obj.faces)

    # --- 3. Get Landmarks on the "Before" Mesh ---
    print("\n[Step 3/4] Finding landmarks on the 'Before' mesh (`econ_tpose`)...")
    before_indices, _ = get_facial_landmarks_on_tpose(
        final_tpose_mesh=mesh_before, smpl_model=smpl_model, smpl_params=smpl_params, device=device
    )
    num_before = len(np.unique(before_indices))
    landmark_coords_before = mesh_before.vertices[before_indices]

    # --- 4. Compare with the Final "After" Mesh ---
    print("\n[Step 4/4] Checking integrity of these landmarks in the final avatar mesh...")
    kdtree_after = cKDTree(final_mesh.vertices)
    distances, indices_in_after = kdtree_after.query(landmark_coords_before)

    moved_mask = distances > 1e-5
    num_moved = np.sum(moved_mask)
    num_unique_after = len(np.unique(indices_in_after))
    num_merged = len(before_indices) - num_unique_after
    
    # --- 5. Report Conclusion ---
    print("\n" + "="*50)
    print("--- FINAL REPORT ---")
    print(f"Landmarks found on recreated 'econ_tpose' mesh: {num_before}")
    
    if num_moved == 0 and num_merged == 0:
        print(colored(f"Conclusion: PASS. All {num_before} landmarks were perfectly preserved in the final mesh.", "green"))
        print("This means the issue is not `texture-defrag`, but rather the initial creation of `econ_tpose`.")
    else:
        print(colored("Conclusion: FAIL. The face geometry was altered AFTER `econ_tpose` was created.", "red"))
        if num_moved > 0: print(f"  - Cause: {num_moved} of the {num_before} landmarks were moved or deleted.")
        if num_merged > 0: print(f"  - Cause: {num_merged} of the {num_before} landmarks were merged into other vertices.")
        print(colored("  - This confirms that a later step, most likely `texture-defrag`, is changing the mesh.", "yellow"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check landmark consistency.")
    parser.add_argument("-n", "--name", required=True, help="Subject name (e.g., carla_Apose)")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    main(args)