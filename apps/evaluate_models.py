import trimesh
import numpy as np
import torch
import pandas as pd
import argparse
import os
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance

# --- Helper function to compute metrics for one pair of meshes ---
def calculate_metrics(pred_mesh, gt_mesh, num_samples=30000):
    """
    Calculates Chamfer, P2S, and Normal consistency between two meshes.
    (Updated for modern trimesh API)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-enabled GPU.")
        
    # --- 1. Sample points and normals from meshes ---
    pred_points, pred_face_indices = trimesh.sample.sample_surface(pred_mesh, num_samples)
    gt_points, gt_face_indices = trimesh.sample.sample_surface(gt_mesh, num_samples)
    
    pred_normals = pred_mesh.face_normals[pred_face_indices]
    
    pred_points_t = torch.from_numpy(pred_points).float().cuda()
    gt_points_t = torch.from_numpy(gt_points).float().cuda()
    
    # --- 2. Chamfer Distance ---
    
    chamfer_dist, _ = chamfer_distance(
        pred_points_t.unsqueeze(0), 
        gt_points_t.unsqueeze(0),
        point_reduction='mean'
    )
    chamfer_dist_val = torch.sqrt(chamfer_dist).item() * 1000 # convert m to mm

    # --- 3. Point-to-Surface (P2S) Distance ---
    # --- FIX IS HERE ---
    # Use trimesh.proximity.closest_point(mesh, points) instead of mesh.proximity.closest_point(points)
    
    # P2S(Predicted -> GT) - Accuracy
    _, pred_to_gt_dist, _ = trimesh.proximity.closest_point(gt_mesh, pred_points)
    p2s_pred_to_gt = np.mean(pred_to_gt_dist) * 1000 # convert m to mm

    # P2S(GT -> Predicted) - Completeness
    _, gt_to_pred_dist, _ = trimesh.proximity.closest_point(pred_mesh, gt_points)
    p2s_gt_to_pred = np.mean(gt_to_pred_dist) * 1000 # convert m to mm
    
    # --- 4. Normal Consistency ---
    # --- FIX IS HERE ---
    # We need the face indices from the closest point query to get the correct normals.
    # The third return value of closest_point is the index of the closest triangle.
    closest_face_indices = trimesh.proximity.closest_point(gt_mesh, pred_points)[2]
    gt_normals_at_closest_points = gt_mesh.face_normals[closest_face_indices]
    
    cos_similarity = np.clip(np.sum(pred_normals * gt_normals_at_closest_points, axis=1), -1.0, 1.0)
    angular_error = np.rad2deg(np.arccos(cos_similarity))
    normal_consistency = np.mean(angular_error)

    return {
        'Chamfer (mm)': chamfer_dist_val,
        'P2S (Acc., mm)': p2s_pred_to_gt,
        'P2S (Comp., mm)': p2s_gt_to_pred,
        'Normal (deg)': normal_consistency
    }

# --- Main script execution (Can remain unchanged) ---
if __name__ == "__main__":
    # ... (the rest of your script is fine) ...
    parser = argparse.ArgumentParser(description="Evaluate 3D reconstruction meshes from a single directory.")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing final.obj, intermediate.obj, and your gt_aligned.obj')
    args = parser.parse_args()

    # --- Define expected filenames ---
    # This part needs to be flexible for different GT filenames
    try:
        gt_filename = next(f for f in os.listdir(args.input_dir) if 'gt_aligned' in f)
        print(f"Found GT file: {gt_filename}")
    except StopIteration:
        print("Error: Could not find a '*gt_aligned.obj' file in the directory.")
        exit()

    updated_econ_filename = 'ext_econ.obj'
    econ_full_filename = 'econ_full.obj'
    ext_econ_double_filename = 'frame_0007_iter_100.obj'
    # --- Construct full paths ---
    gt_path = os.path.join(args.input_dir, gt_filename)
    updated_econ_path = os.path.join(args.input_dir, updated_econ_filename)
    econ_full_path = os.path.join(args.input_dir, econ_full_filename)
    ext_econ_double_path = os.path.join(args.input_dir, ext_econ_double_filename)
    # Check if all files exist
    if not all(os.path.exists(p) for p in [gt_path, updated_econ_path, econ_full_path, ext_econ_double_path]):
        print("Error: One or more required files are missing from the directory.")
        print(f"Looking for: {gt_filename}, {updated_econ_filename}, {econ_full_filename}")
        print(f"Looking for: {gt_filename}, {updated_econ_filename}, {econ_full_filename}, {ext_econ_double_filename}")
        exit()

    all_results = []
    
    try:
        print("Loading meshes...")
        gt_mesh = trimesh.load(gt_path, process=False)
        updated_econ_mesh = trimesh.load(updated_econ_path, process=False)
        econ_full_mesh = trimesh.load(econ_full_path, process=False)
        ext_econ_double_mesh = trimesh.load(ext_econ_double_path, process=False)
        print("Calculating metrics for after Side Surface Refinement (ext_econ.obj)...")
        updated_econ_metrics = calculate_metrics(updated_econ_mesh, gt_mesh)
        updated_econ_metrics['model'] = 'Ours (exECON)'
        updated_econ_metrics['subject'] = os.path.basename(args.input_dir)

        print("Calculating metrics for Original ECON (econ_full.obj)...")
        econ_full_metrics = calculate_metrics(econ_full_mesh, gt_mesh)
        econ_full_metrics['model'] = 'Original ECON'
        econ_full_metrics['subject'] = os.path.basename(args.input_dir)


        print("Calculating metrics for Double Ext. ECON (exEcon_double.obj)...")
        ext_econ_double_metrics = calculate_metrics(ext_econ_double_mesh, gt_mesh)
        ext_econ_double_metrics['model'] = 'Double View exECON'
        ext_econ_double_metrics['subject'] = os.path.basename(args.input_dir)
        
        all_results.append(updated_econ_metrics)
        all_results.append(econ_full_metrics)        
        all_results.append(ext_econ_double_metrics)

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        # Print more details for debugging
        import traceback
        traceback.print_exc()
        exit()

    if not all_results:
        print("No results were generated. Exiting.")
        exit()

    df = pd.DataFrame(all_results)
    
    print("\n\n" + "="*65)
    print("               EVALUATION RESULTS")
    print("="*65)
    print(df.to_string(index=False))
    print("="*65 + "\n")

    csv_path = os.path.join(args.input_dir, 'evaluation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to {csv_path}")