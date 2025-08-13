import os
import numpy as np
import trimesh
import open3d as o3d
import csv
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def load_mesh_as_points_and_normals(file_path, n_points=30000):
    mesh = trimesh.load(file_path, force='mesh')
    if mesh.is_empty or len(mesh.faces) == 0:
        return None, None
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces))
    mesh_o3d.compute_vertex_normals()
    try:
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_points)
    except RuntimeError:
        return None, None
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return points, normals

def chamfer_distance(p1, p2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(p1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(p2)
    d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    return np.mean(d1) * 1000, np.mean(d2) * 1000  # mm

def point_to_surface_distance(points_src, mesh_tgt):
    mesh = o3d.io.read_triangle_mesh(mesh_tgt)
    mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query_points = o3d.core.Tensor(points_src, dtype=o3d.core.Dtype.Float32)
    dists = scene.compute_distance(query_points).numpy()
    return np.mean(dists) * 1000  # mm

def matched_normal_error(points_pred, normals_pred, gt_mesh_file):
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_file)
    gt_mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt_mesh))
    query_points = o3d.core.Tensor(points_pred, dtype=o3d.core.Dtype.Float32)
    closest_points = scene.compute_closest_points(query_points)

    face_ids = closest_points['primitive_ids'].numpy()
    bary_coords = closest_points['primitive_uvs'].numpy()
    gt_faces = np.asarray(gt_mesh.triangles)
    gt_vertex_normals = np.asarray(gt_mesh.vertex_normals)

    matched_normals_gt = []
    for fid, bc in zip(face_ids, bary_coords):
        if fid < 0:
            matched_normals_gt.append([0.0, 0.0, 1.0])
            continue
        v_ids = gt_faces[fid]
        n0, n1, n2 = gt_vertex_normals[v_ids]
        if len(bc) < 3:
            bc = np.pad(bc, (0, 3 - len(bc)), 'constant', constant_values=0)
            bc[2] = 1.0 - bc[0] - bc[1]
        interp_normal = bc[0] * n0 + bc[1] * n1 + bc[2] * n2
        matched_normals_gt.append(interp_normal)

    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=1, keepdims=True)
    matched_normals_gt = np.array(matched_normals_gt)
    matched_normals_gt = matched_normals_gt / np.linalg.norm(matched_normals_gt, axis=1, keepdims=True)

    # Mask out NaNs or invalid normals
    valid_mask = (
        np.isfinite(normals_pred).all(axis=1) &
        np.isfinite(matched_normals_gt).all(axis=1)
    )
    normals_pred = normals_pred[valid_mask]
    matched_normals_gt = matched_normals_gt[valid_mask]

    if len(normals_pred) == 0:
        return np.nan  # No valid normals

    dot = np.clip(np.sum(normals_pred * matched_normals_gt, axis=1), -1.0, 1.0)
    return np.mean(np.degrees(np.arccos(dot)))

def evaluate_pair(gt_file, pred_file):
    p_gt, _ = load_mesh_as_points_and_normals(gt_file)
    p_pred, n_pred = load_mesh_as_points_and_normals(pred_file)
    if p_gt is None or p_pred is None:
        return None
    cd1, cd2 = chamfer_distance(p_pred, p_gt)
    chamfer = (cd1 + cd2) / 2.0
    acc = point_to_surface_distance(p_pred, gt_file)
    comp = point_to_surface_distance(p_gt, pred_file)
    normal_error = matched_normal_error(p_pred, n_pred, gt_file)
    return chamfer, acc, comp, normal_error

def process_subject(args):
    subject, subject_path, pred_filename = args
    gt_file = os.path.join(subject_path, f"{subject}_gt.obj")
    pred_file = os.path.join(subject_path, pred_filename)
    if not os.path.exists(gt_file) or not os.path.exists(pred_file):
        return None
    metrics = evaluate_pair(gt_file, pred_file)
    if metrics is None:
        return None
    chamfer, acc, comp, normal = metrics
    return [subject, chamfer, acc, comp, normal]

def evaluate_all(base_dir, pred_filename="econ.obj", output_csv="evaluation_results_econ.csv"):
    subjects = [(s, os.path.join(base_dir, s), pred_filename)
                for s in sorted(os.listdir(base_dir))
                if os.path.isdir(os.path.join(base_dir, s))]

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_subject, subjects), total=len(subjects)))

    results = [r for r in results if r is not None]
    if not results:
        print("No valid subjects evaluated. Exiting.")
        return

    metrics_array = np.array([r[1:] for r in results], dtype=np.float64)
    mean_metrics = np.nanmean(metrics_array, axis=0)  # NaN-safe mean
    results.append(["AVERAGE", *mean_metrics])

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Chamfer (mm)", "P2S Accuracy (mm)",
                         "P2S Completeness (mm)", "Normal Error (deg)"])
        writer.writerows(results)

    print(f"\n✅ Saved results to {output_csv}")
    print(f"Averages: Chamfer={mean_metrics[0]:.2f} mm | Accuracy={mean_metrics[1]:.2f} mm | "
          f"Completeness={mean_metrics[2]:.2f} mm | Normal={mean_metrics[3]:.2f}°")

if __name__ == "__main__":
    base_dir = "../MExEcon_Outputs"
    evaluate_all(base_dir, pred_filename="econ.obj")
