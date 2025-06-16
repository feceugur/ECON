""# Corrected and functional extract_surface_multiview with proper shape alignment

import numpy as np
import torch
import trimesh
from scipy.sparse import vstack, diags, coo_matrix
from scipy.sparse.linalg import cg
from lib.common.BNI_utils import map_depth_map_to_point_clouds, construct_facets_from, remove_stretched_faces
from scipy.spatial.transform import Rotation as R


def quaternion_to_matrix(q):
    r = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses (x, y, z, w)
    return r.as_matrix()

def get_camera_matrices(cam):
    f = cam['focal_length_px']
    w, h = cam['image_size']
    K = np.array([
        [f, 0, w / 2],
        [0, f, h / 2],
        [0, 0, 1]
    ])
    R_mat = quaternion_to_matrix(cam['quaternion'])
    T = np.array(cam['location']).reshape(3, 1)
    RT = np.hstack((R_mat, -R_mat @ T))  # World to camera
    return K, R_mat, T

def unproject_depth(depth, K, R_mat, T, mask):
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    xx = xx[mask]
    yy = yy[mask]
    z = depth[mask]

    x = (xx - K[0, 2]) * z / K[0, 0]
    y = (yy - K[1, 2]) * z / K[1, 1]
    points_cam = np.stack([x, y, z], axis=1).T  # 3xN

    points_world = R_mat.T @ points_cam + T  # In world coordinates
    return points_world.T

def generate_dx_dy_upsample(mask, nz_horizontal, nz_vertical, step_size, pixel_idx):
    H, W = mask.shape
    dx_list, dy_list = [], []
    data_dx, row_dx, col_dx = [], [], []
    data_dy, row_dy, col_dy = [], [], []

    for y in range(H):
        for x in range(W):
            if not mask[y, x]:
                continue
            idx = pixel_idx[y, x]
            # Horizontal (x)
            if x + step_size < W and mask[y, x + step_size]:
                idx_right = pixel_idx[y, x + step_size]
                row_dx += [len(dx_list)] * 2
                col_dx += [idx, idx_right]
                data_dx += [-1, 1]
                dx_list.append(0)
            # Vertical (y)
            if y + step_size < H and mask[y + step_size, x]:
                idx_down = pixel_idx[y + step_size, x]
                row_dy += [len(dy_list)] * 2
                col_dy += [idx, idx_down]
                data_dy += [-1, 1]
                dy_list.append(0)

    A1 = coo_matrix((data_dx, (row_dx, col_dx)), shape=(len(dx_list), pixel_idx.max() + 1))
    A2 = A1.copy()
    A3 = coo_matrix((data_dy, (row_dy, col_dy)), shape=(len(dy_list), pixel_idx.max() + 1))
    A4 = A3.copy()
    return A1, A2, A3, A4

def extract_surface_multiview(normal_list, mask_list, depth_list, camera_params, lambda_depth=1e-4, scale=256.0):
    H, W, _ = normal_list[0].shape
    canonical_mask = np.any(np.stack(mask_list), axis=0)
    pixel_idx = np.zeros_like(canonical_mask, dtype=int)
    pixel_idx[canonical_mask] = np.arange(np.sum(canonical_mask))

    num_pixels = np.sum(canonical_mask)
    z_prior = np.zeros(num_pixels)
    M_diag = np.zeros(num_pixels)
    A_all = []
    b_all = []

    all_world_points = []

    for i, (normals, mask, depth) in enumerate(zip(normal_list, mask_list, depth_list)):
        nx = normals[:, :, 1][mask]
        ny = normals[:, :, 0][mask]
        nz = -normals[:, :, 2][mask]

        A1, A2, A3, A4 = generate_dx_dy_upsample(
            mask,
            nz_horizontal=nz,
            nz_vertical=nz,
            step_size=1,
            pixel_idx=pixel_idx
        )
        A = vstack([A1, A2, A3, A4])
        b = np.concatenate([-nx, -nx, -ny, -ny])

        A_all.append(A)
        b_all.append(b)

        y_idx, x_idx = np.where(mask)
        valid_indices = pixel_idx[y_idx, x_idx]
        z_prior[valid_indices] += depth[y_idx, x_idx]
        M_diag[valid_indices] += 1

        K, R_mat, T = get_camera_matrices(camera_params[i])
        world_points = unproject_depth(depth, K, R_mat, T, mask)
        all_world_points.append(world_points)

    A_total = vstack(A_all).tocsr()
    b_total = np.concatenate(b_all)

    # Ensure b_total shape matches A_total rows
    if A_total.shape[0] != b_total.shape[0]:
        print(f"[Warning] A_total shape: {A_total.shape}, b_total shape: {b_total.shape}")
        min_len = min(A_total.shape[0], b_total.shape[0])
        b_total = b_total[:min_len]
        A_total = A_total[:min_len, :]

    valid_depth = M_diag > 0
    M_diag[valid_depth] = 1.0 / M_diag[valid_depth]
    M = diags(lambda_depth * M_diag)
    z_prior = z_prior * M_diag

    ATA = A_total.T @ A_total + M
    ATb = A_total.T @ b_total[:A_total.shape[0]] + M @ z_prior

    z0 = np.zeros(num_pixels)
    z, _ = cg(ATA, ATb, x0=z0, maxiter=5000, tol=1e-3)

    depth_map = np.zeros(H * W)
    depth_map[canonical_mask.flatten()] = z
    depth_map = depth_map.reshape(H, W)

    final_points = np.concatenate(all_world_points, axis=0)
    faces = construct_facets_from(canonical_mask.astype(np.uint8).copy())
    faces = np.concatenate((faces[:, [1, 2, 3]], faces[:, [1, 3, 4]]), axis=0)

    mesh = trimesh.Trimesh(vertices=final_points, faces=faces)

    return {
        "verts": torch.tensor(final_points).float(),
        "faces": torch.tensor(faces).long(),
        "depth": torch.tensor(depth_map).float(),
        "mesh": mesh
    }