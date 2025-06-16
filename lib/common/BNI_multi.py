import os
import os.path as osp

import cupy as cp
import cv2
import numpy as np
import torch
import trimesh
from cupyx.scipy.sparse import (
    coo_matrix,
    csr_matrix,
    diags,
    hstack,
    spdiags,
    vstack,
)
from cupyx.scipy.sparse.linalg import cg
from PIL import Image
from tqdm.auto import tqdm
import time
import pyvista as pv


from lib.dataset.mesh_util import clean_floats


def find_max_list(lst):
    list_len = [len(i) for i in lst]
    max_id = np.argmax(np.array(list_len))
    return lst[max_id]


def interpolate_pts(pts, diff_ids):

    pts_extend = np.around((pts[diff_ids] + pts[diff_ids - 1]) * 0.5).astype(np.int32)
    pts = np.insert(pts, diff_ids, pts_extend, axis=0)

    return pts


def align_pts(pts1, pts2):

    diff_num = abs(len(pts1) - len(pts2))
    diff_ids = np.sort(np.random.choice(min(len(pts2), len(pts1)), diff_num, replace=True))

    if len(pts1) > len(pts2):
        pts2 = interpolate_pts(pts2, diff_ids)
    elif len(pts2) > len(pts1):
        pts1 = interpolate_pts(pts1, diff_ids)
    else:
        pass

    return pts1, pts2


def repeat_pts(pts1, pts2):

    coverage_mask = ((pts1[:, None, :] == pts2[None, :, :]).sum(axis=2) == 2.).any(axis=1)

    return coverage_mask


def find_contour(mask, method='all'):

    if method == 'all':

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

    contour_cloth = np.array(find_max_list(contours))[:, 0, :]

    return contour_cloth


def mean_value_cordinates(inner_pts, contour_pts):

    body_edges_a = np.sqrt(((inner_pts[:, None] - contour_pts[None, :])**2).sum(axis=2))
    body_edges_c = np.roll(body_edges_a, shift=-1, axis=1)
    body_edges_b = np.sqrt(((contour_pts - np.roll(contour_pts, shift=-1, axis=0))**2).sum(axis=1))

    body_edges = np.concatenate([
        body_edges_a[..., None], body_edges_c[..., None],
        np.repeat(body_edges_b[None, :, None], axis=0, repeats=len(inner_pts))
    ],
                                axis=-1)

    body_cos = (body_edges[:, :, 0]**2 + body_edges[:, :, 1]**2 -
                body_edges[:, :, 2]**2) / (2 * body_edges[:, :, 0] * body_edges[:, :, 1])
    body_tan_half = np.sqrt(
        (1. - np.clip(body_cos, a_max=1., a_min=-1.)) / np.clip(1. + body_cos, 1e-6, 2.)
    )

    w = (body_tan_half + np.roll(body_tan_half, shift=1, axis=1)) / body_edges_a
    w /= w.sum(axis=1, keepdims=True)

    return w


def get_dst_mat(contour_body, contour_cloth):

    dst_mat = ((contour_body[:, None, :] - contour_cloth[None, :, :])**2).sum(axis=2)

    return dst_mat


def dispCorres(img_size, contour1, contour2, phi, dir_path):

    contour1 = contour1[None, :, None, :].astype(np.int32)
    contour2 = contour2[None, :, None, :].astype(np.int32)

    disp = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cv2.drawContours(disp, contour1, -1, (0, 255, 0), 1)    # green
    cv2.drawContours(disp, contour2, -1, (255, 0, 0), 1)    # blue

    for i in range(contour1.shape[1]):    # do not show all the points when display
        # cv2.circle(disp, (contour1[0, i, 0, 0], contour1[0, i, 0, 1]), 1,
        #            (255, 0, 0), -1)
        corresPoint = contour2[0, phi[i], 0]
        # cv2.circle(disp, (corresPoint[0], corresPoint[1]), 1, (0, 255, 0), -1)
        cv2.line(
            disp, (contour1[0, i, 0, 0], contour1[0, i, 0, 1]), (corresPoint[0], corresPoint[1]),
            (255, 255, 255), 1
        )

    cv2.imwrite(osp.join(dir_path, "corres.png"), disp)


def remove_stretched_faces(verts, faces):

    mesh = trimesh.Trimesh(verts, faces)
    camera_ray = np.array([0.0, 0.0, 1.0])
    faces_cam_angles = np.dot(mesh.face_normals, camera_ray)

    # cos(90-20)=0.34 cos(90-10)=0.17, 10~20 degree
    faces_mask = np.abs(faces_cam_angles) > 2e-1

    mesh.update_faces(faces_mask)
    mesh.remove_unreferenced_vertices()

    return mesh.vertices, mesh.faces


def tensor2arr(t, mask=False):
    if not mask:
        return t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    else:
        mask = t.squeeze(0).abs().sum(dim=0, keepdim=True)
        return (mask != mask[:, 0, 0]).float().squeeze(0).detach().cpu().numpy()


def arr2png(t):
    return ((t + 1.0) * 0.5 * 255.0).astype(np.uint8)


def depth2arr(t):
    """
    Convert depth data to numpy array, handling both PyTorch tensors and numpy arrays.
    
    Args:
        t: Input depth data (PyTorch tensor or numpy array)
    Returns:
        numpy.ndarray: Depth data as numpy array
    """
    if isinstance(t, torch.Tensor):
        return t.float().detach().cpu().numpy()
    elif isinstance(t, np.ndarray):
        return t.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for depth data: {type(t)}")


def depth2png(t):

    t_copy = t.copy()
    t_bg = t_copy[0, 0]
    valid_region = np.logical_and(t > -1.0, t != t_bg)
    t_copy[valid_region] -= t_copy[valid_region].min()
    t_copy[valid_region] /= t_copy[valid_region].max()
    t_copy[valid_region] = (1. - t_copy[valid_region]) * 255.0
    t_copy[~valid_region] = 0.0

    return t_copy[..., None].astype(np.uint8)


def verts_transform(t, depth_scale):

    t_copy = t.clone()
    t_copy *= depth_scale * 0.5
    t_copy += depth_scale * 0.5
    t_copy = t_copy[:, [1, 0, 2]] * torch.Tensor([2.0, 2.0, -2.0]) + torch.Tensor([
        0.0, 0.0, depth_scale
    ])

    return t_copy


def verts_inverse_transform(t, depth_scale):

    t_copy = t.clone()
    t_copy -= torch.tensor([0.0, 0.0, depth_scale])
    t_copy /= torch.tensor([2.0, 2.0, -2.0])
    t_copy -= depth_scale * 0.5
    t_copy /= depth_scale * 0.5
    t_copy = t_copy[:, [1, 0, 2]]

    return t_copy


def depth_inverse_transform(t, depth_scale):

    t_copy = t.clone()
    t_copy -= torch.tensor(depth_scale)
    t_copy /= torch.tensor(-2.0)
    t_copy -= depth_scale * 0.5
    t_copy /= depth_scale * 0.5

    return t_copy


# BNI related


def move_left(mask):
    return cp.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]


def move_right(mask):
    return cp.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]


def move_top(mask):
    return cp.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]


def move_bottom(mask):
    return cp.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]


def move_top_left(mask):
    return cp.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)[1:, 1:]


def move_top_right(mask):
    return cp.pad(mask, ((0, 1), (1, 0)), "constant", constant_values=0)[1:, :-1]


def move_bottom_left(mask):
    return cp.pad(mask, ((1, 0), (0, 1)), "constant", constant_values=0)[:-1, 1:]


def move_bottom_right(mask):
    return cp.pad(mask, ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]


def generate_dx_dy_new(mask, nz_horizontal, nz_vertical, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    num_pixel = cp.sum(mask)

    pixel_idx = cp.zeros_like(mask, dtype=int)
    pixel_idx[mask] = cp.arange(num_pixel)

    has_left_mask = cp.logical_and(move_right(mask), mask)
    has_right_mask = cp.logical_and(move_left(mask), mask)
    has_bottom_mask = cp.logical_and(move_top(mask), mask)
    has_top_mask = cp.logical_and(move_bottom(mask), mask)

    nz_left = nz_horizontal[has_left_mask[mask]]
    nz_right = nz_horizontal[has_right_mask[mask]]
    nz_top = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    data = cp.stack([-nz_left / step_size, nz_left / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_right / step_size, nz_right / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_top / step_size, nz_top / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_bottom / step_size, nz_bottom / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def generate_dx_dy(mask, nz_horizontal, nz_vertical, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    num_pixel = cp.sum(mask)

    pixel_idx = cp.zeros_like(mask, dtype=int)
    pixel_idx[mask] = cp.arange(num_pixel)

    has_left_mask = cp.logical_and(move_right(mask), mask)
    has_right_mask = cp.logical_and(move_left(mask), mask)
    has_bottom_mask = cp.logical_and(move_top(mask), mask)
    has_top_mask = cp.logical_and(move_bottom(mask), mask)

    nz_left = nz_horizontal[has_left_mask[mask]]
    nz_right = nz_horizontal[has_right_mask[mask]]
    nz_top = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    data = cp.stack([-nz_left / step_size, nz_left / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_right / step_size, nz_right / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_top / step_size, nz_top / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_bottom / step_size, nz_bottom / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def construct_facets_from(mask):
    idx = cp.zeros_like(mask, dtype=int)
    idx[mask] = cp.arange(cp.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)
    facet_top_left_mask = (
        facet_move_top_mask * facet_move_left_mask * facet_move_top_left_mask * mask
    )
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    return cp.hstack((
        4 * cp.ones((cp.sum(facet_top_left_mask).item(), 1)),
        idx[facet_top_left_mask][:, None],
        idx[facet_bottom_left_mask][:, None],
        idx[facet_bottom_right_mask][:, None],
        idx[facet_top_right_mask][:, None],
    )).astype(int)


def map_depth_map_to_point_clouds(depth_map, mask, K=None, step_size=1):
    # y
    # |  z
    # | /
    # |/
    # o ---x
    H, W = mask.shape
    yy, xx = cp.meshgrid(cp.arange(W), cp.arange(H))
    xx = cp.flip(xx, axis=0)

    if K is None:
        vertices = cp.zeros((H, W, 3))
        vertices[..., 0] = xx * step_size
        vertices[..., 1] = yy * step_size
        vertices[..., 2] = depth_map
        vertices = vertices[mask]
    else:
        u = cp.zeros((H, W, 3))
        u[..., 0] = xx
        u[..., 1] = yy
        u[..., 2] = 1
        u = u[mask].T    # 3 x m
        vertices = (cp.linalg.inv(K) @ u).T * depth_map[mask, cp.newaxis]    # m x 3

    return vertices


def sigmoid(x, k=1):
    return 1 / (1 + cp.exp(-k * x))


def boundary_excluded_mask(mask):
    top_mask = cp.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    bottom_mask = cp.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    left_mask = cp.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    right_mask = cp.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    be_mask = top_mask * bottom_mask * left_mask * right_mask * mask

    # discard single point
    top_mask = cp.pad(be_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    bottom_mask = cp.pad(be_mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    left_mask = cp.pad(be_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    right_mask = cp.pad(be_mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    bes_mask = (top_mask + bottom_mask + left_mask + right_mask).astype(bool)
    be_mask = cp.logical_and(be_mask, bes_mask)
    return be_mask


def create_boundary_matrix(mask):
    num_pixel = cp.sum(mask)
    pixel_idx = cp.zeros_like(mask, dtype=int)
    pixel_idx[mask] = cp.arange(num_pixel)

    be_mask = boundary_excluded_mask(mask)
    boundary_mask = cp.logical_xor(be_mask, mask)
    diag_data_term = boundary_mask[mask].astype(int)
    B = diags(diag_data_term)

    num_boundary_pixel = cp.sum(boundary_mask).item()
    data_term = cp.concatenate((cp.ones(num_boundary_pixel), -cp.ones(num_boundary_pixel)))
    row_idx = cp.tile(cp.arange(num_boundary_pixel), 2)
    col_idx = cp.concatenate((pixel_idx[boundary_mask], pixel_idx[boundary_mask] + num_pixel))
    B_full = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_boundary_pixel, 2 * num_pixel))
    return B, B_full


def double_side_bilateral_normal_integration(
    normal_list,
    mask_list,
    depth_list,
    k=2,
    lambda_normal_back=1,
    lambda_depth_front=1e-4,
    lambda_depth_back=1e-2,
    lambda_boundary_consistency=1,
    step_size=1,
    max_iter=150,
    tol=1e-4,
    cg_max_iter=5000,
    cg_tol=1e-3,
    cut_intersection=True,
):
    """
    Extract surface from multiple views using bilateral normal integration.
    
    Args:
        normal_list (list): List of normal maps from different views
        mask_list (list): List of masks from different views
        depth_list (list): List of depth maps from different views
        k (int): Sigmoid parameter
        lambda_normal_back (float): Weight for normal consistency between views
        lambda_depth_front (float): Weight for depth consistency with front view
        lambda_depth_back (float): Weight for depth consistency with back view
        lambda_boundary_consistency (float): Weight for boundary consistency
        step_size (int): Step size for generating derivative matrices
        max_iter (int): Maximum number of iterations for solving the linear system
        tol (float): Tolerance for convergence
        cg_max_iter (int): Maximum number of iterations for conjugate gradient solver
        cg_tol (float): Tolerance for conjugate gradient solver
        cut_intersection (bool): Whether to cut the intersection between views
    """
    # Convert all inputs to GPU arrays
    normal_maps = [cp.asarray(normal) for normal in normal_list]
    masks = [cp.asarray(mask) for mask in mask_list]
    depth_maps = [cp.asarray(depth) for depth in depth_list]
    
    # Count total number of valid normals across all views
    num_normals = int(cp.sum(cp.stack([cp.sum(mask) for mask in masks])))
    
    # Initialize lists to store view-specific data
    A_all = []
    b_all = []
    z_prior = cp.zeros(num_normals)
    M_diag = cp.zeros(num_normals)
    
    # Process each view
    for normals, mask, depth in zip(normal_maps, masks, depth_maps):
        # Extract normal components
        nx = normals[:, :, 1][mask]
        ny = normals[:, :, 0][mask]
        nz = -normals[:, :, 2][mask]
        
        # Generate derivative matrices
        A1, A2, A3, A4 = generate_dx_dy(mask, nz_horizontal=nz, nz_vertical=nz, step_size=step_size)
        A = vstack([A1, A2, A3, A4])
        b = cp.concatenate([-nx, -nx, -ny, -ny])
        
        A_all.append(A)
        b_all.append(b)
        
        # Update depth prior
        flat_mask = mask.flatten()
        z_prior[flat_mask] += depth.flatten()[flat_mask]
        M_diag[flat_mask] += 1
    
    # Combine all views
    A_total = vstack(A_all)
    b_total = cp.concatenate(b_all)
    
    # Normalize depth prior
    valid_depth = M_diag > 0
    M_diag[valid_depth] = 1.0 / M_diag[valid_depth]
    M = diags(lambda_normal_back * M_diag)
    z_prior = z_prior * M_diag
    
    # Initialize energy minimization
    z = cp.zeros(num_normals)
    energy_list = []
    
    # Create boundary matrices for each view
    B_mats = []
    for mask in masks:
        B, B_full = create_boundary_matrix(mask)
        B_mats.append(lambda_boundary_consistency * coo_matrix(B_full.get().T @ B_full.get()))
    
    # Combine boundary matrices
    B_total = vstack([hstack([B_mat, csr_matrix((B_mat.shape[0], B_mat.shape[1]))]) for B_mat in B_mats])
    
    # Energy minimization loop
    for i in range(max_iter):
        # Compute energy terms
        normal_energy = (A_total @ z - b_total).T @ (A_total @ z - b_total)
        depth_energy = (z - z_prior).T @ M @ (z - z_prior)
        boundary_energy = z.T @ B_total @ z
        
        # Total energy
        energy = normal_energy + lambda_depth_front * depth_energy + boundary_energy
        energy_list.append(energy)
        
        # Compute system matrices
        ATA = A_total.T @ A_total + lambda_depth_front * M + B_total
        ATb = A_total.T @ b_total + lambda_depth_front * M @ z_prior
        
        # Solve linear system with preconditioner
        D = spdiags(1 / cp.clip(ATA.diagonal(), 1e-5, None), 0, num_normals, num_normals, "csr")
        z_new, _ = cg(ATA, ATb, M=D, x0=z, maxiter=cg_max_iter, tol=cg_tol)
        
        # Check convergence
        if i > 0:
            relative_energy = cp.abs(energy - energy_list[-2]) / energy_list[-2]
            if relative_energy < tol:
                break
        
        z = z_new
    
    # Reshape depth map to original dimensions
    H, W = masks[0].shape
    depth_map = z.reshape(H, W)
    
    # Combine all masks for final mesh creation
    mask_total = cp.any(cp.stack(masks), axis=0)
    
    # Handle intersection cutting if enabled
    if cut_intersection:
        # For multi-view case, we need to handle intersections between all views
        for i in range(len(depth_maps)):
            for j in range(i + 1, len(depth_maps)):
                # Compare depth maps and cut intersections
                intersection_mask = depth_maps[i] >= depth_maps[j]
                mask_total[intersection_mask] = False
                depth_map[~mask_total] = cp.nan
    
    # Create point clouds from depth maps
    vertices = cp.asnumpy(map_depth_map_to_point_clouds(depth_map, mask_total, K=None, step_size=step_size))
    
    # Construct facets and faces
    facets = cp.asnumpy(construct_facets_from(mask_total))
    faces = np.concatenate((facets[:, [1, 2, 3]], facets[:, [1, 3, 4]]), axis=0)
    
    # Remove stretched faces
    vertices, faces = remove_stretched_faces(vertices, faces)
    
    # Create and clean mesh
    mesh = clean_floats(trimesh.Trimesh(vertices, faces))
    mesh.export("mesh.obj")
    
    return {
        "depth_map": depth_map,
        "vertices": vertices,
        "faces": faces,
        "mask": mask_total,
        "energy_list": energy_list,
        "mesh": mesh
    }


def save_normal_tensor_upt(in_tensor_f, in_tensor_b, idx, png_path, thickness=0.0):

    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    normal_F_arr = tensor2arr(in_tensor_f["normal_F"][idx:idx + 1])
    normal_B_arr = tensor2arr(in_tensor_b["normal_F"][idx:idx + 1])
    mask_normal_arr = tensor2arr(in_tensor_f["image"][idx:idx + 1], True)

    depth_F_arr = depth2arr(in_tensor_f["depth_F"][idx])
    depth_B_arr = depth2arr(in_tensor_b["depth_F"][idx])

    BNI_dict = {}

    # clothed human
    BNI_dict["normal_F"] = normal_F_arr
    BNI_dict["normal_B"] = normal_B_arr
    BNI_dict["mask"] = mask_normal_arr > 0.
    BNI_dict["depth_F"] = depth_F_arr - 100. - thickness
    BNI_dict["depth_B"] = 100. - depth_B_arr + thickness
    # BNI_dict["depth_B"] = depth_B_arr - 100. - thickness
    BNI_dict["depth_mask"] = depth_F_arr != -1.0

    np.save(png_path + ".npy", BNI_dict, allow_pickle=True)

    return BNI_dict

def save_normal_tensor(in_tensor, idx, png_path, thickness=0.0):

    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    normal_F_arr = tensor2arr(in_tensor["normal_F"][idx:idx + 1])
    normal_B_arr = tensor2arr(in_tensor["normal_B"][idx:idx + 1])
    mask_normal_arr = tensor2arr(in_tensor["image"][idx:idx + 1], True)

    depth_F_arr = depth2arr(in_tensor["depth_F"][idx])
    # depth_F_arr = depth2arr(in_tensor["depth_map"][idx])
    depth_B_arr = depth2arr(in_tensor["depth_B"][idx])

    BNI_dict = {}

    # clothed human
    BNI_dict["normal_F"] = normal_F_arr
    BNI_dict["normal_B"] = normal_B_arr
    BNI_dict["mask"] = mask_normal_arr > 0.
    BNI_dict["depth_F"] = depth_F_arr - 100. - thickness
    BNI_dict["depth_B"] = 100. - depth_B_arr + thickness
    BNI_dict["depth_mask"] = depth_F_arr != -1.0

    np.save(png_path + ".npy", BNI_dict, allow_pickle=True)

    return BNI_dict


def save_normal_tensor_multi(in_tensor, idx, png_path, thickness=0.0):
    """
    Save normal tensors and depth maps for multi-view data.
    
    Args:
        in_tensor (dict): Dictionary containing all input data including view-specific data
        idx (int): Index of the view to process
        png_path (str): Path to save the output files
        thickness (float): Thickness parameter for depth maps
    """
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    # Get view-specific data
    view_key = f"view_{idx}"
    view_data = in_tensor[view_key]

    # Get normal maps from view-specific data
    normal_F_arr = tensor2arr(view_data["normal_F"])
    normal_B_arr = tensor2arr(view_data["T_normal_B"])
    mask_normal_arr = tensor2arr(view_data["image"], True)

    # Get depth maps from view-specific data
    depth_F_arr = depth2arr(view_data["depth_map"])  # Use depth_map from view data
    depth_B_arr = depth2arr(view_data["depth_map"])  # Use same depth map for back

    BNI_dict = {}

    # clothed human
    BNI_dict["normal_F"] = normal_F_arr
    BNI_dict["normal_B"] = normal_B_arr
    BNI_dict["mask"] = mask_normal_arr > 0.
    BNI_dict["depth_F"] = depth_F_arr - 100. - thickness
    BNI_dict["depth_B"] = 100. - depth_B_arr + thickness
    BNI_dict["depth_mask"] = depth_F_arr != -1.0

    # Add view-specific information
    BNI_dict["view_id"] = idx
    BNI_dict["view_name"] = view_data["name"]
    BNI_dict["T_frame_to_target"] = view_data["T_frame_to_target"].cpu().numpy()
    BNI_dict["smpl_verts_view"] = view_data["smpl_verts_view"].cpu().numpy()

    # Add global information
    BNI_dict["smpl_verts"] = in_tensor["smpl_verts"].cpu().numpy()
    BNI_dict["smpl_joints"] = in_tensor["smpl_joints"].cpu().numpy()
    BNI_dict["smpl_landmarks"] = in_tensor["smpl_landmarks"].cpu().numpy()
    BNI_dict["optimed_betas"] = in_tensor["optimed_betas"].cpu().numpy()
    BNI_dict["optimed_pose_mat"] = in_tensor["optimed_pose_mat"].cpu().numpy()
    BNI_dict["optimed_orient_mat"] = in_tensor["optimed_orient_mat"].cpu().numpy()
    BNI_dict["optimed_trans"] = in_tensor["optimed_trans"].cpu().numpy()

    # Save the dictionary
    np.save(png_path + ".npy", BNI_dict, allow_pickle=True)

    return BNI_dict


