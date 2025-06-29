import os
import os.path as osp
from typing import Optional

from apps.multi_view_d_bini import Camera
from apps.transform_normals import transform_normals
import cupy as cp
import cv2
import numpy as np
import torch
import trimesh
from cupyx.scipy.sparse import (
    coo_matrix,
    csr_matrix,
    csc_matrix,
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
from typing import List, Tuple, Sequence, Optional, Dict, Any

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

def build_per_view_blocks(
    normal: cp.ndarray,
    normal_mask: cp.ndarray,
    depth: Optional[cp.ndarray],
    depth_mask: Optional[cp.ndarray],
    *,
    k: float,
    lambda_normal_back: float,
    lambda_depth_front: float,
    lambda_depth_back: float,
    lambda_boundary_consistency: float,
    step_size: int,
    # state that we carry between IRLS iterations
    z_front_init: Optional[cp.ndarray] = None,
    z_back_init: Optional[cp.ndarray] = None,
):
    # ------------------------
    normal_mask = np.squeeze(normal_mask)
    assert normal_mask.ndim == 2, f"normal_mask shape is {normal_mask.shape}, expected 2D"
    num_normals = cp.sum(normal_mask).item()

    # --- normals ---------------------------------------------------------
    if normal.shape[0] == 3:
        normal = np.transpose(normal, (1, 2, 0))
    nx = normal[normal_mask, 1]
    ny = normal[normal_mask, 0]
    nz = -normal[normal_mask, 2]

    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_horizontal=nz, nz_vertical=nz, step_size=step_size)

    A_data = vstack((A1, A2, A3, A4))
    b = cp.concatenate((-nx, -nx, -ny, -ny))

    # --- W ----------------------------------------------------------------
    W = 0.5 * cp.ones(4 * num_normals)
    W = diags(W)

    # --- M and depth priors ---------------------------------------------
    if depth_mask is not None:
        depth_mask_flat = depth_mask[normal_mask].astype(bool)
        z_prior = depth[normal_mask]
        z_prior[~depth_mask_flat] = 0
        m = depth_mask_flat.astype(int)
        M = diags(m)
    else:
        z_prior = cp.zeros(num_normals)
        M = diags(cp.zeros(num_normals))

    # --- boundary consistency matrix ------------------------------------
    B, B_full = create_boundary_matrix(normal_mask)
    B = lambda_boundary_consistency * B 
    
    # Compute boundary masks for this view
    has_left_mask = cp.logical_and(move_right(normal_mask), normal_mask)
    has_right_mask = cp.logical_and(move_left(normal_mask), normal_mask)
    has_bottom_mask = cp.logical_and(move_top(normal_mask), normal_mask)
    has_top_mask = cp.logical_and(move_bottom(normal_mask), normal_mask)
    
    # Get boundary masks (where we don't have neighbors)
    top_boundary_mask = cp.logical_xor(has_top_mask, normal_mask)[normal_mask]
    bottom_boundary_mask = cp.logical_xor(has_bottom_mask, normal_mask)[normal_mask]
    left_boundary_mask = cp.logical_xor(has_left_mask, normal_mask)[normal_mask]
    right_boundary_mask = cp.logical_xor(has_right_mask, normal_mask)[normal_mask]
    

    return {
        "A_data": A_data,
        "b": b,
        "W": W,
        "B": B,
        "M": M,
        "z_prior": z_prior,
        "num_normals": int(num_normals),
        "A1": A1,
        "A2": A2,
        "A3": A3,
        "A4": A4,
        "top_boundary_mask": top_boundary_mask,
        "bottom_boundary_mask": bottom_boundary_mask,
        "left_boundary_mask": left_boundary_mask,
        "right_boundary_mask": right_boundary_mask,
    }

def _pad(mat, col_offset: int, total_cols: int) -> csr_matrix:
    mat = mat.tocsr(copy=False)
    n = mat.shape[1]
    pad_left  = int(col_offset)
    pad_right = int(total_cols - col_offset - n)
    if pad_left < 0 or pad_right < 0:
        raise ValueError("padding out of range (left={pad_left}, right={pad_right}, total={total_cols})")
    if pad_left == 0 and pad_right == 0:
        return mat
    left  = csr_matrix((mat.shape[0], pad_left),  dtype=mat.dtype)
    right = csr_matrix((mat.shape[0], pad_right), dtype=mat.dtype)
    return hstack([left, mat, right]).tocsr()

def block_diag(blocks):
    """Create a block diagonal matrix from a sequence of sparse matrices.
    This implementation works directly with sparse matrices to avoid memory issues.
    """
    if not blocks:
        return csr_matrix((0, 0))
    
    # Get dimensions of each block
    rows = [b.shape[0] for b in blocks]
    cols = [b.shape[1] for b in blocks]
    
    # Calculate total dimensions
    total_rows = sum(rows)
    total_cols = sum(cols)
    
    # Create arrays for COO format
    data = []
    row_indices = []
    col_indices = []
    
    # Track current position
    row_offset = 0
    col_offset = 0
    
    # Add each block's data
    for block in blocks:
        if not isinstance(block, (csr_matrix, csc_matrix, coo_matrix)):
            block = csr_matrix(block)
        
        # Convert to COO format for easy manipulation
        coo = block.tocoo()
        
        # Add data with appropriate offsets
        data.extend(coo.data.get())  # Convert CuPy array to numpy array
        row_indices.extend(coo.row.get() + row_offset)  # Convert CuPy array to numpy array
        col_indices.extend(coo.col.get() + col_offset)  # Convert CuPy array to numpy array
        
        # Update offsets
        row_offset += block.shape[0]
        col_offset += block.shape[1]
    
    # Create the final sparse matrix
    return coo_matrix((cp.array(data), (cp.array(row_indices), cp.array(col_indices))), 
                     shape=(total_rows, total_cols)).tocsr()


def apply_blender2cv_to_smpl(smpl_verts: np.ndarray) -> np.ndarray:
    """Flip Y and Z on every vertex to match our camera convention."""
    # maps (X, Y, Z) -> ( X, -Z,  Y )
    B2C = np.array([
      [0, 0,  -1],
      [-1, 0, 0],
      [0, 1,  0],
    ], dtype=np.float32)
    return (B2C @ smpl_verts.T).T  # (N,3)

def build_sequential_cross_view_matrix(
    num_views: int,
    offsets: np.ndarray,
    cameras: Sequence[Camera],
    masks: Sequence[np.ndarray],
    smpl_vertices: np.ndarray,
    min_views: int = 2
) -> Tuple[csr_matrix, cp.ndarray]:
    """
    Build cross-view consistency matrix only for sequential views (i.e., view i with view i+1)
    using SMPL vertex correspondences.
    """
    # Apply coordinate transformation
    smpl_vertices_transformed = apply_blender2cv_to_smpl(smpl_vertices)
    
    # Get image dimensions
    H, W = masks[0].shape if masks[0].ndim == 2 else masks[0].squeeze().shape
    
    # Build pixel to flat index mapping for each mask
    pix2flat = []
    for m in masks:
        m2 = np.squeeze(m)
        idx = np.zeros_like(m2, dtype=np.int32)
        idx[m2] = np.arange(m2.sum(), dtype=np.int32)
        pix2flat.append(idx)
    
    rows, cols, data, rhs = [], [], [], []
    total_correspondences = 0
    
    # Only connect sequential views
    for v in range(num_views - 1):
        v_next = v + 1
        
        # Get correspondences between view v and v+1 using SMPL vertices
        sequential_correspondences = []
        
        for P in smpl_vertices_transformed:  # P is (3,) world-space point
            visible_points = []  # will collect (view_idx, flat_idx)
            
            # Check visibility in current view (v) and next view (v+1)
            for view_idx in [v, v_next]:
                cam = cameras[view_idx]
                mask = masks[view_idx]
                mask2 = np.squeeze(mask)
                
                # Extract camera parameters
                R = cam.R.detach().cpu().numpy() if hasattr(cam.R, 'detach') else np.asarray(cam.R)
                t = cam.t.detach().cpu().numpy() if hasattr(cam.t, 'detach') else np.asarray(cam.t)
                K = cam.K.detach().cpu().numpy() if hasattr(cam.K, 'detach') else np.asarray(cam.K)
                
                # Transform to camera coordinates
                Pc = R @ P + t
                if Pc[2] <= 0:  # behind camera
                    continue
                
                # Project to image coordinates
                uv = K @ Pc
                u_coord, v_coord = uv[0] / uv[2], uv[1] / uv[2]
                ui, vi = int(round(u_coord)), int(round(v_coord))
                
                # Check bounds and mask visibility
                if 0 <= ui < W and 0 <= vi < H and mask2[vi, ui]:
                    flat_idx = int(pix2flat[view_idx][vi, ui])
                    visible_points.append((view_idx, flat_idx))
            
            # If visible in both sequential views, create correspondence
            if len(visible_points) == 2:
                v1, p1 = visible_points[0]
                v2, p2 = visible_points[1]
                # Ensure we have the right order (v, v+1)
                if v1 == v and v2 == v_next:
                    sequential_correspondences.append((p1, p2))
                elif v1 == v_next and v2 == v:
                    sequential_correspondences.append((p2, p1))
        
        # Add constraints for sequential correspondences
        for p1, p2 in sequential_correspondences:
            idx1 = offsets[v] + p1
            idx2 = offsets[v_next] + p2
            
            rows.extend([len(rows), len(rows)])
            cols.extend([idx1, idx2])
            data.extend([1.0, -1.0])
            rhs.append(0.0)  # enforce z_v[p1] = z_v+1[p2]
            total_correspondences += 1
    
    print(f"→ Sequential correspondences: {total_correspondences}")
    
    if len(rows) == 0:
        # Return empty matrices if no correspondences found
        return coo_matrix(([], ([], [])), shape=(0, offsets[-1])).tocsr(), cp.array([])
    
    return coo_matrix((data, (rows, cols)), 
                     shape=(len(rows)//2, offsets[-1])).tocsr(), cp.array(rhs)

def build_cross_view_matrix(
    correspondences: Sequence[Tuple[int, int, int, int]],  # (view1, point1, view2, point2)
    offsets: np.ndarray,
    masks: Sequence[np.ndarray],
) -> csr_matrix:
    """
    Build a sparse matrix that enforces consistency at the boundaries where views overlap.
    """
    rows, cols, data = [], [], []
    
    for v1, p1, v2, p2 in correspondences:
        # Get the indices in the global variable vector
        idx1 = offsets[v1] + p1
        idx2 = offsets[v2] + p2
        
        # Add constraint: z_v1[p1] = z_v2[p2]
        rows.extend([len(rows), len(rows)])
        cols.extend([idx1, idx2])
        data.extend([1.0, -1.0])
    
    return coo_matrix((data, (rows, cols)), 
                     shape=(len(correspondences), offsets[-1])).tocsr()

def calculate_energy(energy, A_data_blocks, b_blocks, W_blocks, z_prior_blocks, M_diag_parts, B_blocks, lambda_depth_front, lambda_boundary_consistency, C_full, rhs_full, lambda_cross_view, num_normals_blocks, total_vars):
    # Create global zero vector for cross-view term
    z_global = cp.zeros(total_vars, dtype=cp.float32)
    
    for i in range(len(num_normals_blocks)):
        # Create zero vector with size equal to the number of normals for this view
        z = cp.zeros(num_normals_blocks[i], dtype=cp.float32)
        energy += (A_data_blocks[i] @ z - b_blocks[i]).T @ W_blocks[i] @ (A_data_blocks[i] @ z - b_blocks[i])
        energy += lambda_depth_front * (z - z_prior_blocks[i]).T @ (M_diag_parts[i] * (z - z_prior_blocks[i]))
        if i == 2 or i == 6:
            energy += lambda_boundary_consistency * (z - z_prior_blocks[i]).T @ B_blocks[i] @ (z - z_prior_blocks[i])
    
    # Cross-view term computed once with global vector
    if C_full is not None:
        energy += lambda_cross_view * (C_full @ z_global - rhs_full).T @ (C_full @ z_global - rhs_full)

    return energy

def double_side_bilateral_normal_integration(
    normal_list,
    normal_mask_list,
    depth_list,
    depth_mask_list,
    pairs_list,
    cameras,
    transform_manager,
    correspondences,
    k=2,
    lambda_normal_back=1,
    lambda_depth_front=1e-4,
    lambda_depth_back=1e-2,
    lambda_boundary_consistency=1,
    lambda_cross_view=0.0,
    step_size=1,
    max_iter=150,
    tol=1e-4,
    cg_max_iter=5000,
    cg_tol=1e-3,
    cut_intersection=True,
    in_tensor_dict=None,  
    use_tensor_correspondences=False,
    depth_threshold=0.1,
    debug_output_dir=None,
):

    # To avoid confusion, we list the coordinate systems in this code as follows
    #
    # pixel coordinates         camera coordinates     normal coordinates (the main paper's Fig. 1 (a))
    # u                          x                              y
    # |                          |  z                           |
    # |                          | /                            o -- x
    # |                          |/                            /
    # o --- v                    o --- y                      z
    # (bottom left)
    #                       (o is the optical center;
    #                        xy-plane is parallel to the image plane;
    #                        +z is the viewing direction.)
    #
    # The input normal map should be defined in the normal coordinates.
    # The camera matrix K should be defined in the camera coordinates.
    # K = [[fx, 0,  cx],
    #      [0,  fy, cy],
    #      [0,  0,  1]]

    num_input_views = len(normal_list)

    per_view = []
    for i in range(num_input_views):
        per_view.append(build_per_view_blocks(
                cp.asarray(normal_list[i]),
                cp.asarray(normal_mask_list[i]),
                None if depth_list[i] is None else cp.asarray(depth_list[i]),
                None if depth_mask_list[i] is None else cp.asarray(depth_mask_list[i]),
                k=k,
                lambda_normal_back=lambda_normal_back,
                lambda_depth_front=lambda_depth_front,
                lambda_depth_back=lambda_depth_front,  # same weight for back priors
                lambda_boundary_consistency=lambda_boundary_consistency,
                step_size=step_size,
            )
        )

    num_normals = np.array([blk["num_normals"] for blk in per_view], dtype=np.int64)

    #offsets to find each view's variables ------------------------------------------------
    offsets = np.zeros(num_input_views, dtype=np.int64)
    cumulative = 0
    for v in range(num_input_views):
        offsets[v] = cumulative
        cumulative += num_normals[v]

    total_vars = cumulative

    A_data_blocks, W_blocks, b_blocks = [], [], []
    B_blocks = []
    B_full_rows = []
    z_prior_blocks = []
    M_diag_parts: List[cp.ndarray] = []
    A1_blocks, A2_blocks, A3_blocks, A4_blocks = [], [], [], []
    top_boundary_mask_blocks, bottom_boundary_mask_blocks, left_boundary_mask_blocks, right_boundary_mask_blocks = [], [], [], []
    num_normals_blocks = []

    for ofst, blk in enumerate(per_view):
        A_data_blocks.append(blk["A_data"])
        W_blocks.append(blk["W"])
        b_blocks.append(blk["b"])
        z_prior_blocks.append(blk["z_prior"])
        M_diag_parts.append(blk["M"].diagonal())
        B_blocks.append(blk["B"])
        B_full_rows.append(blk["B"])
        A1_blocks.append(blk["A1"])
        A2_blocks.append(blk["A2"])
        A3_blocks.append(blk["A3"])
        A4_blocks.append(blk["A4"])
        top_boundary_mask_blocks.append(blk["top_boundary_mask"])
        bottom_boundary_mask_blocks.append(blk["bottom_boundary_mask"])
        left_boundary_mask_blocks.append(blk["left_boundary_mask"])
        right_boundary_mask_blocks.append(blk["right_boundary_mask"])
        num_normals_blocks.append(blk["num_normals"])
    
    """
    A = vstack(A_blocks).tocsr()
    W = block_diag(W_blocks)
    b = cp.concatenate(b_blocks)

    M_big = diags(cp.concatenate(M_diag_parts))
    B_big = vstack(B_full_rows).tocsr()
    z_prior = cp.concatenate(z_prior_blocks)
    #B_mat = lambda_boundary_consistency * coo_matrix(B_big.get().T @ B_big.get()) 
    """

    if lambda_cross_view > 0.0:
        assert cameras is not None and offsets is not None
        
        if use_tensor_correspondences and in_tensor_dict is not None:
            print("Using most direct tensor-based sequential correspondences...")
            C_full, rhs_full = build_sequential_cross_view_matrix_most_direct(
                in_tensor_dict=in_tensor_dict,
                num_views=num_input_views,
                offsets=offsets,
                cameras=cameras,
                use_depth_validation=True,
                depth_threshold=depth_threshold,
                debug_output_dir=debug_output_dir,
                normal_mask_list=normal_mask_list,
                total_vars=total_vars,
            )
        else:
            # Fallback to original method with pre-computed correspondences
            print("Using original correspondence method...")
            assert correspondences is not None, "correspondences must be provided when not using tensor-based method"
            C, rhs_C = build_cross_view_matrix(
                correspondences, offsets, normal_mask_list)
            C_full = C.tocsr()
            rhs_full = rhs_C
    else:
        C_full = None
        rhs_full = None

    energy_list = []
    energy = 0.0
    energy_in = 0.0
    """
    energy = (A @ z - b).T @ W @ (A @ z - b) + \
             lambda_depth_front * (z - z_prior).T @ M_big @ (z - z_prior)
    energy += lambda_boundary_consistency * (z - z_prior).T @ B[0] @ (z - z_prior)
    if C_full is not None:
        energy = energy + lambda_cross_view * (C_full @ z - rhs_full).T @ (C_full @ z - rhs_full)
    """
    energy = calculate_energy(energy, A_data_blocks, b_blocks, W_blocks, 
                              z_prior_blocks, M_diag_parts, B_blocks, 
                              lambda_depth_front, lambda_boundary_consistency, 
                              C_full, rhs_full, lambda_cross_view, num_normals_blocks, total_vars)
    
    """
    depth_map_est_list = []
    for i in range(num_input_views):
        depth_map_est = cp.ones_like(normal_mask_list[i], float) * cp.nan
        depth_map_est_list.append(depth_map_est)
    
    facets_list = [cp.asnumpy(construct_facets_from(normal_mask)) for normal_mask in normal_mask_list]
    faces_list = []
    for i in range(len(facets_list)):
        faces = np.concatenate((facets_list[i][:, [1, 4, 3]], facets_list[i][:, [1, 3, 2]]), axis=0)
        faces_list.append(faces)
    """
    A_mat = [None] * len(num_normals_blocks)
    b_vec = [None] * len(num_normals_blocks)
    z_combined = cp.zeros(total_vars, dtype=cp.float32)
    W_v_blocks = []

    for i in range(max_iter):
        for idx in range(len(num_normals_blocks)):
            A_mat[idx] = A_data_blocks[idx].T @ W_blocks[idx] @ A_data_blocks[idx]
            b_vec[idx] = A_data_blocks[idx].T @ W_blocks[idx] @ b_blocks[idx]

            if C_full is not None:
                C_mat = lambda_cross_view * C_full.T @ C_full
                # Extract the portion of the cross-view result relevant to this view
                start_idx = offsets[idx]
                end_idx = start_idx + num_normals[idx]
                cross_view_rhs = C_mat @ z_combined
                b_vec[idx] += cross_view_rhs[start_idx:end_idx]
                # Extract the relevant block of C_mat for this view
                A_mat[idx] += C_mat[start_idx:end_idx, start_idx:end_idx]

            if depth_mask_list is not None:
                b_vec[idx] += lambda_depth_front * (M_diag_parts[idx] * z_prior_blocks[idx])
                M_diag_sparse = diags(M_diag_parts[idx])
                A_mat[idx] += lambda_depth_front * M_diag_sparse

            if idx == 2 or idx == 6:
                b_vec[idx] += lambda_boundary_consistency * B_blocks[idx] @ z_prior_blocks[idx]
                A_mat[idx] += lambda_boundary_consistency * B_blocks[idx]

            z = z_prior_blocks[idx]

            A_mat_combined = A_mat[idx]
            b_vec_combined = b_vec[idx]
            
            D = spdiags(
                1 / cp.clip(A_mat_combined.diagonal(), 1e-5, None), 0, num_normals_blocks[idx], num_normals_blocks[idx],
                "csr"
            )    # Jacob preconditioner

            # Extract the relevant portion of z_combined for this view
            start_idx = offsets[idx]
            end_idx = start_idx + num_normals[idx]
            z_view_init = z_combined[start_idx:end_idx]
            
            z_view_solution, _ = cg(
                A_mat_combined, b_vec_combined, M=D, x0=z_view_init, maxiter=cg_max_iter, tol=cg_tol
            )
            
            # Update the global z_combined with this view's solution
            z_combined[start_idx:end_idx] = z_view_solution
            z_slice = z_view_solution
            
            # Compute weights for this view using the stored derivative matrices
            wu_v = sigmoid((A2_blocks[idx].dot(z_slice))**2 - (A1_blocks[idx].dot(z_slice))**2, k)  # top
            wv_v = sigmoid((A4_blocks[idx].dot(z_slice))**2 - (A3_blocks[idx].dot(z_slice))**2, k)  # right
            
            # Set boundary weights to 0.5
            wu_v[top_boundary_mask_blocks[idx]] = 0.5
            wu_v[bottom_boundary_mask_blocks[idx]] = 0.5
            wv_v[left_boundary_mask_blocks[idx]] = 0.5
            wv_v[right_boundary_mask_blocks[idx]] = 0.5
            
            # Build weight matrix for this view
            W_v = spdiags(
                cp.concatenate((wu_v, 1 - wu_v, wv_v, 1 - wv_v)),
                0,
                4 * num_normals[idx],
                4 * num_normals[idx],
                format="csr"
            )
            W_v_blocks.append(W_v)
            
        energy_old = energy

        energy = calculate_energy(energy_in, A_data_blocks, b_blocks, W_v_blocks, 
                              z_prior_blocks, M_diag_parts, B_blocks, 
                              lambda_depth_front, lambda_boundary_consistency, 
                              C_full, rhs_full, lambda_cross_view, num_normals_blocks, total_vars)
        
        energy_list.append(energy)
        relative_energy = cp.abs(energy - energy_old) / energy_old
        if relative_energy < tol:
            break


    result = []
    for pair_idx in range(len(pairs_list)):
        idx_f, idx_b = pairs_list[pair_idx]
        front_mask = cp.asarray(normal_mask_list[idx_f])
        back_mask = cp.asarray(normal_mask_list[idx_b])
        
        # Get the depth values for this pair
        start_front = offsets[idx_f]
        end_front = start_front + num_normals[idx_f]
        start_back = offsets[idx_b]
        end_back = start_back + num_normals[idx_b]
        
        z_front = z_combined[start_front:end_front]
        z_back = z_combined[start_back:end_back]
        
        # Create depth maps
        depth_map_front_est = cp.ones_like(front_mask, float) * cp.nan
        depth_map_back_est = cp.ones_like(back_mask, float) * cp.nan
        
        depth_map_front_est[front_mask] = z_front
        depth_map_back_est[back_mask] = z_back

        facets_front = cp.asnumpy(construct_facets_from(front_mask))
        facets_back = cp.asnumpy(construct_facets_from(back_mask))
        faces_front = np.concatenate((facets_front[:, [1, 4, 3]], facets_front[:, [1, 3, 2]]), axis=0)
        faces_back = np.concatenate((facets_back[:, [1, 4, 3]], facets_back[:, [1, 3, 2]]), axis=0)
        
        vertices_front = cp.asnumpy(
            map_depth_map_to_point_clouds(depth_map_front_est, front_mask, K=None, step_size=step_size)
        )
        vertices_back = cp.asnumpy(
            map_depth_map_to_point_clouds(depth_map_back_est, back_mask, K=None, step_size=step_size)
        )
        
        vertices_front, faces_front = remove_stretched_faces(vertices_front, faces_front)
        vertices_back, faces_back = remove_stretched_faces(vertices_back, faces_back)

        #result [{f_verts, f_faces, b_verts, b_faces, f_depth, b_depth}, ...] for each pair

        result_pair = {
            "F_verts": torch.as_tensor(vertices_front).float(),
            "F_faces": torch.as_tensor(faces_front).long(),
            "B_verts": torch.as_tensor(vertices_back).float(),
            "B_faces": torch.as_tensor(faces_back).long(),
            "F_depth": torch.as_tensor(depth_map_front_est).float(),
            "B_depth": torch.as_tensor(depth_map_back_est).float()
        }
        result.append(result_pair)

    return result
    """
    all_world_vertices_list = []
    all_faces_list = []
    bni_mesh_list = []

    T_bcam_from_ocam_3x3 = torch.tensor([
        [0,  0, -1],
        [-1, 0,  0],
        [0,  1,  0],
    ], dtype=torch.float32)

    for v_idx in range(num_input_views):
        # 1. Extract this view's depth map from the solution
        start = offsets[v_idx]
        end = start + num_normals[v_idx]
        z_view = z_combined[start:end]
        mask = cp.asarray(normal_mask_list[v_idx])
        depth_map_est = cp.ones_like(mask, dtype=float) * cp.nan
        depth_map_est[mask] = z_view

        # 2. Generate mesh in this camera's local coordinate system
        vertices_camera = map_depth_map_to_point_clouds(depth_map_est, mask, K=None, step_size=step_size)
        vertices_camera = cp.asnumpy(vertices_camera)
        if vertices_camera.shape[0] == 0: continue

        facets_view = cp.asnumpy(construct_facets_from(mask))
        if facets_view.shape[0] == 0: continue
        faces_view = np.concatenate((facets_view[:, [1, 4, 3]], facets_view[:, [1, 3, 2]]), axis=0)
        vertices_camera, faces_view = remove_stretched_faces(vertices_camera, faces_view)
        if vertices_camera.shape[0] == 0: continue

        # 3. Get the INVERSE transform (camera-to-world) from the 'cameras' object
        cam = cameras[v_idx]
        device = cam.R.device
        R = cam.R.detach().cpu().numpy()
        t = cam.t.detach().cpu().numpy()
        
        # Convert points from OpenCV Camera space to Blender Camera space.
        verts_ocam_torch = torch.from_numpy(vertices_camera).float().to(device)
        verts_bcam_torch = (T_bcam_from_ocam_3x3.to(device) @ verts_ocam_torch.T).T

        # Subtract the camera's internal translation vector 't'.
        # This correctly applies the first part of the inverse: (P_cam - t)
        t_torch = torch.from_numpy(t.flatten()).float().to(device)
        verts_bcam_translated = verts_bcam_torch - t_torch

        # Apply the inverse rotation R.T to the translated points.
        # This completes the formula: P_world = R.T @ (P_cam - t)
        R_world_from_bcam = torch.from_numpy(R.T).float().to(device)
        vertices_world_torch = (R_world_from_bcam @ verts_bcam_translated.T).T
        
        vertices_world = vertices_world_torch.cpu().numpy()
        vertices_world = vertices_world * np.array([1, 1, -1])
        bni_mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces_view)
        bni_mesh.export(os.path.join(debug_output_dir, f"bni_mesh_{v_idx}.ply"))
        print(f"bni_mesh_{v_idx}.ply saved")
        bni_mesh_list.append(bni_mesh)

        # 5. Store the correctly transformed vertices and faces
        all_world_vertices_list.append(vertices_world)
        all_faces_list.append(faces_view)

    # 6. Combine all mesh pieces into a single mesh
    if not all_world_vertices_list:
        print("Warning: No vertices were generated from any view.")
        return []

    final_faces_list = []
    vertex_offset = 0
    for i, vertices_part in enumerate(all_world_vertices_list):
        faces_part = all_faces_list[i]
        if faces_part.shape[0] > 0:
            final_faces_list.append(faces_part + vertex_offset)
        vertex_offset += vertices_part.shape[0]

    final_vertices = np.vstack(all_world_vertices_list)
    final_faces = np.vstack(final_faces_list) if final_faces_list else np.empty((0, 3), dtype=np.int64)

    # 7. Format the final result
    result = {
        "bni_mesh_list": bni_mesh_list,
        "verts": torch.as_tensor(final_vertices).float(),
        "faces": torch.as_tensor(final_faces).long(),
    }
    
    final_trimesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
    final_trimesh.export(os.path.join(debug_output_dir, "final_mesh.ply"))
    print("final_mesh.ply saved")

    return result
    """



def save_normal_tensor_upt(in_tensor_f, in_tensor_b, idx, png_path, thickness=0.0, back_view_idx=0, transform_manager=None):
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
    BNI_dict["depth_F"] = depth_F_arr - 100.
    BNI_dict["depth_B"] = 100. - depth_B_arr
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


def save_normal_tensor_multi(in_tensor, idx, png_path, thickness=10.0):
    """
    Save normal tensors and depth maps for multi-view data and prepare inputs for bilateral normal integration.
    
    Args:
        in_tensor (dict): Dictionary containing all input data including view-specific data
        idx (int): Index of the view to process
        png_path (str): Path to save the output files
        thickness (float): Thickness parameter for depth maps
    """
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    # Initialize lists for multi-view integration
    normal_list = []
    mask_list = []
    depth_list = []
    depth_mask_list = []
    # Process all views
    
    for view_idx in range(len([k for k in in_tensor.keys() if k.startswith('view_')])):
        view_key = f"view_{view_idx}"
        view_data = in_tensor[view_key]

        # Get normal maps from view-specific data (already 2D)
        normal_F_arr = tensor2arr(view_data["normal_F"])
        
        # Get mask (already 2D)
        mask_normal_arr = tensor2arr(view_data["image"], True)
        mask = mask_normal_arr > 0.

        # Get depth maps and remove batch dimension if needed
        depth_F_arr = depth2arr(view_data["depth_F"])
        if depth_F_arr.ndim == 3 and depth_F_arr.shape[0] == 1:
            depth_F_arr = depth_F_arr.squeeze(0)  # Remove batch dimension

        # Store data for integration
        normal_list.append(normal_F_arr)  
        mask_list.append(mask)
        depth_list.append(depth_F_arr - 120. - thickness) 
        #depth_list.append(depth_F_arr)
        depth_mask_list.append(depth_F_arr != -1.0)

    BNI_dict = {}

    # Add multi-view integration data
    BNI_dict["normal_list"] = normal_list
    BNI_dict["mask_list"] = mask_list
    BNI_dict["depth_list"] = depth_list
    BNI_dict["depth_mask_list"] = depth_mask_list
    # Save the dictionary
    np.save(png_path + ".npy", BNI_dict, allow_pickle=True)

    return BNI_dict


class BNI:
    def __init__(self, dir_path, name, BNI_dict, cfg, device, cameras, correspondences, pairs_list, transform_manager, 
                 in_tensor=None, use_tensor_correspondences=False):

        self.scale = 256.0
        self.cfg = cfg
        self.name = name

        self.normal_list = BNI_dict["normal_list"]
        self.mask_list = BNI_dict["mask_list"]
        self.depth_list = BNI_dict["depth_list"]
        self.depth_mask_list = BNI_dict["depth_mask_list"]

        # hparam:
        # k --> smaller, keep continuity
        # lambda --> larger, more depth-awareness
        self.pairs_list = pairs_list
        self.cameras = cameras
        self.correspondences = correspondences

        self.k = self.cfg['k']
        self.lambda1 = self.cfg['lambda1']
        self.boundary_consist = self.cfg['boundary_consist']
        self.cut_intersection = self.cfg['cut_intersection']

        self.F_B_surface = None
        self.F_B_trimesh = None
        self.F_depth = None
        self.B_depth = None

        self.device = device
        self.export_dir = dir_path
        self.transform_manager = transform_manager
        
        # NEW: Store tensor-based correspondence parameters
        self.in_tensor_dict = {k: v for k, v in in_tensor.items()} if in_tensor is not None else None
        self.use_tensor_correspondences = use_tensor_correspondences

    # code: https://github.com/hoshino042/bilateral_normal_integration
    # paper: Bilateral Normal Integration

    def extract_surface_multi(self, verbose=True, lambda_cross_view=0.0, 
                              depth_threshold=0.1, debug_output_dir=None):

        bni_result = double_side_bilateral_normal_integration(
            normal_list=self.normal_list,
            normal_mask_list=self.mask_list,
            depth_list=self.depth_list,
            depth_mask_list=self.depth_mask_list,
            pairs_list=self.pairs_list,
            cameras=self.cameras,
            transform_manager=self.transform_manager,
            correspondences=self.correspondences,
            k=self.k,
            lambda_normal_back=1.0,
            lambda_depth_front=self.lambda1,
            lambda_depth_back=self.lambda1,
            lambda_boundary_consistency=self.boundary_consist,
            lambda_cross_view=lambda_cross_view, 
            cut_intersection=self.cut_intersection,
            in_tensor_dict=self.in_tensor_dict,
            use_tensor_correspondences=self.use_tensor_correspondences,
            depth_threshold=depth_threshold,
            debug_output_dir=debug_output_dir,
        )

        """
        bni_mesh_list = bni_result["bni_mesh_list"]
        bni_rot_mesh_list = []

        for i in (0,2,4,6):
            T_view_i = self.transform_manager.get_transform_to_target(i)
            R = T_view_i[:3, :3].cpu().numpy()
            R_4x4 = np.eye(4)
            R_4x4[:3, :3] = R
            bni_mesh_list[i].apply_transform(R_4x4)
            bni_mesh_list[i].export(os.path.join(self.export_dir, f"{self.name}_rotated_BNI_{i}.obj"))
            bni_rot_mesh_list.append(bni_mesh_list[i])

        fused_bni_rot_mesh = trimesh.util.concatenate(bni_rot_mesh_list)
        fused_bni_rot_mesh.export(os.path.join(self.export_dir, f"{self.name}_fused_BNI_rot.obj"))

        return fused_bni_rot_mesh
    
        """
        i=0
        
        for pair in bni_result:
            f_frame_id = self.pairs_list[i][0]
            b_frame_id = self.pairs_list[i][1]
            T_view_f = self.transform_manager.get_transform_to_target(f_frame_id)
            R_f = T_view_f[:3, :3]
            R_f_4x4 = torch.eye(4, device=R_f.device)
            R_f_4x4[:3, :3] = R_f
            T_view_b = self.transform_manager.get_transform_to_target(b_frame_id)
            R_b = T_view_b[:3, :3]
            R_b_4x4 = torch.eye(4, device=R_b.device)
            R_b_4x4[:3, :3] = R_b

            F_verts = verts_inverse_transform(pair["F_verts"], self.scale)
            B_verts = verts_inverse_transform(pair["B_verts"], self.scale)
            
            F_verts = F_verts.to(R_f.device)
            B_verts = B_verts.to(R_b.device)
            
            F_verts = torch.matmul(F_verts, R_f_4x4[:3, :3])
            B_verts = torch.matmul(B_verts, R_b_4x4[:3, :3])
            F_depth = depth_inverse_transform(pair["F_depth"], self.scale)
            B_depth = depth_inverse_transform(pair["B_depth"], self.scale)
            
            F_B_verts = torch.cat((F_verts, B_verts), dim=0)
            F_B_faces = torch.cat(
                (pair["F_faces"], pair["B_faces"] + pair["F_faces"].max() + 1), dim=0
            )
            F_trimesh = trimesh.Trimesh(
                F_verts.cpu().float(), pair["F_faces"].long(), process=False, maintain_order=True
            )
            B_trimesh = trimesh.Trimesh(
                B_verts.cpu().float(), pair["B_faces"].long(), process=False, maintain_order=True
            )
           
            # Apply to trimesh object
            F_trimesh.apply_transform(R_f_4x4.cpu())
            F_trimesh.export(os.path.join(self.export_dir, f"{self.name}_fused_pair{i}_BNI_F.obj"))
            B_trimesh.apply_transform(R_b_4x4.cpu())
            B_trimesh.export(os.path.join(self.export_dir, f"{self.name}_fused_pair{i}_BNI_B.obj"))
            
            F_B_trimesh = trimesh.Trimesh(
                F_B_verts.cpu().float(), F_B_faces.long(), process=False, maintain_order=True
            )

            F_B_trimesh.export(os.path.join(self.export_dir, f"{self.name}_fused_pair{i}_BNI.obj"))
            i+=1
            
        print(f"BNI result saved")
        



def build_sequential_cross_view_matrix_most_direct(
    in_tensor_dict: Dict[str, Any],
    num_views: int,
    offsets: np.ndarray,
    total_vars: int,
    use_depth_validation: bool = True,
    depth_threshold: float = 0.1,
    cameras: List[Camera] = None,
    debug_output_dir: str = None,
    normal_mask_list: List[np.ndarray] = None,
) -> Tuple[csr_matrix, cp.ndarray]:
    """
    Most direct version - assumes SMPL vertices are already mapped to surface pixels.
    No camera projections needed at all.
    """
    
    # Extract view data
    views = []
    for v in range(num_views):
        view_key = f"view_{v}"
        if view_key in in_tensor_dict:
            views.append(in_tensor_dict[view_key])
        else:
            raise ValueError(f"Missing {view_key} in in_tensor_dict")
    
    # Build pixel to flat index mapping for each view
    pix2flat = []
    for v in range(num_views):
        # Use the SAME mask processing as the BNI integration
        mask_for_bni = cp.asarray(normal_mask_list[v])  # This comes from BNI integration
        mask_for_bni = np.squeeze(mask_for_bni)
        
        idx = np.zeros_like(mask_for_bni, dtype=np.int32)
        idx[mask_for_bni] = np.arange(mask_for_bni.sum(), dtype=np.int32)
        pix2flat.append(idx)
    
    rows, cols, data, rhs = [], [], [], []
    validated_correspondences = 0
    correspondences_debug = []  # For debugging: (v1, u1, v1_pixel, v2, u2, v2_pixel)
    constraint_count = 0  # Track constraint index properly
    
    # Process sequential view pairs
    for v in range(num_views - 1):
        v_next = v + 1
        
        # vertex_idx i in view v corresponds to vertex_idx i in view v_next
        
        # Get number of vertices
        smpl_verts_v = views[v]["smpl_verts"]*torch.tensor([-1.0, 1.0, 1.0]).to(device=views[v]["smpl_verts"].device)
        smpl_verts_v_next = views[v_next]["smpl_verts"]*torch.tensor([-1.0, 1.0, 1.0]).to(device=views[v]["smpl_verts"].device)
        
        if hasattr(smpl_verts_v, 'detach'):
            smpl_verts_v = smpl_verts_v.detach().cpu().numpy()
        if hasattr(smpl_verts_v_next, 'detach'):
            smpl_verts_v_next = smpl_verts_v_next.detach().cpu().numpy()
            
        if smpl_verts_v.ndim == 3:
            smpl_verts_v = smpl_verts_v.squeeze(0)
        if smpl_verts_v_next.ndim == 3:
            smpl_verts_v_next = smpl_verts_v_next.squeeze(0)
        
        # Get masks
        mask_v = views[v]["img_mask"][0] if views[v]["img_mask"].ndim == 3 else views[v]["img_mask"]
        mask_v = mask_v.detach().cpu().numpy() if hasattr(mask_v, 'detach') else np.asarray(mask_v)
        mask_v = np.squeeze(mask_v).astype(bool)
        
        mask_v_next = views[v_next]["img_mask"][0] if views[v_next]["img_mask"].ndim == 3 else views[v_next]["img_mask"]
        mask_v_next = mask_v_next.detach().cpu().numpy() if hasattr(mask_v_next, 'detach') else np.asarray(mask_v_next)
        mask_v_next = np.squeeze(mask_v_next).astype(bool)
        
        # Get depth maps for validation
        depth_map_v = None
        depth_map_v_next = None
        if use_depth_validation:
            depth_map_v = views[v]["depth_F"]
            depth_map_v = depth_map_v.detach().cpu().numpy() if hasattr(depth_map_v, 'detach') else np.asarray(depth_map_v)
            if depth_map_v.ndim == 3:
                depth_map_v = depth_map_v.squeeze(0)
            
            depth_map_v_next = views[v_next]["depth_F"]
            depth_map_v_next = depth_map_v_next.detach().cpu().numpy() if hasattr(depth_map_v_next, 'detach') else np.asarray(depth_map_v_next)
            if depth_map_v_next.ndim == 3:
                depth_map_v_next = depth_map_v_next.squeeze(0)
        
        # Apply coordinate transformation to both views' vertices
        smpl_verts_v_transformed = apply_blender2cv_to_smpl(smpl_verts_v)
        smpl_verts_v_next_transformed = apply_blender2cv_to_smpl(smpl_verts_v_next)
        
        # Get camera parameters for both views
        cam_v = cameras[v]
        cam_v_next = cameras[v_next]
        
        R_v = cam_v.R.detach().cpu().numpy() if hasattr(cam_v.R, 'detach') else np.asarray(cam_v.R)
        t_v = cam_v.t.detach().cpu().numpy() if hasattr(cam_v.t, 'detach') else np.asarray(cam_v.t)
        K_v = cam_v.K.detach().cpu().numpy() if hasattr(cam_v.K, 'detach') else np.asarray(cam_v.K)
        
        R_v_next = cam_v_next.R.detach().cpu().numpy() if hasattr(cam_v_next.R, 'detach') else np.asarray(cam_v_next.R)
        t_v_next = cam_v_next.t.detach().cpu().numpy() if hasattr(cam_v_next.t, 'detach') else np.asarray(cam_v_next.t)
        K_v_next = cam_v_next.K.detach().cpu().numpy() if hasattr(cam_v_next.K, 'detach') else np.asarray(cam_v_next.K)
        
        H, W = mask_v.shape
        
        # Process each SMPL vertex (same vertex index = same physical point)
        num_vertices = min(len(smpl_verts_v_transformed), len(smpl_verts_v_next_transformed))
        
        for vertex_idx in range(num_vertices):
            pt_v = smpl_verts_v_transformed[vertex_idx]
            pt_v_next = smpl_verts_v_next_transformed[vertex_idx]
            
            # Project vertex in view v (same as debug_projection_overlay)
            pt_cam_v = R_v @ pt_v + t_v
            if pt_cam_v[2] <= 0:  # behind camera
                continue
                
            uv_v = K_v @ pt_cam_v
            u_v, v_pixel_v = int(round(uv_v[0] / uv_v[2])), int(round(uv_v[1] / uv_v[2]))
            
            # Check bounds and mask visibility for view v
            if not (0 <= u_v < W and 0 <= v_pixel_v < H and mask_v[v_pixel_v, u_v]):
                continue
            
            # Project vertex in view v_next (same as debug_projection_overlay)
            pt_cam_v_next = R_v_next @ pt_v_next + t_v_next
            if pt_cam_v_next[2] <= 0:  # behind camera
                continue
                
            uv_v_next = K_v_next @ pt_cam_v_next
            u_v_next, v_pixel_v_next = int(round(uv_v_next[0] / uv_v_next[2])), int(round(uv_v_next[1] / uv_v_next[2]))
            
            # Check bounds and mask visibility for view v_next
            if not (0 <= u_v_next < W and 0 <= v_pixel_v_next < H and mask_v_next[v_pixel_v_next, u_v_next]):
                continue
            
            # Optional depth validation
            valid = True
            if use_depth_validation:
                if depth_map_v is not None:
                    projected_depth_v = pt_cam_v[2]
                    actual_depth_v = depth_map_v[v_pixel_v, u_v]
                    if actual_depth_v != -1 and not np.isnan(actual_depth_v):
                        depth_diff_v = abs(projected_depth_v - actual_depth_v)
                        valid &= depth_diff_v < depth_threshold
                
                if depth_map_v_next is not None:
                    projected_depth_v_next = pt_cam_v_next[2]
                    actual_depth_v_next = depth_map_v_next[v_pixel_v_next, u_v_next]
                    if actual_depth_v_next != -1 and not np.isnan(actual_depth_v_next):
                        depth_diff_v_next = abs(projected_depth_v_next - actual_depth_v_next)
                        valid &= depth_diff_v_next < depth_threshold
            
            if valid:
                # Get flat indices for correspondence
                flat_idx_v = int(pix2flat[v][v_pixel_v, u_v])
                flat_idx_v_next = int(pix2flat[v_next][v_pixel_v_next, u_v_next])
                max_pixels_v = offsets[v+1] - offsets[v] if v+1 < len(offsets) else float('inf')
                max_pixels_v_next = offsets[v_next+1] - offsets[v_next] if v_next+1 < len(offsets) else float('inf')
                
                if flat_idx_v >= max_pixels_v:
                    print(f"ERROR: flat_idx_v ({flat_idx_v}) >= max_pixels_v ({max_pixels_v}) for view {v}")
                    continue
                if flat_idx_v_next >= max_pixels_v_next:
                    print(f"ERROR: flat_idx_v_next ({flat_idx_v_next}) >= max_pixels_v_next ({max_pixels_v_next}) for view {v_next}")
                    continue
                
                # Create correspondence constraint
                idx1 = offsets[v] + flat_idx_v
                idx2 = offsets[v_next] + flat_idx_v_next
                
                # Use constraint_count instead of len(rows) to avoid index overflow
                rows.extend([constraint_count, constraint_count])
                cols.extend([idx1, idx2])
                data.extend([1.0, -1.0])
                rhs.append(0.0)
                
                # Store debug information
                correspondences_debug.append((v, u_v, v_pixel_v, v_next, u_v_next, v_pixel_v_next))
                
                constraint_count += 1
                validated_correspondences += 1
    
    print(f"→ Most direct correspondences: {validated_correspondences} validated")
    
    # Save debug visualization if requested
    if debug_output_dir is not None and correspondences_debug:
        debug_save_correspondence_points(
            in_tensor_dict, num_views, cameras, debug_output_dir, correspondences_debug
        )
    
    if len(rows) == 0:
        return coo_matrix((cp.array([]), (cp.array([]), cp.array([]))), shape=(0, offsets[-1])).tocsr(), cp.array([])
    
    return coo_matrix((cp.array(data), (cp.array(rows), cp.array(cols))), 
                     shape=(constraint_count, total_vars)).tocsr(), cp.array(rhs)

def debug_save_correspondence_points(
    in_tensor_dict: Dict[str, Any],
    num_views: int,
    cameras: List[Camera],
    output_dir: str,
    correspondences_debug: List[Tuple[int, int, int, int, int, int]],  # (v1, u1, v1_pixel, v2, u2, v2_pixel)
    use_depth_validation: bool = True,
    depth_threshold: float = 0.1
):
    """
    Save correspondence points overlaid on masks for debugging.
    
    Args:
        correspondences_debug: List of (view1, u1, v1_pixel, view2, u2, v2_pixel) tuples
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract view data
    views = []
    for v in range(num_views):
        view_key = f"view_{v}"
        if view_key in in_tensor_dict:
            views.append(in_tensor_dict[view_key])
        else:
            raise ValueError(f"Missing {view_key} in in_tensor_dict")
    
    # Create debug images for each view
    debug_images = []
    for v in range(num_views):
        # Get mask and convert to 3-channel image
        mask = views[v]["img_mask"][0] if views[v]["img_mask"].ndim == 3 else views[v]["img_mask"]
        mask = mask.detach().cpu().numpy() if hasattr(mask, 'detach') else np.asarray(mask)
        mask = np.squeeze(mask).astype(np.uint8)
        
        # Create RGB image from mask
        debug_img = np.stack([mask * 255] * 3, axis=-1)  # Shape: (H, W, 3)
        debug_images.append(debug_img)
    
    # Draw correspondence points
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    print(f"Drawing {len(correspondences_debug)} correspondence pairs...")
    
    for i, (v1, u1, v1_pixel, v2, u2, v2_pixel) in enumerate(correspondences_debug):
        color = colors[i % len(colors)]
        
        # Draw points on both views
        cv2.circle(debug_images[v1], (u1, v1_pixel), radius=3, color=color, thickness=-1)
        cv2.circle(debug_images[v2], (u2, v2_pixel), radius=3, color=color, thickness=-1)
        
        # Add text labels
        cv2.putText(debug_images[v1], f"C{i}", (u1+5, v1_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(debug_images[v2], f"C{i}", (u2+5, v2_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save individual view images
    for v in range(num_views):
        output_path = os.path.join(output_dir, f"correspondences_view_{v}.png")
        cv2.imwrite(output_path, debug_images[v])
        print(f"Saved correspondence debug image: {output_path}")
    
    # Create side-by-side visualization for sequential pairs
    for v in range(num_views - 1):
        v_next = v + 1
        
        # Get images for this pair
        img1 = debug_images[v].copy()
        img2 = debug_images[v_next].copy()
        
        # Ensure same height
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        max_h = max(h1, h2)
        
        if h1 < max_h:
            img1 = np.pad(img1, ((0, max_h - h1), (0, 0), (0, 0)), mode='constant')
        if h2 < max_h:
            img2 = np.pad(img2, ((0, max_h - h2), (0, 0), (0, 0)), mode='constant')
        
        # Create side-by-side image
        combined = np.hstack([img1, img2])
        
        # Draw lines connecting correspondence points
        for i, (view1, u1, v1_pixel, view2, u2, v2_pixel) in enumerate(correspondences_debug):
            if view1 == v and view2 == v_next:
                color = colors[i % len(colors)]
                pt1 = (u1, v1_pixel)
                pt2 = (u2 + w1, v2_pixel)  # Offset by width of first image
                cv2.line(combined, pt1, pt2, color, 1)
        
        # Save combined image
        output_path = os.path.join(output_dir, f"correspondences_pair_{v}_{v_next}.png")
        cv2.imwrite(output_path, combined)
        print(f"Saved correspondence pair image: {output_path}")
    
    print(f"Debug correspondence images saved to {output_dir}")
