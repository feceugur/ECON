# -*- coding: utf-8 -*-
"""
Multi‑view Double‑Sided Bilateral Normal Integration (d‑BiNI)
------------------------------------------------------------
This is a *reference* (not minimal‑reproducible) implementation that extends the
original two‑view d‑BiNI codebase to an arbitrary number **N** of calibrated views.

* Dependencies
  - cupy / cupyx                 (GPU arrays & sparse linear algebra)
  - numpy, scipy.sparse          (CPU fall‑back for small pieces)
  - trimesh                      (meshing utilities)
  - PyTorch (optional)           (for returning tensors in the result dict)

The code is written so that the *per‑view* parts remain identical to the open
source d‑BiNI (see build_per_view_blocks).
Only two new ingredients are added:
  1. Block‑diagonal concatenation of all per‑view sparse blocks.
  2. A *cross‑view* coupling term that glues together depths that back‑project
     to the same 3‑D point (see build_cross_view_matrix).

If you simply set `lambda_c = 0` the algorithm degrades gracefully to a pure
per‑view variant whose results can later be fused (e.g. Poisson).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Dict, Any
from lib.dataset.mesh_util import clean_floats

import numpy as np
import cupy as cp
import cupyx
from cupyx.scipy.sparse import csr_matrix, coo_matrix, csc_matrix, hstack, vstack, diags  # type: ignore
from scipy.sparse import block_diag as scipy_block_diag
from cupyx.scipy.sparse.linalg import cg  # conjugate gradient on GPU

try:
    import trimesh
except ImportError:  # noqa: D401
    trimesh = None  # mesh export can be disabled if not available

# -----------------------------------------------------------------------------
# Helper data structures ------------------------------------------------------
# -----------------------------------------------------------------------------

@dataclass
class Camera:
    """Simple pin‑hole camera with intrinsics & extrinsics."""

    K: np.ndarray  # 3×3
    Rt: np.ndarray  # 4×4 (world ← camera)

    def backproject_direction(self, uv: np.ndarray) -> np.ndarray:
        """Return the (unnormalised) 3‑D *world‑space* ray direction that goes
        through pixel (u,v).
        ``uv`` shape (...,2) in **pixel** coordinates.
        """
        uv1 = np.concatenate([uv, np.ones_like(uv[..., :1])], axis=-1)  # (...,3)
        d_cam = (np.linalg.inv(self.K) @ uv1.T).T                       # cam‑space
        R = self.Rt[:3, :3]                                             # world←cam
        return (R @ d_cam.T).T                                          # to world

    @property
    def origin(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.Rt[:3, :3].T @ self.Rt[:3, 3]


# -----------------------------------------------------------------------------
# Sigmoid for bilateral weights (same as d‑BiNI) ------------------------------
# -----------------------------------------------------------------------------

def sigmoid(x: cp.ndarray, k: float = 2.0) -> cp.ndarray:
    return 1.0 / (1.0 + cp.exp(-k * x))

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


def map_depth_map_to_point_clouds(depth_map: cp.ndarray,
                                   mask: cp.ndarray,
                                   *,
                                   K: Optional[np.ndarray] = None,
                                   Rt: Optional[np.ndarray] = None,
                                   step_size: int = 1,
                                   scale: float = 1.0) -> cp.ndarray:
    #Convert a depth map into a *point cloud*.

    # Parameters
    # ----------
    # depth_map : (H,W)   depth *in the camera metric*.
    # mask      : (H,W)   boolean – which pixels to keep.
    # K, Rt     : intrinsics and *world ← camera* extrinsics.  If ``None`` the
    #              function falls back to the original pixel‑grid embedding that
    #              d‑BiNI used for its internal solver.
    # step_size : pixel size in the solver's internal units.
    # scale     : global scale factor. All depths are multiplied by this value
    #              **before** back‑projection so you can move from pixel units
    #              → mm → m, etc.
    
    H, W = depth_map.shape

    # pre‑compute pixel coordinates (y,x) → (u,v)
    yy, xx = cp.meshgrid(cp.arange(W), cp.arange(H))   # note: xx = rows, yy = cols
    xx = cp.flip(xx, axis=0)

    if K is None or Rt is None:
        # --- legacy solver coordinate frame -------------------------
        verts = cp.zeros((H, W, 3), dtype=cp.float32)
        verts[..., 0] = xx * step_size
        verts[..., 1] = yy * step_size
        verts[..., 2] = depth_map * scale
        return verts[mask]

    # --- calibrated back‑projection (camera → world) ---------------
    pix_idx = cp.argwhere(mask)                      # (N,2)  (y,x)
    u = pix_idx[:, 1]
    v = pix_idx[:, 0]
    ones = cp.ones_like(u)
    pix_h = cp.stack([u, v, ones], axis=1).astype(cp.float32)  # (N,3)

    Kinv = cp.asarray(np.linalg.inv(K), dtype=cp.float32)
    rays_cam = (Kinv @ pix_h.T).T                     # (N,3) direction in camera frame

    rays_cam *= depth_map[mask, None] * scale        # scale & depth –> metric cam‑space pts

    # camera → world
    R = cp.asarray(Rt[:3, :3], dtype=cp.float32)      # world ← camera
    t = cp.asarray(Rt[:3, 3], dtype=cp.float32)
    verts_world = (R @ rays_cam.T).T + t              # (N,3)
    return verts_world
"""

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
"""
def remove_stretched_faces(verts, faces):

    mesh = trimesh.Trimesh(verts, faces)
    camera_ray = np.array([0.0, 0.0, 1.0])
    faces_cam_angles = np.dot(mesh.face_normals, camera_ray)

    # cos(90-20)=0.34 cos(90-10)=0.17, 10~20 degree
    faces_mask = np.abs(faces_cam_angles) > 2e-1

    mesh.update_faces(faces_mask)
    mesh.remove_unreferenced_vertices()

    return mesh.vertices, mesh.faces


# -----------------------------------------------------------------------------
# Per‑view block builder -------------------------------------------------------
# -----------------------------------------------------------------------------

def build_per_view_blocks(
    normal_front: cp.ndarray,
    normal_back: cp.ndarray,
    normal_mask: cp.ndarray,
    depth_front: Optional[cp.ndarray],
    depth_back: Optional[cp.ndarray],
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
    """Replicates the original d‑BiNI *per‑view* computation but returns only
    the static matrices that enter the global normal equation.  Anything that
    changes across IRLS iterations (weights **W**) is returned separately so the
    caller can update it in‑place.

    Returns
    -------
    dict with keys
        'A_f', 'A_b',  # gradient–depth matrices (csr)
        'b_f', 'b_b',  # RHS built from normals           (1‑D cp arrays)
        'B',           # boundary consistency matrix       (csr)
        'M',           # depth prior diagonal (csr)
        'z_prior_f', 'z_prior_b',
        'W_f', 'W_b',  # **initial** bilateral weights     (csr, diagonal)
        'state'        # any extra per‑view state you need later
    """
    # ------------------------
    normal_mask = np.squeeze(normal_mask)
    assert normal_mask.ndim == 2, f"normal_mask shape is {normal_mask.shape}, expected 2D"
    num_normals = cp.sum(normal_mask).item()

    # --- normals ---------------------------------------------------------
    if normal_front.shape[0] == 3:
        normal_front = np.transpose(normal_front, (1, 2, 0))
    nx_f = normal_front[normal_mask, 1]
    ny_f = normal_front[normal_mask, 0]
    nz_f = -normal_front[normal_mask, 2]
    nx_b = normal_back[normal_mask, 1]
    ny_b = normal_back[normal_mask, 0]
    nz_b = -normal_back[normal_mask, 2]

    A3_f, A4_f, A1_f, A2_f = generate_dx_dy(normal_mask, nz_horizontal=nz_f, nz_vertical=nz_f, step_size=step_size)
    A3_b, A4_b, A1_b, A2_b = generate_dx_dy(normal_mask, nz_horizontal=nz_b, nz_vertical=nz_b, step_size=step_size)

    A_front_data = vstack((A1_f, A2_f, A3_f, A4_f))
    A_back_data = vstack((A1_b, A2_b, A3_b, A4_b))

    # build block‑diag view matrices A_f|A_b by padding with zeros  (we add those
    # zeros later in the global builder to avoid redundant storage)

    b_f = cp.concatenate((-nx_f, -nx_f, -ny_f, -ny_f))
    b_b = cp.concatenate((-nx_b, -nx_b, -ny_b, -ny_b))

    # --- W ----------------------------------------------------------------
    W_f = 0.5 * cp.ones(4 * num_normals)
    W_b = 0.5 * cp.ones(4 * num_normals)

    W_f = diags(W_f)
    W_b = diags(W_b)

    # --- M and depth priors ---------------------------------------------
    if depth_mask is not None:
        depth_mask_flat = depth_mask[normal_mask].astype(bool)
        z_prior_f = depth_front[normal_mask]
        z_prior_b = depth_back[normal_mask]
        z_prior_f[~depth_mask_flat] = 0
        z_prior_b[~depth_mask_flat] = 0
        m = depth_mask_flat.astype(int)
        M = diags(m)
    else:
        z_prior_f = cp.zeros(num_normals)
        z_prior_b = cp.zeros(num_normals)
        M = diags(cp.zeros(num_normals))

    # --- boundary consistency matrix ------------------------------------
    B, B_full = create_boundary_matrix(normal_mask)
    B = lambda_boundary_consistency * B  # incorporate weight so caller needn't

    return {
        "A_f": A_front_data,  # no zero padding here
        "A_b": A_back_data,
        "b_f": b_f,
        "b_b": b_b,
        "W_f": W_f,
        "W_b": W_b,
        "B": B,
        "M": M,
        "z_prior_f": z_prior_f,
        "z_prior_b": z_prior_b,
        "n_valid": int(num_normals),
    }


# -----------------------------------------------------------------------------
# Cross‑view coupling ---------------------------------------------------------
# -----------------------------------------------------------------------------
"""
def build_cross_view_matrix(
    correspondences: Sequence[Tuple[int, int, int, int]],
    offsets: np.ndarray,
    cameras: Sequence[Camera],
    masks: Sequence[np.ndarray],
) -> csr_matrix:
    #Build a sparse matrix **C** such that for every correspondence we have
    #(p,v) ↔ (q,w):
    #    || X_v(p) − X_w(q) ||²       with   X_v(p) = o_v + d_v(p)·r_v(p)
    #where o_v is the camera origin and r_v(p) is the *unit* ray direction.

    #For each *axis* (x,y,z) we add one row in C, containing
    #    +r_v[axis]   at column idx(depth_v[p])
    #    −r_w[axis]   at column idx(depth_w[q]).

    #The RHS is always  −(o_v[axis] − o_w[axis])  ,  so that the equation
    #    C * z  =  d   enforces equality of the two 3‑D points.

    #Parameters
    #----------
    #correspondences : list of (v, p_flat, w, q_flat)
        #Pixel index *within the masked list*, **not** absolute (u,v).
    #offsets         : prefix sum that maps view‑local flat index → global index.
    #cameras         : list of Camera objects.
    #masks           : original boolean masks (CPU) so we can convert p_flat→uv.
    
    rows, cols, data, rhs = [], [], [], []

    for corr_id, (v, p_flat, w, q_flat) in enumerate(correspondences):
        v = int(v)
        w = int(w)
        p_flat = p_flat.flatten()
        q_flat = q_flat.flatten()
        # --- per‑correspondence constants --------------------------------
        cam_v, cam_w = cameras[v], cameras[w]
        mask_v, mask_w = masks[v], masks[w]

        # flat‑index → (u,v) pixel coordinate in that view
        vidx_v = np.stack(np.nonzero(mask_v), axis=1)  # (N_v,2), CPU np.arrays
        vidx_w = np.stack(np.nonzero(mask_w), axis=1)
        uv_v = vidx_v[p_flat]  # (y,x)
        uv_w = vidx_w[q_flat]
        # back‑projection directions (world)
        r_v = cam_v.backproject_direction(uv_v[[1, 0]]).astype(np.float32)  # (u,v)→(x,y)
        r_w = cam_w.backproject_direction(uv_w[[1, 0]]).astype(np.float32)
        # camera origins
        o_v = cam_v.origin.astype(np.float32)
        o_w = cam_w.origin.astype(np.float32)

        for axis in range(3):
            rows.append(3 * corr_id + axis)
            cols.append(int(offsets[v] + p_flat))
            data.append(float(r_v[axis]))

            rows.append(3 * corr_id + axis)
            cols.append(int(offsets[w] + q_flat))
            data.append(float(-r_w[axis]))

            rhs.append(float(o_w[axis] - o_v[axis]))

    C = coo_matrix((data, (rows, cols)),
                   shape=(3 * len(correspondences), int(offsets[-1])))
    rhs = cp.asarray(rhs, dtype=cp.float32)
    return C.tocsr(), rhs

"""
def build_cross_view_matrix(
    correspondences: Sequence[Tuple[int, int, int, int]],
    offsets: np.ndarray,
    cameras: Sequence[Camera],
    masks: Sequence[np.ndarray],
) -> Tuple[csr_matrix, cp.ndarray]:
    """
    Build a sparse matrix C_front of shape (3*num_corr, num_front_variables)
    so that C_front * z_front ≈ d enforces X_v(p)=X_w(q) in the front layer.
    """
    rows, cols, data, rhs = [], [], [], []
    for cid, (v, p_flat, w, q_flat) in enumerate(correspondences):
        v, w = int(v), int(w)
        p_flat, q_flat = int(p_flat), int(q_flat)

        # flat-index → (row, col) in the mask
        ij_v = np.stack(np.nonzero(masks[v]), axis=1)
        ij_w = np.stack(np.nonzero(masks[w]), axis=1)
        yx_v, yx_w = ij_v[p_flat], ij_w[q_flat]

        # ensure we're only sending a 2-vector (u,v) into backproject_direction
        uv_v = np.array(yx_v[::-1], dtype=np.float32)
        if uv_v.ndim > 1 or uv_v.shape[-1] > 2:
            uv_v = uv_v[..., :2]
        r_v = cameras[v].backproject_direction(uv_v).astype(np.float32)

        uv_w = np.array(yx_w[::-1], dtype=np.float32)
        if uv_w.ndim > 1 or uv_w.shape[-1] > 2:
            uv_w = uv_w[..., :2]
        r_w = cameras[w].backproject_direction(uv_w).astype(np.float32)

        o_v = cameras[v].origin.astype(np.float32)
        o_w = cameras[w].origin.astype(np.float32)

        for axis in range(3):
            rows.append(3 * cid + axis)
            cols.append(p_flat)         # depth index for view v
            data.append(float(r_v[axis]))

            rows.append(3 * cid + axis)
            cols.append(q_flat)         # depth index for view w
            data.append(float(-r_w[axis]))

            rhs.append(float(o_w[axis] - o_v[axis]))

    # move to CuPy arrays
    rows_cp = cp.asarray(rows, dtype=cp.int32)
    cols_cp = cp.asarray(cols, dtype=cp.int32)
    data_cp = cp.asarray(data, dtype=cp.float32)
    rhs_cp  = cp.asarray(rhs,  dtype=cp.float32)

    num_front = int(offsets[-1])
    C_front = coo_matrix((data_cp, (rows_cp, cols_cp)),
                         shape=(3 * len(correspondences), num_front))
    return C_front.tocsr(), rhs_cp




# -----------------------------------------------------------------------------
# Padding helper ----------------------------------------------------------------
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Main algorithm ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def multi_view_d_bini(
    normals_front: Sequence[np.ndarray],
    normals_back: Sequence[np.ndarray],
    normal_masks: Sequence[np.ndarray],
    *,
    depth_front: Optional[Sequence[np.ndarray]] = None,
    depth_back: Optional[Sequence[np.ndarray]] = None,
    depth_masks: Optional[Sequence[np.ndarray]] = None,
    cameras: Optional[Sequence[Camera]] = None,
    correspondences: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    offsets: Optional[np.ndarray] = None,
    # hyper‑parameters ---------------------------------------------------
    k: float = 2.0,
    lambda_normal_back: float = 1.0,
    lambda_depth_front: float = 1e-4,
    lambda_depth_back: float = 1e-2,
    lambda_boundary_consistency: float = 1.0,
    lambda_cross_view: float = 0.0,
    # solver -------------------------------------------------------------
    step_size: int = 1,
    irls_max_iter: int = 150,
    cg_max_iter: int = 5000,
    cg_tol: float = 1e-3,
) -> List[Dict[str, Any]]:

    V = len(normals_front)
    assert len(normals_back) == V == len(normal_masks)

    depth_front = depth_front or [None] * V
    depth_back = depth_back or [None] * V
    depth_masks = depth_masks or [None] * V

    # ------------------------------------------------------------------
    # 1. per‑view static blocks ----------------------------------------
    # ------------------------------------------------------------------
    per_view = []
    for v in range(V):
        per_view.append(
            build_per_view_blocks(
                cp.asarray(normals_front[v]),
                cp.asarray(normals_back[v]),
                cp.asarray(normal_masks[v]),
                None if depth_front[v] is None else cp.asarray(depth_front[v]),
                None if depth_back[v] is None else cp.asarray(depth_back[v]),
                None if depth_masks[v] is None else cp.asarray(depth_masks[v]),
                k=k,
                lambda_normal_back=lambda_normal_back,
                lambda_depth_front=lambda_depth_front,
                lambda_depth_back=lambda_depth_front,  # same weight for back priors
                lambda_boundary_consistency=lambda_boundary_consistency,
                step_size=step_size,
            )
        )

    n_pix = np.array([blk["n_valid"] for blk in per_view], dtype=np.int64)

    # front/back offsets ------------------------------------------------
    ofs_front = np.zeros(V, dtype=np.int64)
    cumulative = 0
    for v in range(V):
        ofs_front[v] = cumulative
        cumulative += 2 * n_pix[v]
    total_vars = cumulative
    ofs_back = ofs_front + n_pix

    # ------------------------------------------------------------------
    # 2. assemble global matrices --------------------------------------
    # ------------------------------------------------------------------
    A_blocks, W_blocks, b_blocks = [], [], []
    S_rows = []
    z_prior_blocks = []
    M_diag_parts: List[cp.ndarray] = []  # store only diagonals

    for v, blk in enumerate(per_view):
        # gradient–depth matrices -------------------------------------
        A_blocks.append(_pad(blk["A_f"], int(ofs_front[v]), total_vars))
        A_blocks.append(_pad(blk["A_b"], int(ofs_back[v]),  total_vars))

        # bilateral weights (block‑diag) ------------------------------
        W_blocks.append(blk["W_f"])
        W_blocks.append(lambda_normal_back * blk["W_b"])

        # RHS from normals -------------------------------------------
        b_blocks.append(blk["b_f"])
        b_blocks.append(blk["b_b"])

        # depth‑prior diagonal (just vector) --------------------------
        M_diag_parts.append(blk["M"].diagonal())  # front
        M_diag_parts.append(blk["M"].diagonal())  # back (same diag)
        z_prior_blocks.append(blk["z_prior_f"])
        z_prior_blocks.append(blk["z_prior_b"])

        # silhouette S rows ------------------------------------------
        Bv_front = _pad(blk["B"], int(ofs_front[v]), total_vars)
        Bv_back  = _pad(blk["B"], int(ofs_back[v]),  total_vars)
        S_rows.append(Bv_front - Bv_back)
        S_rows.append(-Bv_front + Bv_back)

    # convert to big sparse objects -----------------------------------
    A = vstack(A_blocks).tocsr()
    W = block_diag(W_blocks)
    b = cp.concatenate(b_blocks)

    M_big = diags(cp.concatenate(M_diag_parts))
    S_big = vstack(S_rows).tocsr()
    z_prior = cp.concatenate(z_prior_blocks)

    # ------------------------------------------------------------------
    # 3. cross‑view coupling 
    # ------------------------------------------------------------------
    
    if lambda_cross_view > 0.0:
        assert cameras is not None and correspondences is not None and offsets is not None
        C_front, rhs_C_front = build_cross_view_matrix(
            correspondences, offsets, cameras, normal_masks)
        # pad zero columns for back variables
        Nf = int(offsets[-1])
        C_full = hstack([
            C_front,
            csr_matrix((C_front.shape[0], Nf))
        ]).tocsr()
        rhs_full = cp.concatenate([rhs_C_front, cp.zeros(Nf, dtype=rhs_C_front.dtype)])
    else:
        C_full = None
        rhs_full = None

    # ------------------------------------------------------------------
    # 4. IRLS --------------------------------
    # ------------------------------------------------------------------
    z = cp.zeros(2 * int(offsets[-1]), dtype=cp.float32)
    for it in range(irls_max_iter):
        energy = (A @ z - b).T @ W @ (A @ z - b) + \
                 lambda_depth_front * (z - z_prior).T @ M_big @ (z - z_prior) + \
                 lambda_boundary_consistency * (z - z_prior).T @ S_big @ (z - z_prior)
        if C_full is not None:
            energy = energy + lambda_cross_view * (C_full @ z - rhs_C_front).T @ (C_full @ z - rhs_C_front)
        #print(f"IRLS iteration {it}, energy: {energy}")

        lhs = A.T @ W @ A
        lhs = lhs + lambda_depth_front * M_big
        lhs = lhs + lambda_boundary_consistency * S_big
        rhs = A.T @ W @ b
        rhs = rhs + lambda_depth_front * (M_big @ z_prior)
        if C_full is not None:
            lhs = lhs + lambda_cross_view * (C_full.T @ C_full)
            rhs = rhs + lambda_cross_view * (C_full.T @ rhs_C_front)

        D_inv = diags(1.0 / cp.clip(lhs.diagonal(), 1e-5, None))
        z, _ = cg(lhs, rhs, x0=z, M=D_inv,
                  maxiter=cg_max_iter, tol=cg_tol)

        # TODO: update bilateral weights per view 

    # ------------------------------------------------------------------
    # 5. unpack + build meshes -----------------------------------------
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    for v in range(len(normal_masks)):
        n = n_pix[v]
        z_f = z[ofs_front[v] : ofs_front[v] + n]
        z_b = z[ofs_back[v]  : ofs_back[v]  + n]
        mask_v = cp.asarray(normal_masks[v])

        depth_f = cp.full(mask_v.shape, cp.nan, dtype=cp.float32)
        depth_b = cp.full(mask_v.shape, cp.nan, dtype=cp.float32)
        depth_f[mask_v] = z_f
        depth_b[mask_v] = z_b

        # --- optional mesh export -------------------------------------
        if trimesh is not None:
            cam = cameras[v] if cameras is not None else None
            K_v  = None if cam is None else cam.K
            Rt_v = None if cam is None else cam.Rt
            #verts_f = cp.asnumpy(map_depth_map_to_point_clouds(depth_f, mask_v, K=None, step_size=step_size))
            #verts_b = cp.asnumpy(map_depth_map_to_point_clouds(depth_b, mask_v, K=None, step_size=step_size))
            verts_f = cp.asnumpy(map_depth_map_to_point_clouds(depth_f, mask_v, K=K_v, Rt=Rt_v, step_size=step_size))
            verts_b = cp.asnumpy(map_depth_map_to_point_clouds(depth_b, mask_v, K=K_v, Rt=Rt_v, step_size=step_size))


            facets  = cp.asnumpy(construct_facets_from(mask_v))
            faces_b = np.concatenate((facets[:, [1, 4, 3]], facets[:, [1, 3, 2]]))
            faces_f = np.concatenate((facets[:, [1, 2, 3]], facets[:, [1, 3, 4]]))
            from lib.dataset.mesh_util import clean_floats  # local import to avoid hard dep if unavailable
            mesh_f = clean_floats(trimesh.Trimesh(verts_f, faces_f))
            mesh_b = clean_floats(trimesh.Trimesh(verts_b, faces_b))
            result_v: Dict[str, Any] = {
                "F_verts": mesh_f.vertices.astype(np.float32),
                "F_faces": mesh_f.faces.astype(np.int32),
                "B_verts": mesh_b.vertices.astype(np.float32),
                "B_faces": mesh_b.faces.astype(np.int32),
                "F_depth": depth_f.get(),
                "B_depth": depth_b.get(),
            }
        else:
            result_v = {
                "F_verts": None,
                "F_faces": None,
                "B_verts": None,
                "B_faces": None,
                "F_depth": depth_f.get(),
                "B_depth": depth_b.get(),
            }
        results.append(result_v)

    return results

# Helper function to convert scipy sparse matrix to cupy sparse matrix
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
