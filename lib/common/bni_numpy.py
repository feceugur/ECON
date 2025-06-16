"""
Bilateral Normal Integration (BiNI)
"""
__author__ = "Xu Cao <xucao.42@gmail.com>"
__copyright__ = "Copyright (C) 2022 Xu Cao"
__version__ = "2.0"

from lib.common.BNI_utils import create_boundary_matrix, remove_stretched_faces, generate_dx_dy, map_depth_map_to_point_clouds, construct_facets_from
from lib.dataset.mesh_util import clean_floats
from scipy.sparse import spdiags, csr_matrix, vstack
from scipy.sparse.linalg import cg
import numpy as np
from tqdm.auto import tqdm
import time
import pyvista as pv
import torch
import cupy as cp
from cupyx.scipy.sparse import (
    coo_matrix,
    csr_matrix,
    diags,
    hstack,
    spdiags,
    vstack,
)
import trimesh
# from pyamg.aggregation import smoothed_aggregation_solver


# Define helper functions for moving masks in different directions
def move_left(mask): return np.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:,1:]  # Shift the input mask array to the left by 1, filling the right edge with zeros.
def move_right(mask): return np.pad(mask,((0,0),(1,0)),'constant',constant_values=0)[:,:-1]  # Shift the input mask array to the right by 1, filling the left edge with zeros.
def move_top(mask): return np.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:,:]  # Shift the input mask array up by 1, filling the bottom edge with zeros.
def move_bottom(mask): return np.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1,:]  # Shift the input mask array down by 1, filling the top edge with zeros.
def move_top_left(mask): return np.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:,1:]  # Shift the input mask array up and to the left by 1, filling the bottom and right edges with zeros.
def move_top_right(mask): return np.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:,:-1]  # Shift the input mask array up and to the right by 1, filling the bottom and left edges with zeros.
def move_bottom_left(mask): return np.pad(mask,((1,0),(0,1)),'constant',constant_values=0)[:-1,1:]  # Shift the input mask array down and to the left by 1, filling the top and right edges with zeros.
def move_bottom_right(mask): return np.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1,:-1]  # Shift the input mask array down and to the right by 1, filling the top and left edges with zeros.

"""
def generate_dx_dy(mask, nz_horizontal, nz_vertical, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    num_pixel = np.sum(mask)

    # Generate an integer index array with the same shape as the mask.
    pixel_idx = np.zeros_like(mask, dtype=int)
    # Assign a unique integer index to each True value in the mask.
    pixel_idx[mask] = np.arange(num_pixel)

    # Create boolean masks representing the presence of neighboring pixels in each direction.
    has_left_mask = np.logical_and(move_right(mask), mask)
    has_right_mask = np.logical_and(move_left(mask), mask)
    has_bottom_mask = np.logical_and(move_top(mask), mask)
    has_top_mask = np.logical_and(move_bottom(mask), mask)

    # Extract the horizontal and vertical components of the normal vectors for the neighboring pixels.
    nz_left = nz_horizontal[has_left_mask[mask]]
    nz_right = nz_horizontal[has_right_mask[mask]]
    nz_top = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    # Create sparse matrices representing the partial derivatives for each direction.
    # top/bottom/left/right = vertical positive/vertical negative/horizontal negative/horizontal positive
    # The matrices are constructed using the extracted normal components and pixel indices.
    data = np.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    # Return the four sparse matrices representing the partial derivatives for each direction.
    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg

def generate_dx_dy_upsample(mask, nz_horizontal, nz_vertical, step_size=1, pixel_idx=None):
    assert pixel_idx is not None, "Global pixel_idx must be passed in."
    num_pixel = pixel_idx.max() + 1

    def build_sparse_pairs(mask_dir, shift_axis, shift_val, nz_values):
        # Convert CuPy arrays to NumPy if needed
        if hasattr(nz_values, 'get'):
            nz_values = nz_values.get()
        
        shifted_mask = np.roll(mask_dir, shift_val, axis=shift_axis)
        valid_pairs = mask_dir & shifted_mask

        current_indices = pixel_idx[valid_pairs]
        neighbor_indices = pixel_idx[np.roll(valid_pairs, shift_val, axis=shift_axis)]

        min_len = min(len(current_indices), len(neighbor_indices), len(nz_values))
        current_indices = current_indices[:min_len]
        neighbor_indices = neighbor_indices[:min_len]
        nz_values = nz_values[:min_len]

        data = np.stack([-nz_values / step_size, nz_values / step_size], axis=-1).flatten()
        indices = np.stack((neighbor_indices, current_indices), axis=-1).flatten()
        indptr = np.arange(0, len(current_indices) * 2 + 1, 2)

        return csr_matrix((data, indices, indptr), shape=(len(current_indices), num_pixel))

    has_left = np.logical_and(np.roll(mask, -1, axis=1), mask)
    has_right = np.logical_and(np.roll(mask, 1, axis=1), mask)
    has_top = np.logical_and(np.roll(mask, 1, axis=0), mask)
    has_bottom = np.logical_and(np.roll(mask, -1, axis=0), mask)

    # Convert CuPy arrays to NumPy if needed
    if hasattr(nz_horizontal, 'get'):
        nz_horizontal = nz_horizontal.get()
    if hasattr(nz_vertical, 'get'):
        nz_vertical = nz_vertical.get()

    nz_left = nz_horizontal[has_left[mask]]
    nz_right = nz_horizontal[has_right[mask]]
    nz_top = nz_vertical[has_top[mask]]
    nz_bottom = nz_vertical[has_bottom[mask]]

    A1 = build_sparse_pairs(has_left, shift_axis=1, shift_val=1, nz_values=nz_left)
    A2 = build_sparse_pairs(has_right, shift_axis=1, shift_val=-1, nz_values=nz_right)
    A3 = build_sparse_pairs(has_top, shift_axis=0, shift_val=1, nz_values=nz_top)
    A4 = build_sparse_pairs(has_bottom, shift_axis=0, shift_val=-1, nz_values=nz_bottom)

    return A1, A2, A3, A4



def construct_facets_from(mask):
    # Initialize an array 'idx' of the same shape as 'mask' with integers
    # representing the indices of valid pixels in the mask.
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    # Generate masks for neighboring pixels to define facets
    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    # Identify the top-left pixel of each facet by performing a logical AND operation
    # on the masks of neighboring pixels and the input mask.
    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))

    # Create masks for the other three vertices of each facet by shifting the top-left mask.
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    # Return a numpy array of facets by stacking the indices of the four vertices
    # of each facet along the last dimension. Each row of the resulting array represents
    # a single facet with the format [4, idx_top_left, idx_bottom_left, idx_bottom_right, idx_top_right].
    return np.stack((4 * np.ones(np.sum(facet_top_left_mask)),
               idx[facet_top_left_mask],
               idx[facet_bottom_left_mask],
               idx[facet_bottom_right_mask],
               idx[facet_top_right_mask]), axis=-1).astype(int)


def map_depth_map_to_point_clouds(depth_map, mask, K=None, step_size=1):
    # y
    # |  z
    # | /
    # |/
    # o ---x
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, axis=0)

    if K is None:
        vertices = np.zeros((H, W, 3))
        vertices[..., 0] = xx * step_size
        vertices[..., 1] = yy * step_size
        vertices[..., 2] = depth_map
        vertices = vertices[mask]
    else:
        u = np.zeros((H, W, 3))
        u[..., 0] = xx
        u[..., 1] = yy
        u[..., 2] = 1
        u = u[mask].T  # 3 x m
        vertices = (np.linalg.inv(K) @ u).T * depth_map[mask, np.newaxis]  # m x 3

    return vertices

"""
def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k * x))


def bilateral_normal_integration(normal_map,
        normal_mask,
        k=2,
        depth_map=None,
        depth_mask=None,
        lambda1=0,
        K=None,
        step_size=1,
        max_iter=150,
        tol=1e-4,
        cg_max_iter=5000,
        cg_tol=1e-3):
    """
    This function performs the bilateral normal integration algorithm, as described in the paper.
    It takes as input the normal map, normal mask, and several optional parameters to control the integration process.

    :param normal_map: A normal map, which is an image where each pixel's color encodes the corresponding 3D surface normal.
    :param normal_mask: A binary mask that indicates the region of interest in the normal_map to be integrated.
    :param k: A parameter that controls the stiffness of the surface.
              The smaller the k value, the smoother the surface appears (fewer discontinuities).
              If set as 0, a smooth surface is obtained (No discontinuities), and the iteration should end at step 2 since the surface will not change with iterations.

    :param depth_map: (Optional) An initial depth map to guide the integration process.
    :param depth_mask: (Optional) A binary mask that indicates the valid depths in the depth_map.

    :param lambda1 (Optional): A regularization parameter that controls the influence of the depth_map on the final result.
                               Required when depth map is input.
                               The larger the lambda1 is, the result more close to the initial depth map (fine details from the normal map are less reflected)

    :param K: (Optional) A 3x3 camera intrinsic matrix, used for perspective camera models. If not provided, the algorithm assumes an orthographic camera model.
    :param step_size: (Optional) The pixel size in the world coordinates. Default value is 1.
                                 Used only in the orthographic camera mdoel.
                                 Default value should be fine, unless you know the true value of the pixel size in the world coordinates.
                                 Do not adjust it in perspective camera model.

    :param max_iter: (Optional) The maximum number of iterations for the optimization process. Default value is 150.
                                If set as 1, a smooth surface is obtained (No discontinuities).
                                Default value should be fine.
    :param tol:  (Optional) The tolerance for the relative change in energy to determine the convergence of the optimization process. Default value is 1e-4.
                            The larger, the iteration stops faster, but the discontinuity preservation quality might be worse. (fewer discontinuities)
                            Default value should be fine.

    :param cg_max_iter: (Optional) The maximum number of iterations for the Conjugate Gradient solver. Default value is 5000.
                                   Default value should be fine.
    :param cg_tol: (Optional) The tolerance for the Conjugate Gradient solver. Default value is 1e-3.
                              Default value should be fine.

    :return: depth_map: The resulting depth map after the bilateral normal integration process.
             surface: A pyvista PolyData mesh representing the 3D surface reconstructed from the depth map.
             wu_map: A 2D image that represents the horizontal smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             wv_map: A 2D image that represents the vertical smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             energy_list: A list of energy values during the optimization process.
    """
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
    # I forgot why I chose the awkward coordinate system after getting used to opencv convention :(
    # but I won't touch the working code.

    num_normals = np.sum(normal_mask)
    projection = "orthographic" if K is None else "perspective"
    print(f"Running bilateral normal integration with k={k} in the {projection} case. \n"
          f"The number of normal vectors is {num_normals}.")
    # Transform the normal map from the normal coordinates to the camera coordinates
    nx = normal_map[normal_mask, 1]
    ny = normal_map[normal_mask, 0]
    nz = - normal_map[normal_mask, 2]

    # Handle perspective and orthographic cases separately
    if K is not None:  # perspective
        img_height, img_width = normal_mask.shape[:2]

        yy, xx = np.meshgrid(range(img_width), range(img_height))
        xx = np.flip(xx, axis=0)

        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy

        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz
        del xx, yy, uu, vv
    else:  # orthographic
        nz_u = nz.copy()
        nz_v = nz.copy()

    # get partial derivative matrices
    # right, left, top, bottom
    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_horizontal=nz_v, nz_vertical=nz_u, step_size=step_size)

    # Construct the linear system
    A = vstack((A1, A2, A3, A4))
    b = np.concatenate((-nx, -nx, -ny, -ny))

    # Initialize variables for the optimization process
    W = spdiags(0.5 * np.ones(4*num_normals), 0, 4*num_normals, 4*num_normals, format="csr")
    z = np.zeros(np.sum(normal_mask))
    energy = (A @ z - b).T @ W @ (A @ z - b)

    tic = time.time()

    energy_list = []
    if depth_map is not None:
        m = depth_mask[normal_mask].astype(int)
        M = spdiags(m, 0, num_normals, num_normals, format="csr")
        z_prior = np.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]

    pbar = tqdm(range(max_iter))

    # Optimization loop
    for i in pbar:
        # fix weights and solve for depths
        A_mat = A.T @ W @ A
        b_vec = A.T @ W @ b
        if depth_map is not None:
            depth_diff = M @ (z_prior - z)
            depth_diff[depth_diff==0] = np.nan
            offset = np.nanmean(depth_diff)
            z = z + offset
            A_mat += lambda1 * M
            b_vec += lambda1 * M @ z_prior

        D = spdiags(1/np.clip(A_mat.diagonal(), 1e-5, None), 0, num_normals, num_normals, format="csr")  # Jacob preconditioner

        # ml = smoothed_aggregation_solver(A_mat, max_levels=4)  # AMG preconditioner, not very stable but faster than Jacob preconditioner.
        # D = ml.aspreconditioner(cycle='W')
        z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=cg_max_iter, tol=cg_tol)

        # Update the weight matrices
        wu = sigmoid((A2 @ z) ** 2 - (A1 @ z) ** 2, k)
        wv = sigmoid((A4 @ z) ** 2 - (A3 @ z) ** 2, k)
        W = spdiags(np.concatenate((wu, 1-wu, wv, 1-wv)), 0, 4*num_normals, 4*num_normals, format="csr")

        # Check for convergence
        energy_old = energy
        energy = (A @ z - b).T @ W @ (A @ z - b)
        energy_list.append(energy)
        relative_energy = np.abs(energy - energy_old) / energy_old
        pbar.set_description(
            f"step {i + 1}/{max_iter} energy: {energy:.3f} relative energy: {relative_energy:.3e}")
        if relative_energy < tol:
            break
    toc = time.time()

    print(f"Total time: {toc - tic:.3f} sec")

    # Reconstruct the depth map and surface
    depth_map = np.ones_like(normal_mask, float) * np.nan
    depth_map[normal_mask] = z

    if K is not None:  # perspective
        depth_map = np.exp(depth_map)
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=K)
    else:  # orthographic
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=None, step_size=step_size)

    facets = construct_facets_from(normal_mask)
    if normal_map[:, :, -1].mean() < 0:
        facets = facets[:, [0, 1, 4, 3, 2]]
    surface = pv.PolyData(vertices, facets)

    # In the main paper, wu indicates the horizontal direction; wv indicates the vertical direction
    wu_map = np.ones_like(normal_mask) * np.nan
    wu_map[normal_mask] = wv

    wv_map = np.ones_like(normal_mask) * np.nan
    wv_map[normal_mask] = wu

    result = {
        "F_verts": torch.as_tensor(vertices).float(), 
        "F_faces": torch.as_tensor(facets).long(), 
        "F_depth": torch.as_tensor(depth_map).float()
    }
    # return depth_map, surface, wu_map, wv_map, energy_list
    return result


def bilateral_normal_integration_upt(normal_map,
        normal_mask,
        k=2,
        depth_map=None,
        depth_mask=None,
        lambda1=0,
        K=None,
        step_size=1,
        max_iter=150,
        tol=1e-4,
        cg_max_iter=5000,
        cg_tol=1e-3,
        lambda_boundary_consistency=1):
    
    num_normals = cp.sum(normal_mask)
    normal_map = cp.asarray(normal_map)
    normal_mask = cp.asarray(normal_mask)
    if depth_mask is not None:
        depth_map = cp.asarray(depth_map)
        depth_mask = cp.asarray(depth_mask)
        
    # Transform the normal map from the normal coordinates to the camera coordinates
    nx = normal_map[normal_mask, 1]
    ny = normal_map[normal_mask, 0]
    nz = - normal_map[normal_mask, 2]
    del normal_map

    # get partial derivative matrices
    # right, left, top, bottom
    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_horizontal=nz, nz_vertical=nz, step_size=step_size)
    
    has_left_mask = cp.logical_and(move_right(normal_mask), normal_mask)
    has_right_mask = cp.logical_and(move_left(normal_mask), normal_mask)
    has_bottom_mask = cp.logical_and(move_top(normal_mask), normal_mask)
    has_top_mask = cp.logical_and(move_bottom(normal_mask), normal_mask)

    top_boundnary_mask = cp.logical_xor(has_top_mask, normal_mask)[normal_mask]
    bottom_boundary_mask = cp.logical_xor(has_bottom_mask, normal_mask)[normal_mask]
    left_boundary_mask = cp.logical_xor(has_left_mask, normal_mask)[normal_mask]
    right_boudnary_mask = cp.logical_xor(has_right_mask, normal_mask)[normal_mask]

    A_data = vstack((A1, A2, A3, A4))
    A_zero = csr_matrix(A_data.shape)
    A = hstack([A_data, A_zero])

    b = cp.concatenate((-nx, -nx, -ny, -ny))

    # initialization
    W = spdiags(
        0.5 * cp.ones(4 * num_normals), 0, 4 * num_normals, 4 * num_normals, format="csr"
    )

    z = cp.zeros(num_normals, float)
    B, B_full = create_boundary_matrix(normal_mask)
    B_mat = lambda_boundary_consistency * coo_matrix(B_full.get().T @ B_full.get())    #bug

    energy_list = []
    if depth_mask is not None:
       # depth_mask_flat = depth_mask[normal_mask].astype(bool)    # shape: (num_normals,)
       # z_prior = depth_map[normal_mask]    # shape: (num_normals,)
       # z_prior[~depth_mask_flat] = 0
       # m = depth_mask[normal_mask].astype(int)
       # M = diags(m)
        # Reshape depth_mask to match normal_mask's shape if it's 1D
        if depth_mask.ndim == 1:
            # Create a new 2D mask initialized with False
            new_depth_mask = cp.zeros(normal_mask.shape, dtype=bool)
            # Set True values in the first row up to the length of the 1D mask
            new_depth_mask[0, :len(depth_mask)] = depth_mask
            depth_mask = new_depth_mask
        depth_mask_flat = depth_mask[normal_mask].astype(bool)    # shape: (num_normals,)
        z_prior = depth_map[normal_mask]    # shape: (num_normals,)
        z_prior[~depth_mask_flat] = 0
        m = depth_mask[normal_mask].astype(int)
        M = diags(m)
    
    energy = (A @ z - b).T @ W @ (A @ z - b) + \
             lambda_boundary_consistency * (z - z_prior).T @ B_mat @ (z - z_prior)

    depth_map_est = cp.ones_like(normal_mask, float) * cp.nan
    facets = construct_facets_from(normal_mask)

    for i in range(max_iter):
        A_mat = A.T @ W @ A
        b_vec = A.T @ W @ b
        if depth_mask is not None:
            b_vec += M @ z_prior
            A_mat += M
            offset = cp.mean((z_prior - z)[depth_mask_flat])
            z = z + offset

        D = spdiags(
            1 / cp.clip(A_mat.diagonal(), 1e-5, None), 0, 2 * num_normals, 2 * num_normals,
            "csr"
        )    # Jacob preconditioner

        z, _ = cg(
            A_mat, b_vec, M=D, x0=z, maxiter=cg_max_iter, tol=cg_tol
        )

        wu = sigmoid((A2 @ z) ** 2 - (A1 @ z) ** 2, k)
        wv = sigmoid((A4 @ z) ** 2 - (A3 @ z) ** 2, k)
        wu[top_boundnary_mask] = 0.5
        wu[bottom_boundary_mask] = 0.5
        wv[left_boundary_mask] = 0.5
        wv[right_boudnary_mask] = 0.5
        W = spdiags(
            cp.concatenate((wu, 1 - wu, wv, 1 - wv)), 0, 4 * num_normals, 4 * num_normals, format="csr"
        )

        energy_old = energy
        energy = (A @ z - b).T @ W @ (A @ z - b) + \
             lambda_boundary_consistency * (z - z_prior).T @ B_mat @ (z - z_prior)

        energy_list.append(energy)
        relative_energy = cp.abs(energy - energy_old) / energy_old

        if relative_energy < tol:
            break

    depth_map_est[normal_mask] = z

    vertices = cp.asnumpy(
        map_depth_map_to_point_clouds(depth_map_est, normal_mask, K=None, step_size=step_size)
    )
    facets = cp.asnumpy(construct_facets_from(normal_mask))
    faces = np.concatenate((facets[:, [1, 4, 3]], facets[:, [1, 3, 2]]), axis=0)
    vertices, faces = remove_stretched_faces(vertices, faces)

    mesh = clean_floats(trimesh.Trimesh(vertices, faces))
    # Save mesh as obj
    mesh_path = "output_mesh.obj"
    mesh.export(mesh_path)

    result = {
        "F_verts": torch.as_tensor(mesh.vertices).float(), 
        "F_faces": torch.as_tensor(mesh.faces).long(), 
        "F_depth": torch.as_tensor(depth_map_est).float()
    }
    return result


            
        
    

def extract_surface_multiview(normal_list, mask_list, depth_list, K_list, scale=1.0, k=2, lambda1=0.1, max_iter=150, tol=1e-4):
    """
    Extract surfaces from multiple views using bilateral normal integration.
    
    Args:
        normal_list (list): List of normal maps from different views
        mask_list (list): List of masks from different views
        depth_list (list): List of depth maps from different views
        K_list (list): List of camera intrinsic matrices for each view
        scale (float): Scale factor for depth values
        k (float): Stiffness parameter for bilateral normal integration
        lambda1 (float): Regularization parameter for depth guidance
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence
        
    Returns:
        list: List of dictionaries containing reconstruction results for each view
    """
    results = []
    
    for i, (normals, mask, depth, K) in enumerate(zip(normal_list, mask_list, depth_list, K_list)):
        print(f"\nProcessing view {i}")
        
        # Scale depth values
        depth_scaled = depth * scale
        
        # Run bilateral normal integration for this view
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(
            normal_map=normals,
            normal_mask=mask,
            k=k,
            depth_map=depth_scaled,
            depth_mask=mask,
            lambda1=lambda1,
            K=K,
            max_iter=max_iter,
            tol=tol
        )
        
        # Store results for this view
        view_result = {
            "depth_map": depth_map,
            "surface": surface,
            "wu_map": wu_map,
            "wv_map": wv_map,
            "energy_list": energy_list,
            "camera_matrix": K
        }
        
        results.append(view_result)
        
    return results


if __name__ == '__main__':
    import cv2
    import argparse, os
    import warnings
    warnings.filterwarnings('ignore')
    # To ignore the possible overflow runtime warning: overflow encountered in exp return 1 / (1 + np.exp(-k * x)).
    # This overflow issue does not affect our results as np.exp will correctly return 0.0 when -k * x is massive.

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise FileNotFoundError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=dir_path)
    parser.add_argument('-k', type=float, default=2)
    parser.add_argument('-i', '--iter', type=np.uint, default=150)
    parser.add_argument('-t', '--tol', type=float, default=1e-4)
    arg = parser.parse_args()

    normal_map = cv2.cvtColor(cv2.imread(os.path.join(arg.path, "normal_map.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    if normal_map.dtype is np.dtype(np.uint16):
        normal_map = normal_map/65535 * 2 - 1
    else:
        normal_map = normal_map/255 * 2 - 1

    try:
        mask = cv2.imread(os.path.join(arg.path, "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)
    except:
        mask = np.ones(normal_map.shape[:2], bool)

    if os.path.exists(os.path.join(arg.path, "K.txt")):
        K =np.loadtxt(os.path.join(arg.path, "K.txt"))
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                       normal_mask=mask,
                                                                                       k=arg.k,
                                                                                       K=K,
                                                                                       max_iter=arg.iter,
                                                                                       tol=arg.tol)
    else:
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                       normal_mask=mask,
                                                                                       k=arg.k,
                                                                                       K=None,
                                                                                       max_iter=arg.iter,
                                                                                       tol=arg.tol)

    # save the resultant polygon mesh and discontinuity maps.
    np.save(os.path.join(arg.path, "energy"), np.array(energy_list))
    surface.save(os.path.join(arg.path, f"mesh_k_{arg.k}.ply"), binary=False)
    wu_map = cv2.applyColorMap((255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
    wv_map = cv2.applyColorMap((255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
    wu_map[~mask] = 255
    wv_map[~mask] = 255
    cv2.imwrite(os.path.join(arg.path, f"wu_k_{arg.k}.png"), wu_map)
    cv2.imwrite(os.path.join(arg.path, f"wv_k_{arg.k}.png"), wv_map)
    print(f"saved {arg.path}")
