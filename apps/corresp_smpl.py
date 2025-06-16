# apps/corresp_smpl.py ---------------------------------------------------------
import numpy as np
import trimesh

def build_pixel_vertex_lookup(mask, K, Rt, smpl_mesh):
    """
    Returns two arrays of shape (N_pix,)
        verts_idx   – integer index of the SMPL vertex each pixel sees  (-1 = none)
        rays_world  – 3-D position hit on the SMPL surface (world frame)
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    # 3-D ray direction for every valid pixel  (world frame)
    pix_h = np.stack([xs, ys, np.ones_like(xs)], 1).T
    dir_cam = np.linalg.inv(K) @ pix_h          # (3,N)
    R, t = Rt[:3, :3], Rt[:3, 3]
    dirs_world = (R @ dir_cam).T
    origins_world = np.repeat((-R.T @ t)[None], len(xs), 0)

    # trimesh ray intersector
    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(smpl_mesh)
    loc, idx, _ = ray_intersector.intersects_location(
        origins_world, dirs_world, multiple_hits=False)

    verts_idx = -np.ones(len(xs), dtype=np.int32)
    if len(idx):
        # barycentric hit → closest vertex
        faces = smpl_mesh.faces[idx]
        # distance to each of the 3 vertices, pick nearest
        tri_verts = smpl_mesh.vertices[faces]            # (M,3,3)
        dists = ((tri_verts - loc[:, None])**2).sum(-1)  # (M,3)
        nearest = faces[np.arange(len(idx)), dists.argmin(1)]
        verts_idx[idx] = nearest

    rays_world = loc                                      # (<=N,3)
    return xs, ys, verts_idx, rays_world
