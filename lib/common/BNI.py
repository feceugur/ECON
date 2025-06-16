import torch
import trimesh
import cv2
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.sparse import (
    coo_matrix,
    csr_matrix,
    diags,
    hstack,
    spdiags,
    vstack,
)
from cupyx.scipy.sparse.linalg import cg

from lib.common.bni_numpy import bilateral_normal_integration, bilateral_normal_integration_upt, construct_facets_from, map_depth_map_to_point_clouds
from lib.common.BNI_utils import (
    depth_inverse_transform,
    double_side_bilateral_normal_integration,
    remove_stretched_faces,
    verts_inverse_transform,
)



class BNI:
    def __init__(self, dir_path, name, BNI_dict, cfg, device):

        self.scale = 256.0
        self.cfg = cfg
        self.name = name

        self.normal_front = BNI_dict["normal_F"]
        self.normal_back = BNI_dict["normal_B"]
        self.mask = BNI_dict["mask"]

        self.depth_front = BNI_dict["depth_F"]
        self.depth_back = BNI_dict["depth_B"]
        self.depth_mask = BNI_dict["depth_mask"]

        # hparam:
        # k --> smaller, keep continuity
        # lambda --> larger, more depth-awareness

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

    # code: https://github.com/hoshino042/bilateral_normal_integration
    # paper: Bilateral Normal Integration

    def extract_surface(self, verbose=True):

        bni_result = double_side_bilateral_normal_integration(
            normal_front=self.normal_front,
            normal_back=self.normal_back,
            normal_mask=self.mask,
            depth_front=self.depth_front * self.scale,
            depth_back=self.depth_back * self.scale,
            depth_mask=self.depth_mask,
            k=self.k,
            lambda_normal_back=1.0,
            lambda_depth_front=self.lambda1,
            lambda_depth_back=self.lambda1,
            lambda_boundary_consistency=self.boundary_consist,
            cut_intersection=self.cut_intersection,
        )

        F_verts = verts_inverse_transform(bni_result["F_verts"], self.scale)
        B_verts = verts_inverse_transform(bni_result["B_verts"], self.scale)

        self.F_depth = depth_inverse_transform(bni_result["F_depth"], self.scale)
        self.B_depth = depth_inverse_transform(bni_result["B_depth"], self.scale)

        F_B_verts = torch.cat((F_verts, B_verts), dim=0)
        F_B_faces = torch.cat(
            (bni_result["F_faces"], bni_result["B_faces"] + bni_result["F_faces"].max() + 1), dim=0
        )

        self.F_B_trimesh = trimesh.Trimesh(
            F_B_verts.float(), F_B_faces.long(), process=False, maintain_order=True
        )

        self.F_trimesh = trimesh.Trimesh(
            F_verts.float(), bni_result["F_faces"].long(), process=False, maintain_order=True
        )

        #self.B_trimesh = trimesh.Trimesh(
        #     B_verts.float(), bni_result["B_faces"].long(), process=False, maintain_order=True
        # )


    
    def extract_surface_multiview(self, normal_list, mask_list, depth_list):
        H, W, _ = normal_list[0].shape
        N_pix = H * W
        scale = self.scale

        # Step 1: Canonical pixel index map
        canonical_mask = np.any(np.stack(mask_list), axis=0)
        num_pixels = np.sum(canonical_mask)
        pixel_idx = np.zeros_like(canonical_mask, dtype=int)
        pixel_idx[canonical_mask] = np.arange(num_pixels)

        z_prior = np.zeros(N_pix)
        M_diag = np.zeros(N_pix)
        A_all = []
        b_all = []

        for normals, mask, depth in zip(normal_list, mask_list, depth_list):
            nx = normals[:, :, 1][mask]
            ny = normals[:, :, 0][mask]
            nz = -normals[:, :, 2][mask]

            A1, A2, A3, A4 = generate_dx_dy_upsample(mask, nz_horizontal=nz, nz_vertical=nz, step_size=1, pixel_idx=pixel_idx)
            A = vstack([A1, A2, A3, A4])
            b = np.concatenate([-nx, -nx, -ny, -ny])

            A_all.append(A)
            b_all.append(b)

            flat_mask = mask.flatten()
            z_prior[flat_mask] += depth.flatten()[flat_mask]
            M_diag[flat_mask] += 1

        A_total = vstack(A_all)
        b_total = np.concatenate(b_all)

        valid_depth = M_diag > 0
        M_diag[valid_depth] = 1.0 / M_diag[valid_depth]
        M = diags(self.cfg['lambda1'] * M_diag)
        z_prior = z_prior * M_diag

        ATA = A_total.T @ A_total + M
        ATb = A_total.T @ b_total + M @ z_prior

        z0 = np.zeros(N_pix)
        z, _ = cg(ATA, ATb, x0=z0, maxiter=5000, tol=1e-3)
        depth_map = z.reshape(H, W)

        # Combine all masks into one for final mesh creation
        mask_total = np.any(np.stack(mask_list), axis=0)
        verts = map_depth_map_to_point_clouds(depth_map, mask_total, K=None, step_size=1)
        faces = construct_facets_from(mask_total)
        faces = np.concatenate((faces[:, [1, 2, 3]], faces[:, [1, 3, 4]]), axis=0)
        verts, faces = remove_stretched_faces(verts, faces)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        self.F_B_trimesh = mesh
        self.F_B_surface = verts
        self.F_depth = depth_map

        return {
            "verts": torch.as_tensor(verts).float(),
            "faces": torch.as_tensor(faces).long(),
            "depth": torch.as_tensor(depth_map).float(),
            "mesh": mesh
        }


if __name__ == "__main__":

    root = "/home/yxiu/Code/ECON/results/examples/BNI"
    npy_file = f"{root}/304e9c4798a8c3967de7c74c24ef2e38.npy"
    bni_dict = np.load(npy_file, allow_pickle=True).item()

    default_cfg = {'k': 2, 'lambda1': 1e-4, 'boundary_consist': 1e-6}

    # for k in [1, 2, 4, 10, 100]:
    #     default_cfg['k'] = k
    # for k in [1e-8, 1e-4, 1e-2, 1e-1, 1]:
    # default_cfg['lambda1'] = k
    # for k in [1e-4, 1e-2, 0]:
    # default_cfg['boundary_consist'] = k

    bni_object = BNI(
        osp.dirname(npy_file), osp.basename(npy_file), bni_dict, default_cfg,
        torch.device('cuda:0')
    )

    bni_object.extract_surface()
    bni_object.F_trimesh_list[0].export(osp.join(osp.dirname(npy_file), "F.obj"))
    bni_object.B_trimesh.export(osp.join(osp.dirname(npy_file), "B.obj"))
    bni_object.F_B_trimesh.export(osp.join(osp.dirname(npy_file), "BNI.obj"))
