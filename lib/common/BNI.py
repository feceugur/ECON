import torch
import trimesh
import cv2
import os

from lib.common.bni_numpy import bilateral_normal_integration
from lib.common.BNI_utils import (
    depth_inverse_transform,
    double_side_bilateral_normal_integration,
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
        

        """
        # Front estimation
        depth_front_est, front_surface, wu_map_front, wv_map_front, energy_list_front = bilateral_normal_integration(
            normal_map=self.normal_front,
            normal_mask=self.mask,
            k=2,  # or other value you prefer
            depth_map=self.depth_front,  # optional, if you have it
            depth_mask=self.depth_mask,  # optional, if you have it
            lambda1=1e-4,  # or other value depending on your need
            K=None,
            step_size=1,
            max_iter=150
        )

        # Back estimation
        depth_back_est, back_surface, wu_map_back, wv_map_back, energy_list_back = bilateral_normal_integration(
            normal_map=self.normal_back,
            normal_mask=self.mask,
            k=2,
            depth_map=self.depth_back,  # optional
            depth_mask=self.depth_mask,  # optional
            lambda1=1e-4,  # typically stronger if you have a good prior for back
            K=None,
            step_size=1,
            max_iter=150
        )

        bni_result = {
            "F_verts": torch.as_tensor(front_surface.points).float(), 
            "F_faces": torch.as_tensor(front_surface.faces.reshape(-1, 5)[:, 1:]).long(), 
            "B_verts": torch.as_tensor(back_surface.points).float(), 
            "B_faces": torch.as_tensor(back_surface.faces.reshape(-1, 5)[:, 1:]).long(), 
            "F_depth": torch.as_tensor(depth_front_est).float(), 
            "B_depth": torch.as_tensor(depth_back_est).float()
        }

        def save_mesh_as_obj(vertices, faces, save_path):
            mesh = trimesh.Trimesh(vertices, faces)
            mesh.export(save_path)
            print(f"Saved mesh to {save_path}")

        # Get numpy arrays from result tensors
        F_verts_np = bni_result["F_verts"].cpu().numpy()
        F_faces_np = bni_result["F_faces"].cpu().numpy()
        B_verts_np = bni_result["B_verts"].cpu().numpy()
        B_faces_np = bni_result["B_faces"].cpu().numpy()

        # Save paths
        save_dir = "./results_thuman/infer_two/carla/obj"
        os.makedirs(save_dir, exist_ok=True)

        save_mesh_as_obj(F_verts_np, F_faces_np, os.path.join(save_dir, "front_mesh.obj"))
        save_mesh_as_obj(B_verts_np, B_faces_np, os.path.join(save_dir, "back_mesh.obj"))
        """

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

        self.B_trimesh = trimesh.Trimesh(
            B_verts.float(), bni_result["B_faces"].long(), process=False, maintain_order=True
        )

        # Save front mesh
        self.F_trimesh.export("front_mesh.obj")

        # Save back mesh
        self.B_trimesh.export("back_mesh.obj")

if __name__ == "__main__":

    import os.path as osp

    import numpy as np
    from tqdm import tqdm

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
    bni_object.F_trimesh.export(osp.join(osp.dirname(npy_file), "F.obj"))
    bni_object.B_trimesh.export(osp.join(osp.dirname(npy_file), "B.obj"))
    bni_object.F_B_trimesh.export(osp.join(osp.dirname(npy_file), "BNI.obj"))
