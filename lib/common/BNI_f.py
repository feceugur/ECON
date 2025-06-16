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

from lib.common.BNI_multi import double_side_bilateral_normal_integration
from lib.common.BNI_utils import (
    depth_inverse_transform,
    remove_stretched_faces,
    verts_inverse_transform,
)


class BNI_f:
    def __init__(self, dir_path, name, BNI_dict, cfg, device):
        """
        Initialize BNI class for multi-view bilateral normal integration.
        
        Args:
            dir_path (str): Directory path for saving results
            name (str): Name identifier for the current processing
            BNI_dict (dict): Dictionary containing normal maps, masks, and depth maps
            cfg (dict): Configuration parameters
            device (torch.device): Device to use for computations
        """
        self.scale = 256.0
        self.cfg = cfg
        self.name = name
        self.device = device
        self.export_dir = dir_path

        # Extract multi-view data
        self.normal_list = BNI_dict["normal_list"]
        self.mask_list = BNI_dict["mask_list"]
        self.depth_list = BNI_dict["depth_list"]

        # Extract single view data for backward compatibility
        self.normal_front = BNI_dict["normal_F"]
        self.normal_back = BNI_dict["normal_B"]
        self.mask = BNI_dict["mask"]
        self.depth_front = BNI_dict["depth_F"]
        self.depth_back = BNI_dict["depth_B"]
        self.depth_mask = BNI_dict["depth_mask"]

        # Hyperparameters
        self.k = self.cfg.get('k', 2)
        self.lambda1 = self.cfg.get('lambda1', 1e-4)
        self.boundary_consist = self.cfg.get('boundary_consist', 1e-6)
        self.cut_intersection = self.cfg.get('cut_intersection', True)

        # Initialize result storage
        self.F_B_surface = None
        self.F_B_trimesh = None
        self.F_depth = None
        self.B_depth = None

    def extract_surface(self, verbose=True):
        """
        Extract surface using multi-view bilateral normal integration.
        
        Args:
            verbose (bool): Whether to print progress information
        """
        # Scale depth maps
        scaled_depth_list = [depth * self.scale for depth in self.depth_list]
        
        # Perform multi-view integration
        bni_result = double_side_bilateral_normal_integration(
            normal_list=self.normal_list,
            mask_list=self.mask_list,
            depth_list=scaled_depth_list,
            k=self.k,
            lambda_normal_back=1.0,
            lambda_depth_front=self.lambda1,
            lambda_depth_back=self.lambda1,
            lambda_boundary_consistency=self.boundary_consist,
            cut_intersection=self.cut_intersection,
        )

        # Process results
        if "mesh" in bni_result:
            self.F_B_trimesh = bni_result["mesh"]
            self.F_B_surface = bni_result["vertices"]
            self.F_depth = depth_inverse_transform(bni_result["depth_map"], self.scale)
            
            # Save results if export directory is provided
            if self.export_dir:
                os.makedirs(self.export_dir, exist_ok=True)
                self.F_B_trimesh.export(osp.join(self.export_dir, f"{self.name}_multi.obj"))
                
                # Save depth maps
                np.save(osp.join(self.export_dir, f"{self.name}_depth.npy"), self.F_depth)
                
                if verbose:
                    print(f"Saved results to {self.export_dir}")

        return bni_result

    def extract_surface_single_view(self, verbose=True):
        """
        Extract surface using single-view bilateral normal integration (for backward compatibility).
        
        Args:
            verbose (bool): Whether to print progress information
        """
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

        self.B_trimesh = trimesh.Trimesh(
            B_verts.float(), bni_result["B_faces"].long(), process=False, maintain_order=True
        )

        return bni_result


if __name__ == "__main__":
    # Example usage
    root = "/path/to/results"
    npy_file = f"{root}/example.npy"
    bni_dict = np.load(npy_file, allow_pickle=True).item()

    default_cfg = {
        'k': 2,
        'lambda1': 1e-4,
        'boundary_consist': 1e-6,
        'cut_intersection': True
    }

    bni_object = BNI_f(
        osp.dirname(npy_file),
        osp.basename(npy_file),
        bni_dict,
        default_cfg,
        torch.device('cuda:0')
    )

    # Use multi-view integration
    bni_object.extract_surface()
    
    # Or use single-view integration for backward compatibility
    # bni_object.extract_surface_single_view()
