# -*- coding: utf-8 -*-
"""
BNIPipeline – end‑to‑end thin wrapper around *multi_view_d_bini*.

Changes compared with the previous draft
========================================
* **multi_view_d_bini signature has cameras=**, not Ks=/extrinsics=.  The
  wrapper now builds a list of Camera objects (imported from the solver file)
  and passes it via the correct keyword.
* Tidied tensor→NumPy conversion and made channel‑order explicit.
* Added a trivial front‑mesh fusion step so the caller gets **one** watertight
  surface in world space (you can replace it with Poisson if preferred).

Copy‑paste this file next to *multi_view_d_bini.py*.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from apps.CameraTransformManager import CameraTransformManager
import numpy as np
import torch
import trimesh
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from apps.multi_view_d_bini import multi_view_d_bini, Camera

# -----------------------------------------------------------------------------
# Quaternion utils ------------------------------------------------------------
# -----------------------------------------------------------------------------

def quat_wxyz_to_R(q: List[float]) -> np.ndarray:
    """Convert quaternion **(w,x,y,z)** to a 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),         2 * (x * z + y * w)],
        [2 * (x * y + z * w),         1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [2 * (x * z - y * w),         2 * (y * z + x * w),         1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)

def apply_blender2cv_to_smpl(smpl_verts: np.ndarray) -> np.ndarray:
    """Flip Y and Z on every vertex to match our camera convention."""
    # maps (X, Y, Z) -> ( X, -Z,  Y )
    B2C = np.array([
      [0, 0,  -1],
      [-1, 0, 0],
      [0, 1,  0],
    ], dtype=np.float32)
    return (B2C @ smpl_verts.T).T  # (N,3)

def debug_projection_overlay(smpl_vertices, cameras, masks, image_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    smpl_vertices = apply_blender2cv_to_smpl(smpl_vertices)
    H, W = image_size
    for i, cam in enumerate(cameras):
        gray = (masks[i].astype(np.uint8) * 255)
        img = np.stack([gray] * 3, axis=-1)  # Shape: (H, W, 3)
        img = np.squeeze(img)  # Ensure shape is (H, W, 3)

        # Ensure cam.R, cam.t, and cam.K are on CPU and numpy
        R = cam.R.detach().cpu().numpy() if hasattr(cam.R, 'detach') else np.asarray(cam.R)
        t = cam.t.detach().cpu().numpy() if hasattr(cam.t, 'detach') else np.asarray(cam.t)
        K = cam.K.detach().cpu().numpy() if hasattr(cam.K, 'detach') else np.asarray(cam.K)

        num_points = 0
        for pt in smpl_vertices:
            pt_cam = R @ pt + t
            if pt_cam[2] <= 0:
                continue
            uv = K @ pt_cam
            u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(img, (int(u), int(v)), radius=2, color=(0, 255, 0), thickness=-1)
                num_points += 1
        print(f"View {i}: Drew {num_points} points.")
        print(f"Saving overlay image to {os.path.join(output_dir, f'proj_overlay_view_{i}.png')}, img shape: {img.shape}")
        cv2.imwrite(os.path.join(output_dir, f"proj_overlay_view_{i}.png"), img)
        print("Image saved.")

def visualize_uv_projection_with_crop(
    smpl_vertices: np.ndarray,
    cameras: List[Camera],
    masks: List[np.ndarray],
    M_crop: List[np.ndarray],
    output_dir: str,
    max_points: int = 100
):
    os.makedirs(output_dir, exist_ok=True)
    H, W = masks[0].shape

    for view_idx, cam in enumerate(cameras):
        mask = masks[view_idx]
        img = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

        M = M_crop[view_idx]
        pre_crop_pts, post_crop_pts = [], []

        for i, pt in enumerate(smpl_vertices[:max_points]):
            pt_cam = cam.R @ pt + cam.t
            if pt_cam[2] <= 0:
                continue

            uv = cam.K @ pt_cam
            u, v = uv[0] / uv[2], uv[1] / uv[2]
            uv_h = np.array([u, v, 1.0], dtype=np.float32)
            uv_crop = M @ uv_h

            pre_crop_pts.append((u, v))
            uv_crop = (M_crop[view_idx] @ uv_h).cpu().numpy().squeeze()
            u_crop, v_crop = float(uv_crop[0]), float(uv_crop[1])
            post_crop_pts.append((u_crop, v_crop))
            #print(f"View {view_idx} M_crop: {M_crop[view_idx]}")

        # --- Plot ---
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap="gray")
        ax.set_title(f"View {view_idx} – Pre/Post-Crop Projection")

        for (u, v), (uc, vc) in zip(pre_crop_pts, post_crop_pts):
            # Green: pre-crop (raw projection)
            ax.add_patch(patches.Circle((u, v), radius=1.5, color='lime', label='pre-crop'))
            # Red: post-crop (after M_crop)
            ax.add_patch(patches.Circle((uc, vc), radius=1.5, color='red', label='post-crop'))

        ax.set_xlim([0, W])
        ax.set_ylim([H, 0])
        fig.savefig(os.path.join(output_dir, f"uv_crop_debug_view_{view_idx}.png"))
        plt.close()

def detailed_debug_projection(smpl_vertices, cameras, masks, image_size, n_samples=5):
    """
    For each camera:
      1) Print world‐space centre & axes
      2) Print mask bbox
      3) Compute all P_cam = R @ P + t
         • Print min/max for X_cam, Y_cam, Z_cam (front only & all)
         • Print how many are in front (Z_cam>0)
      4) Take the first n_samples verts and print their P_cam
      5) Compute uv on just those n_samples and print them
    """
    H, W = image_size

    # 1) get verts as (N,3)
    if isinstance(smpl_vertices, np.ndarray):
        verts = smpl_vertices.reshape(-1, 3)
    else:
        verts = smpl_vertices.detach().cpu().numpy().reshape(-1, 3)

    for i, (cam, mask) in enumerate(zip(cameras, masks)):
        print(f"\n—— CAMERA {i} ——")
        # a) world‐space centre & axes
        R = cam.R.detach().cpu().numpy()
        t = cam.t.detach().cpu().numpy()
        C = -R.T @ t
        print("  world‐space centre:", np.round(C,3))
        axes = R.T
        print("  axes (world) →  X:", np.round(axes[:,0],3),
              " Y:", np.round(axes[:,1],3),
              " Z:", np.round(axes[:,2],3))

        # b) mask bbox
        m2 = mask
        ys, xs = np.where(m2>0)
        if len(xs):
            print(f"  mask bbox x [{xs.min()},{xs.max()}], y [{ys.min()},{ys.max()}]")
        else:
            print("  mask empty!")

        # c) camera‐space coords
        #    we can vectorize this for all points
        P_cam_all = (R @ verts.T) + t[:,None]   # shape (3, N)
        Xc, Yc, Zc = P_cam_all[0], P_cam_all[1], P_cam_all[2]

        # stats on all points
        print(f"  P_cam     X range: [{Xc.min():.3f},{Xc.max():.3f}]")
        print(f"            Y range: [{Yc.min():.3f},{Yc.max():.3f}]")
        print(f"            Z range: [{Zc.min():.3f},{Zc.max():.3f}]")
        front_mask = Zc > 0
        print(f"  in front  {front_mask.sum()}/{len(verts)} points")

        # stats on front points only
        if front_mask.any():
            Xf, Yf, Zf = Xc[front_mask], Yc[front_mask], Zc[front_mask]
            print(f"  front‐only X range: [{Xf.min():.3f},{Xf.max():.3f}]")
            print(f"            Y range: [{Yf.min():.3f},{Yf.max():.3f}]")
            print(f"            Z range: [{Zf.min():.3f},{Zf.max():.3f}]")
        else:
            print("  no points in front of camera!")

        # d) sample some points to see exact values
        print(f"\n  Sample P_cam for first {n_samples} verts:")
        for idx in range(min(n_samples, len(verts))):
            pc = P_cam_all[:, idx]
            print(f"    V[{idx}]: (X={pc[0]:.3f}, Y={pc[1]:.3f}, Z={pc[2]:.3f})")

        # e) project those samples into uv
        print(f"\n  Sample uv (first {n_samples} verts):")
        K = cam.K.detach().cpu().numpy()
        for idx in range(min(n_samples, len(verts))):
            pc = P_cam_all[:, idx]
            if pc[2] <= 0:
                print(f"    V[{idx}] behind camera, Z={pc[2]:.3f}")
                continue
            uv_h = K @ pc
            u, v = uv_h[0]/uv_h[2], uv_h[1]/uv_h[2]
            print(f"    V[{idx}]  u={u:.1f}, v={v:.1f}")

def visualize_smpl_projections(
    smpl_vertices: np.ndarray,
    cameras: List[Camera],
    masks: List[np.ndarray],
    output_dir: str,
    max_points: int = 50
):
    os.makedirs(output_dir, exist_ok=True)
    H, W = masks[0].shape
    visible_counts = []

    for v_idx, cam in enumerate(cameras):
        mask = masks[v_idx]
        img = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

        visible_points_uv = []

        for point in smpl_vertices:
            pt_cam = cam.R @ point + cam.t
            uv = cam.K @ pt_cam
            u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))

            if 0 <= u < W and 0 <= v < H and mask[v, u]:
                visible_points_uv.append((u, v))

        visible_counts.append(len(visible_points_uv))

        # Visualize with projected points
        plt.figure()
        plt.imshow(mask, cmap="gray")
        if visible_points_uv:
            u_coords, v_coords = zip(*visible_points_uv)
            plt.scatter(u_coords, v_coords, c="red", s=3, marker="o", label="SMPL Proj")
            plt.legend()
        plt.title(f"SMPL Proj Overlay – View {v_idx}")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"proj_overlay_{v_idx}.png"))
        plt.close()

    print(f"DEBUG: Visible SMPL points per view: {visible_counts}")
    print(f"DEBUG: Total visible SMPL points across all views: {sum(visible_counts)}")


def project_point_to_mask_space(
    pt_world: np.ndarray,
    cam_K: np.ndarray,
    cam_R: np.ndarray,
    cam_t: np.ndarray,
    image_size: tuple[int, int],        # (H_orig, W_orig)
    bbox: tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max)
    target_square_res: int = 1024,
    final_mask_res: int = 512
) -> tuple[float, float] | None:
    """
    Projects a 3D point to 2D 512×512 mask space, manually simulating the effect of M_square and M_crop.
    Returns (u, v) in mask space, or None if behind camera.
    """
    H_orig, W_orig = image_size
    x_min, y_min, x_max, y_max = bbox

    # Step 1: Project to camera image plane
    pt_cam = cam_R @ pt_world + cam_t
    if pt_cam[2] <= 0:
        return None  # behind camera

    uv = cam_K @ pt_cam
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]

    # Step 2: Resize original image to centered square
    scale_square = target_square_res / max(W_orig, H_orig)
    u_square = u * scale_square + (target_square_res - W_orig * scale_square) / 2
    v_square = v * scale_square + (target_square_res - H_orig * scale_square) / 2

    # Step 3: Crop to bbox and scale to 512×512
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    scale_crop = final_mask_res / max(crop_w, crop_h)

    u_crop = (u_square - x_min) * scale_crop
    v_crop = (v_square - y_min) * scale_crop

    return u_crop, v_crop

def get_bbox_from_mask(mask: np.ndarray) -> list[float]:
    """Compute tight [x_min, y_min, x_max, y_max] bbox from a binary mask."""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, 0, 512, 512]  # fallback if empty mask
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


# -----------------------------------------------------------------------------
# Pipeline --------------------------------------------------------------------
# -----------------------------------------------------------------------------

class BNIPipeline:
    """High‑level helper that hides all boiler‑plate around *multi_view_d_bini*.

    Parameters
    ----------
    camera_json_path : str | Path
        Path to the *camera_parameters.json* blob you pasted in chat.
    lambda_c         : float, optional
        Weight of the cross‑view consistency term. 0 ⇒ disabled.
    lambda_d, lambda_s, k : float, optional
        Same symbols as in the BiNI paper (depth prior, silhouette prior, IRLS‑k).
    """

    def __init__(
        self,
        *,
        lambda_c: float = 1e-4,
        lambda_d: float = 1e-4,
        lambda_s: float = 1e-6,
        k: float = 2.0,
        cameras: List[Camera],
    ) -> None:
        self.lambda_c = float(lambda_c)
        self.lambda_d = float(lambda_d)
        self.lambda_s = float(lambda_s)
        self.k = float(k)
        self.cameras = cameras
        if cameras is None:
            cam_meta = json.loads(Path(camera_json_path).read_text())
            cam_meta = sorted(cam_meta, key=lambda x: x["frame"])  # guarantee order

            self.cameras: List[Camera] = []
            target_frame_id = cam_meta[0]["frame"]
            transform_manager = CameraTransformManager(camera_json_path, target_frame=target_frame_id, device="cpu")
            for cam in cam_meta:
                f = cam["focal_length_px"]
                K = np.array(
                    [[f*0.28, 0, 256], [0, f*0.28, 256], [0, 0, 1]],
                    dtype=np.float32,
                )

                frame_id = cam["frame"]
                T_view_to_canonical = transform_manager.get_transform_to_target(frame_id)
                T_canonical_to_view = torch.inverse(T_view_to_canonical)
                R_final, t_final = T_canonical_to_view[:3, :3], T_canonical_to_view[:3, 3]
                Rt = np.eye(4, dtype=np.float32)
                Rt[:3, :3] = R_final
                Rt[:3, 3] = t_final
                self.cameras.append(Camera(K=K, Rt=Rt))

            for i, cam in enumerate(self.cameras):
                print(f"Camera {i} position: {cam.Rt[:3, 3]}")

    # ------------------------------------------------------------------
    # private helpers ---------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _to_np(x: Any) -> np.ndarray:
        """Detach & move to CPU, then to NumPy."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return np.asarray(x)

    def _split_views(self, in_tensor_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return a list sorted by **view_i** key."""
        keys = sorted([k for k in in_tensor_dict if k.startswith("view_")],
                      key=lambda s: int(s.split("_")[1]))
        return [in_tensor_dict[k] for k in keys]
    
    
    """
    def _generate_correspondences_from_smpl(
        self,
        smpl_vertices: np.ndarray,
        cameras: List[Camera],
        masks: List[np.ndarray],
        image_size: Tuple[int, int],
        min_views: int = 2
    ) -> List[Tuple[int, np.ndarray, int, np.ndarray]]:
        H, W = image_size
        V = len(cameras)
        correspondences = []
        kept_points = 0

        for i, point in enumerate(smpl_vertices):  # shared across all views
            visible = []

            for view_idx in range(V):
                cam = cameras[view_idx]
                mask = masks[view_idx]
                pt_cam = cam.Rt[:3, :3] @ point + cam.Rt[:3, 3]
                if pt_cam[2] <= 0:
                    continue
                uv = cam.K @ pt_cam
                u, v = uv[0] / uv[2], uv[1] / uv[2]
                u_int, v_int = int(round(u)), int(round(v))
                if 0 <= u_int < W and 0 <= v_int < H and mask[v_int, u_int]:
                    visible.append((view_idx, point.copy()))  # point is in world space

            if len(visible) >= min_views:
                for i in range(len(visible)):
                    for j in range(i + 1, len(visible)):
                        v1, p1 = visible[i]
                        v2, p2 = visible[j]
                        correspondences.append((v1, p1, v2, p2))
                kept_points += 1

        print(f"→ Kept {kept_points} SMPL points with ≥{min_views} views")
        print(f"Total correspondences: {len(correspondences)}")
        return correspondences
    """
    def _generate_correspondences_from_smpl(
        self,
        smpl_vertices: np.ndarray,
        cameras: List[Camera],
        masks: List[np.ndarray],
        min_views: int = 2
    ) -> List[Tuple[int, int, int, int]]:
        """
        For each SMPL vertex (in world space), project into each mask.
        If it falls inside ≥min_views masks, emit all pairwise correspondences
        (view_i, idx_i, view_j, idx_j) where idx is the flattened mask index.
        """
        smpl_vertices = apply_blender2cv_to_smpl(smpl_vertices)

        # image size
        H, W = masks[0].shape if masks[0].ndim == 2 else masks[0].squeeze().shape
        V = len(cameras)

        # build a map from (y,x) to flat index inside each mask
        pix2flat = []
        for m in masks:
            m2 = np.squeeze(m)
            idx = np.zeros_like(m2, dtype=np.int32)
            idx[m2] = np.arange(m2.sum(), dtype=np.int32)
            pix2flat.append(idx)

        correspondences = []
        kept_points = 0

        for P in smpl_vertices:  # P is (3,) world-space point
            visible = []         # will collect (view_idx, flat_idx)

            for v_idx, (cam, mask) in enumerate(zip(cameras, masks)):
                mask2 = np.squeeze(mask)
                # grab extrinsics & intrinsics as numpy
                R = cam.R.detach().cpu().numpy() if hasattr(cam.R, 'detach') else np.asarray(cam.R)
                t = cam.t.detach().cpu().numpy() if hasattr(cam.t, 'detach') else np.asarray(cam.t)
                K = cam.K.detach().cpu().numpy() if hasattr(cam.K, 'detach') else np.asarray(cam.K)

                # ==== CHANGE: world→camera ====
                Pc = R @ P + t       # shape (3,)
                if Pc[2] <= 0:
                    continue         # behind the camera

                # project to homogeneous pixel coords
                uv = K @ Pc         # shape (3,)
                u, v = uv[0]/uv[2], uv[1]/uv[2]
                ui, vi = int(round(u)), int(round(v))

                # check bounds + mask
                if 0 <= ui < W and 0 <= vi < H and mask2[vi, ui]:
                    flat_idx = int(pix2flat[v_idx][vi, ui])
                    visible.append((v_idx, flat_idx))

            if len(visible) >= min_views:
                kept_points += 1
                # emit all unique pairs among the views where P was visible
                for i in range(len(visible)):
                    for j in range(i+1, len(visible)):
                        v1, p1 = visible[i]
                        v2, p2 = visible[j]
                        correspondences.append((v1, p1, v2, p2))

        print(f"→ Kept {kept_points} SMPL verts with ≥{min_views} views")
        print(f"Total correspondences: {len(correspondences)}")
        return correspondences




    # ------------------------------------------------------------------
    # public API --------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, in_tensor_dict: Dict[str, Any], smpl_vertices: np.ndarray) -> trimesh.Trimesh:
        """Execute the multi‑view solver and fuse the front meshes."""
        views = self._split_views(in_tensor_dict)
        V = len(views)

        normals_front, normals_back, normal_masks = [], [], []
        depth_front, depth_back, depth_masks = [], [], []
        M_crop_list = []

        for v in range(V):
            blob = views[v]

            # normals: (1,C,H,W) → (H,W,C)
            nF = self._to_np(blob["normal_F"])[0].transpose(1, 2, 0)
            nB = self._to_np(blob["normal_B"])[0].transpose(1, 2, 0)
            normals_front.append(nF)
            normals_back.append(nB)

            # mask, depth ------------------------------------------------
            m = self._to_np(blob["img_mask"])[0].astype(bool)
            normal_masks.append(m)

            dF = self._to_np(blob["depth_F"])[0]
            dB = self._to_np(blob["depth_B"])[0]
            depth_front.append(dF)
            depth_back.append(dB)
            depth_masks.append((dF != -1).astype(np.uint8))
            #M_crop_list.append(blob["M_crop"])
        # --------------------------------------------------------------
        # Generate correspondences from SMPL
        # --------------------------------------------------------------
        H, W = normal_masks[0].shape
        smpl_vertices_np = smpl_vertices.squeeze(0) if smpl_vertices.ndim == 3 else smpl_vertices
        
        n_valid = [int(m.sum()) for m in normal_masks]      # e.g. [N₀, N₁, …]
        offsets = np.zeros(len(n_valid)+1, dtype=int)
        offsets[1:] = np.cumsum(n_valid)                    # offsets[v] is start index of view v

        detailed_debug_projection(smpl_vertices, self.cameras, normal_masks, (H, W))
        debug_projection_overlay(smpl_vertices_np, self.cameras, normal_masks, (H, W), "debug_overlay_all")


        #visualize_smpl_projections(
        #    smpl_vertices=smpl_vertices_np,
        #    cameras=self.cameras,
        #    masks=normal_masks,
        #    max_points=50,
        #    output_dir="projections_debug"
        #)

        correspondences = self._generate_correspondences_from_smpl(
           smpl_vertices=smpl_vertices_np[::10],
           cameras=self.cameras,
           masks=normal_masks)
    

        #  visualize_uv_projection_with_crop(
        #    smpl_vertices=smpl_vertices_np,
        #    cameras=self.cameras,
        #    masks=normal_masks,
        #    M_crop=M_crop_list,
        #    output_dir="uv_crop_debug"
        #)

        # --------------------------------------------------------------
        # Call solver
        # --------------------------------------------------------------

        # 3) wrap them
        dbini_cams = []
        for cam in self.cameras:
            # extract R, t, K from your loader‐Camera
            R = cam.R.detach().cpu().numpy() if hasattr(cam.R, 'detach') else cam.R
            t = cam.t.detach().cpu().numpy() if hasattr(cam.t, 'detach') else cam.t
            K = cam.K.detach().cpu().numpy()

            # build the 4×4 world←camera matrix
            Rt = np.eye(4, dtype=np.float32)
            Rt[:3, :3] = R
            Rt[:3,  3] = t

            # instantiate the d-BiNI Camera
            dbini_cams.append(Camera(K, Rt))

        results = multi_view_d_bini(
            normals_front,
            normals_back,
            normal_masks,
            depth_front=depth_front,
            depth_back=depth_back,
            depth_masks=depth_masks,
            cameras=dbini_cams,
            correspondences=correspondences,
            lambda_depth_front=self.lambda_d,
            lambda_depth_back=self.lambda_d * 100.0,
            lambda_boundary_consistency=self.lambda_s,
            lambda_cross_view=self.lambda_c,  # ENABLED
            k=self.k,
            offsets=offsets,
        )

        # --------------------------------------------------------------
        # Merge per‑view *front* meshes into one world‑space mesh
        # --------------------------------------------------------------
        verts_all, faces_all = [], []
        v_offset = 0
        for res in results:
            vtx = res["F_verts"]
            if isinstance(vtx, torch.Tensor):
                vtx = vtx.detach().cpu().numpy()
            faces = res["F_faces"]
            if isinstance(faces, torch.Tensor):
                faces = faces.detach().cpu().numpy()
            verts_all.append(vtx)
            faces_all.append(faces + v_offset)
            v_offset += vtx.shape[0]

        verts_all = np.vstack(verts_all)
        faces_all = np.vstack(faces_all)
        fused = trimesh.Trimesh(verts_all, faces_all)
        fused.remove_unreferenced_vertices()
        fused.apply_translation(-fused.bounding_box.centroid)

        # Print fused mesh bounding box
        min_fused = verts_all.min(axis=0)
        max_fused = verts_all.max(axis=0)
        print(f"[DEBUG] Fused mesh bounding box: Min {min_fused}, Max {max_fused}, Size {max_fused - min_fused}")

        print("Camera 0 R:\n", self.cameras[0].R[:3, :3])
        print("Camera 0 t:", self.cameras[0].t[:3])
        print("Sample SMPL vertex:", smpl_vertices_np[0])
        
        # Convert camera parameters to numpy arrays
        R = self.cameras[0].R[:3, :3].detach().cpu().numpy() if hasattr(self.cameras[0].R, 'detach') else np.asarray(self.cameras[0].R[:3, :3])
        t = self.cameras[0].t[:3].detach().cpu().numpy() if hasattr(self.cameras[0].t, 'detach') else np.asarray(self.cameras[0].t[:3])
        
        pt_cam = R @ (smpl_vertices_np[0] - t)
        print("Transformed pt_cam:", pt_cam)

        return fused

        

# -----------------------------------------------------------------------------
# Quick test ------------------------------------------------------------------
# -----------------------------------------------------------------------------
"""
if __name__ == "__main__":
    import pickle, sys

    if len(sys.argv) != 3:
        print("Usage: python bni_pipeline.py <camera_json> <in_tensor.pkl>")
        sys.exit(0)

    cam_json, pkl_path = sys.argv[1:]

    with open(pkl_path, "rb") as fh:
        in_tensor = pickle.load(fh)

    pipeline = BNIPipeline(cam_json, lambda_c=0.0)
    mesh = pipeline.run(in_tensor)
    mesh.export("bni_mesh.ply")
    print("✓ Wrote bni_mesh.ply")
"""