from apps.CameraTransformManager import CameraTransformManager
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import mcubes
import json
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

# ===============================================================================================
# 1. Device Configuration
# ===============================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# ===============================================================================================
# 2. Utility and Visualization Functions
# ===============================================================================================
def quaternion_to_rotation_matrix(q):
    """Converts a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

def load_cameras_from_json(json_path, device="cpu", use_blender_to_cv=True):
    with open(json_path, 'r') as f:
        cam_params = json.load(f)
    B2C = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device))
    cameras = []
    for cam_data in cam_params:
        loc = np.array(cam_data["location"], dtype=np.float32)
        quat = cam_data["quaternion"]
        R_bw = quaternion_to_rotation_matrix(quat)
        pose_c2w = np.eye(4, dtype=np.float32)
        pose_c2w[:3, :3] = R_bw
        pose_c2w[:3, 3] = loc
        pose_c2w = torch.tensor(pose_c2w, device=device)
        pose_w2c = torch.linalg.inv(pose_c2w)
        if use_blender_to_cv:
            pose_w2c = B2C @ pose_w2c
        R = pose_w2c[:3, :3]
        t = pose_w2c[:3, 3]
        w, h = cam_data["image_size"]
        sx, sy = 512.0 / w, 512.0 / h
        f = cam_data["focal_length_px"]
        K = torch.tensor([[f * 0.28, 0.0, 256.0], [0.0, f * 0.28, 256.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        cameras.append(Camera(R, t, K))
        C = (-R.T @ t).cpu().numpy()
        print(f"Camera {cam_data['frame']} center: {C}, JSON loc: {loc}")
    return cameras

def visualize_progress(sdf_net, camera, gt_mask, gt_normals, smpl_points, iteration, folder="debug_visuals", vis_res=256):
    os.makedirs(folder, exist_ok=True)
    H, W = gt_mask.shape
    u_grid, v_grid = torch.meshgrid(torch.linspace(0, W - 1, vis_res), torch.linspace(0, H - 1, vis_res), indexing='xy')
    pixels_uv = torch.stack([u_grid.ravel(), v_grid.ravel()], dim=-1).to(device)
    
    with torch.no_grad():
        rays_o, rays_d = camera.generate_rays(H, W, pixels_uv)
        near, far, bbox_hit_mask = get_ray_intersections(rays_o, rays_d, smpl_points)
        print(f"bbox_hit_mask sum: {bbox_hit_mask.sum().item()}")  # Debug: Check if any rays hit the bounding box
        surface_points = torch.zeros_like(rays_o)
        st_hit_full = torch.zeros_like(bbox_hit_mask)
        secant_hit_full = torch.zeros_like(bbox_hit_mask)
        if bbox_hit_mask.any():
            rays_o_valid, rays_d_valid = rays_o[bbox_hit_mask], rays_d[bbox_hit_mask]
            near_valid, far_valid = near[bbox_hit_mask], far[bbox_hit_mask]
            final_points, final_hit_mask, st_hits, secant_hits = hybrid_ray_tracer_for_visuals(sdf_net, rays_o_valid, rays_d_valid, near_valid, far_valid)
            print(f"final_hit_mask sum: {final_hit_mask.sum().item()}")  # Debug: Check if any surface points are found
            surface_points = surface_points.index_put((bbox_hit_mask,), final_points)
            st_hit_full = st_hit_full.index_put((bbox_hit_mask,), st_hits)
            secant_hit_full = secant_hit_full.index_put((bbox_hit_mask,), secant_hits)
        else:
            print("No rays hit the bounding box. Check camera setup or SMPL points.")
        
        bbox_hit_img = bbox_hit_mask.reshape(vis_res, vis_res).cpu().numpy()
        st_hit_img = st_hit_full.reshape(vis_res, vis_res).cpu().numpy()
        secant_hit_img = secant_hit_full.reshape(vis_res, vis_res).cpu().numpy()
        sdf_values = sdf_net.sdf(surface_points).reshape(vis_res, vis_res)
        rendered_silhouette = torch.sigmoid(-10.0 * sdf_values).cpu().numpy() if sdf_values.numel() > 0 else np.zeros((vis_res, vis_res))
        plane_coords = torch.stack([(pixels_uv[:, 0] / (vis_res - 1) - 0.5) * 2.0, (pixels_uv[:, 1] / (vis_res - 1) - 0.5) * 2.0, torch.zeros_like(pixels_uv[:, 0])], dim=-1)
        sdf_slice = sdf_net.sdf(plane_coords).reshape(vis_res, vis_res).cpu().numpy() if plane_coords.numel() > 0 else np.zeros((vis_res, vis_res))

    rendered_normals_img = np.zeros((vis_res, vis_res, 3))
    near_surface_mask = sdf_values.abs() < 0.1 if sdf_values.numel() > 0 else np.zeros((vis_res, vis_res), dtype=bool)
    valid_points = surface_points.reshape(vis_res, vis_res, 3)[near_surface_mask]
    if valid_points.shape[0] > 0:
        with torch.set_grad_enabled(True):
            valid_points = valid_points.detach().requires_grad_(True)
            sdf_vals, pred_normals_world = sdf_net.forward_with_grad(valid_points)
            if sdf_vals.numel() > 0 and pred_normals_world is not None:
                pred_normals_cam = F.normalize(torch.matmul(pred_normals_world, camera.R.T), dim=-1)
        with torch.no_grad():  # Detach gradients before converting to NumPy
            rendered_normals_img_flat = (pred_normals_cam.detach().cpu().numpy() + 1) / 2.0
            valid_indices = np.where(near_surface_mask)
            if len(valid_indices[0]) == rendered_normals_img_flat.shape[0]:
                rendered_normals_img[valid_indices] = rendered_normals_img_flat
    else:
        print("No valid points for normal rendering. Check surface_points or near_surface_mask.")

    gt_mask_vis = gt_mask.cpu().numpy()
    gt_normals_vis = ((gt_normals.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2.0)
    ray_diag_img = np.zeros((vis_res, vis_res, 3))
    ray_diag_img[bbox_hit_img] = [0, 0, 0.5]  # Dark Blue
    ray_diag_img[st_hit_img] = [0, 1, 0]      # Green
    ray_diag_img[secant_hit_img] = [1, 0, 0]  # Red

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Iteration {iteration}', fontsize=16)

    axes[0, 0].imshow(gt_mask_vis, cmap='gray'); axes[0, 0].set_title('GT Mask')
    axes[0, 1].imshow(rendered_silhouette, cmap='gray'); axes[0, 1].set_title('Rendered Silhouette')
    im = axes[0, 2].imshow(sdf_slice, cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[0, 2].set_title('SDF Slice at z=0'); fig.colorbar(im, ax=axes[0, 2])
    axes[0, 3].imshow(ray_diag_img); axes[0, 3].set_title('Ray Tracer Diagnostics')
    axes[0, 3].text(0.95, 0.07, 'Red: Secant Hit', color='white', ha='right', transform=axes[0, 3].transAxes, bbox=dict(facecolor='red', alpha=0.8))
    axes[0, 3].text(0.95, 0.14, 'Green: Sphere Trace Hit', color='black', ha='right', transform=axes[0, 3].transAxes, bbox=dict(facecolor='lime', alpha=0.8))

    axes[1, 0].imshow(gt_normals_vis); axes[1, 0].set_title('GT Normals')
    axes[1, 1].imshow(rendered_normals_img); axes[1, 1].set_title('Rendered Normals')
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')

    for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(folder, f'progress_{iteration:05d}.png')); plt.close(fig)
    print(f"--- Saved debug visualization for iteration {iteration} ---")

# ===============================================================================================
# 3. Core Model and Camera Classes
# ===============================================================================================
class SDFNetwork(nn.Module):
    def __init__(self, d_in=3, d_out=1, d_hidden=256, n_layers=8, skip_in=(4,), bias=0.5, geometric_init=True, weight_norm=True):
        super(SDFNetwork, self).__init__()
        self.skip_in = skip_in
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        for l in range(self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - d_in
            else:
                out_dim = dims[l + 1]
            in_dim = dims[l]
            lin = nn.Linear(in_dim, out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(in_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, f"lin{l}", lin)
        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        x = inputs
        for l in range(self.num_layers - 1):
            lin = getattr(self, f"lin{l}")
            if l in self.skip_in:
                x = torch.cat([x, inputs], dim=-1)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x

    def sdf(self, x):
        return self.forward(x)

    def forward_with_grad(self, x):
        """Compute SDF and gradients in one pass."""
        x = x.detach().requires_grad_(True)
        sdf = self.sdf(x)
        if sdf.requires_grad:
            grad = torch.autograd.grad(sdf.sum(), x, create_graph=True)[0]
        else:
            grad = torch.zeros_like(x)
        return sdf, grad

class Camera(nn.Module):
    def __init__(self, R, t, K):
        super(Camera, self).__init__()
        self.R = nn.Parameter(R)
        self.t = nn.Parameter(t)
        self.K = K.to(device)
        self.offset_2d = nn.Parameter(torch.zeros(1, 2, device=R.device), requires_grad=True)

    def generate_rays(self, H, W, pixels_uv):
        u, v = pixels_uv[:, 0], pixels_uv[:, 1]
        dirs_cam = torch.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], torch.ones_like(u)], -1)
        rays_d = F.normalize(torch.matmul(dirs_cam, self.R.T), dim=-1)
        rays_o = -torch.matmul(self.R.T, self.t).expand_as(rays_d)
        return rays_o, rays_d

    def project_points(self, points_3d):
        points_in_camera_space = points_3d @ self.R.T + self.t
        points_2d_homo = points_in_camera_space @ self.K.T
        depth = points_2d_homo[..., 2:3]
        projected_pixels = points_2d_homo[..., :2] / (depth + 1e-8)
        projected_pixels = projected_pixels + self.offset_2d
        return projected_pixels, depth > 0

# ===============================================================================================
# 4. Core Logic: Ray Tracing, Sampling, and Interpolation
# ===============================================================================================
def find_surface_secant(sdf_net, origins, directions, near, far, n_steps=64, n_secant_steps=8):
    t_vals = torch.linspace(0.0, 1.0, steps=n_steps, device=origins.device)
    z_vals = near.unsqueeze(-1) + t_vals * (far - near).unsqueeze(-1)
    points = origins.unsqueeze(-2) + directions.unsqueeze(-2) * z_vals.unsqueeze(-1)
    sdf_values = sdf_net.sdf(points.reshape(-1, 3)).reshape(origins.shape[0], n_steps)
    sign_change = (sdf_values[:, :-1] > 0) & (sdf_values[:, 1:] < 0)
    has_sign_change = sign_change.any(dim=1)
    if not has_sign_change.any():
        return torch.zeros_like(origins), torch.zeros_like(has_sign_change)
    first_idx = torch.argmax(sign_change[has_sign_change].int(), dim=1)
    z_low = z_vals[has_sign_change][torch.arange(first_idx.shape[0]), first_idx]
    z_high = z_vals[has_sign_change][torch.arange(first_idx.shape[0]), first_idx + 1]
    sdf_low = sdf_values[has_sign_change][torch.arange(first_idx.shape[0]), first_idx]
    sdf_high = sdf_values[has_sign_change][torch.arange(first_idx.shape[0]), first_idx + 1]
    origins_secant = origins[has_sign_change]
    directions_secant = directions[has_sign_change]
    for _ in range(n_secant_steps):
        z_pred = z_low - sdf_low * (z_high - z_low) / (sdf_high - sdf_low + 1e-8)
        p_mid = origins_secant + directions_secant * z_pred.unsqueeze(-1)
        sdf_mid = sdf_net.sdf(p_mid).squeeze(-1)
        ind_low = sdf_mid > 0
        z_low[ind_low] = z_pred[ind_low]
        sdf_low[ind_low] = sdf_mid[ind_low]
        ind_high = sdf_mid < 0
        z_high[ind_high] = z_pred[ind_high]
        sdf_high[ind_high] = sdf_mid[ind_high]
    final_z = z_low - sdf_low * (z_high - z_low) / (sdf_high - sdf_low + 1e-8)
    final_points = torch.zeros_like(origins)
    final_hit_mask = torch.zeros_like(has_sign_change)
    final_points[has_sign_change] = origins[has_sign_change] + directions[has_sign_change] * final_z.unsqueeze(-1)
    final_hit_mask[has_sign_change] = True
    return final_points, final_hit_mask

def hybrid_ray_tracer_for_visuals(sdf_net, origins, directions, near, far):
    st_points, st_hit_mask = ray_marcher(sdf_net, origins, directions, near, far, num_steps=16, threshold=5e-3)
    failed_mask = ~st_hit_mask
    secant_points = torch.zeros_like(origins)
    secant_hit_mask = torch.zeros_like(failed_mask)
    if failed_mask.any():
        secant_points_subset, secant_hit_mask_subset = find_surface_secant(sdf_net, origins[failed_mask], directions[failed_mask], near[failed_mask], far[failed_mask])
        secant_points[failed_mask] = secant_points_subset
        secant_hit_mask[failed_mask] = secant_hit_mask_subset
    final_points = torch.where(st_hit_mask.unsqueeze(-1), st_points, secant_points)
    final_hit_mask = st_hit_mask | secant_hit_mask
    return final_points, final_hit_mask, st_hit_mask, secant_hit_mask

def hybrid_ray_tracer_for_training(sdf_net, origins, directions, near, far):
    st_points, st_hit_mask = ray_marcher(sdf_net, origins, directions, near, far, num_steps=16, threshold=5e-3)
    failed_mask = ~st_hit_mask
    if not failed_mask.any():
        return st_points, st_hit_mask
    secant_points, secant_hit_mask = find_surface_secant(sdf_net, origins[failed_mask], directions[failed_mask], near[failed_mask], far[failed_mask])
    full_secant_points = torch.zeros_like(origins)
    full_secant_points[failed_mask] = secant_points
    full_secant_hit_mask = torch.zeros_like(failed_mask)
    full_secant_hit_mask[failed_mask] = secant_hit_mask
    final_points = torch.where(st_hit_mask.unsqueeze(-1), st_points, full_secant_points)
    final_hit_mask = st_hit_mask | full_secant_hit_mask
    return final_points, final_hit_mask

def get_ray_intersections(origins, directions, smpl_points, padding=0.2):
    if isinstance(smpl_points, np.ndarray):
        smpl_points = torch.from_numpy(smpl_points).to(origins.device)
    if smpl_points.ndim == 3:
        min_bound = smpl_points.min(axis=(0, 1))[0] - padding
        max_bound = smpl_points.max(axis=(0, 1))[0] + padding
    else:
        min_bound = smpl_points.min(dim=0)[0] - padding
        max_bound = smpl_points.max(dim=0)[0] + padding
    bounds = torch.stack([min_bound, max_bound])
    t_min = (bounds[0] - origins) / (directions + 1e-8)
    t_max = (bounds[1] - origins) / (directions + 1e-8)
    t1, t2 = torch.min(t_min, t_max), torch.max(t_min, t_max)
    near = torch.max(torch.stack([t1[:, 0], t1[:, 1], t1[:, 2]], dim=1), dim=1)[0]
    far = torch.min(torch.stack([t2[:, 0], t2[:, 1], t2[:, 2]], dim=1), dim=1)[0]
    valid_mask = (near < far) & (far > 0)
    return near, far, valid_mask

def ray_marcher(sdf_net, origins, directions, near, far, num_steps=64, threshold=1e-4):
    points = origins + directions * near.unsqueeze(-1)
    damping_factor = 0.5
    for i in range(num_steps):
        sdf = sdf_net.sdf(points).squeeze(-1)
        hit_mask_step = sdf.abs() < threshold
        if hit_mask_step.all():
            break
        step_size = sdf * damping_factor
        points = points + directions * step_size.unsqueeze(-1)
        dist_from_origin = torch.norm(points - origins, dim=-1)
        past_far_bound = dist_from_origin > far
        if past_far_bound.any():
            break
    final_sdf = sdf_net.sdf(points).squeeze(-1)
    hit_mask = final_sdf.abs() < threshold
    return points, hit_mask

def sample_pixels(mask, num_samples):
    fg_coords = torch.argwhere(mask > 0.5)
    bg_coords = torch.argwhere(mask <= 0.5)
    num_fg_samples = min(len(fg_coords), int(num_samples * 0.75))
    num_bg_samples = num_samples - num_fg_samples
    fg_indices = torch.randperm(len(fg_coords), device=device)[:num_fg_samples]
    bg_indices = torch.randperm(len(bg_coords), device=device)[:num_bg_samples]
    sampled_fg = fg_coords[fg_indices]
    sampled_bg = bg_coords[bg_indices]
    return torch.cat([sampled_fg, sampled_bg], dim=0)[:, [1, 0]].float()

def get_interp_values(values_map, points_2d, H, W):
    grid = points_2d.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
    grid = grid.unsqueeze(0).unsqueeze(0)
    return F.grid_sample(values_map.unsqueeze(0), grid, mode='bilinear', align_corners=True, padding_mode='border').squeeze().T

# ===============================================================================================
# 5. Main Optimization and Mesh Extraction Functions
# ===============================================================================================
def optimize_sdf(in_tensor, cameras, smpl_vertices, num_iterations=10000, num_samples_per_view=4096, vis_interval=500):
    debug_vis_folder = "debug_visuals"
    os.makedirs(debug_vis_folder, exist_ok=True)
    H, W = 512, 512
    sdf_net = SDFNetwork().to(device)
    num_views = len(cameras)
    normals_list = []
    for i in range(num_views):
        normal_map = in_tensor[f"view_{i}"]["normal_F"].to(device).squeeze()
        if normal_map.shape[-1] == 3:
            normal_map = normal_map.permute(2, 0, 1)
        normals_list.append(normal_map)
    masks_list = [torch.from_numpy(in_tensor[f"view_{i}"]["img_mask"]).to(device, dtype=torch.float32).squeeze() for i in range(num_views)]
    smpl_points = smpl_vertices.squeeze().to(device)
    phase1_iters = 2000
    phase2_iters = 5000
    optimizer = Adam(sdf_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    for iteration in range(num_iterations):
        phase = 1 if iteration < phase1_iters else (2 if iteration < phase1_iters + phase2_iters else 3)
        print(f"Phase: {phase}, Iteration: {iteration}")
        
        if iteration % vis_interval == 0:
            view_idx_vis = np.random.randint(0, num_views)
            with torch.no_grad():
                visualize_progress(sdf_net, cameras[view_idx_vis], masks_list[view_idx_vis], normals_list[view_idx_vis], smpl_points, iteration, folder=debug_vis_folder)
        
        optimizer.zero_grad()
        
        total_loss = 0.0
        mask_loss_val, normal_loss_val, eikonal_loss_val = 0.0, 0.0, 0.0
        
        if phase == 1:
            lambda_init, lambda_eikonal = 10.0, 0.1
            on_surface_points = smpl_points[torch.randperm(len(smpl_points), device=device)[:4096]]
            off_surface_points = on_surface_points + torch.randn_like(on_surface_points) * 0.05
            
            sdf_on, _ = sdf_net.forward_with_grad(on_surface_points)
            loss_init = F.l1_loss(sdf_on.squeeze(), torch.zeros_like(sdf_on.squeeze()))
            total_loss += lambda_init * loss_init
            mask_loss_val = loss_init.item()
            
            _, eikonal_grad = sdf_net.forward_with_grad(off_surface_points)
            loss_eikonal = ((eikonal_grad.norm(dim=-1) - 1.0) ** 2).mean()
            total_loss += lambda_eikonal * loss_eikonal
            eikonal_loss_val = loss_eikonal.item()
            
        else:
            if phase == 2 and iteration == phase1_iters:
                print("\n--- PHASE 1 COMPLETE | STARTING PHASE 2: Coarse Shape (Mask Loss) ---\n")
                optimizer.add_param_group({'params': [p for cam in cameras for p in cam.parameters()], 'lr': 1e-4})
            elif phase == 3 and iteration == phase1_iters + phase2_iters:
                print("\n--- PHASE 2 COMPLETE | STARTING PHASE 3: Fine Detail (All Losses) ---\n")
            
            view_idx = np.random.randint(0, num_views)
            cam, mask, normals_map = cameras[view_idx], masks_list[view_idx], normals_list[view_idx]
            pixels_uv = sample_pixels(mask, num_samples_per_view)
            rays_o, rays_d = cam.generate_rays(H, W, pixels_uv)
            near, far, bbox_hit_mask = get_ray_intersections(rays_o, rays_d, smpl_points)
            if bbox_hit_mask.sum() == 0: continue
            
            rays_o, rays_d, pixels_uv = rays_o[bbox_hit_mask], rays_d[bbox_hit_mask], pixels_uv[bbox_hit_mask]
            near, far = near[bbox_hit_mask], far[bbox_hit_mask]
            final_points, final_hit_mask = hybrid_ray_tracer_for_training(sdf_net, rays_o, rays_d, near, far)
            
            if final_hit_mask.sum() > 0:
                points_2d, in_front = cam.project_points(final_points[final_hit_mask])
                valid_proj_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H) & in_front.any(dim=-1)
                if valid_proj_mask.sum() > 0:
                    surface_points = final_points[final_hit_mask][valid_proj_mask]
                    pixels_uv_valid = pixels_uv[final_hit_mask][valid_proj_mask]
                    points_2d = points_2d[valid_proj_mask]
                    gt_mask = get_interp_values(mask.unsqueeze(0), pixels_uv_valid, H, W).squeeze() > 0.5
                    sdf_vals, _ = sdf_net.forward_with_grad(surface_points)
                    sdf_vals = sdf_vals.squeeze()
                    fg_mask = gt_mask
                    bg_mask = ~gt_mask
                    loss_fg = F.l1_loss(sdf_vals[fg_mask], torch.zeros_like(sdf_vals[fg_mask])) if fg_mask.sum() > 0 else 0.0
                    loss_bg = F.relu(-sdf_vals[bg_mask]).mean() if bg_mask.sum() > 0 else 0.0
                    mask_loss = loss_fg + 0.5 * loss_bg
                    total_loss += mask_loss
                    mask_loss_val = mask_loss.item()
                    
                    if phase == 3 and fg_mask.sum() > 0:
                        surface_points_fg = surface_points[fg_mask]
                        _, pred_normals_world = sdf_net.forward_with_grad(surface_points_fg)
                        pred_normals_camera = F.normalize(torch.matmul(pred_normals_world, cam.R.T), dim=-1)
                        gt_normals = get_interp_values(normals_map, points_2d[fg_mask], H, W)
                        gt_normals = F.normalize(gt_normals, dim=-1)
                        normal_loss = (1.0 - F.cosine_similarity(pred_normals_camera, gt_normals, dim=-1)).mean()
                        total_loss += 0.1 * normal_loss
                        normal_loss_val = normal_loss.item()
            
            rand_points = (torch.rand(2048, 3, device=device) - 0.5) * 2.5
            _, eikonal_grad = sdf_net.forward_with_grad(rand_points)
            eikonal_loss = ((eikonal_grad.norm(dim=-1) - 1.0) ** 2).mean()
            total_loss += 0.1 * eikonal_loss
            eikonal_loss_val = eikonal_loss.item()
        
        if not torch.isnan(total_loss) and total_loss > 0:
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        
        if iteration % 100 == 0:
            log_str = f"Iter: {iteration:04d}, Phase: {phase}, Total Loss: {total_loss.item():.4f}, Mask: {mask_loss_val:.4f}, Eikonal: {eikonal_loss_val:.4f}"
            if phase == 3:
                log_str += f", Normal: {normal_loss_val:.4f}"
            print(log_str)
    
    with torch.no_grad():
        visualize_progress(sdf_net, cameras[-1], masks_list[-1], normals_list[-1], smpl_points, num_iterations, folder=debug_vis_folder)
    return sdf_net, cameras

def extract_mesh_from_sdf(sdf_net, smpl_vertices, grid_size=256, padding=0.2, filename="output_mesh.obj"):
    print(f"\nExtracting mesh with grid size {grid_size}...")
    smpl_np = smpl_vertices.detach().cpu().numpy()
    if smpl_np.ndim == 3:
        min_coords, max_coords = smpl_np.min(axis=(0, 1)) - padding, smpl_np.max(axis=(0, 1)) + padding
    else:
        min_coords, max_coords = smpl_np.min(axis=0) - padding, smpl_np.max(axis=0) + padding
    grid = np.mgrid[min_coords[0]:max_coords[0]:complex(0, grid_size), min_coords[1]:max_coords[1]:complex(0, grid_size), min_coords[2]:max_coords[2]:complex(0, grid_size)]
    grid_pts_flat = grid.reshape(3, -1).T
    sdf_values_flat = []
    batch_size = 32768
    with torch.no_grad():
        for i in range(0, len(grid_pts_flat), batch_size):
            batch_pts = torch.tensor(grid_pts_flat[i:i+batch_size], device=device, dtype=torch.float32)
            sdf_values_flat.append(sdf_net.sdf(batch_pts).detach().cpu().numpy())
    sdf_volume = np.concatenate(sdf_values_flat, axis=0).reshape(grid_size, grid_size, grid_size)
    print("\n" + "="*50 + "\nSDF Field Diagnostics:")
    print(f"  Min/Max SDF: {sdf_volume.min():.6f} / {sdf_volume.max():.6f}")
    print(f"  Mean SDF:    {sdf_volume.mean():.6f}")
    print(f"  % Positive:  {100 * (sdf_volume > 0).sum() / sdf_volume.size:.2f}%")
    print("="*50 + "\n")
    print("Running Marching Cubes...")
    try:
        vertices, triangles = mcubes.marching_cubes(sdf_volume.transpose(2, 1, 0), 0)
    except ValueError:
        print("Marching cubes failed."); return
    if vertices.shape[0] == 0:
        print("WARNING: Marching cubes generated 0 vertices."); return
    print(f"Generated mesh with {len(vertices)} vertices and {len(triangles)} triangles.")
    scale = (max_coords - min_coords) / (grid_size - 1)
    vertices_world = vertices * scale + min_coords
    mcubes.export_obj(vertices_world, triangles, filename)
    print(f"Mesh successfully saved to {filename}")