# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import math
import os

import cv2
import numpy as np
import torch
from PIL import ImageColor
from pytorch3d.renderer import (
    AlphaCompositor,
    BlendParams,
    FoVOrthographicCameras,
    MeshRasterizer,
    MeshRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
    blending,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes
from termcolor import colored
from tqdm import tqdm

import lib.common.render_utils as util
from lib.common.imutils import blend_rgb_norm
from lib.dataset.mesh_util import get_visibility


def image2vid(images, vid_path):

    os.makedirs(os.path.dirname(vid_path), exist_ok=True)

    w, h = images[0].size
    videodims = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(vid_path, fourcc, len(images) / 5.0, videodims)
    for image in images:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()

# def query_color(verts, faces, image, device, paint_normal=True):
#     """query colors from points and image

#     Args:
#         verts ([B, 3]): [query verts]
#         faces ([M, 3]): [query faces]
#         image ([B, 3, H, W]): [full image]

#     Returns:
#         [np.float]: [return colors]
#     """

#     verts = verts.float().to(device)
#     faces = faces.long().to(device)

#     (xy, z) = verts.split([2, 1], dim=1)
#     visibility = get_visibility(xy, z, faces[:, [0, 2, 1]]).flatten()
#     uv = xy.unsqueeze(0).unsqueeze(2)    # [B, N, 2]
#     uv = uv * torch.tensor([1.0, -1.0]).type_as(uv)
#     colors = ((
#         torch.nn.functional.grid_sample(image, uv, align_corners=True)[0, :, :, 0].permute(1, 0) +
#         1.0
#     ) * 0.5 * 255.0)
#     if paint_normal:
#         colors[visibility == 0.0] = ((
#             Meshes(verts.unsqueeze(0), faces.unsqueeze(0)).verts_normals_padded().squeeze(0) + 1.0
#         ) * 0.5 * 255.0)[visibility == 0.0]
#     else:
#         colors[visibility == 0.0] = (torch.tensor([0.5, 0.5, 0.5]) * 255.0).to(device)

#     return colors.detach().cpu()

# @SSH - works well
def query_avatar_color(verts, faces, front_image, back_image, device):
    """
    Query colors from two images (front_image and back_image) based on vertex visibility.
    
    For vertices that are visible from the front, we sample colors from front_image.
    For vertices that are not visible (i.e. on the back side), we sample from back_image.
    
    Args:
        verts (Tensor): [N, 3] vertex coordinates.
        faces (Tensor): [M, 3] face indices.
        front_image (Tensor): [1, C, H, W] tensor representing the front image (values in [-1, 1]).
        back_image (Tensor): [1, C, H, W] tensor representing the back image (values in [-1, 1]).
        device (torch.device): The device on which tensors should reside.
    
    Returns:
        Tensor: [N, 3] colors in the range [0, 255] for each vertex.
    """
    verts = verts.float().to(device)
    faces = faces.long().to(device)
    
    # Split vertex coordinates: xy for UV computation and z for depth
    (xy, z) = verts.split([2, 1], dim=1)
    
    # Compute the visibility mask using your existing function.
    # Here, visibility > 0.5 indicates the vertex is seen from the front.
    visibility = get_visibility(xy, z, faces[:, [0, 2, 1]]).flatten().to(device)  # shape: [N]
    
    # Compute UV coordinates based on xy.
    # We assume that the front and back images share the same UV mapping.
    # The grid_sample function expects a grid of shape [N, H, W, 2].
    uv = xy.unsqueeze(0).unsqueeze(2)  # shape: [1, N, 1, 2]
    # Adjust the y-axis if necessary (flip y) so that the UV mapping aligns with image coordinates.
    uv = uv * torch.tensor([1.0, -1.0], device=device).view(1, 1, 1, 2)
    
    # Sample from the front image.
    front_sample = torch.nn.functional.grid_sample(
        front_image, uv, align_corners=True, padding_mode='border', mode='nearest',
    )
    # front_sample: [1, C, N, 1] --> reshape to [N, C]
    front_colors = front_sample.squeeze(3).permute(2, 1, 0).squeeze(2)
    front_colors = (front_colors + 1.0) * 0.5 * 255.0  # Map from [-1, 1] to [0, 255]
    
    # Sample from the back image.
    back_sample = torch.nn.functional.grid_sample(
        back_image, uv, align_corners=True, padding_mode='border', mode='nearest',
    )
    # back_sample: [1, C, N, 1] --> reshape to [N, C]
    back_colors = back_sample.squeeze(3).permute(2, 1, 0).squeeze(2)
    back_colors = (back_colors + 1.0) * 0.5 * 255.0
    
    # Create a boolean mask: True for vertices visible from the front.
    visibility_mask = (visibility > 0.6).unsqueeze(1).expand(-1, 3)
    
    # Use torch.where to choose front_colors if visible, otherwise use back_colors.
    colors = torch.where(visibility_mask, front_colors, back_colors)
    
    return colors.detach().cpu()

# IDK IF THIS WORKS WELL
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points

def query_color(verts, faces, side_verts, side_faces,
                front_image, back_image, device,
                threshold=0.8):
    """
    1) Classify main mesh (verts) as front/back using single visibility.
    2) Paint front/back from images.
    3) Paint side faces red.
    4) Return ONE color buffer for all vertices: [N_main + N_side, 3].

    Args:
        verts (FloatTensor): [N_main, 3] main mesh vertices
        faces (LongTensor):  [M_main, 3] main mesh faces (used in get_visibility)
        side_verts (FloatTensor): [N_side, 3] side mesh vertices
        side_faces (LongTensor): [M_side, 3] side mesh faces (indices into side_verts)
        front_image (FloatTensor): [1, C, H, W], in [-1,1]
        back_image  (FloatTensor): [1, C, H, W], in [-1,1]
        device (torch.device)
        threshold (float): e.g. 0.8 for front/back classification

    Returns:
        colors: [N_main + N_side, 3], each in [0,255], on CPU
    """
    import torch.nn.functional as F

    # ---------------------------------------------------------
    # PART A: MAIN MESH -- VISIBILITY AND FRONT/BACK PAINTING
    # ---------------------------------------------------------
    # 1) Move main mesh to device
    verts  = verts.float().to(device)   # [N_main, 3]
    faces  = faces.long().to(device)    # [M_main, 3]
    N_main = verts.shape[0]

    # 2) Compute visibility in [0,1] or {0,1}.
    #    (Assuming get_visibility returns a [N_main]-shaped tensor.)
    xy, z = verts.split([2, 1], dim=1)
    visibility = get_visibility(xy, z, faces[:, [0, 2, 1]])  # -> [N_main]
    visibility = visibility.flatten().to(device)

    # 3) Sample front/back textures with grid_sample.
    #    Create UV coordinates from the xy positions.
    uv = xy.unsqueeze(0).unsqueeze(2)  # -> [1, N_main, 1, 2]
    # Flip Y for PyTorch grid_sample (if needed):
    uv = uv * torch.tensor([1.0, -1.0], device=device).view(1, 1, 1, 2)

    front_sample = F.grid_sample(front_image, uv, align_corners=True,
                                 padding_mode='border', mode='nearest')
    back_sample  = F.grid_sample(back_image,  uv, align_corners=True,
                                 padding_mode='border', mode='nearest')

    # Reshape to [N_main, C] and map from [-1, 1] to [0, 255]
    front_colors = (front_sample.squeeze(3).permute(2, 1, 0).squeeze(2) + 1.0) * 0.5 * 255.0
    back_colors  = (back_sample.squeeze(3).permute(2, 1, 0).squeeze(2) + 1.0) * 0.5 * 255.0

    # 4) Create front/back masks based on visibility.
    front_mask   = (visibility > threshold)
    back_mask    = (visibility < (1.0 - threshold))
    inpaint_mask = ~(front_mask | back_mask)

    # 5) Initialize color buffer for the main mesh.
    colors_main = torch.zeros((N_main, 3), device=device)

    # 6) Assign front and back colors.
    colors_main[front_mask] = front_colors[front_mask]
    colors_main[back_mask]  = back_colors[back_mask]

    # 7) Optionally fill "in-between" vertices (fractional visibility)
    inpaint_inds = inpaint_mask.nonzero().flatten()
    if len(inpaint_inds) > 0:
        front_inds = front_mask.nonzero().flatten()
        back_inds  = back_mask.nonzero().flatten()
        if len(front_inds) > 0 and len(back_inds) > 0:
            inpaint_positions = verts[inpaint_inds].unsqueeze(0)  # [1, N_inpaint, 3]
            front_positions   = verts[front_inds].unsqueeze(0)    # [1, N_front, 3]
            back_positions    = verts[back_inds].unsqueeze(0)     # [1, N_back, 3]

            nn_front = knn_points(inpaint_positions, front_positions, K=1)
            nn_back  = knn_points(inpaint_positions, back_positions,  K=1)

            idx_front = nn_front.idx[0, :, 0]
            idx_back  = nn_back.idx[0, :, 0]
            c_front = colors_main[front_inds[idx_front]]
            c_back  = colors_main[back_inds[idx_back]]

            c_inpaint = 0.5 * (c_front + c_back)
            colors_main[inpaint_inds] = c_inpaint

    # ---------------------------------------------------------
    # PART B: SIDE MESH -- PAINT SIDE FACES RED
    # ---------------------------------------------------------
    # Process side mesh vertices.
    side_verts = side_verts.float().to(device)  # [N_side, 3]
    N_side = side_verts.shape[0]
    colors_side = torch.zeros((N_side, 3), device=device)

    # If side_faces is provided (and nonempty), use it to assign red to the vertices
    # that participate in side faces. (Here, we simply set all side vertices to red.)
    if side_faces is not None and side_faces.numel() > 0:
        print(side_faces is not None, side_faces.numel()>0)
        side_faces = side_faces.long().to(device)
        # Optionally, determine the vertices used by side_faces:
        side_vertex_indices = torch.unique(side_faces.view(-1))
        colors_side[side_vertex_indices] = torch.tensor([1.0, 0.0, 0.0], device=device)
        # For simplicity, we assign red to all side vertices:
        colors_side[:] = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        colors_side[:] = torch.tensor([1.0, 0.0, 0.0], device=device)

    # ---------------------------------------------------------
    # PART C: COMBINE MAIN AND SIDE MESH COLORS
    # ---------------------------------------------------------
    colors = torch.cat([colors_main, colors_side], dim=0)  # [N_main + N_side, 3]
    return colors.detach().cpu()

# @SSH END

def query_normal_color(verts, faces, device):
    """query normal colors

    Args:
        verts ([B, 3]): [query verts]
        faces ([M, 3]): [query faces]

    Returns:
        [np.float]: [return colors]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)

    colors = (
        (Meshes(verts.unsqueeze(0), faces.unsqueeze(0)).verts_normals_padded().squeeze(0) + 1.0) *
        0.5 * 255.0
    )

    return colors.detach().cpu()


class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)

        return images


class Render:
    def __init__(self, size=512, device=torch.device("cuda:0")):
        self.device = device
        self.size = size

        # camera setting
        self.dis = 100.0
        self.scale = 100.0
        self.mesh_y_center = 0.0

        # speed control
        self.fps = 30
        self.step = 3

        self.cam_pos = {
            "front":
            torch.tensor([
                (0, self.mesh_y_center, self.dis),
                (0, self.mesh_y_center, -self.dis),
            ]), "frontback":
            torch.tensor([
                (0, self.mesh_y_center, self.dis),
                (0, self.mesh_y_center, -self.dis),
            ]), "four":
            torch.tensor([
                (0, self.mesh_y_center, self.dis),
                (self.dis, self.mesh_y_center, 0),
                (0, self.mesh_y_center, -self.dis),
                (-self.dis, self.mesh_y_center, 0),
            ]), "around":
            torch.tensor([(
                100.0 * math.cos(np.pi / 180 * angle), self.mesh_y_center,
                100.0 * math.sin(np.pi / 180 * angle)
            ) for angle in range(0, 360, self.step)])
        }

        self.type = "color"

        self.mesh = None
        self.deform_mesh = None
        self.pcd = None
        self.renderer = None
        self.meshRas = None

        self.uv_rasterizer = util.Pytorch3dRasterizer(self.size)

    def get_camera_batch(self, type="four", idx=None):

        if idx is None:
            idx = np.arange(len(self.cam_pos[type]))

        R, T = look_at_view_transform(
            eye=self.cam_pos[type][idx],
            at=((0, self.mesh_y_center, 0), ),
            up=((0, 1, 0), ),
        )

        cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            znear=100.0,
            zfar=-100.0,
            max_y=100.0,
            min_y=-100.0,
            max_x=100.0,
            min_x=-100.0,
            scale_xyz=(self.scale * np.ones(3), ) * len(R),
        )

        return cameras

    def init_renderer(self, camera, type="mesh", bg="gray"):

        blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)

        if ("mesh" in type) or ("depth" in type) or ("rgb" in type):

            # rasterizer
            self.raster_settings_mesh = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                bin_size=-1,
                faces_per_pixel=30,
            )
            self.meshRas = MeshRasterizer(cameras=camera, raster_settings=self.raster_settings_mesh)

            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=cleanShader(blend_params=blendparam),
            )

        elif type == "mask":

            self.raster_settings_silhouette = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * 5e-5,
                faces_per_pixel=50,
                bin_size=-1,
                cull_backfaces=True,
            )

            self.silhouetteRas = MeshRasterizer(
                cameras=camera, raster_settings=self.raster_settings_silhouette
            )
            self.renderer = MeshRenderer(
                rasterizer=self.silhouetteRas, shader=SoftSilhouetteShader()
            )

        elif type == "pointcloud":
            self.raster_settings_pcd = PointsRasterizationSettings(
                image_size=self.size, radius=0.006, points_per_pixel=10
            )

            self.pcdRas = PointsRasterizer(cameras=camera, raster_settings=self.raster_settings_pcd)
            self.renderer = PointsRenderer(
                rasterizer=self.pcdRas,
                compositor=AlphaCompositor(background_color=(0, 0, 0)),
            )

    def load_meshes(self, verts, faces):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3] / [B,N,3]): array or tensor
            faces ([N,3]/ [B,N,3]): array or tensor
        """

        if isinstance(verts, list):
            V_lst = []
            F_lst = []
            for V, F in zip(verts, faces):
                if not torch.is_tensor(V):
                    V_lst.append(torch.tensor(V).float().to(self.device))
                    F_lst.append(torch.tensor(F).long().to(self.device))
                else:
                    V_lst.append(V.float().to(self.device))
                    F_lst.append(F.long().to(self.device))
            self.meshes = Meshes(V_lst, F_lst).to(self.device)
        else:
            # array or tensor
            if not torch.is_tensor(verts):
                verts = torch.tensor(verts)
                faces = torch.tensor(faces)
            if verts.ndimension() == 2:
                verts = verts.float().unsqueeze(0).to(self.device)
                faces = faces.long().unsqueeze(0).to(self.device)
            if verts.shape[0] != faces.shape[0]:
                faces = faces.repeat(len(verts), 1, 1).to(self.device)
            self.meshes = Meshes(verts, faces).to(self.device)

        # texture only support single mesh
        if len(self.meshes) == 1:
            self.meshes.textures = TexturesVertex(
                verts_features=(self.meshes.verts_normals_padded() + 1.0) * 0.5
            )

    def get_image(self, cam_type="frontback", type="rgb", bg="gray"):

        self.init_renderer(self.get_camera_batch(cam_type), type, bg)

        img_lst = []

        for mesh_id in range(len(self.meshes)):

            current_mesh = self.meshes[mesh_id]
            current_mesh.textures = TexturesVertex(
                verts_features=(current_mesh.verts_normals_padded() + 1.0) * 0.5
            )

            if type == "depth":
                fragments = self.meshRas(current_mesh.extend(len(self.cam_pos[cam_type])))
                images = fragments.zbuf[..., 0]

            elif type == "rgb":
                images = self.renderer(current_mesh.extend(len(self.cam_pos[cam_type])))
                images = (images[:, :, :, :3].permute(0, 3, 1, 2) - 0.5) * 2.0

            elif type == "mask":
                images = self.renderer(current_mesh.extend(len(self.cam_pos[cam_type])))[:, :, :, 3]
            else:
                print(f"unknown {type}")

            if cam_type == 'frontback':
                images[1] = torch.flip(images[1], dims=(-1, ))

            # images [N_render, 3, res, res]
            img_lst.append(images.unsqueeze(1))

        # meshes [N_render, N_mesh, 3, res, res]
        meshes = torch.cat(img_lst, dim=1)

        return list(meshes)

    def get_rendered_video_multi(self, data, save_path):

        height, width = data["img_raw"].shape[2:]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            save_path,
            fourcc,
            self.fps,
            (width * 3, int(height)),
        )

        pbar = tqdm(range(len(self.meshes)))
        pbar.set_description(colored(f"Normal Rendering {os.path.basename(save_path)}...", "blue"))

        mesh_renders = []    #[(N_cam, 3, res, res)*N_mesh]

        # render all the normals
        for mesh_id in pbar:

            current_mesh = self.meshes[mesh_id]
            current_mesh.textures = TexturesVertex(
                verts_features=(current_mesh.verts_normals_padded() + 1.0) * 0.5
            )

            norm_lst = []

            for batch_cams_idx in np.array_split(np.arange(len(self.cam_pos["around"])), 12):

                batch_cams = self.get_camera_batch(type='around', idx=batch_cams_idx)

                self.init_renderer(batch_cams, "mesh", "gray")

                norm_lst.append(
                    self.renderer(current_mesh.extend(len(batch_cams_idx))
                                 )[..., :3].permute(0, 3, 1, 2)
                )
            mesh_renders.append(torch.cat(norm_lst).detach().cpu())

        # generate video frame by frame
        pbar = tqdm(range(len(self.cam_pos["around"])))
        pbar.set_description(colored(f"Video Exporting {os.path.basename(save_path)}...", "blue"))

        for cam_id in pbar:
            img_raw = data["img_raw"]
            num_obj = len(mesh_renders) // 2
            img_smpl = blend_rgb_norm((torch.stack(mesh_renders)[:num_obj, cam_id] - 0.5) * 2.0,
                                      data)
            img_cloth = blend_rgb_norm((torch.stack(mesh_renders)[num_obj:, cam_id] - 0.5) * 2.0,
                                       data)
            final_img = torch.cat([img_raw, img_smpl, img_cloth],
                                  dim=-1).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

            video.write(final_img[:, :, ::-1])

        video.release()