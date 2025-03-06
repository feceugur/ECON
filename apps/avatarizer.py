import argparse
import os
import os.path as osp

import numpy as np
import torch
import trimesh
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree
from termcolor import colored

import lib.smplx as smplx
from lib.common.local_affine import register
from lib.dataset.mesh_util import (
    SMPLX,
    export_obj,
    keep_largest,
    poisson,
)
from lib.smplx.lbs import general_lbs

# loading cfg file
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-uv", action="store_true")
parser.add_argument("-dress", action="store_true")
args = parser.parse_args()

smplx_container = SMPLX()
device = torch.device(f"cuda:{args.gpu}")

# loading SMPL-X and econ objs inferred with ECON
# prefix = f"./results_fulden/econ/obj/{args.name}"
prefix = f"./results/Eric/IFN+_face_thresh_0.31/econ/obj/{args.name}"

smpl_path = f"{prefix}_smpl_00.npy"
smplx_param = np.load(smpl_path, allow_pickle=True).item()

# export econ obj with pre-computed normals
econ_path = f"{prefix}_0_full_soups.ply"
econ_obj = trimesh.load(econ_path)
assert econ_obj.vertex_normals.shape[1] == 3
os.makedirs(f"{prefix}/", exist_ok=True)

# align econ with SMPL-X
econ_obj.vertices *= np.array([1.0, -1.0, -1.0])
econ_obj.vertices /= smplx_param["scale"].cpu().numpy()
econ_obj.vertices -= smplx_param["transl"].cpu().numpy()

for key in smplx_param.keys():
    smplx_param[key] = smplx_param[key].cpu().view(1, -1)

smpl_model = smplx.create(
    smplx_container.model_dir,
    model_type="smplx",
    gender="neutral",
    age="adult",
    use_face_contour=False,
    use_pca=False,
    num_betas=smplx_param["betas"].shape[1],
    num_expression_coeffs=smplx_param["expression"].shape[1],
    ext="pkl",
)

smpl_out_lst = []

# obtain the pose params of T-pose, DA-pose, and the original pose
for pose_type in ["a-pose", "t-pose", "da-pose", "pose"]:
    smpl_out_lst.append(
        smpl_model(
            body_pose=smplx_param["body_pose"],
            global_orient=smplx_param["global_orient"],
            betas=smplx_param["betas"],
            expression=smplx_param["expression"],
            jaw_pose=smplx_param["jaw_pose"],
            left_hand_pose=smplx_param["left_hand_pose"],
            right_hand_pose=smplx_param["right_hand_pose"],
            return_verts=True,
            return_full_pose=True,
            return_joint_transformation=True,
            return_vertex_transformation=True,
            pose_type=pose_type,
        )
    )

# -------------------------- align econ and SMPL-X in DA-pose space ------------------------- #
# 1. find the vertex-correspondence between SMPL-X and econ
# 2. ECON + SMPL-X: posed space --> T-pose space --> DA-pose space
# 3. ECON (w/o hands & over-streched faces) + SMPL-X (w/ hands & registered inpainting parts)
# ------------------------------------------------------------------------------------------- #

smpl_verts = smpl_out_lst[3].vertices.detach()[0]
smpl_tree = cKDTree(smpl_verts.cpu().numpy())
dist, idx = smpl_tree.query(econ_obj.vertices, k=3)

if not osp.exists(f"{prefix}/econ_da.obj") or not osp.exists(f"{prefix}/smpl_da.obj"):

    # t-pose for ECON
    econ_verts = torch.tensor(econ_obj.vertices).float()
    rot_mat_t = smpl_out_lst[3].vertex_transformation.detach()[0][idx[:, 0]]
    homo_coord = torch.ones_like(econ_verts)[..., :1]
    econ_cano_verts = torch.inverse(rot_mat_t) @ torch.cat([econ_verts, homo_coord],
                                                        dim=1).unsqueeze(-1)
    econ_cano_verts = econ_cano_verts[:, :3, 0].cpu()
    econ_cano = trimesh.Trimesh(econ_cano_verts, econ_obj.faces)

    # da-pose for ECON
    rot_mat_da = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
    econ_da_verts = rot_mat_da @ torch.cat([econ_cano_verts, homo_coord], dim=1).unsqueeze(-1)
    econ_da = trimesh.Trimesh(econ_da_verts[:, :3, 0].cpu(), econ_obj.faces)

    # da-pose for SMPL-X
    smpl_da = trimesh.Trimesh(
        smpl_out_lst[2].vertices.detach()[0],
        smpl_model.faces,
        maintain_orders=True,
        process=False,
    )
    smpl_da.export(f"{prefix}/smpl_da.obj")

    # ignore parts: hands, front_flame, eyeball
    ignore_vid = np.concatenate([
        smplx_container.smplx_mano_vid,
        smplx_container.smplx_front_flame_vid,
        smplx_container.smplx_eyeball_vid,
    ])

    # a trick to avoid torn dress/skirt
    if args.dress:
        ignore_vid = np.concatenate([ignore_vid, smplx_container.smplx_leg_vid])

    # remove ignore parts from ECON
    econ_da_body = econ_da.copy()
    mano_mask = ~np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
    econ_da_body.update_faces(mano_mask[econ_da.faces].all(axis=1))
    econ_da_body.remove_unreferenced_vertices()
    econ_da_body = keep_largest(econ_da_body)

    # remove ignore parts from SMPL-X
    register_mask = np.ones(smpl_da.vertices.shape[0], dtype=bool)  # All True
    smpl_da_body = smpl_da.copy()
    smpl_da_body.update_faces(register_mask[smpl_da.faces].all(axis=1))
    smpl_da_body.remove_unreferenced_vertices()
    smpl_da_body = keep_largest(smpl_da_body)

    # upsample the smpl_da_body and do registeration
    smpl_da_body = Meshes(
        verts=[torch.tensor(smpl_da_body.vertices).float()],
        faces=[torch.tensor(smpl_da_body.faces).long()],
    ).to(device)
    sm = SubdivideMeshes(smpl_da_body)
    smpl_da_body = register(econ_da_body, sm(smpl_da_body), device)

    # remove over-streched+hand faces from ECON
    econ_da_body = econ_da.copy()
    edge_before = np.sqrt(
        ((econ_obj.vertices[econ_cano.edges[:, 0]] -
        econ_obj.vertices[econ_cano.edges[:, 1]])**2).sum(axis=1)
    )
    edge_after = np.sqrt(
        ((econ_da.vertices[econ_cano.edges[:, 0]] -
        econ_da.vertices[econ_cano.edges[:, 1]])**2).sum(axis=1)
    )
    edge_diff = edge_after / edge_before.clip(1e-2)

    streched_vid = np.unique(econ_cano.edges[edge_diff > 6])
    mano_mask[streched_vid] = False
    econ_da_body.update_faces(mano_mask[econ_cano.faces].all(axis=1))
    econ_da_body.remove_unreferenced_vertices()

    # stitch the registered SMPL-X body and floating hands to ECON
    econ_da_tree = cKDTree(econ_da.vertices)
    dist, idx = econ_da_tree.query(smpl_da_body.vertices, k=1)
    smpl_da_body.update_faces((dist > 0.02)[smpl_da_body.faces].all(axis=1))
    smpl_da_body.remove_unreferenced_vertices()

    smpl_hand = smpl_da.copy()
    smpl_hand.update_faces(
        smplx_container.smplx_mano_vertex_mask.numpy()[smpl_hand.faces].all(axis=1)
    )
    smpl_hand.remove_unreferenced_vertices()

    # combination of ECON body, SMPL-X side parts, SMPL-X hands
    econ_da = sum([smpl_hand, smpl_da_body, econ_da_body])
    econ_da = poisson(
        econ_da, f"{prefix}/econ_da.obj", depth=10, face_count=1e5, laplacian_remeshing=True
    )
else:
    econ_da = trimesh.load(f"{prefix}/econ_da.obj")
    smpl_da = trimesh.load(f"{prefix}/smpl_da.obj", maintain_orders=True, process=False)

# ---------------------- SMPL-X compatible ECON ---------------------- #
# 1. Find the new vertex-correspondence between NEW ECON and SMPL-X
# 2. Build the new J_regressor, lbs_weights, posedirs
# 3. canonicalize the NEW ECON
# ------------------------------------------------------------------- #

print("Start building the SMPL-X compatible ECON model...")

smpl_tree = cKDTree(smpl_da.vertices)
dist, idx = smpl_tree.query(econ_da.vertices, k=3)
knn_weights = np.exp(-(dist**2))
knn_weights /= knn_weights.sum(axis=1, keepdims=True)

econ_J_regressor = (smpl_model.J_regressor[:, idx] * knn_weights[None]).sum(dim=-1)
econ_lbs_weights = (smpl_model.lbs_weights.T[:, idx] * knn_weights[None]).sum(dim=-1).T

num_posedirs = smpl_model.posedirs.shape[0]
econ_posedirs = ((
    smpl_model.posedirs.view(num_posedirs, -1, 3)[:, idx, :] * knn_weights[None, ..., None]
).sum(dim=-2).view(num_posedirs, -1).float())

econ_J_regressor /= econ_J_regressor.sum(dim=1, keepdims=True).clip(min=1e-10)
econ_lbs_weights /= econ_lbs_weights.sum(dim=1, keepdims=True)

rot_mat_da = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
econ_da_verts = torch.tensor(econ_da.vertices).float()
econ_cano_verts = torch.inverse(rot_mat_da) @ torch.cat([
    econ_da_verts, torch.ones_like(econ_da_verts)[..., :1]
],
                                                        dim=1).unsqueeze(-1)
econ_cano_verts = econ_cano_verts[:, :3, 0].double()

# ----------------------------------------------------
# use original pose to animate ECON reconstruction
# ----------------------------------------------------

rot_mat_pose = smpl_out_lst[3].vertex_transformation.detach()[0][idx[:, 0]]
posed_econ_verts = rot_mat_pose @ torch.cat([
    econ_cano_verts.float(),
    torch.ones_like(econ_cano_verts.float())[..., :1]
],
                                            dim=1).unsqueeze(-1)
posed_econ_verts = posed_econ_verts[:, :3, 0].double()

aligned_econ_verts = posed_econ_verts.detach().cpu().numpy()
aligned_econ_verts += smplx_param["transl"].cpu().numpy()
aligned_econ_verts *= smplx_param["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
econ_pose = trimesh.Trimesh(aligned_econ_verts, econ_da.faces)
assert econ_pose.vertex_normals.shape[1] == 3
econ_pose.export(f"{prefix}/econ_pose.ply")

cache_path = f"{prefix.replace('obj','cache')}"
os.makedirs(cache_path, exist_ok=True)

# -----------------------------------------------------------------
# create UV texture (.obj .mtl .png) from posed ECON reconstruction
# -----------------------------------------------------------------
print("Start Color mapping...")

from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import trimesh
import os.path as osp
from lib.common.render import query_color, query_normal_color
from lib.common.render_utils import Pytorch3dRasterizer
from lib.dataset.mesh_util import export_obj

##########################################
# First Pass: Using _cloth_front and _cloth_back_red images
##########################################

cloth_front_red_path = f"./results/Eric/IFN+_face_thresh_0.31/econ/png/{args.name}_cloth_front_red.png"
cloth_back_red_path  = f"./results/Eric/IFN+_face_thresh_0.31/econ/png/{args.name}_cloth_back_red.png"

tensor_front_1 = transforms.ToTensor()(Image.open(cloth_front_red_path))[:, :, :512]
tensor_back_1  = transforms.ToTensor()(Image.open(cloth_back_red_path))[:, :, :512]
H, W = tensor_front_1.shape[1], tensor_front_1.shape[2]

front_image_1 = ((tensor_front_1 - 0.5) * 2.0).unsqueeze(0).to(device)
back_image_1  = ((tensor_back_1  - 0.5) * 2.0).unsqueeze(0).to(device)

verts_tensor = torch.tensor(econ_pose.vertices).float().to(device)
faces_tensor = torch.tensor(econ_pose.faces).long().to(device)

final_rgb_pass1 = query_color(
    verts_tensor,
    faces_tensor,
    front_image_1,
    back_image_1,
    device=device,
).numpy()

uv_rasterizer = Pytorch3dRasterizer(image_size=8192, device=device)
if not ('vt' in globals() and 'ft' in globals() and 'vmapping' in globals()):
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(econ_pose.vertices, econ_pose.faces)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    chart_options.max_iterations = 4
    pack_options.resolution = 8192
    pack_options.bruteForce = True
    atlas.generate(chart_options=chart_options)
    vmapping, ft_np, vt_np = atlas[0]
    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

v_np = econ_pose.vertices
f_np = econ_pose.faces
texture_map1 = uv_rasterizer.get_texture(
    torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
    ft,
    torch.tensor(v_np).unsqueeze(0).float(),
    torch.tensor(f_np).unsqueeze(0).long(),
    torch.tensor(final_rgb_pass1).unsqueeze(0).float() / 255.0,
)
texture_map1_8bit = (texture_map1 * 255.0).astype(np.uint8)
Image.fromarray(texture_map1_8bit).save(f"{cache_path}/texture_map1.png")
print("First-pass texture map saved as texture_map1.png.")

##########################################
# Second Pass: Using _cloth_front_red_blue and _cloth_back_blue images
##########################################

cloth_front_path_blue = f"./results/Eric/IFN+_face_thresh_0.31/econ/png/{args.name}_cloth_front_blue.png"
cloth_back_path_blue  = f"./results/Eric/IFN+_face_thresh_0.31/econ/png/{args.name}_cloth_back_blue.png"

tensor_front_2 = transforms.ToTensor()(Image.open(cloth_front_path_blue))[:, :, :512]
tensor_back_2  = transforms.ToTensor()(Image.open(cloth_back_path_blue))[:, :, :512]

front_image_2 = ((tensor_front_2 - 0.5) * 2.0).unsqueeze(0).to(device)
back_image_2  = ((tensor_back_2  - 0.5) * 2.0).unsqueeze(0).to(device)

final_rgb_pass2 = query_color(
    verts_tensor,
    faces_tensor,
    front_image_2,
    back_image_2,
    device=device,
).numpy()

texture_map2 = uv_rasterizer.get_texture(
    torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
    ft,
    torch.tensor(v_np).unsqueeze(0).float(),
    torch.tensor(f_np).unsqueeze(0).long(),
    torch.tensor(final_rgb_pass2).unsqueeze(0).float() / 255.0,
)
texture_map2_8bit = (texture_map2 * 255.0).astype(np.uint8)
Image.fromarray(texture_map2_8bit).save(f"{cache_path}/texture_map2.png")
print("Second-pass texture map saved as texture_map2.png.")

##########################################
# Compare Texture Maps and Create Difference Mask
##########################################

diff_map = np.abs(texture_map1_8bit.astype(np.float32) - texture_map2_8bit.astype(np.float32)) / 255.0
# threshold = 0.1
mask_diff = np.any(diff_map > 0.01, axis=2)
Image.fromarray((mask_diff.astype(np.uint8) * 255)).save(f"{cache_path}/diff_mask.png")
print("Difference mask saved as diff_mask.png.")

##########################################
# Assign Final Vertex Colors and Export Mesh
##########################################

final_rgb = final_rgb_pass2
econ_pose.visual.vertex_colors = final_rgb
econ_pose.export(f"{prefix}/econ_icp_rgb.ply")

##########################################
# Normal-based Color Mapping (Choice 2, unchanged)
##########################################

if not osp.exists(f"{prefix}/econ_icp_normal.ply"):
    file_normal = query_normal_color(
        verts_tensor,
        faces_tensor,
        device=device,
    ).numpy()
    econ_pose.visual.vertex_colors = file_normal
    econ_pose.export(f"{prefix}/econ_icp_normal.ply")
else:
    mesh = trimesh.load(f"{prefix}/econ_icp_normal.ply")
    file_normal = mesh.visual.vertex_colors[:, :3]

##########################################
# Save econ Data for Further Processing
##########################################

econ_dict = {
    "v_template": econ_cano_verts.unsqueeze(0),
    "posedirs": econ_posedirs,
    "J_regressor": econ_J_regressor,
    "parents": smpl_model.parents,
    "lbs_weights": econ_lbs_weights,
    "final_rgb": final_rgb,
    "final_normal": file_normal,
    "faces": econ_pose.faces,
}

torch.save(econ_dict, f"{cache_path}/econ.pt")

print(
    "If the dress/skirt is torn in `<file_name>/econ_da.obj`, please delete ./file_name and regenerate them with `-dress`\n"
    "python -m apps.avatarizer -n <file_name> -dress"
)

##########################################
# UV Texture Generation (if enabled)
##########################################

args.uv = True
args.dress = False

if args.uv:
    print("Start UV texture generation...")
    v_np = econ_pose.vertices
    f_np = econ_pose.faces

    vt_cache = osp.join(cache_path, "vt.pt")
    ft_cache = osp.join(cache_path, "ft.pt")

    if osp.exists(vt_cache) and osp.exists(ft_cache):
        vt = torch.load(vt_cache).to(device)
        ft = torch.load(ft_cache).to(device)
    else:
        import xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        pack_options = xatlas.PackOptions()
        chart_options.max_iterations = 4
        pack_options.resolution = 8192
        pack_options.bruteForce = True
        atlas.generate(chart_options=chart_options)
        vmapping, ft_np, vt_np = atlas[0]
        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
        torch.save(vt.cpu(), vt_cache)
        torch.save(ft.cpu(), ft_cache)

    uv_rasterizer = Pytorch3dRasterizer(image_size=8192, device=device)
    texture_npy = uv_rasterizer.get_texture(
        torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
        ft,
        torch.tensor(v_np).unsqueeze(0).float(),
        torch.tensor(f_np).unsqueeze(0).long(),
        torch.tensor(final_rgb).unsqueeze(0).float() / 255.0,
    )
    
    import cv2
    # Create a mask for missing (zero-sum) pixels.
    missing_mask = (texture_npy.sum(axis=2) == 0).astype(np.uint8) * 255

    # Resize the diff mask from 512x512 to the texture resolution.
    diff_mask_resized = cv2.resize((mask_diff.astype(np.uint8) * 255),
                                   (texture_npy.shape[1], texture_npy.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

    # Combine the missing mask with the diff mask.
    combined_mask = cv2.bitwise_or(missing_mask, diff_mask_resized)
    
    # Instead of erosion, we now dilate the mask.
    kernel = np.ones((7, 7), np.uint8)  # 7x7 kernel
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    # Save the dilated mask for reference.
    cv2.imwrite(f"{cache_path}/mask_dilated.png", dilated_mask)
    
    # Convert texture_npy to an 8-bit image.
    texture_8bit = (texture_npy * 255).astype(np.uint8)
    
    # Inpaint the texture using the dilated mask, inpaintRadius=3, and TELEA method.
    inpainted_image = cv2.inpaint(texture_8bit, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    result = inpainted_image.astype(np.float32) / 255.0
    filename = f"{cache_path}/texture_TELEA.png"
    Image.fromarray((result * 255.0).astype(np.uint8)).save(filename)
    print(f"Saved inpainted texture: {filename}")

    print("UV texture generation complete.")

    with open(f"{cache_path}/material.mtl", "w") as fp:
        fp.write("newmtl mat0 \n")
        fp.write("Ka 1.000000 1.000000 1.000000 \n")
        fp.write("Kd 1.000000 1.000000 1.000000 \n")
        fp.write("Ks 0.000000 0.000000 0.000000 \n")
        fp.write("Tr 1.000000 \n")
        fp.write("illum 1 \n")
        fp.write("Ns 0.000000 \n")
        fp.write("map_Kd texture.png \n")
    
    # Export the mesh using the original vertices.
    export_obj(np.array(econ_pose.vertices), f_np, vt, ft, f"{cache_path}/mesh.obj")
