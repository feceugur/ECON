from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import trimesh
import os.path as osp
from lib.common.render import query_color, query_normal_color
from lib.common.render_utils import Pytorch3dRasterizer
from lib.dataset.mesh_util import export_obj
import os

name = "carla_Apose"
device = torch.device(f"cuda:0")
prefix = f"./results/Carla/IFN+_face_thresh_0.30/econ/obj/{name}"


cache_path = f"{prefix.replace('obj','cache')}"
os.makedirs(cache_path, exist_ok=True)

econ_pose = trimesh.load("/home/ubuntu/projects/induxr/econ_s/ECON/results/Carla/IFN+_face_thresh_0.30/econ/obj/carla_Apose/econ_pose.ply")

# Now econ_pose is a Trimesh object; you can inspect its vertices, faces, etc.
print("Number of vertices:", len(econ_pose.vertices))
print("Number of faces:", len(econ_pose.faces))

cloth_front_red_path = f"./results/Carla/IFN+_face_thresh_0.30/econ/png/{name}_cloth_front_red.png"
cloth_back_red_path  = f"./results/Carla/IFN+_face_thresh_0.30/econ/png/{name}_cloth_back_red.png"

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

cloth_front_path_blue = f"./results/Carla/IFN+_face_thresh_0.30/econ/png/{name}_cloth_front_blue.png"
cloth_back_path_blue  = f"./results/Carla/IFN+_face_thresh_0.30/econ/png/{name}_cloth_back_blue.png"

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
# Defragment Texture Maps Using mesh.obj Files
##########################################

import pymeshlab

# --- Export first-pass mesh with texture_map1_8bit ---
# export_obj should create a .obj file (and corresponding .mtl) that references texture_map1.png.
export_obj(np.array(econ_pose.vertices), f_np, vt, ft, f"{cache_path}/mesh1.obj")
# Now load the mesh from mesh1.obj.
ms = pymeshlab.MeshSet()
ms.load_new_mesh(f"{cache_path}/mesh1.obj")
ms.apply_filter('apply_texmap_defragmentation')
defrag_texture1 = ms.current_mesh().texture_image()  # Expect a PIL image or similar.
defrag_texture1_np = np.array(defrag_texture1)
Image.fromarray(defrag_texture1_np).save(f"{cache_path}/defrag_texture_map1.png")
print("Defragmented first-pass texture map saved as defrag_texture_map1.png.")

# --- Export second-pass mesh with texture_map2_8bit ---
export_obj(np.array(econ_pose.vertices), f_np, vt, ft, f"{cache_path}/mesh2.obj")
ms.clear()  # Reset the MeshSet.
ms.load_new_mesh(f"{cache_path}/mesh2.obj")
ms.apply_filter('apply_texmap_defragmentation')
defrag_texture2 = ms.current_mesh().texture_image()
defrag_texture2_np = np.array(defrag_texture2)
Image.fromarray(defrag_texture2_np).save(f"{cache_path}/defrag_texture_map2.png")
print("Defragmented second-pass texture map saved as defrag_texture_map2.png.")

##########################################
# Compare Defragmented Texture Maps and Create Difference Mask
##########################################

diff_map = np.abs(defrag_texture1_np.astype(np.float32) - defrag_texture2_np.astype(np.float32)) / 255.0
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

# econ_dict = {
#     "v_template": econ_cano_verts.unsqueeze(0),
#     "posedirs": econ_posedirs,
#     "J_regressor": econ_J_regressor,
#     "parents": smpl_model.parents,
#     "lbs_weights": econ_lbs_weights,
#     "final_rgb": final_rgb,
#     "final_normal": file_normal,
#     "faces": econ_pose.faces,
# }

# torch.save(econ_dict, f"{cache_path}/econ.pt")

print(
    "If the dress/skirt is torn in <file_name>/econ_da.obj, please delete ./file_name and regenerate them with -dress\n"
    "python -m apps.avatarizer -n <file_name> -dress"
)

##########################################
# UV Texture Generation (if enabled)
##########################################

uv = True

if uv:
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
    # Use the defragmented second-pass texture as the final texture.
    final_texture = torch.tensor(defrag_texture2_np.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    
    texture_npy = uv_rasterizer.get_texture(
        torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
        ft,
        torch.tensor(v_np).unsqueeze(0).float(),
        torch.tensor(f_np).unsqueeze(0).long(),
        final_texture,
    )
    
    import cv2
    missing_mask = (texture_npy.sum(axis=2) == 0).astype(np.uint8) * 255
    diff_mask_resized = cv2.resize((mask_diff.astype(np.uint8) * 255),
                                   (texture_npy.shape[1], texture_npy.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
    combined_mask = cv2.bitwise_or(missing_mask, diff_mask_resized)
    kernel = np.ones((7, 7), np.uint8)
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    cv2.imwrite(f"{cache_path}/mask_dilated.png", dilated_mask)
    texture_8bit = (texture_npy * 255.0).astype(np.uint8)
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
    
    export_obj(np.array(econ_pose.vertices), f_np, vt, ft, f"{cache_path}/mesh.obj")
