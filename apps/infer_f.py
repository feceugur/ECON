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

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
import os

import numpy as np
import torch
import torchvision
import trimesh
from pytorch3d.ops import SubdivideMeshes
from termcolor import colored
from tqdm.auto import tqdm

from apps.IFGeo import IFGeo
from apps.Normal_f import Normal
from apps.sapiens import ImageProcessor
from apps.clean_mesh import MeshWatertightifier

from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.common.config import cfg
from lib.common.imutils import blend_rgb_norm, load_img, transform_to_tensor, wrap
from lib.common.local_affine import register
from lib.common.render import query_color
from lib.common.train_util import Format, init_loss
from lib.common.voxelize import VoxelGrid
from lib.dataset.mesh_util import *
from lib.dataset.TestDataset_f import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis
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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=50)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-in_b_dir", "--in_b_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ_f.yaml")
    parser.add_argument("-multi", action="store_false")
    parser.add_argument("-novis", action="store_true")
    
    parser.add_argument("-n", "--name", type=str, default="fulden_tpose_f1")
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-uv", action="store_true")
    parser.add_argument("-dress", action="store_true")
    args = parser.parse_args()
    
    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymafx/configs/pymafx_config.yaml")
    device = torch.device(f"cuda:{args.gpu_device}")

    # setting for testing on in-the-wild images
    cfg_show_list = [
        "test_gpus", [args.gpu_device], "mcube_res", 512, "clean_mesh", True, "test_mode", True,
        "batch_size", 1
    ]
 
    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()
    normal_path = "/home/ubuntu/Data/Fulden/ckpt/normal.ckpt"
    # load normal model
    normal_net = Normal.load_from_checkpoint(
        cfg=cfg, checkpoint_path=normal_path, map_location=device, strict=False
    )
    normal_net = normal_net.to(device)
    normal_net.netG.eval()
    print(
        colored(
            f"Resume Normal Estimator from {Format.start} {cfg.normal_path} {Format.end}", "green"
        )
    )

    if cfg.sapiens.use:
        sapiens_normal_net = ImageProcessor(device=device)

    # SMPLX object
    SMPLX_object = SMPLX()

    dataset_param = {
        "image_dir": args.in_dir,
        "image_b_dir": args.in_b_dir,
        "seg_dir": args.seg_dir,
        "use_seg": True,    # w/ or w/o segmentation
        "hps_type": cfg.bni.hps_type,    # pymafx/pixie
        "vol_res": cfg.vol_res,
        "single": args.multi,
    }

    if cfg.bni.use_ifnet:
        # load IFGeo model
        ifnet = IFGeo.load_from_checkpoint(
            cfg=cfg, checkpoint_path=cfg.ifnet_path, map_location=device, strict=False
        )
        ifnet = ifnet.to(device)
        ifnet.netG.eval()

        print(colored(f"Resume IF-Net+ from {Format.start} {cfg.ifnet_path} {Format.end}", "green"))
        print(colored(f"Complete with {Format.start} IF-Nets+ (Implicit) {Format.end}", "green"))
    else:
        print(colored(f"Complete with {Format.start} SMPL-X (Explicit) {Format.end}", "green"))

    dataset = TestDataset(dataset_param, device)
    front_arr_dict, back_arr_dict = dataset[0]

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)

    # @SSH
    people=['Fulden', 'Rafael', 'Roger', 'Albert', 'Stefan']
    if cfg.bni.use_ifnet:
        args.out_dir = f'{args.out_dir}/{people[0]}/IFN+_face_thresh_{cfg.bni.face_thres:.2f}'
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        args.out_dir = f'{args.out_dir}/{people[0]}/face_thresh_{cfg.bni.face_thres:.2f}'
        os.makedirs(args.out_dir, exist_ok=True)
       
    for data, data_b in pbar:

        losses = init_loss()

        pbar.set_description(f"{data['name']}")

        # final results rendered as image (PNG)
        # 1. Render the final fitted SMPL (xxx_smpl.png)
        # 2. Render the final reconstructed clothed human (xxx_cloth.png)
        # 3. Blend the original image with predicted cloth normal (xxx_overlap.png)
        # 4. Blend the cropped image with predicted cloth normal (xxx_crop.png)

        os.makedirs(osp.join(args.out_dir, cfg.name, "png"), exist_ok=True)

        # final reconstruction meshes (OBJ)
        # 1. SMPL mesh (xxx_smpl_xx.obj)
        # 2. SMPL params (xxx_smpl.npy)
        # 3. d-BiNI surfaces (xxx_BNI.obj)
        # 4. seperate face/hand mesh (xxx_hand/face.obj)
        # 5. full shape impainted by IF-Nets+ after remeshing (xxx_IF.obj)
        # 6. sideded or occluded parts (xxx_side.obj)
        # 7. final reconstructed clothed human (xxx_full.obj)

        os.makedirs(osp.join(args.out_dir, cfg.name, "obj"), exist_ok=True)

        in_tensor = {
            "smpl_faces": data["smpl_faces"], 
            "image": data["img_icon"].to(device), 
            "image_back": data_b["img_icon"].to(device),
            "mask": data["img_mask"].to(device),
            "mask_back": data_b["img_mask"].to(device)
        }

        # The optimizer and variables
        optimed_pose = data["body_pose"].requires_grad_(True)
        optimed_trans = data["trans"].requires_grad_(True)
        optimed_betas = data["betas"].requires_grad_(True)
        optimed_orient = data["global_orient"].requires_grad_(True)

        optimizer_smpl = torch.optim.Adam([
            optimed_pose, optimed_trans, optimed_betas, optimed_orient
        ],
                                          lr=1e-2,
                                          amsgrad=True)
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=args.patience,
        )

        # [result_loop_1, result_loop_2, ...]
        per_data_lst = []

        N_body, N_pose = optimed_pose.shape[:2]
   
        smpl_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_00.obj"

        # sapiens inference for current batch data

        if cfg.sapiens.use:
            
            sapiens_normal = sapiens_normal_net.process_image(
                Image.fromarray(
                    data["img_raw"].squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                ), "1b", cfg.sapiens.seg_model
            )
            print(colored("Estimating normal maps from input image, using Sapiens-normal", "green"))

            sapiens_normal_square_lst = []
            for idx in range(len(data["img_icon"])):
                sapiens_normal_square_lst.append(wrap(sapiens_normal, data["uncrop_param"], idx))
            sapiens_normal_square = torch.cat(sapiens_normal_square_lst)

        # remove this line if you change the loop_smpl and obtain different SMPL-X fits
        if osp.exists(smpl_path) and (not cfg.force_smpl_optim):
            smpl_verts_lst = []
            smpl_faces_lst = []

            for idx in range(N_body):

                smpl_obj = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_{idx:02d}.obj"
                smpl_mesh = trimesh.load(smpl_obj)
                smpl_verts = torch.tensor(smpl_mesh.vertices).to(device).float()
                smpl_faces = torch.tensor(smpl_mesh.faces).to(device).long()
                smpl_verts_lst.append(smpl_verts)
                smpl_faces_lst.append(smpl_faces)

            batch_smpl_verts = torch.stack(smpl_verts_lst)
            batch_smpl_faces = torch.stack(smpl_faces_lst)
            # render optimized mesh as normal [-1,1]
            in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                batch_smpl_verts, batch_smpl_faces
            )

            with torch.no_grad():
                in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)
            

            # Flip the x-component and invert the entire back normal map
            # in_tensor["normal_B"][:, 0, :, :] = -in_tensor["normal_B"][:, 0, :, :]  # Flip x component
            # in_tensor["normal_B"] = -in_tensor["normal_B"]  # Invert entire normal map

            # # Normalize to maintain unit length
            # in_tensor["normal_B"] = in_tensor["normal_B"] / torch.norm(in_tensor["normal_B"], dim=1, keepdim=True)
            # normal_net.netG.save_nml_image(in_tensor["normal_F"], in_tensor["normal_B"], out_dir=args.out_dir)

            in_tensor["smpl_verts"] = batch_smpl_verts * torch.tensor([1., -1., 1.]).to(device)
            in_tensor["smpl_faces"] = batch_smpl_faces[:, :, [0, 2, 1]]

        else:
            # smpl optimization
            loop_smpl = tqdm(range(args.loop_smpl))

            for i in loop_smpl:

                per_loop_lst = []

                optimizer_smpl.zero_grad()

                N_body, N_pose = optimed_pose.shape[:2]

                # 6d_rot to rot_mat
                optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1,
                                                                         6)).view(N_body, 1, 3, 3)
                optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1,
                                                                     6)).view(N_body, N_pose, 3, 3)

                smpl_verts, smpl_landmarks, smpl_joints = dataset.smpl_model(
                    shape_params=optimed_betas,
                    expression_params=tensor2variable(data["exp"], device),
                    body_pose=optimed_pose_mat,
                    global_pose=optimed_orient_mat,
                    jaw_pose=tensor2variable(data["jaw_pose"], device),
                    left_hand_pose=tensor2variable(data["left_hand_pose"], device),
                    right_hand_pose=tensor2variable(data["right_hand_pose"], device),
                )

                smpl_verts = (smpl_verts + optimed_trans) * data["scale"]
                smpl_joints = (smpl_joints + optimed_trans) * data["scale"] * torch.tensor([
                    1.0, 1.0, -1.0
                ]).to(device)

                # landmark errors
                smpl_joints_3d = (
                    smpl_joints[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0
                ) * 0.5
                in_tensor["smpl_joint"] = smpl_joints[:,
                                                      dataset.smpl_data.smpl_joint_ids_24_pixie, :]

                ghum_lmks = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                ghum_conf = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
                smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]

                # render optimized mesh as normal [-1,1]
                in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                    smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
                    in_tensor["smpl_faces"],
                )

                T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

                with torch.no_grad():
                    # [1, 3, 512, 512], (-1.0, 1.0)
                    in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor, args.out_dir)
                    
                # Flip the x-component and invert the entire back normal map
                # in_tensor["normal_B"][:, 0, :, :] = -in_tensor["normal_B"][:, 0, :, :]  # Flip x component
                # in_tensor["normal_B"] = -in_tensor["normal_B"]  # Invert entire normal map

                # # Normalize to maintain unit length
                # in_tensor["normal_B"] = in_tensor["normal_B"] / torch.norm(in_tensor["normal_B"], dim=1, keepdim=True)
                # normal_net.netG.save_nml_image(in_tensor["normal_F"], in_tensor["normal_B"], out_dir=args.out_dir)


                """
                img_norm_path_f = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_front_normal.png")
                torchvision.utils.save_image((in_tensor['normal_F'].detach().cpu()),img_norm_path_f)
                img_norm_path_b = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_back_normal.png")
                torchvision.utils.save_image((in_tensor['normal_B'].detach().cpu()),img_norm_path_b)"""

                img_crop_path = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_normal_and_mask.png")
                row1 = torch.cat([data["img_crop"][:, :3], data_b["img_crop"][:, :3]], dim=3)  # Concatenate front and back images
                row2 = torch.cat([(in_tensor['normal_F'].detach().cpu() + 1.0) * 0.5, 
                                (in_tensor['normal_B'].detach().cpu() + 1.0) * 0.5], dim=3)  # Concatenate normal front and back
                row3 = torch.cat([in_tensor["mask"].unsqueeze(1).repeat(1, 3, 1, 1).detach().cpu(),
                                in_tensor["mask_back"].unsqueeze(1).repeat(1, 3, 1, 1).detach().cpu()], dim=3)  # Concatenate mask and mask_back

                # Now stack the rows vertically (along dim=2)
                final_tensor = torch.cat([row1, row2, row3], dim=2)

                # Save the final vertically stacked image
                torchvision.utils.save_image(final_tensor, img_crop_path)

                # only replace the front cloth normals, and the back cloth normals will get improved accordingly
                # as the back cloth normals are conditioned on the body cloth normals

                if cfg.sapiens.use:
                    in_tensor["normal_F"] = sapiens_normal_square

                diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
                diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

                # silhouette loss
                smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
                gt_arr = in_tensor["mask"].repeat(1, 1, 2)
                diff_S = torch.abs(smpl_arr - gt_arr)
                losses["silhouette"]["value"] = diff_S.mean()

                # large cloth_overlap --> big difference between body and cloth mask
                # for loose clothing, reply more on landmarks instead of silhouette+normal loss
                cloth_overlap = diff_S.sum(dim=[1, 2]) / gt_arr.sum(dim=[1, 2])
                cloth_overlap_flag = cloth_overlap > cfg.cloth_overlap_thres
                losses["joint"]["weight"] = [10.0 if flag else 1.0 for flag in cloth_overlap_flag]

                # small body_overlap --> large occlusion or out-of-frame
                # for highly occluded body, reply only on high-confidence landmarks, no silhouette+normal loss

                # BUG: PyTorch3D silhouette renderer generates dilated mask
                bg_value = in_tensor["T_normal_F"][0, 0, 0, 0]
                smpl_arr_fake = torch.cat([
                    in_tensor["T_normal_F"][:, 0].ne(bg_value).float(),
                    in_tensor["T_normal_B"][:, 0].ne(bg_value).float()
                ],
                                          dim=-1)

                body_overlap = (gt_arr * smpl_arr_fake.gt(0.0)
                               ).sum(dim=[1, 2]) / smpl_arr_fake.gt(0.0).sum(dim=[1, 2])
                body_overlap_mask = (gt_arr * smpl_arr_fake).unsqueeze(1)
                body_overlap_flag = body_overlap < cfg.body_overlap_thres

                if not cfg.sapiens.use:
                    losses["normal"]["value"] = (
                        diff_F_smpl * body_overlap_mask[..., :512] +
                        diff_B_smpl * body_overlap_mask[..., 512:]
                    ).mean() / 2.0
                else:
                    losses["normal"]["value"] = diff_F_smpl * body_overlap_mask[..., :512]

                losses["silhouette"]["weight"] = [0 if flag else 1.0 for flag in body_overlap_flag]
                occluded_idx = torch.where(body_overlap_flag)[0]
                ghum_conf[occluded_idx] *= ghum_conf[occluded_idx] > 0.95
                losses["joint"]["value"] = (torch.norm(ghum_lmks - smpl_lmks, dim=2) *
                                            ghum_conf).mean(dim=1)

                # Weighted sum of the losses
                smpl_loss = 0.0
                pbar_desc = "Body Fitting -- "
                for k in ["normal", "silhouette", "joint"]:
                    per_loop_loss = (
                        losses[k]["value"] * torch.tensor(losses[k]["weight"]).to(device)
                    ).mean()
                    pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                    smpl_loss += per_loop_loss
                pbar_desc += f"Total: {smpl_loss:.3f}"
                loose_str = ''.join([str(j) for j in cloth_overlap_flag.int().tolist()])
                occlude_str = ''.join([str(j) for j in body_overlap_flag.int().tolist()])
                pbar_desc += colored(f"| loose:{loose_str}, occluded:{occlude_str}", "yellow")
                loop_smpl.set_description(pbar_desc)

                # save intermediate results
                if (i == args.loop_smpl - 1) and (not args.novis):

                    per_loop_lst.extend([
                        in_tensor["image"],
                        in_tensor["T_normal_F"],
                        in_tensor["normal_F"],
                        diff_S[:, :, :512].unsqueeze(1).repeat(1, 3, 1, 1),
                    ])
                    per_loop_lst.extend([
                        # @SSH
                        # in_tensor["image"],
                        in_tensor["image_back"],
                        # @SSH END
                        in_tensor["T_normal_B"],
                        in_tensor["normal_B"],
                        diff_S[:, :, 512:].unsqueeze(1).repeat(1, 3, 1, 1),
                    ])
                    per_data_lst.append(
                        get_optim_grid_image(per_loop_lst, None, nrow=N_body * 2, type="smpl")
                    )

                smpl_loss.backward()
                optimizer_smpl.step()
                scheduler_smpl.step(smpl_loss)

            in_tensor["smpl_verts"] = smpl_verts * torch.tensor([1.0, 1.0, -1.0]).to(device)
            in_tensor["smpl_faces"] = in_tensor["smpl_faces"][:, :, [0, 2, 1]]

            if not args.novis:
                per_data_lst[-1].save(
                    osp.join(args.out_dir, cfg.name, f"png/{data['name']}_smpl.png")
                )

        if not args.novis:
            img_crop_path = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_crop.png")
            torchvision.utils.save_image(
                torch.cat([
                    data["img_crop"][:, :3], (in_tensor['normal_F'].detach().cpu() + 1.0) * 0.5,
                    (in_tensor['normal_B'].detach().cpu() + 1.0) * 0.5
                ],
                          dim=3), img_crop_path
            )

            rgb_norm_F = blend_rgb_norm(in_tensor["normal_F"], data)
            rgb_norm_B = blend_rgb_norm(in_tensor["normal_B"], data)

            img_overlap_path = osp.join(args.out_dir, cfg.name, f"png/{data['name']}_overlap.png")
            torchvision.utils.save_image(
                torch.cat([data["img_raw"], rgb_norm_F, rgb_norm_B], dim=-1) / 255.,
                img_overlap_path
            )

        smpl_obj_lst = []

        for idx in range(N_body):

            smpl_obj = trimesh.Trimesh(
                in_tensor["smpl_verts"].detach().cpu()[idx] * torch.tensor([1.0, -1.0, 1.0]),
                in_tensor["smpl_faces"].detach().cpu()[0][:, [0, 2, 1]],
                process=False,
                maintains_order=True,
            )

            smpl_obj_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_{idx:02d}.obj"

            if not osp.exists(smpl_obj_path) or cfg.force_smpl_optim:
                smpl_obj.export(smpl_obj_path)
                smpl_info = {
                    "betas":
                    optimed_betas[idx].detach().cpu().unsqueeze(0),
                    "body_pose":
                    rotation_matrix_to_angle_axis(optimed_pose_mat[idx].detach()
                                                 ).cpu().unsqueeze(0),
                    "global_orient":
                    rotation_matrix_to_angle_axis(optimed_orient_mat[idx].detach()
                                                 ).cpu().unsqueeze(0),
                    "transl":
                    optimed_trans[idx].detach().cpu(),
                    "expression":
                    data["exp"][idx].cpu().unsqueeze(0),
                    "jaw_pose":
                    rotation_matrix_to_angle_axis(data["jaw_pose"][idx]).cpu().unsqueeze(0),
                    "left_hand_pose":
                    rotation_matrix_to_angle_axis(data["left_hand_pose"][idx]).cpu().unsqueeze(0),
                    "right_hand_pose":
                    rotation_matrix_to_angle_axis(data["right_hand_pose"][idx]).cpu().unsqueeze(0),
                    "scale":
                    data["scale"][idx].cpu(),
                }
                np.save(
                    smpl_obj_path.replace(".obj", ".npy"),
                    smpl_info,
                    allow_pickle=True,
                )
            smpl_obj_lst.append(smpl_obj)

        del optimizer_smpl
        del optimed_betas
        del optimed_orient
        del optimed_pose
        del optimed_trans

        torch.cuda.empty_cache()

        # ------------------------------------------------------------------------------------------------------------------
        # clothing refinement

        per_data_lst = []

        batch_smpl_verts = in_tensor["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0],
                                                                           device=device)
        batch_smpl_faces = in_tensor["smpl_faces"].detach()[:, :, [0, 2, 1]]

        in_tensor["depth_F"], in_tensor["depth_B"] = dataset.render_depth(
            batch_smpl_verts, batch_smpl_faces
        )

        per_loop_lst = []

        in_tensor["BNI_verts"] = []
        in_tensor["BNI_faces"] = []
        in_tensor["body_verts"] = []
        in_tensor["body_faces"] = []

        for idx in range(N_body):

            final_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full.obj"

            side_mesh = smpl_obj_lst[idx].copy()
            face_mesh = smpl_obj_lst[idx].copy()
            hand_mesh = smpl_obj_lst[idx].copy()
            smplx_mesh = smpl_obj_lst[idx].copy()


            # save normals, depths and masks
            BNI_dict = save_normal_tensor(
                in_tensor,
                idx,
                osp.join(args.out_dir, cfg.name, f"BNI/{data['name']}_{idx}"),
                cfg.bni.thickness,
            )

            # BNI process
            BNI_object = BNI(
                dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
                name=data["name"],
                BNI_dict=BNI_dict,
                cfg=cfg.bni,
                device=device
            )

            BNI_object.extract_surface(False)

            in_tensor["body_verts"].append(torch.tensor(smpl_obj_lst[idx].vertices).float())
            in_tensor["body_faces"].append(torch.tensor(smpl_obj_lst[idx].faces).long())

            # requires shape completion when low overlap
            # replace SMPL by completed mesh as side_mesh

            if cfg.bni.use_ifnet:
                side_mesh_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_IF.obj"
                side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)
                # mesh completion via IF-net
                in_tensor.update(
                    dataset.depth_to_voxel({
                        "depth_F": BNI_object.F_depth.unsqueeze(0), 
                        "depth_B": BNI_object.B_depth.unsqueeze(0)
                    })
                )
                occupancies = VoxelGrid.from_mesh(side_mesh, cfg.vol_res, loc=[
                    0,
                ] * 3, scale=2.0).data.transpose(2, 1, 0)
                occupancies = np.flip(occupancies, axis=1)

                in_tensor["body_voxels"] = torch.tensor(occupancies.copy()
                                                       ).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    sdf = ifnet.reconEngine(netG=ifnet.netG, batch=in_tensor)
                    verts_IF, faces_IF = ifnet.reconEngine.export_mesh(sdf)
                if ifnet.clean_mesh_flag:
                    verts_IF, faces_IF = clean_mesh(verts_IF, faces_IF)
                side_mesh = trimesh.Trimesh(verts_IF, faces_IF)
                side_mesh = remesh_laplacian(side_mesh, side_mesh_path)

            else:
                side_mesh = apply_vertex_mask(
                    side_mesh,
                    (
                        SMPLX_object.front_flame_vertex_mask + SMPLX_object.smplx_mano_vertex_mask +
                        SMPLX_object.eyeball_vertex_mask
                    ).eq(0).float(),
                )

                #register side_mesh to BNI surfaces
                side_mesh = Meshes(
                    verts=[torch.tensor(side_mesh.vertices).float()],
                    faces=[torch.tensor(side_mesh.faces).long()],
                ).to(device)
                sm = SubdivideMeshes(side_mesh)
                side_mesh = register(BNI_object.F_B_trimesh, sm(side_mesh), device)

            side_verts = torch.tensor(side_mesh.vertices).float().to(device)
            side_faces = torch.tensor(side_mesh.faces).long().to(device)

            # Possion Fusion between SMPLX and BNI
            # 1. keep the faces invisible to front+back cameras
            # 2. keep the front-FLAME+MANO faces
            # 3. remove eyeball faces

            # export intermediate meshes
            BNI_object.F_B_trimesh.export(
                f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj"
            )
            full_lst = []

            if "face" in cfg.bni.use_smpl:
                
                # only face
                face_mesh = apply_vertex_mask(face_mesh, SMPLX_object.front_flame_vertex_mask)

                if not face_mesh.is_empty:
                    face_mesh.vertices = face_mesh.vertices - np.array([0, 0, cfg.bni.thickness])

                    # remove face neighbor triangles
                    BNI_object.F_B_trimesh = part_removal(
                        BNI_object.F_B_trimesh,
                        face_mesh,
                        cfg.bni.face_thres,
                        device,
                        smplx_mesh,
                        region="face"
                    )
                    side_mesh = part_removal(
                        side_mesh, face_mesh, cfg.bni.face_thres, device, smplx_mesh, region="face"
                    )
                    face_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_face.obj")
                    full_lst += [face_mesh]

            if "hand" in cfg.bni.use_smpl:
                hand_mask = torch.zeros(SMPLX_object.smplx_verts.shape[0], )

                if data['hands_visibility'][idx][0]:

                    mano_left_vid = np.unique(
                        np.concatenate([
                            SMPLX_object.smplx_vert_seg["leftHand"],
                            SMPLX_object.smplx_vert_seg["leftHandIndex1"],
                        ])
                    )

                    hand_mask.index_fill_(0, torch.tensor(mano_left_vid), 1.0)

                if data['hands_visibility'][idx][1]:

                    mano_right_vid = np.unique(
                        np.concatenate([
                            SMPLX_object.smplx_vert_seg["rightHand"],
                            SMPLX_object.smplx_vert_seg["rightHandIndex1"],
                        ])
                    )

                    hand_mask.index_fill_(0, torch.tensor(mano_right_vid), 1.0)

                # only hands
                hand_mesh = apply_vertex_mask(hand_mesh, hand_mask)

                if not hand_mesh.is_empty:
                    # remove hand neighbor triangles
                    BNI_object.F_B_trimesh = part_removal(
                        BNI_object.F_B_trimesh,
                        hand_mesh,
                        cfg.bni.hand_thres,
                        device,
                        smplx_mesh,
                        region="hand"
                    )
                    side_mesh = part_removal(
                        side_mesh, hand_mesh, cfg.bni.hand_thres, device, smplx_mesh, region="hand"
                    )
                    hand_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_hand.obj")
                    full_lst += [hand_mesh]

            full_lst += [BNI_object.F_B_trimesh]

            # initial side_mesh could be SMPLX or IF-net
            side_mesh = part_removal(
                side_mesh, sum(full_lst), 2e-2, device, smplx_mesh, region="", clean=False
            )

            full_lst += [side_mesh]

            # # export intermediate meshes
            BNI_object.F_B_trimesh.export(
                f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj"
            )
            side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_side.obj")

            if cfg.bni.use_poisson:
                final_mesh = poisson(
                    sum(full_lst),
                    final_path,
                    cfg.bni.poisson_depth,
                )
                print(
                    colored(
                        f"\n Poisson completion to {Format.start} {final_path} {Format.end}",
                        "yellow"
                    )
                )
            else:
                final_mesh = sum(full_lst)
                final_mesh.export(final_path)

        if not args.novis:
            # Load rendered images from multiple views.
            dataset.render.load_meshes(final_mesh.vertices, final_mesh.faces)
            rotate_recon_lst = dataset.render.get_image(cam_type="four")
            # Create two separate lists for the cloth images: one for front and one for back.
            per_loop_lst_front = [in_tensor['image'][idx:idx + 1]] + rotate_recon_lst
            per_loop_lst_back  = [in_tensor['image_back'][idx:idx + 1]] + rotate_recon_lst

        # @SSH
        # --- Color the final mesh ---
            if cfg.bni.texture_src == 'image':

                # coloring the final mesh (front: RGB pixels, back: normal colors)
                final_colors = query_color(
                    torch.tensor(final_mesh.vertices).float(),
                    torch.tensor(final_mesh.faces).long(),
                torch.tensor(side_mesh.vertices).float(),
                torch.tensor(side_mesh.faces).long(),
                in_tensor["image"][idx:idx + 1],
                in_tensor["image_back"][idx:idx + 1],
                device=device,
            )
                final_mesh.visual.vertex_colors = final_colors
                final_mesh.export(final_path)
            

            elif cfg.bni.texture_src == 'SD':

                # !TODO: add texture from Stable Diffusion
                pass

        if not args.novis and len(per_loop_lst_front) > 0 and len(per_loop_lst_back) > 0:
            # Save the front cloth image.
            cloth_front = get_optim_grid_image(per_loop_lst_front, None, nrow=5, type="cloth_front")
            cloth_front_path = osp.join(args.out_dir, cfg.name, f"png/{data['name']}_cloth_front.png")
            cloth_front.save(cloth_front_path)

            # Save the back cloth image.
            cloth_back = get_optim_grid_image(per_loop_lst_back, None, nrow=5, type="cloth_back")
            cloth_back_path = osp.join(args.out_dir, cfg.name, f"png/{data['name']}_cloth_back.png")
            cloth_back.save(cloth_back_path)
        # @SSH END
            # for video rendering
            in_tensor["BNI_verts"].append(torch.tensor(final_mesh.vertices).float())
            in_tensor["BNI_faces"].append(torch.tensor(final_mesh.faces).long())

            os.makedirs(osp.join(args.out_dir, cfg.name, "vid"), exist_ok=True)
            in_tensor["uncrop_param"] = data["uncrop_param"]
            in_tensor["img_raw"] = data["img_raw"]
            torch.save(
                in_tensor, osp.join(args.out_dir, cfg.name, f"vid/{data['name']}_in_tensor.pt")
            )

        final_watertight_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full_wt.obj"
        watertightifier = MeshWatertightifier(final_path, final_watertight_path)
        result = watertightifier.process(reconstruction_method='poisson', depth=10)

        if result:
            print("The mesh is watertight and has been saved successfully!")
        else:
            print("The mesh is not watertight. Further inspection may be needed.")


print("avatarizer")

smplx_container = SMPLX()
device = torch.device(f"cuda:{args.gpu}")

# loading SMPL-X and econ objs inferred with ECON
# prefix = f"./results_fulden/econ/obj/{args.name}"
prefix = f"./results/Fulden/IFN+_face_thresh_0.31/econ/obj/{args.name}"

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

print(smplx_container.model_dir)
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

from lib.common.render import query_color, query_normal_color, query_avatar_color
from lib.common.render_utils import Pytorch3dRasterizer

# choice 1: pixels to visible regions, normals to invisible regions

# @SSH
# if not osp.exists(f"{prefix}/econ_icp_rgb.ply"):
#     masked_image = f"./results/Fulden/face_thresh_0.30/econ/png/{args.name}_cloth.png"
#     # masked_image = f"./results/econ/png/{args.name}_cloth.png"

#     tensor_image = transforms.ToTensor()(Image.open(masked_image))[:, :, :512]
#     final_rgb = query_color(
#         torch.tensor(econ_pose.vertices).float(),
#         torch.tensor(econ_pose.faces).long(),
#         ((tensor_image - 0.5) * 2.0).unsqueeze(0).to(device),
#         ((tensor_image - 0.5) * 2.0).unsqueeze(0).to(device),   # back image
#         device=device,
#         paint_normal=False,
#     ).numpy()
#     final_rgb[final_rgb == tensor_image[:, 0, 0] * 255.0] = 0.5 * 255.0

#     econ_pose.visual.vertex_colors = final_rgb
#     econ_pose.export(f"{prefix}/econ_icp_rgb.ply")
# else:
#     mesh = trimesh.load(f"{prefix}/econ_icp_rgb.ply")
#     final_rgb = mesh.visual.vertex_colors[:, :3]
if not osp.exists(f"{prefix}/econ_icp_rgb.ply"):
    # Load the separate cloth images saved from infer_f.py.
    cloth_front_path = f"./results/Fulden/IFN+_face_thresh_0.31/econ/png/{args.name}_cloth_front.png"
    cloth_back_path  = f"./results/Fulden/IFN+_face_thresh_0.31/econ/png/{args.name}_cloth_back.png"

    # Load the images as tensors.
    tensor_front = transforms.ToTensor()(Image.open(cloth_front_path))[:, :, :512]
    tensor_back  = transforms.ToTensor()(Image.open(cloth_back_path))[:, :, :512]

    # Normalize the textures from [0,1] to [-1,1] and add a batch dimension.
    front_image = ((tensor_front - 0.5) * 2.0).unsqueeze(0).to(device)
    back_image  = ((tensor_back  - 0.5) * 2.0).unsqueeze(0).to(device)

    # Call the updated query_color that now accepts two images.
    final_rgb = query_color(
        torch.tensor(econ_pose.vertices).float(),
        torch.tensor(econ_pose.faces).long(),
        torch.tensor(side_mesh.vertices).float(),
        torch.tensor(side_mesh.faces).long(),
    # in_tensor["image"][idx:idx + 1],
    # in_tensor["image_back"][idx:idx + 1],
    front_image,
    back_image,
    device=device,
    ).numpy()

    # Optionally: adjust any pixels that still match a default color if needed.
    # For example:
    final_rgb[final_rgb == tensor_front[:, 0, 0] * 255.0] = 0.5 * 255.0

    econ_pose.visual.vertex_colors = final_rgb
    econ_pose.export(f"{prefix}/econ_icp_rgb.ply")
else:
    mesh = trimesh.load(f"{prefix}/econ_icp_rgb.ply")
    final_rgb = mesh.visual.vertex_colors[:, :3]

# @SSH END 

# choice 2: normals to all the regions

if not osp.exists(f"{prefix}/econ_icp_normal.ply"):

    file_normal = query_normal_color(
        torch.tensor(econ_pose.vertices).float(),
        torch.tensor(econ_pose.faces).long(),
        device=device,
    ).numpy()

    econ_pose.visual.vertex_colors = file_normal
    econ_pose.export(f"{prefix}/econ_icp_normal.ply")
else:
    mesh = trimesh.load(f"{prefix}/econ_icp_normal.ply")
    file_normal = mesh.visual.vertex_colors[:, :3]

# econ data used for animation and rendering

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
    colored(
        "If the dress/skirt is torn in `<file_name>/econ_da.obj`, please delete ./file_name and regenerate them with `-dress` \n \
    python -m apps.avatarizer -n <file_name> -dress", "yellow"
    )
)

args.uv = True
args.dress = False

if args.uv:

    print("Start UV texture generation...")

    # Generate UV coords
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

    # UV texture rendering

    uv_rasterizer = Pytorch3dRasterizer(image_size=8192, device=device)

    texture_npy = uv_rasterizer.get_texture(
        torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
        ft,
        torch.tensor(v_np).unsqueeze(0).float(),
        torch.tensor(f_np).unsqueeze(0).long(),
        torch.tensor(final_rgb).unsqueeze(0).float() / 255.0,
    )

    gray_texture = texture_npy.copy()
    gray_texture[texture_npy.sum(axis=2) == 0.0] = 0.5
    Image.fromarray((gray_texture * 255.0).astype(np.uint8)).save(f"{cache_path}/texture.png")


    # UV mask for TEXTure (https://readpaper.com/paper/4720151447010820097)
    white_texture = texture_npy.copy()
    white_texture[texture_npy.sum(axis=2) == 0.0] = 1.0
    Image.fromarray((white_texture * 255.0).astype(np.uint8)).save(f"{cache_path}/mask.png")
    
    # generate a-pose vertices
    new_pose = smpl_out_lst[0].full_pose
    new_pose[:, :3] = 0.0

    posed_econ_verts, _ = general_lbs(
        pose=new_pose,
        v_template=econ_cano_verts.unsqueeze(0),
        posedirs=econ_posedirs,
        J_regressor=econ_J_regressor,
        parents=smpl_model.parents,
        lbs_weights=econ_lbs_weights,
    )

    # export mtl file
    with open(f"{cache_path}/material.mtl", "w") as fp:
        fp.write(f"newmtl mat0 \n")
        fp.write(f"Ka 1.000000 1.000000 1.000000 \n")
        fp.write(f"Kd 1.000000 1.000000 1.000000 \n")
        fp.write(f"Ks 0.000000 0.000000 0.000000 \n")
        fp.write(f"Tr 1.000000 \n")
        fp.write(f"illum 1 \n")
        fp.write(f"Ns 0.000000 \n")
        fp.write(f"map_Kd texture.png \n")

    export_obj(posed_econ_verts[0].detach().cpu().numpy(), f_np, vt, ft, f"{cache_path}/mesh.obj")