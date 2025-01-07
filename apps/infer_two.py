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
from apps.Normal import Normal
from apps.sapiens import ImageProcessor

from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.common.BNI_utils import save_normal_tensor_default
from lib.common.config import cfg
from lib.common.imutils import blend_rgb_norm, load_img, transform_to_tensor, wrap
from lib.common.local_affine import register
from lib.common.render import query_color
from lib.common.train_util import Format, init_loss
from lib.common.voxelize import VoxelGrid
from lib.dataset.mesh_util import *
from lib.dataset.TestDataset import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=50)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-in_f_dir", "--in_f_dir", type=str, default="./examples/front")
    parser.add_argument("-in_b_dir", "--in_b_dir", type=str, default="./examples/back")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ.yaml")
    parser.add_argument("-multi", action="store_false")
    parser.add_argument("-novis", action="store_true")

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
        "front_image_dir": args.in_f_dir,
        "back_image_dir": args.in_b_dir,
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
    front_dict, back_dict = dataset[0]

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)

    for data_f, data_b in pbar:

        losses_f = init_loss()
        losses_b = init_loss()

        pbar.set_description(f"{data_f['name']}")

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

        in_tensor_f = {
            "smpl_faces": data_f["smpl_faces"], "image": data_f["img_icon"].to(device), "mask":
            data_f["img_mask"].to(device)
        }

        in_tensor_b = {
            "smpl_faces": data_b["smpl_faces"], "image": data_b["img_icon"].to(device), "mask":
            data_b["img_mask"].to(device)
        }

        # The optimizer and variables
        optimed_pose_f = data_f["body_pose"].requires_grad_(True)
        optimed_trans_f = data_f["trans"].requires_grad_(True)
        optimed_betas_f = data_f["betas"].requires_grad_(True)
        optimed_orient_f = data_f["global_orient"].requires_grad_(True)

        # The optimizer and variables
        optimed_pose_b = data_b["body_pose"].requires_grad_(True)
        optimed_trans_b = data_b["trans"].requires_grad_(True)
        optimed_betas_b = data_b["betas"].requires_grad_(True)
        optimed_orient_b = data_b["global_orient"].requires_grad_(True)

        optimizer_smpl_f = torch.optim.Adam([
            optimed_pose_f, optimed_trans_f, optimed_betas_f, optimed_orient_f
        ],
                                          lr=1e-2,
                                          amsgrad=True)
        scheduler_smpl_f = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl_f,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=args.patience,
        )

        optimizer_smpl_b = torch.optim.Adam([
            optimed_pose_b, optimed_trans_b, optimed_betas_b, optimed_orient_b
        ],
                                          lr=1e-2,
                                          amsgrad=True)
        scheduler_smpl_b = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl_b,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=args.patience,
        )

        # [result_loop_1, result_loop_2, ...]
        per_data_lst = []

        N_body_f, N_pose_f = optimed_pose_f.shape[:2]
        N_body_b, N_pose_b = optimed_pose_b.shape[:2]

        smpl_path_f = f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_f_smpl_00.obj"
        smpl_path_b = f"{args.out_dir}/{cfg.name}/obj/{data_b['name']}_b_smpl_00.obj"

        # sapiens inference for current batch data

        if cfg.sapiens.use:
            
            sapiens_normal_f = sapiens_normal_net.process_image(
                Image.fromarray(
                    data_f["img_raw"].squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                ), "1b", cfg.sapiens.seg_model
            )
            sapiens_normal_b = sapiens_normal_net.process_image(
                Image.fromarray(
                    data_b["img_raw"].squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                ), "1b", cfg.sapiens.seg_model
            )
            print(colored("Estimating normal maps from input image, using Sapiens-normal", "green"))

            sapiens_normal_square_lst = []
            for idx in range(len(data_f["img_icon"])):
                sapiens_normal_square_lst.append(wrap(sapiens_normal_f, data_f["uncrop_param"], idx))
            sapiens_normal_square = torch.cat(sapiens_normal_square_lst)

        # remove this line if you change the loop_smpl and obtain different SMPL-X fits
        if osp.exists(smpl_path_f) and osp.exists(smpl_path_b) and (not cfg.force_smpl_optim):

            smpl_verts_f_lst = []
            smpl_faces_f_lst = []

            smpl_verts_b_lst = []
            smpl_faces_b_lst = []

            for idx in range(N_body_f):

                smpl_obj_f = f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_smpl_{idx:02d}.obj"
                smpl_mesh_f = trimesh.load(smpl_obj_f)
                smpl_verts_f = torch.tensor(smpl_mesh_f.vertices).to(device).float()
                smpl_faces_f = torch.tensor(smpl_mesh_f.faces).to(device).long()
                smpl_verts_f_lst.append(smpl_verts_f)
                smpl_faces_f_lst.append(smpl_faces_f)

            batch_smpl_f_verts = torch.stack(smpl_verts_f_lst)
            batch_smpl_f_faces = torch.stack(smpl_faces_f_lst)

            for idx in range(N_body_b):

                smpl_obj_b = f"{args.out_dir}/{cfg.name}/obj/{data_b['name']}_smpl_{idx:02d}.obj"
                smpl_mesh_b = trimesh.load(smpl_obj_b)
                smpl_verts_b = torch.tensor(smpl_mesh_b.vertices).to(device).float()
                smpl_faces_b = torch.tensor(smpl_mesh_b.faces).to(device).long()
                smpl_verts_b_lst.append(smpl_verts_b)
                smpl_faces_b_lst.append(smpl_faces_b)

            batch_smpl_b_verts = torch.stack(smpl_verts_b_lst)
            batch_smpl_b_faces = torch.stack(smpl_faces_b_lst)

            # render optimized mesh as normal [-1,1] FRONT
            in_tensor_f["T_normal_F"], in_tensor_f["T_normal_B"] = dataset.render_normal(
                batch_smpl_f_verts, batch_smpl_f_faces
            )

            with torch.no_grad():
                in_tensor_f["normal_F"], in_tensor_f["normal_B"] = normal_net.netG(in_tensor_f)

            in_tensor_f["smpl_verts"] = batch_smpl_f_verts * torch.tensor([1., -1., 1.]).to(device)
            in_tensor_f["smpl_faces"] = batch_smpl_f_faces[:, :, [0, 2, 1]]

            # render optimized mesh as normal [-1,1] BACK
            in_tensor_b["T_normal_F"], in_tensor_b["T_normal_B"] = dataset.render_normal(
                batch_smpl_b_verts, batch_smpl_b_faces
            )

            with torch.no_grad():
                in_tensor_b["normal_F"], in_tensor_b["normal_B"] = normal_net.netG(in_tensor_b)

            in_tensor_b["smpl_verts"] = batch_smpl_b_verts * torch.tensor([1., -1., 1.]).to(device)
            in_tensor_b["smpl_faces"] = batch_smpl_b_faces[:, :, [0, 2, 1]]

        else:
            # smpl optimization
            loop_smpl = tqdm(range(args.loop_smpl))

            for i in loop_smpl:

                per_loop_lst = []

                optimizer_smpl_f.zero_grad()
                N_body_f, N_pose_f = optimed_pose_f.shape[:2]

                optimizer_smpl_b.zero_grad()
                N_body_b, N_pose_b = optimed_pose_b.shape[:2]

                # 6d_rot to rot_mat FRONT
                optimed_orient_mat_f = rot6d_to_rotmat(optimed_orient_f.view(-1,
                                                                         6)).view(N_body_f, 1, 3, 3)
                optimed_pose_mat_f = rot6d_to_rotmat(optimed_pose_f.view(-1,
                                                                     6)).view(N_body_f, N_pose_f, 3, 3)

                smpl_verts_f, smpl_landmarks_f, smpl_joints_f = dataset.smpl_model(
                    shape_params=optimed_betas_f,
                    expression_params=tensor2variable(data_f["exp"], device),
                    body_pose=optimed_pose_mat_f,
                    global_pose=optimed_orient_mat_f,
                    jaw_pose=tensor2variable(data_f["jaw_pose"], device),
                    left_hand_pose=tensor2variable(data_f["left_hand_pose"], device),
                    right_hand_pose=tensor2variable(data_f["right_hand_pose"], device),
                )

                smpl_verts_f = (smpl_verts_f + optimed_trans_f) * data_f["scale"]
                smpl_joints_f = (smpl_joints_f + optimed_trans_f) * data_f["scale"] * torch.tensor([
                    1.0, 1.0, -1.0
                ]).to(device)

                # 6d_rot to rot_mat BACK
                optimed_orient_mat_b = rot6d_to_rotmat(optimed_orient_b.view(-1,
                                                                         6)).view(N_body_b, 1, 3, 3)
                optimed_pose_mat_b = rot6d_to_rotmat(optimed_pose_b.view(-1,
                                                                     6)).view(N_body_b, N_pose_b, 3, 3)

                smpl_verts_b, smpl_landmarks_b, smpl_joints_b = dataset.smpl_model(
                    shape_params=optimed_betas_b,
                    expression_params=tensor2variable(data_b["exp"], device),
                    body_pose=optimed_pose_mat_b,
                    global_pose=optimed_orient_mat_b,
                    jaw_pose=tensor2variable(data_b["jaw_pose"], device),
                    left_hand_pose=tensor2variable(data_b["left_hand_pose"], device),
                    right_hand_pose=tensor2variable(data_b["right_hand_pose"], device),
                )

                smpl_verts_b = (smpl_verts_b + optimed_trans_b) * data_b["scale"]
                smpl_joints_b = (smpl_joints_b + optimed_trans_b) * data_b["scale"] * torch.tensor([
                    1.0, 1.0, -1.0
                ]).to(device)

                # landmark errors FRONT
                smpl_joints_3d_f = (
                    smpl_joints_f[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0
                ) * 0.5
                in_tensor_f["smpl_joint"] = smpl_joints_f[:,
                                                      dataset.smpl_data.smpl_joint_ids_24_pixie, :]

                ghum_lmks_f = data_f["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                ghum_conf_f = data_f["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
                smpl_lmks_f = smpl_joints_3d_f[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]

                # landmark errors BACK
                smpl_joints_3d_b = (
                    smpl_joints_b[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0
                ) * 0.5
                in_tensor_b["smpl_joint"] = smpl_joints_b[:,
                                                      dataset.smpl_data.smpl_joint_ids_24_pixie, :]

                ghum_lmks_b = data_b["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                ghum_conf_b = data_b["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
                smpl_lmks_b = smpl_joints_3d_b[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]


                # render optimized mesh as normal [-1,1] FRONT
                in_tensor_f["T_normal_F"], in_tensor_f["T_normal_B"] = dataset.render_normal(
                    smpl_verts_f * torch.tensor([1.0, -1.0, -1.0]).to(device),
                    in_tensor_f["smpl_faces"],
                )

                # render optimized mesh as normal [-1,1] BACK
                in_tensor_b["T_normal_F"], in_tensor_b["T_normal_B"] = dataset.render_normal(
                    smpl_verts_b * torch.tensor([1.0, -1.0, -1.0]).to(device),
                    in_tensor_b["smpl_faces"],
                )

                T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

                with torch.no_grad():
                    # [1, 3, 512, 512], (-1.0, 1.0)
                    in_tensor_f["normal_F"], in_tensor_f["normal_B"] = normal_net.netG(in_tensor_f)

                with torch.no_grad():
                    # [1, 3, 512, 512], (-1.0, 1.0)
                    in_tensor_b["normal_F"], in_tensor_b["normal_B"] = normal_net.netG(in_tensor_b)


                # only replace the front cloth normals, and the back cloth normals will get improved accordingly
                # as the back cloth normals are conditioned on the body cloth normals

                if cfg.sapiens.use:
                    in_tensor_f["normal_F"] = sapiens_normal_square

                ##########################################################
                ###### front(T_normal_F )- back(normal_F)#################
                diff_F_smpl = torch.abs(in_tensor_f["T_normal_F"] - in_tensor_f["normal_F"])
                diff_B_smpl = torch.abs(in_tensor_b["T_normal_F"] - in_tensor_b["normal_F"])

                # silhouette loss FRONT
                smpl_arr_f = torch.cat([T_mask_F, T_mask_B], dim=-1)
                gt_arr_f = in_tensor_f["mask"].repeat(1, 1, 2)
                diff_S_f = torch.abs(smpl_arr_f - gt_arr_f)
                losses_f["silhouette"]["value"] = diff_S_f.mean()

                # silhouette loss BACK
                smpl_arr_b = torch.cat([T_mask_F, T_mask_B], dim=-1)
                gt_arr_b = in_tensor_b["mask"].repeat(1, 1, 2)
                diff_S_b = torch.abs(smpl_arr_b - gt_arr_b)
                losses_f["silhouette"]["value"] = diff_S_b.mean()

                # large cloth_overlap --> big difference between body and cloth mask
                # for loose clothing, reply more on landmarks instead of silhouette+normal loss
                cloth_overlap_f = diff_S_f.sum(dim=[1, 2]) / gt_arr_f.sum(dim=[1, 2])
                cloth_overlap_flag_f = cloth_overlap_f > cfg.cloth_overlap_thres
                losses_f["joint"]["weight"] = [10.0 if flag else 1.0 for flag in cloth_overlap_flag_f]

                cloth_overlap_b = diff_S_b.sum(dim=[1, 2]) / gt_arr_b.sum(dim=[1, 2])
                cloth_overlap_flag_b = cloth_overlap_b > cfg.cloth_overlap_thres
                losses_b["joint"]["weight"] = [10.0 if flag else 1.0 for flag in cloth_overlap_flag_b]

                # small body_overlap --> large occlusion or out-of-frame
                # for highly occluded body, reply only on high-confidence landmarks, no silhouette+normal loss

                ########mask for FRONT##########
                # BUG: PyTorch3D silhouette renderer generates dilated mask
                bg_value_f = in_tensor_f["T_normal_F"][0, 0, 0, 0]
                smpl_arr_f_fake = torch.cat([
                    in_tensor_f["T_normal_F"][:, 0].ne(bg_value_f).float(),
                    in_tensor_f["T_normal_B"][:, 0].ne(bg_value_f).float()
                ],
                                          dim=-1)

                body_overlap_f = (gt_arr_f * smpl_arr_f_fake.gt(0.0)
                               ).sum(dim=[1, 2]) / smpl_arr_f_fake.gt(0.0).sum(dim=[1, 2])
                body_overlap_mask_f = (gt_arr_f * smpl_arr_f_fake).unsqueeze(1)
                body_overlap_flag_f = body_overlap_f < cfg.body_overlap_thres

                ########mask for BACK##########
                bg_value_b = in_tensor_b["T_normal_F"][0, 0, 0, 0]
                smpl_arr_b_fake = torch.cat([
                    in_tensor_b["T_normal_F"][:, 0].ne(bg_value_b).float(),
                    in_tensor_b["T_normal_B"][:, 0].ne(bg_value_b).float()
                ],
                                          dim=-1)

                body_overlap_b = (gt_arr_b * smpl_arr_b_fake.gt(0.0)
                               ).sum(dim=[1, 2]) / smpl_arr_b_fake.gt(0.0).sum(dim=[1, 2])
                body_overlap_mask_b = (gt_arr_b * smpl_arr_b_fake).unsqueeze(1)
                body_overlap_flag_b = body_overlap_b < cfg.body_overlap_thres

                if not cfg.sapiens.use:
                    losses_f["normal"]["value"] = (
                        diff_F_smpl * body_overlap_mask_f[..., :512] +
                        diff_B_smpl * body_overlap_mask_b[..., 512:]
                    ).mean() / 2.0
                else:
                    losses_f["normal"]["value"] = diff_F_smpl * body_overlap_mask_f[..., :512]
                    losses_b["normal"]["value"] = diff_B_smpl * body_overlap_mask_b[..., :512]

                losses_f["silhouette"]["weight"] = [0 if flag else 1.0 for flag in body_overlap_flag_f]
                occluded_idx_f = torch.where(body_overlap_flag_f)[0]
                ghum_conf_f[occluded_idx_f] *= ghum_conf_f[occluded_idx_f] > 0.95
                losses_f["joint"]["value"] = (torch.norm(ghum_lmks_f - smpl_lmks_f, dim=2) *
                                            ghum_conf_f).mean(dim=1)

                losses_b["silhouette"]["weight"] = [0 if flag else 1.0 for flag in body_overlap_flag_b]
                occluded_idx_b = torch.where(body_overlap_flag_b)[0]
                ghum_conf_b[occluded_idx_b] *= ghum_conf_b[occluded_idx_b] > 0.95
                losses_b["joint"]["value"] = (torch.norm(ghum_lmks_b - smpl_lmks_b, dim=2) *
                                            ghum_conf_b).mean(dim=1)
                
                # Weighted sum of the losses
                smpl_loss = 0.0
                pbar_desc = "Body Fitting -- "
                for k in ["normal", "silhouette", "joint"]:
                    per_loop_loss = (
                        losses_f[k]["value"] * torch.tensor(losses_f[k]["weight"]).to(device)
                    ).mean()
                    pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                    smpl_loss += per_loop_loss
                pbar_desc += f"Total: {smpl_loss:.3f}"
                loose_str = ''.join([str(j) for j in cloth_overlap_flag_f.int().tolist()])
                occlude_str = ''.join([str(j) for j in body_overlap_flag_f.int().tolist()])
                pbar_desc += colored(f"| loose:{loose_str}, occluded:{occlude_str}", "yellow")
                loop_smpl.set_description(pbar_desc)

                # save intermediate results
                if (i == args.loop_smpl - 1) and (not args.novis):

                    per_loop_lst.extend([
                        in_tensor_f["image"],
                        in_tensor_f["T_normal_F"],
                        in_tensor_f["normal_F"],
                        diff_S_f[:, :, :512].unsqueeze(1).repeat(1, 3, 1, 1),
                    ])
                    per_loop_lst.extend([
                        in_tensor_b["image"],
                        in_tensor_b["T_normal_F"],
                        in_tensor_b["normal_F"],
                        diff_S_b[:, :, 512:].unsqueeze(1).repeat(1, 3, 1, 1),
                    ])
                    per_data_lst.append(
                        get_optim_grid_image(per_loop_lst, None, nrow=N_body_f * 2, type="smpl")
                    )

                smpl_loss.backward()
                optimizer_smpl_f.step()
                scheduler_smpl_f.step(smpl_loss)

            in_tensor_f["smpl_verts"] = smpl_verts_f * torch.tensor([1.0, 1.0, -1.0]).to(device)
            in_tensor_f["smpl_faces"] = in_tensor_f["smpl_faces"][:, :, [0, 2, 1]]

            in_tensor_b["smpl_verts"] = smpl_verts_b * torch.tensor([1.0, 1.0, -1.0]).to(device)
            in_tensor_b["smpl_faces"] = in_tensor_b["smpl_faces"][:, :, [0, 2, 1]]

            if not args.novis:
                per_data_lst[-1].save(
                    osp.join(args.out_dir, cfg.name, f"png/{data_f['name']}_f_smpl.png")
                )
            
            if not args.novis:
                per_data_lst[-1].save(
                    osp.join(args.out_dir, cfg.name, f"png/{data_b['name']}_b_smpl.png")
                )

        if not args.novis:
            img_crop_path_f = osp.join(args.out_dir, cfg.name, "png", f"{data_f['name']}_f_crop.png")
            img_crop_path_b = osp.join(args.out_dir, cfg.name, "png", f"{data_b['name']}_f_crop.png")
            img_crop_path = osp.join(args.out_dir, cfg.name, "png", f"{data_f['name']}_final_crop.png")
            torchvision.utils.save_image(
                torch.cat([
                    data_f["img_crop"][:, :3], (in_tensor_f['normal_F'].detach().cpu() + 1.0) * 0.5,
                    (in_tensor_f['normal_B'].detach().cpu() + 1.0) * 0.5
                ],
                          dim=3), img_crop_path_f
            )

            torchvision.utils.save_image(
                torch.cat([
                    data_b["img_crop"][:, :3], (in_tensor_b['normal_F'].detach().cpu() + 1.0) * 0.5,
                    (in_tensor_b['normal_B'].detach().cpu() + 1.0) * 0.5
                ],
                          dim=3), img_crop_path_b
            )

            torchvision.utils.save_image(
                torch.cat([
                    data_f["img_crop"][:, :3], (in_tensor_f['normal_F'].detach().cpu() + 1.0) * 0.5,
                    (in_tensor_b['normal_F'].detach().cpu() + 1.0) * 0.5
                ],
                          dim=3), img_crop_path
            )
            
            rgb_norm_F = blend_rgb_norm(in_tensor_f["normal_F"], data_f)
            rgb_norm_B = blend_rgb_norm(in_tensor_b["normal_F"], data_b)
            rgb_norm_F_b = blend_rgb_norm(in_tensor_f["normal_B"], data_f)
            rgb_norm_B_b = blend_rgb_norm(in_tensor_b["normal_B"], data_b)

            img_overlap_path_f = osp.join(args.out_dir, cfg.name, f"png/{data_f['name']}_f_overlap.png")
            torchvision.utils.save_image(
                torch.cat([data_f["img_raw"], rgb_norm_F, rgb_norm_F_b], dim=-1) / 255.,
                img_overlap_path_f
            )

            img_overlap_path_b = osp.join(args.out_dir, cfg.name, f"png/{data_b['name']}_b_overlap.png")
            torchvision.utils.save_image(
                torch.cat([data_b["img_raw"], rgb_norm_B, rgb_norm_B_b], dim=-1) / 255.,
                img_overlap_path_b
            )

            img_overlap_path = osp.join(args.out_dir, cfg.name, f"png/{data_f['name']}_overlap.png")
            torchvision.utils.save_image(
                torch.cat([data_f["img_raw"], rgb_norm_F, rgb_norm_B], dim=-1) / 255.,
                img_overlap_path
            )
        ####################################################################################################################################
        smpl_obj_lst = []
        smpl_obj_lst_b = []

        for idx in range(N_body_f):

            smpl_obj_f = trimesh.Trimesh(
                in_tensor_f["smpl_verts"].detach().cpu()[idx] * torch.tensor([1.0, -1.0, 1.0]),
                in_tensor_f["smpl_faces"].detach().cpu()[0][:, [0, 2, 1]],
                process=False,
                maintains_order=True,
            )

            smpl_obj_path_f = f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_f_smpl_{idx:02d}.obj"

            if not osp.exists(smpl_obj_path_f) or cfg.force_smpl_optim:
                smpl_obj_f.export(smpl_obj_path_f)
                smpl_info = {
                    "betas":
                    optimed_betas_f[idx].detach().cpu().unsqueeze(0),
                    "body_pose":
                    rotation_matrix_to_angle_axis(optimed_pose_mat_f[idx].detach()
                                                 ).cpu().unsqueeze(0),
                    "global_orient":
                    rotation_matrix_to_angle_axis(optimed_orient_mat_f[idx].detach()
                                                 ).cpu().unsqueeze(0),
                    "transl":
                    optimed_trans_f[idx].detach().cpu(),
                    "expression":
                    data_f["exp"][idx].cpu().unsqueeze(0),
                    "jaw_pose":
                    rotation_matrix_to_angle_axis(data_f["jaw_pose"][idx]).cpu().unsqueeze(0),
                    "left_hand_pose":
                    rotation_matrix_to_angle_axis(data_f["left_hand_pose"][idx]).cpu().unsqueeze(0),
                    "right_hand_pose":
                    rotation_matrix_to_angle_axis(data_f["right_hand_pose"][idx]).cpu().unsqueeze(0),
                    "scale":
                    data_f["scale"][idx].cpu(),
                }
                np.save(
                    smpl_obj_path_f.replace(".obj", ".npy"),
                    smpl_info,
                    allow_pickle=True,
                )
            smpl_obj_lst.append(smpl_obj_f)


            smpl_obj_b = trimesh.Trimesh(
                in_tensor_b["smpl_verts"].detach().cpu()[idx] * torch.tensor([1.0, -1.0, 1.0]),
                in_tensor_b["smpl_faces"].detach().cpu()[0][:, [0, 2, 1]],
                process=False,
                maintains_order=True,
            )

            smpl_obj_path_b = f"{args.out_dir}/{cfg.name}/obj/{data_b['name']}_b_smpl_{idx:02d}.obj"

            if not osp.exists(smpl_obj_path_b) or cfg.force_smpl_optim:
                smpl_obj_b.export(smpl_obj_path_b)
                smpl_info = {
                    "betas":
                    optimed_betas_b[idx].detach().cpu().unsqueeze(0),
                    "body_pose":
                    rotation_matrix_to_angle_axis(optimed_pose_mat_b[idx].detach()
                                                 ).cpu().unsqueeze(0),
                    "global_orient":
                    rotation_matrix_to_angle_axis(optimed_orient_mat_b[idx].detach()
                                                 ).cpu().unsqueeze(0),
                    "transl":
                    optimed_trans_b[idx].detach().cpu(),
                    "expression":
                    data_b["exp"][idx].cpu().unsqueeze(0),
                    "jaw_pose":
                    rotation_matrix_to_angle_axis(data_b["jaw_pose"][idx]).cpu().unsqueeze(0),
                    "left_hand_pose":
                    rotation_matrix_to_angle_axis(data_b["left_hand_pose"][idx]).cpu().unsqueeze(0),
                    "right_hand_pose":
                    rotation_matrix_to_angle_axis(data_b["right_hand_pose"][idx]).cpu().unsqueeze(0),
                    "scale":
                    data_b["scale"][idx].cpu(),
                }
                np.save(
                    smpl_obj_path_b.replace(".obj", ".npy"),
                    smpl_info,
                    allow_pickle=True,
                )
            smpl_obj_lst_b.append(smpl_obj_b)

        del optimizer_smpl_f
        del optimed_betas_f
        del optimed_orient_f
        del optimed_pose_f
        del optimed_trans_f
        del optimizer_smpl_b
        del optimed_betas_b
        del optimed_orient_b
        del optimed_pose_b
        del optimed_trans_b

        torch.cuda.empty_cache()

        # ------------------------------------------------------------------------------------------------------------------
        # clothing refinement

        per_data_lst = []

        batch_smpl_verts_f = in_tensor_f["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0],
                                                                           device=device)
        batch_smpl_faces_f = in_tensor_f["smpl_faces"].detach()[:, :, [0, 2, 1]]

        batch_smpl_verts_b = in_tensor_b["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0],
                                                                           device=device)
        batch_smpl_faces_b = in_tensor_b["smpl_faces"].detach()[:, :, [0, 2, 1]]

        in_tensor_f["depth_F"], in_tensor_b["depth_F"] = dataset.render_depth(
            batch_smpl_verts_f, batch_smpl_faces_f
        )
        in_tensor_f["depth_B"], in_tensor_b["depth_B"] = dataset.render_depth(
            batch_smpl_verts_b, batch_smpl_faces_b
        )
        per_loop_lst = []

        in_tensor_f["BNI_verts"] = []
        in_tensor_f["BNI_faces"] = []
        in_tensor_f["body_verts"] = []
        in_tensor_f["body_faces"] = []

        for idx in range(N_body_f):

            final_path = f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_{idx}_full.obj"

            side_mesh = smpl_obj_lst[idx].copy()
            face_mesh = smpl_obj_lst[idx].copy()
            hand_mesh = smpl_obj_lst[idx].copy()
            smplx_mesh = smpl_obj_lst[idx].copy()

            # save normals, depths and masks
            BNI_dict_f = save_normal_tensor(
                in_tensor_f,
                in_tensor_b,
                idx,
                osp.join(args.out_dir, cfg.name, f"BNI/{data_f['name']}_{idx}"),
                cfg.bni.thickness,
            )

            BNI_dict_b = save_normal_tensor_default(
                in_tensor_b,
                idx,
                osp.join(args.out_dir, cfg.name, f"BNI/{data_b['name']}_{idx}"),
                cfg.bni.thickness,
            )
            
            # BNI process
            BNI_object_f = BNI(
                dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
                name=data_f["name"],
                BNI_dict=BNI_dict_f,
                cfg=cfg.bni,
                device=device
            )

            # BNI process
            BNI_object_b = BNI(
                dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
                name=data_b["name"],
                BNI_dict=BNI_dict_b,
                cfg=cfg.bni,
                device=device
            )

            BNI_object_f.extract_surface(False)

            in_tensor_f["body_verts"].append(torch.tensor(smpl_obj_lst[idx].vertices).float())
            in_tensor_f["body_faces"].append(torch.tensor(smpl_obj_lst[idx].faces).long())

            # requires shape completion when low overlap
            # replace SMPL by completed mesh as side_mesh

            if cfg.bni.use_ifnet:

                side_mesh_path = f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_{idx}_IF.obj"

                side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)

                # mesh completion via IF-net
                in_tensor_f.update(
                    dataset.depth_to_voxel({
                        "depth_F": BNI_object_f.F_depth.unsqueeze(0), "depth_B":
                        BNI_object_b._depth.unsqueeze(0)
                    })
                )

                occupancies = VoxelGrid.from_mesh(side_mesh, cfg.vol_res, loc=[
                    0,
                ] * 3, scale=2.0).data.transpose(2, 1, 0)
                occupancies = np.flip(occupancies, axis=1)

                in_tensor_f["body_voxels"] = torch.tensor(occupancies.copy()
                                                       ).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    sdf = ifnet.reconEngine(netG=ifnet.netG, batch=in_tensor_f)
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
                side_mesh = register(BNI_object_f.F_B_trimesh, sm(side_mesh), device)

            side_verts = torch.tensor(side_mesh.vertices).float().to(device)
            side_faces = torch.tensor(side_mesh.faces).long().to(device)

            # Possion Fusion between SMPLX and BNI
            # 1. keep the faces invisible to front+back cameras
            # 2. keep the front-FLAME+MANO faces
            # 3. remove eyeball faces

            # export intermediate meshes
            BNI_object_f.F_B_trimesh.export(
                f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_{idx}_BNI.obj"
            )
            full_lst = []

            if "face" in cfg.bni.use_smpl:

                # only face
                face_mesh = apply_vertex_mask(face_mesh, SMPLX_object.front_flame_vertex_mask)

                if not face_mesh.is_empty:
                    face_mesh.vertices = face_mesh.vertices - np.array([0, 0, cfg.bni.thickness])

                    # remove face neighbor triangles
                    BNI_object_f.F_B_trimesh = part_removal(
                        BNI_object_f.F_B_trimesh,
                        face_mesh,
                        cfg.bni.face_thres,
                        device,
                        smplx_mesh,
                        region="face"
                    )
                    side_mesh = part_removal(
                        side_mesh, face_mesh, cfg.bni.face_thres, device, smplx_mesh, region="face"
                    )
                    face_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_{idx}_face.obj")
                    full_lst += [face_mesh]

            if "hand" in cfg.bni.use_smpl:
                hand_mask = torch.zeros(SMPLX_object.smplx_verts.shape[0], )

                if data_f['hands_visibility'][idx][0]:

                    mano_left_vid = np.unique(
                        np.concatenate([
                            SMPLX_object.smplx_vert_seg["leftHand"],
                            SMPLX_object.smplx_vert_seg["leftHandIndex1"],
                        ])
                    )

                    hand_mask.index_fill_(0, torch.tensor(mano_left_vid), 1.0)

                if data_f['hands_visibility'][idx][1]:

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
                    BNI_object_f.F_B_trimesh = part_removal(
                        BNI_object_f.F_B_trimesh,
                        hand_mesh,
                        cfg.bni.hand_thres,
                        device,
                        smplx_mesh,
                        region="hand"
                    )
                    side_mesh = part_removal(
                        side_mesh, hand_mesh, cfg.bni.hand_thres, device, smplx_mesh, region="hand"
                    )
                    hand_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_{idx}_hand.obj")
                    full_lst += [hand_mesh]

            full_lst += [BNI_object_f.F_B_trimesh]

            # initial side_mesh could be SMPLX or IF-net
            side_mesh = part_removal(
                side_mesh, sum(full_lst), 2e-2, device, smplx_mesh, region="", clean=False
            )

            full_lst += [side_mesh]

            # # export intermediate meshes
            BNI_object_f.F_B_trimesh.export(
                f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_{idx}_BNI.obj"
            )
            side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data_f['name']}_{idx}_side.obj")

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
                dataset.render.load_meshes(final_mesh.vertices, final_mesh.faces)
                rotate_recon_lst = dataset.render.get_image(cam_type="four")
                per_loop_lst.extend([in_tensor_f['image'][idx:idx + 1]] + rotate_recon_lst)

            if cfg.bni.texture_src == 'image':

                # coloring the final mesh (front: RGB pixels, back: normal colors)
                final_colors = query_color(
                    torch.tensor(final_mesh.vertices).float(),
                    torch.tensor(final_mesh.faces).long(),
                    in_tensor_f["image"][idx:idx + 1],
                    device=device,
                )
                final_mesh.visual.vertex_colors = final_colors
                final_mesh.export(final_path)

            elif cfg.bni.texture_src == 'SD':

                # !TODO: add texture from Stable Diffusion
                pass

        if len(per_loop_lst) > 0 and (not args.novis):

            per_data_lst.append(get_optim_grid_image(per_loop_lst, None, nrow=5, type="cloth"))
            per_data_lst[-1].save(osp.join(args.out_dir, cfg.name, f"png/{data_f['name']}_cloth.png"))

            # for video rendering
            in_tensor_f["BNI_verts"].append(torch.tensor(final_mesh.vertices).float())
            in_tensor_f["BNI_faces"].append(torch.tensor(final_mesh.faces).long())

            os.makedirs(osp.join(args.out_dir, cfg.name, "vid"), exist_ok=True)
            in_tensor_f["uncrop_param"] = data_f["uncrop_param"]
            in_tensor_f["img_raw"] = data_f["img_raw"]
            torch.save(
                in_tensor_f, osp.join(args.out_dir, cfg.name, f"vid/{data_f['name']}_in_tensor.pt")
            )
