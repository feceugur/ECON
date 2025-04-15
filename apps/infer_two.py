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


from apps.transform_normals import transform_normals
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis
from lib.dataset.TestDataset_f import TestDataset
from lib.dataset.mesh_util import *
from lib.common.voxelize import VoxelGrid
from lib.common.train_util import Format, init_loss
from lib.common.render import query_color
from lib.common.local_affine import register
from lib.common.imutils import blend_rgb_norm
from lib.common.config import cfg
from lib.common.BNI_utils import save_normal_tensor, save_normal_tensor_upt
from lib.common.BNI import BNI
from apps.Normal import Normal
from apps.IFGeo import IFGeo
from tqdm.auto import tqdm
from termcolor import colored
from pytorch3d.ops import SubdivideMeshes
import trimesh
import torchvision
import torch
import numpy as np
import os
import argparse
import logging
import warnings
import math

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)


torch.backends.cudnn.benchmark = True

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

    pbar = tqdm(dataset)

    for data, data2 in pbar:

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

        in_tensor_f = {
            "smpl_faces": data["smpl_faces"], "image": data["img_icon"].to(device), "mask":
            data["img_mask"].to(device)
        }

        in_tensor_b = {
            "smpl_faces": data2["smpl_faces"], "image": data2["img_icon"].to(device), "mask":
            data2["img_mask"].to(device)
        }

        # The optimizer and variables
        optimed_pose_f = data["body_pose"].requires_grad_(True)
        optimed_trans_f = data["trans"].requires_grad_(True)
        optimed_betas_f = data["betas"].requires_grad_(True)
        optimed_orient_f = data["global_orient"].requires_grad_(True)

        # The optimizer and variables
        optimed_pose_b = data2["body_pose"].requires_grad_(True)
        optimed_trans_b = data2["trans"].requires_grad_(True)
        optimed_betas_b = data2["betas"].requires_grad_(True)
        optimed_orient_b = data2["global_orient"].requires_grad_(True)

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

        smpl_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_00.obj"
        """
        # remove this line if you change the loop_smpl and obtain different SMPL-X fits
        if osp.exists(smpl_path):

            smpl_verts_f_lst = []
            smpl_faces_f_lst = []

            smpl_verts_b_lst = []
            smpl_faces_b_lst = []

            for idx in range(N_body_f):

                smpl_obj_f = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_{idx:02d}.obj"
                smpl_mesh_f = trimesh.load(smpl_obj_f)
                smpl_verts_f = torch.tensor(smpl_mesh_f.vertices).to(device).float()
                smpl_faces_f = torch.tensor(smpl_mesh_f.faces).to(device).long()
                smpl_verts_f_lst.append(smpl_verts_f)
                smpl_faces_f_lst.append(smpl_faces_f)

            for idx in range(N_body_f):

                smpl_obj_b = f"{args.out_dir}/{cfg.name}/obj/{data2['name']}_smpl_{idx:02d}.obj"
                smpl_mesh_b = trimesh.load(smpl_obj_b)
                smpl_verts_b = torch.tensor(smpl_mesh_b.vertices).to(device).float()
                smpl_faces_b = torch.tensor(smpl_mesh_b.faces).to(device).long()
                smpl_verts_b_lst.append(smpl_verts_b)
                smpl_faces_b_lst.append(smpl_faces_b)

            batch_smpl_verts_f = torch.stack(smpl_verts_f_lst)
            batch_smpl_faces_f = torch.stack(smpl_faces_f_lst)

            batch_smpl_verts_b = torch.stack(smpl_verts_b_lst)
            batch_smpl_faces_b = torch.stack(smpl_faces_b_lst)

            # render optimized mesh as normal [-1,1]
            in_tensor_f["T_normal_F"], in_tensor_f["T_normal_B"] = dataset.render_normal(
                batch_smpl_verts_f, batch_smpl_faces_f
            )

            # render optimized mesh as normal [-1,1]
            in_tensor_b["T_normal_F"], in_tensor_b["T_normal_B"] = dataset.render_normal(
                batch_smpl_verts_b, batch_smpl_faces_b
            )

            with torch.no_grad():
                in_tensor_f["normal_F"], in_tensor_f["normal_B"] = normal_net.netG(
                    in_tensor_f)

            with torch.no_grad():
                in_tensor_b["normal_F"], in_tensor_b["normal_B"] = normal_net.netG(
                    in_tensor_b)

            in_tensor_f["smpl_verts"] = batch_smpl_verts_f * \
                torch.tensor([1., -1., 1.]).to(device)
            in_tensor_f["smpl_faces"] = batch_smpl_faces_f[:, :, [0, 2, 1]]

            in_tensor_b["smpl_verts"] = batch_smpl_verts_b * \
                torch.tensor([1., -1., 1.]).to(device)
            in_tensor_b["smpl_faces"] = batch_smpl_faces_b[:, :, [0, 2, 1]]

        else:
        """
        # smpl optimization
        loop_smpl = tqdm(range(args.loop_smpl))
        
        for i in loop_smpl:

            per_loop_lst = []

            optimizer_smpl_f.zero_grad()

            # ---------------------------------------------------------------
            # Derive back SMPL by rotating the front SMPL 
            # ---------------------------------------------------------------

            T_back_tensor = torch.tensor([
                [-1.00000000e+00, -3.22578062e-08,  1.23496643e-07, -6.84291201e-08],
                [ 3.22577886e-08, -1.00000000e+00, -1.42392824e-07,  6.84291228e-08],
                [ 1.23496647e-07, -1.42392820e-07,  1.00000000e+00, -4.00848333e-08],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
            ], dtype=torch.float32).to(device)
            
            R_back_tensor = T_back_tensor[:3, :3]  # Extract 3x3 rotation matrix

            R_back = torch.tensor([
                [-1.0,  0.0,  0.0],  # Keep X negative
                [ 0.0,  1.0,  0.0],  # Keep Y unchanged
                [ 0.0,  0.0, -1.0],  # Flip Z
            ], dtype=torch.float32).to(device)

            N_body_f, N_pose_f = optimed_pose_f.shape[:2]
            N_body_b, N_pose_b = optimed_pose_b.shape[:2]

            # 6d_rot to rot_mat for front
            optimed_orient_mat_f = rot6d_to_rotmat(optimed_orient_f.view(-1, 6)).view(N_body_f, 1, 3, 3)
            optimed_pose_mat_f = rot6d_to_rotmat(optimed_pose_f.view(-1, 6)).view(N_body_f, N_pose_f, 3, 3)
           
            optimed_orient_mat_b = rot6d_to_rotmat(optimed_orient_b.view(-1, 6)).view(N_body_b, 1, 3, 3)
            optimed_pose_mat_b = rot6d_to_rotmat(optimed_pose_b.view(-1, 6)).view(N_body_b, N_pose_b, 3, 3)
            # ---------------------------
            # Compute front SMPL outputs
            # ---------------------------
            smpl_verts_f, smpl_landmarks_f, smpl_joints_f = dataset.smpl_model(
                shape_params=optimed_betas_f,
                expression_params=tensor2variable(data["exp"], device),
                body_pose=optimed_pose_mat_f,
                global_pose=optimed_orient_mat_f,
                jaw_pose=tensor2variable(data["jaw_pose"], device),
                left_hand_pose=tensor2variable(data["left_hand_pose"], device),
                right_hand_pose=tensor2variable(data["right_hand_pose"], device),
            )

            # ---------------------------
            # Compute back SMPL outputs
            # ---------------------------
            smpl_verts_b, smpl_landmarks_b, smpl_joints_b = dataset.smpl_model(
                shape_params=optimed_betas_b,
                expression_params=tensor2variable(data2["exp"], device),
                body_pose=optimed_pose_mat_b,
                global_pose=optimed_orient_mat_b,
                jaw_pose=tensor2variable(data2["jaw_pose"], device),
                left_hand_pose=tensor2variable(data2["left_hand_pose"], device),
                right_hand_pose=tensor2variable(data2["right_hand_pose"], device),
            )

            # Apply translation and scaling for front
            smpl_verts_f = (smpl_verts_f + optimed_trans_f) * data["scale"]
            smpl_joints_f = (smpl_joints_f + optimed_trans_f) * data["scale"] * torch.tensor([1.0, 1.0, -1.0]).to(device)
            """
            # Apply translation and scaling for back
            smpl_verts_b = (smpl_verts_b + optimed_trans_b) * data["scale"]
            smpl_joints_b = (smpl_joints_b + optimed_trans_b) * data["scale"] * torch.tensor([1.0, 1.0, -1.0]).to(device)
            """
            smpl_verts_b = (torch.matmul(smpl_verts_f, R_back) + optimed_trans_b) * data2["scale"]
            smpl_joints_b = (torch.matmul(smpl_joints_f, R_back) + optimed_trans_b) * data2["scale"] * torch.tensor([1.0, 1.0, -1.0]).to(device)

            # ----------------------
            # Process landmarks and joints
            # ----------------------
            # For the front
            smpl_joints_3d_f = (smpl_joints_f[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0) * 0.5
            in_tensor_f["smpl_joint"] = smpl_joints_f[:, dataset.smpl_data.smpl_joint_ids_24_pixie, :]

            ghum_lmks_f = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
            ghum_conf_f = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
            smpl_lmks_f = smpl_joints_3d_f[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]

            # For the back (using rotated joints)
            smpl_joints_3d_b = (smpl_joints_b[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0) * 0.5
            in_tensor_b["smpl_joint"] = smpl_joints_b[:, dataset.smpl_data.smpl_joint_ids_24_pixie, :]

            ghum_lmks_b = data2["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
            ghum_conf_b = data2["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
            smpl_lmks_b = smpl_joints_3d_b[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]

            # --------------------------------------------
            # Render the optimized mesh normals for both views
            # --------------------------------------------
            in_tensor_f["T_normal_F"], in_tensor_f["T_normal_B"] = dataset.render_normal(
                smpl_verts_f * torch.tensor([1.0, -1.0, -1.0]).to(device),
                in_tensor_f["smpl_faces"],
            )

            in_tensor_b["T_normal_F"], in_tensor_b["T_normal_B"] = dataset.render_normal(
                smpl_verts_b * torch.tensor([1.0, -1.0, -1.0]).to(device),
                in_tensor_b["smpl_faces"],
            )

            T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

            # --------------------------------------------
            # Save SMPL meshes for visual inspection
            # --------------------------------------------
            def save_obj(obj_path, vertices, faces):
                with open(obj_path, "w") as f:
                    for v in vertices:
                        f.write("v {:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))
                    for face in faces:
                        # OBJ files are 1-indexed; adjust indices accordingly
                        f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))

            # paths for saving meshes
            smpl_path_f = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_front.obj"
            smpl_path_b = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_back.obj"

            # If processing batches, here we save only the first instance in the batch.
            save_obj(smpl_path_f,
                    smpl_verts_f[0].detach().cpu().numpy(),
                    in_tensor_f["smpl_faces"].detach().cpu().numpy())
            save_obj(smpl_path_b,
                    smpl_verts_b[0].detach().cpu().numpy(),
                    in_tensor_b["smpl_faces"].detach().cpu().numpy())
            

            face_vids = SMPLX_object.smplx_front_flame_vid
            face_verts = smpl_verts_f[:, face_vids, :]  # [B, N_face, 3]

            # Define facial joint indices (standard from SMPL-X)
            facial_joint_indices = [66, 67, 68, 69, 70, 71, 72, 73, 127]
            facial_kpt3d = data["smplx_kpt3d"][0, facial_joint_indices].detach().cpu().numpy().tolist()
            facial_kpt2d = data["smplx_kpt"][0, facial_joint_indices].detach().cpu().numpy().tolist()

            with torch.no_grad():
                rig_data_json = {
                    # Expression & pose
                    'expression_params': data["exp"].detach().cpu().numpy().tolist(),
                    'jaw_pose': data["jaw_pose"].detach().cpu().numpy().tolist(),
                    'head_pose': data["head_pose"].detach().cpu().numpy().tolist(),
                    'abs_head_pose': data["abs_head_pose"].detach().cpu().numpy().tolist(),
                    'neck_pose': data["neck_pose"].detach().cpu().numpy().tolist(),

                    # Geometry
                    'shape_params': data["shape"].detach().cpu().numpy().tolist(),
                    'face_vertex_ids': face_vids.tolist(),
                    'face_verts': face_verts[0].detach().cpu().numpy().tolist(),

                    # Joints and keypoints
                    'smplx_kpt3d': data["smplx_kpt3d"][0].detach().cpu().numpy().tolist(),
                    'smplx_kpt': data["smplx_kpt"][0].detach().cpu().numpy().tolist(),
                    'facial_joint_indices': facial_joint_indices,
                    'facial_kpt3d': facial_kpt3d,
                    'facial_kpt2d': facial_kpt2d,
                }

                out_dir_rig = osp.join(args.out_dir, cfg.name, "rig_params_json")
                os.makedirs(out_dir_rig, exist_ok=True)
                json_path = f"{out_dir_rig}/{data['name']}_face_rig.json"

                with open(json_path, 'w') as f:
                    json.dump(rig_data_json, f, indent=4)  # indent=4 for pretty-printing



            with torch.no_grad():
                # [1, 3, 512, 512], (-1.0, 1.0)
                in_tensor_f["normal_F"], in_tensor_f["normal_B"] = normal_net.netG(in_tensor_f)

            with torch.no_grad():
                # [1, 3, 512, 512], (-1.0, 1.0)
                in_tensor_b["normal_F"], in_tensor_b["normal_B"] = normal_net.netG(in_tensor_b)

            # Apply the transformation
            R_back_tensor = T_back_tensor[:3, :3] 
            in_tensor_b["normal_F"] = transform_normals(in_tensor_b["normal_F"], R_back_tensor)
            
            ##########################################################
            ###### front(T_normal_F )- back(normal_F)#################
            diff_F_smpl = torch.abs(
                in_tensor_f["T_normal_F"] - in_tensor_f["normal_F"])
            diff_B_smpl = torch.abs(
                in_tensor_b["T_normal_F"] - in_tensor_b["normal_F"])

            # silhouette loss
            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
            gt_arr = in_tensor_f["mask"].repeat(1, 1, 2)
            diff_S = torch.abs(smpl_arr - gt_arr)
            losses["silhouette"]["value"] = diff_S.mean()

            # large cloth_overlap --> big difference between body and cloth mask
            # for loose clothing, reply more on landmarks instead of silhouette+normal loss
            cloth_overlap = diff_S.sum(
                dim=[1, 2]) / gt_arr.sum(dim=[1, 2])
            cloth_overlap_flag = cloth_overlap > cfg.cloth_overlap_thres
            losses["joint"]["weight"] = [
                50.0 if flag else 5.0 for flag in cloth_overlap_flag]

            # small body_overlap --> large occlusion or out-of-frame
            # for highly occluded body, reply only on high-confidence landmarks, no silhouette+normal loss

            # BUG: PyTorch3D silhouette renderer generates dilated mask
            bg_value = in_tensor_f["T_normal_F"][0, 0, 0, 0]
            smpl_arr_fake = torch.cat([
                in_tensor_f["T_normal_F"][:, 0].ne(bg_value).float(),
                in_tensor_b["T_normal_F"][:, 0].ne(bg_value).float()
            ],
                dim=-1)

            body_overlap = (gt_arr * smpl_arr_fake.gt(0.0)
                            ).sum(dim=[1, 2]) / smpl_arr_fake.gt(0.0).sum(dim=[1, 2])
            body_overlap_mask = (gt_arr * smpl_arr_fake).unsqueeze(1)
            body_overlap_flag = body_overlap < cfg.body_overlap_thres

            losses["normal"]["value"] = (
                diff_F_smpl * body_overlap_mask[..., :512] +
                diff_B_smpl * body_overlap_mask[..., 512:]
            ).mean() / 2.0

            losses["silhouette"]["weight"] = [
                0 if flag else 1.0 for flag in body_overlap_flag]
            occluded_idx = torch.where(body_overlap_flag)[0]
            ghum_conf_f[occluded_idx] *= ghum_conf_f[occluded_idx] > 0.95
            losses["joint"]["value"] = (torch.norm(ghum_lmks_f - smpl_lmks_f, dim=2) *
                                        ghum_conf_f).mean(dim=1)

            # Weighted sum of the losses
            smpl_loss = 0.0
            pbar_desc = "Body Fitting -- "
            for k in ["normal", "silhouette", "joint"]:
                per_loop_loss = (
                    losses[k]["value"] *
                    torch.tensor(losses[k]["weight"]).to(device)
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
                    in_tensor_f["image"],
                    in_tensor_f["T_normal_F"],
                    in_tensor_f["normal_F"],
                    diff_S[:, :, :512].unsqueeze(1).repeat(1, 3, 1, 1),
                ])
                per_loop_lst.extend([
                    in_tensor_b["image"],
                    in_tensor_b["T_normal_F"],
                    in_tensor_b["normal_F"],
                    diff_S[:, :, 512:].unsqueeze(1).repeat(1, 3, 1, 1),
                ])
                per_data_lst.append(
                    get_optim_grid_image(per_loop_lst, None, nrow=N_body_f * 2, type="smpl")
                )

            smpl_loss.backward()
            optimizer_smpl_f.step()
            scheduler_smpl_f.step(smpl_loss)

        in_tensor_f["smpl_verts"] = smpl_verts_f * torch.tensor([1.0, 1.0, -1.0]).to(device)
        in_tensor_f["smpl_faces"] = in_tensor_f["smpl_faces"][:, :, [0, 2, 1]]

        if not args.novis:
            per_data_lst[-1].save(
                osp.join(args.out_dir, cfg.name, f"png/{data['name']}_smpl.png")
            )

        if not args.novis:
            img_crop_path = osp.join(
                args.out_dir, cfg.name, "png", f"{data['name']}_crop.png")
            torchvision.utils.save_image(
                torch.cat([
                    data["img_crop"][:, :3], (in_tensor_f['normal_F'].detach(
                    ).cpu() + 1.0) * 0.5,
                    (in_tensor_b['normal_F'].detach().cpu() + 1.0) * 0.5
                ],
                    dim=3), img_crop_path
            )

            rgb_norm_F = blend_rgb_norm(in_tensor_f["normal_F"], data)
            rgb_norm_B = blend_rgb_norm(in_tensor_b["normal_F"], data2)

            img_overlap_path = osp.join(
                args.out_dir, cfg.name, f"png/{data['name']}_overlap.png")
            torchvision.utils.save_image(
                torch.cat([data["img_raw"], rgb_norm_F,
                          rgb_norm_B], dim=-1) / 255.,
                img_overlap_path
            )

        smpl_obj_lst = []

        for idx in range(N_body_f):

            smpl_obj_f = trimesh.Trimesh(
                in_tensor_f["smpl_verts"].detach().cpu()[idx] * torch.tensor([1.0, -1.0, 1.0]),
                in_tensor_f["smpl_faces"].detach().cpu()[0][:, [0, 2, 1]],
                process=False,
                maintains_order=True,
            )

            smpl_obj_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_{idx:02d}.obj"

            if not osp.exists(smpl_obj_path):
                smpl_obj_f.export(smpl_obj_path)
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
            smpl_obj_lst.append(smpl_obj_f)

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

        in_tensor_f["depth_F"], in_tensor_b["depth_F"] = dataset.render_depth(
            batch_smpl_verts_f, batch_smpl_faces_f
        )
        per_loop_lst = []

        in_tensor_f["BNI_verts"] = []
        in_tensor_f["BNI_faces"] = []
        in_tensor_f["body_verts"] = []
        in_tensor_f["body_faces"] = []

        for idx in range(N_body_f):

            final_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full.obj"

            side_mesh = smpl_obj_lst[idx].copy()
            face_mesh = smpl_obj_lst[idx].copy()
            hand_mesh = smpl_obj_lst[idx].copy()
            smplx_mesh = smpl_obj_lst[idx].copy()

            """
            img_norm_path_f = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_front_normal.png")
            torchvision.utils.save_image((in_tensor_f['normal_F'].detach().cpu()),img_norm_path_f)
            img_norm_path_b = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_back_normal.png")
            torchvision.utils.save_image((in_tensor_b['normal_F'].detach().cpu()),img_norm_path_b)
            img_depth_path_f = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_front_depth.png")
            torchvision.utils.save_image((in_tensor_f['depth_F'].detach().cpu()),img_depth_path_f)
            img_depth_path_b = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_back_depth.png")
            torchvision.utils.save_image((in_tensor_b['depth_F'].detach().cpu()),img_depth_path_b)
            """

            img_crop_path = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_normal_and_depth.png")
            row1 = torch.cat([data["img_crop"][:, :3], data2["img_crop"][:, :3]], dim=3)  # Concatenate front and back images
            row2 = torch.cat([(in_tensor_f['normal_F'].detach().cpu() + 1.0) * 0.5, 
                            (in_tensor_b['normal_F'].detach().cpu() + 1.0) * 0.5], dim=3)  # Concatenate normal front and back
            row3 = torch.cat([in_tensor_f["depth_F"].unsqueeze(1).repeat(1, 3, 1, 1).detach().cpu(),
                            in_tensor_b["depth_F"].unsqueeze(1).repeat(1, 3, 1, 1).detach().cpu()], dim=3)  # Concatenate mask and mask_back

            # save the normal and depth images
            torchvision.utils.save_image(
                torch.cat([row1, row2, row3], dim=2) / 255., img_crop_path
            )

            # save normals, depths and masks
            BNI_dict= save_normal_tensor_upt(
                in_tensor_f,
                in_tensor_b,
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

            # BNI_object.extract_surface_single(False)
            BNI_object.extract_surface(False)

            in_tensor_f["body_verts"].append(torch.tensor(smpl_obj_lst[idx].vertices).float())
            in_tensor_f["body_faces"].append(torch.tensor(smpl_obj_lst[idx].faces).long())

            # requires shape completion when low overlap
            # replace SMPL by completed mesh as side_mesh

            if cfg.bni.use_ifnet:

                side_mesh_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_IF.obj"

                side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)

                # mesh completion via IF-net
                in_tensor_f.update(
                    dataset.depth_to_voxel({
                        "depth_F": BNI_object.F_depth.unsqueeze(0), 
                        "depth_B": BNI_object.B_depth.unsqueeze(0)
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
            per_data_lst[-1].save(osp.join(args.out_dir, cfg.name, f"png/{data['name']}_cloth.png"))

            # for video rendering
            in_tensor_f["BNI_verts"].append(torch.tensor(final_mesh.vertices).float())
            in_tensor_f["BNI_faces"].append(torch.tensor(final_mesh.faces).long())

            os.makedirs(osp.join(args.out_dir, cfg.name, "vid"), exist_ok=True)
            in_tensor_f["uncrop_param"] = data["uncrop_param"]
            in_tensor_f["img_raw"] = data["img_raw"]
            torch.save(
                in_tensor_f, osp.join(args.out_dir, cfg.name, f"vid/{data['name']}_in_tensor.pt")
            )