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

import glob
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFile
from termcolor import colored
from torchvision import transforms
from torchvision.models import detection

from lib.common.config import cfg
from lib.common.imutils import process_image
from lib.common.render import Render
from lib.common.train_util import Format
from lib.dataset.mesh_util import SMPLX, get_visibility
from lib.pixielib.models.SMPLX import SMPLX as PIXIE_SMPLX
from lib.pixielib.pixie import PIXIE
from lib.pixielib.utils.config import cfg as pixie_cfg
from lib.pymafx.core import path_config
from lib.pymafx.models import pymaf_net

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestDataset:
    def __init__(self, cfg, device):
        self.image_dir = cfg["image_dir"]
        self.image_b_dir = cfg["image_b_dir"]
        self.seg_dir = cfg["seg_dir"]
        self.use_seg = cfg["use_seg"]
        self.hps_type = cfg["hps_type"]
        self.smpl_type = "smplx"
        self.smpl_gender = "neutral"
        self.vol_res = cfg["vol_res"]
        self.single = cfg["single"]

        self.device = device

        # Define valid image formats
        img_fmts = ["jpg", "png", "jpeg", "JPG", "bmp", "exr"]

        # Gather front and back images
        keep_lst_a = sorted(glob.glob(f"{self.image_dir}/*"))
        keep_lst_b = sorted(glob.glob(f"{self.image_b_dir}/*"))

        # Filter files by valid image formats
        valid_images_a = [item for item in keep_lst_a if item.split(".")[-1] in img_fmts]
        valid_images_b = [item for item in keep_lst_b if item.split(".")[-1] in img_fmts]

        # Pair front and back images based on sorting
        self.subject_list = [
            {"front": front, "back": back}
            for front, back in zip(sorted(valid_images_a), sorted(valid_images_b))
        ]

        # SMPL related setup
        self.smpl_data = SMPLX()

        if self.hps_type == "pymafx":
            self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
            self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)["model"], strict=True)
            self.hps.eval()
            pixie_cfg.merge_from_list(["model.n_shape", 10, "model.n_exp", 10])
        elif self.hps_type == "pixie":
            self.hps = PIXIE(config=pixie_cfg, device=self.device)

        self.smpl_model = PIXIE_SMPLX(pixie_cfg.model).to(self.device)

        self.detector = detection.maskrcnn_resnet50_fpn(
            weights=detection.MaskRCNN_ResNet50_FPN_V2_Weights
        )
        self.detector.eval()

        self.render = Render(size=512, device=self.device)

    def __len__(self):
        return len(self.subject_list)

    def compute_vis_cmap(self, smpl_verts, smpl_faces):

        (xy, z) = torch.as_tensor(smpl_verts).split([2, 1], dim=-1)
        smpl_vis = get_visibility(xy, z,
                                  torch.as_tensor(smpl_faces).long()[:, :,
                                                                     [0, 2, 1]]).unsqueeze(-1)
        smpl_cmap = self.smpl_data.cmap_smpl_vids(self.smpl_type).unsqueeze(0)

        return {
            "smpl_vis": smpl_vis.to(self.device),
            "smpl_cmap": smpl_cmap.to(self.device),
            "smpl_verts": smpl_verts,
        }

    def depth_to_voxel(self, data_dict):

        data_dict["depth_F"] = transforms.Resize(self.vol_res)(data_dict["depth_F"])
        data_dict["depth_B"] = transforms.Resize(self.vol_res)(data_dict["depth_B"])

        depth_mask = (~torch.isnan(data_dict['depth_F']))
        depth_FB = torch.cat([data_dict['depth_F'], data_dict['depth_B']], dim=0)
        depth_FB[:, ~depth_mask[0]] = 0.

        # Important: index_long = depth_value - 1
        index_z = (((depth_FB + 1.) * 0.5 * self.vol_res) - 1).clip(0, self.vol_res -
                                                                    1).permute(1, 2, 0)
        index_z_ceil = torch.ceil(index_z).long()
        index_z_floor = torch.floor(index_z).long()
        index_z_frac = torch.frac(index_z)

        index_mask = index_z[..., 0] == torch.tensor(self.vol_res * 0.5 - 1).long()
        voxels = F.one_hot(index_z_ceil[..., 0], self.vol_res) * index_z_frac[..., 0] + \
            F.one_hot(index_z_floor[..., 0], self.vol_res) * (1.0-index_z_frac[..., 0]) + \
            F.one_hot(index_z_ceil[..., 1], self.vol_res) * index_z_frac[..., 1]+ \
            F.one_hot(index_z_floor[..., 1], self.vol_res) * (1.0 - index_z_frac[..., 1])

        voxels[index_mask] *= 0
        voxels = torch.flip(voxels, [2]).permute(2, 0, 1).float()    #[x-2, y-0, z-1]

        return {
            "depth_voxels": voxels.flip([
                0,
            ]).unsqueeze(0).to(self.device),
        }

    def __getitem__(self, index):
        # Get paths for the front and back images
        img_paths = self.subject_list[index]
        front_img_path = img_paths["front"]
        back_img_path = img_paths["back"]

        # Process the front image
        front_img_name = front_img_path.split("/")[-1].rsplit(".", 1)[0]
        front_arr_dict = process_image(front_img_path, self.hps_type, self.single, 512, self.detector)
        front_arr_dict.update({"name": front_img_name})

        # Process the back image
        back_img_name = back_img_path.split("/")[-1].rsplit(".", 1)[0]
        back_arr_dict = process_image(back_img_path, self.hps_type, self.single, 512, self.detector)
        back_arr_dict.update({"name": back_img_name})

        # Perform HPS inference for both front and back images
        with torch.no_grad():
            # Front image processing
            if self.hps_type == "pixie":
                front_preds_dict = self.hps.forward(front_arr_dict["img_hps"].to(self.device))
                front_arr_dict.update(front_preds_dict)
                front_arr_dict["global_orient"] = front_preds_dict["global_pose"]
                front_arr_dict["betas"] = front_preds_dict["shape"]
                front_arr_dict["smpl_verts"] = front_preds_dict["vertices"]
                scale, tranX, tranY = front_preds_dict["cam"].split(1, dim=1)
            elif self.hps_type == "pymafx":
                front_batch = {k: v.to(self.device) for k, v in front_arr_dict["img_pymafx"].items()}
                front_preds_dict, _ = self.hps.forward(front_batch)
                front_output = front_preds_dict["mesh_out"][-1]
                front_arr_dict.update({
                    "betas": front_output["pred_shape"],
                    "body_pose": front_output["rotmat"][:, 1:22],
                    "global_orient": front_output["rotmat"][:, 0:1],
                    "smpl_verts": front_output["smplx_verts"],
                    "left_hand_pose": front_output["pred_lhand_rotmat"],
                    "right_hand_pose": front_output["pred_rhand_rotmat"],
                    "jaw_pose": front_output["pred_face_rotmat"][:, 0:1],
                    "exp": front_output["pred_exp"],
                })
                scale, tranX, tranY = front_output["pred_cam"].split(1, dim=1)

            front_arr_dict["scale"] = scale.unsqueeze(1)
            front_arr_dict["trans"] = (
                torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                        dim=1).unsqueeze(1).to(self.device).float()
            )

            # Back image processing
            if self.hps_type == "pixie":
                back_preds_dict = self.hps.forward(back_arr_dict["img_hps"].to(self.device))
                back_arr_dict.update(back_preds_dict)
                back_arr_dict["global_orient"] = back_preds_dict["global_pose"]
                back_arr_dict["betas"] = back_preds_dict["shape"]
                back_arr_dict["smpl_verts"] = back_preds_dict["vertices"]
                scale, tranX, tranY = back_preds_dict["cam"].split(1, dim=1)
            elif self.hps_type == "pymafx":
                back_batch = {k: v.to(self.device) for k, v in back_arr_dict["img_pymafx"].items()}
                back_preds_dict, _ = self.hps.forward(back_batch)
                back_output = back_preds_dict["mesh_out"][-1]
                back_arr_dict.update({
                    "betas": back_output["pred_shape"],
                    "body_pose": back_output["rotmat"][:, 1:22],
                    "global_orient": back_output["rotmat"][:, 0:1],
                    "smpl_verts": back_output["smplx_verts"],
                    "left_hand_pose": back_output["pred_lhand_rotmat"],
                    "right_hand_pose": back_output["pred_rhand_rotmat"],
                    "jaw_pose": back_output["pred_face_rotmat"][:, 0:1],
                    "exp": back_output["pred_exp"],
                })
                scale, tranX, tranY = back_output["pred_cam"].split(1, dim=1)

            back_arr_dict["scale"] = scale.unsqueeze(1)
            back_arr_dict["trans"] = (
                torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                        dim=1).unsqueeze(1).to(self.device).float()
            )

        # Add shared SMPL-X data for both images
        smpl_faces = (
            torch.as_tensor(self.smpl_data.smplx_faces.astype(np.int64))
            .unsqueeze(0)
            .long()
            .to(self.device)
        )
        for arr_dict in (front_arr_dict, back_arr_dict):
            arr_dict["smpl_faces"] = smpl_faces
            arr_dict["type"] = self.smpl_type

            # Convert rotation matrices to 6D for optimization
            N_body, N_pose = arr_dict["body_pose"].shape[:2]
            arr_dict["body_pose"] = arr_dict["body_pose"][:, :, :, :2].reshape(N_body, N_pose, -1)
            arr_dict["global_orient"] = arr_dict["global_orient"][:, :, :, :2].reshape(N_body, 1, -1)

        return front_arr_dict, back_arr_dict


    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="rgb")

    def render_depth(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="depth")