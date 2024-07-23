
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

        self.front_image_dir = cfg["front_image_dir"]
        self.back_image_dir = cfg["back_image_dir"]
        self.seg_dir = cfg["seg_dir"]
        self.use_seg = cfg["use_seg"]
        self.hps_type = cfg["hps_type"]
        self.smpl_type = "smplx"
        self.smpl_gender = "neutral"
        self.vol_res = cfg["vol_res"]
        self.single = cfg["single"]

        self.device = device

        front_img_list = sorted(glob.glob(f"{self.front_image_dir}/*"))
        back_img_list = sorted(glob.glob(f"{self.back_image_dir}/*"))
        img_fmts = ["jpg", "png", "jpeg", "JPG", "bmp", "exr"]

        self.front_subject_list = sorted([item for item in front_img_list if item.split(".")[-1] in img_fmts],
                                         reverse=False)
        self.back_subject_list = sorted([item for item in back_img_list if item.split(".")[-1] in img_fmts],
                                        reverse=False)

        assert len(self.front_subject_list) == len(
            self.back_subject_list), "The number of front and back images must be the same."

        # smpl related
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

        print(
            colored(
                f"SMPL-X estimate with {Format.start} {self.hps_type.upper()} {Format.end}", "green"
            )
        )

        self.render = Render(size=512, device=self.device)

    def __len__(self):
        return len(self.front_subject_list)

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
                 F.one_hot(index_z_floor[..., 0], self.vol_res) * (1.0 - index_z_frac[..., 0]) + \
                 F.one_hot(index_z_ceil[..., 1], self.vol_res) * index_z_frac[..., 1] + \
                 F.one_hot(index_z_floor[..., 1], self.vol_res) * (1.0 - index_z_frac[..., 1])

        voxels[index_mask] *= 0
        voxels = torch.flip(voxels, [2]).permute(2, 0, 1).float()  # [x-2, y-0, z-1]

        return {
            "depth_voxels": voxels.flip([
                0,
            ]).unsqueeze(0).to(self.device),
        }

    def __getitem__(self, index):
        front_img_path = self.front_subject_list[index]
        front_img_name = front_img_path.split("/")[-1].rsplit(".", 1)[0]
        back_img_path = self.back_subject_list[index]
        back_img_name = back_img_path.split("/")[-1].rsplit(".", 1)[0]

        front_arr_dict = process_image(front_img_path, self.hps_type, self.single, 512, self.detector)
        back_arr_dict = process_image(back_img_path, self.hps_type, self.single, 512, self.detector)

        front_arr_dict.update({"name": front_img_name})
        back_arr_dict.update({"name": back_img_name})

        with torch.no_grad():
            if self.hps_type == "pixie":
                front_preds_dict = self.hps.forward(front_arr_dict["img_hps"].to(self.device))
                back_preds_dict = self.hps.forward(back_arr_dict["img_hps"].to(self.device))
            elif self.hps_type == 'pymafx':
                front_batch = {k: v.to(self.device) for k, v in front_arr_dict["img_pymafx"].items()}
                back_batch = {k: v.to(self.device) for k, v in back_arr_dict["img_pymafx"].items()}
                front_preds_dict, _ = self.hps.forward(front_batch)
                back_preds_dict, _ = self.hps.forward(back_batch)

        front_arr_dict["smpl_faces"] = (
            torch.as_tensor(self.smpl_data.smplx_faces.astype(np.int64)).unsqueeze(0).long().to(self.device)
        )
        back_arr_dict["smpl_faces"] = (
            torch.as_tensor(self.smpl_data.smplx_faces.astype(np.int64)).unsqueeze(0).long().to(self.device)
        )
        front_arr_dict["type"] = self.smpl_type
        back_arr_dict["type"] = self.smpl_type

        if self.hps_type == "pymafx":
            front_output = front_preds_dict["mesh_out"][-1]
            back_output = back_preds_dict["mesh_out"][-1]
            scale, tranX, tranY = front_output["pred_cam"].split(1, dim=1)
            
            front_arr_dict["betas"] = front_output["pred_shape"]  # 10
            front_arr_dict["body_pose"] = front_output["rotmat"][:, 1:22]
            front_arr_dict["global_orient"] = front_output["rotmat"][:, 0:1]
            front_arr_dict["smpl_verts"] = front_output["smplx_verts"]
            front_arr_dict["left_hand_pose"] = front_output["pred_lhand_rotmat"]
            front_arr_dict["right_hand_pose"] = front_output["pred_rhand_rotmat"]
            front_arr_dict['jaw_pose'] = front_output['pred_face_rotmat'][:, 0:1]
            front_arr_dict["exp"] = front_output["pred_exp"]

            # back_arr_dict["betas"] = back_output["pred_shape"]  # 10
            back_arr_dict["betas"] = front_output["pred_shape"]  # 10
            back_arr_dict["body_pose"] = front_output["rotmat"][:, 1:22]
            back_arr_dict["global_orient"] = front_output["rotmat"][:, 0:1]
            back_arr_dict["smpl_verts"] = front_output["smplx_verts"]
            back_arr_dict["left_hand_pose"] = front_output["pred_lhand_rotmat"]
            back_arr_dict["right_hand_pose"] = front_output["pred_rhand_rotmat"]
            back_arr_dict['jaw_pose'] = front_output['pred_face_rotmat'][:, 0:1]
            back_arr_dict["exp"] = front_output["pred_exp"]

        elif self.hps_type == "pixie":
            front_arr_dict.update(front_preds_dict)
            back_arr_dict.update(front_preds_dict)
            front_arr_dict["global_orient"] = front_preds_dict["global_pose"]
            front_arr_dict["betas"] = front_preds_dict["shape"]  # 200
            front_arr_dict["smpl_verts"] = front_preds_dict["vertices"]
            scale, tranX, tranY = front_preds_dict["cam"].split(1, dim=1)

            back_arr_dict["global_orient"] = front_preds_dict["global_pose"]
            back_arr_dict["betas"] = front_preds_dict["shape"]  # 200
            back_arr_dict["smpl_verts"] = front_preds_dict["vertices"]
            back_scale, back_tranX, back_tranY = front_preds_dict["cam"].split(1, dim=1)

        front_arr_dict["scale"] = scale.unsqueeze(1)
        front_arr_dict["trans"] = (
            torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                      dim=1).unsqueeze(1).to(self.device).float()
        )
        back_arr_dict["scale"] = back_scale.unsqueeze(1)
        back_arr_dict["trans"] = (
            torch.cat([back_tranX, back_tranY, torch.zeros_like(back_tranX)],
                      dim=1).unsqueeze(1).to(self.device).float()
        )

        # from rot_mat to rot_6d for better optimization
        N_body, N_pose = front_arr_dict["body_pose"].shape[:2]
        front_arr_dict["body_pose"] = front_arr_dict["body_pose"][:, :, :, :2].reshape(N_body, N_pose, -1)
        front_arr_dict["global_orient"] = front_arr_dict["global_orient"][:, :, :, :2].reshape(N_body, 1, -1)

        N_body, N_pose = back_arr_dict["body_pose"].shape[:2]
        back_arr_dict["body_pose"] = back_arr_dict["body_pose"][:, :, :, :2].reshape(N_body, N_pose, -1)
        back_arr_dict["global_orient"] = back_arr_dict["global_orient"][:, :, :, :2].reshape(N_body, 1, -1)

        # Debugging output
        print(front_arr_dict.keys)
        print(back_arr_dict.keys)
        print(f"Type of front_arr_dict: {type(front_arr_dict)}")
        print(f"Type of back_arr_dict: {type(back_arr_dict)}")

        return front_arr_dict, back_arr_dict


    def render_normal(self, verts, faces, view='front'):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="rgb")

    def render_depth(self, verts, faces, view='front'):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="depth")

