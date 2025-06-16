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

from apps.HGFilters import HGFilter
from lib.net.net_util import init_net
from lib.net.voxelize import Voxelization
from lib.dataset.mesh_util import feat_select, read_smpl_constants
from lib.net.NormalNet import NormalNet
from lib.net.MLP import MLP
from lib.net.spatial import SpatialEncoder
from lib.dataset.PointFeat import PointFeat
from lib.dataset.mesh_util import SMPLX
from apps.VE import VolumeEncoder
from termcolor import colored
from lib.net.BasePIFuNet import BasePIFuNet
import torch.nn as nn
import torch
import torch.nn.functional as F


class HGPIFuNet(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """
    def __init__(self, cfg, projection_mode="orthogonal", error_term=nn.MSELoss()):

        super(HGPIFuNet, self).__init__(projection_mode=projection_mode, error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()
        self.opt = cfg.net
        self.root = cfg.root
        self.overfit = cfg.overfit

        channels_IF = self.opt.mlp_dim

        self.use_filter = self.opt.use_filter
        self.prior_type = self.opt.prior_type
        self.smpl_feats = self.opt.smpl_feats

        self.smpl_dim = self.opt.smpl_dim
        self.voxel_dim = self.opt.voxel_dim
        self.hourglass_dim = self.opt.hourglass_dim

        self.in_geo = [item[0] for item in self.opt.in_geo]
        self.in_nml = [item[0] for item in self.opt.in_nml]

        self.in_geo_dim = sum([item[1] for item in self.opt.in_geo])
        self.in_nml_dim = sum([item[1] for item in self.opt.in_nml])

        self.in_total = self.in_geo + self.in_nml
        self.smpl_feat_dict = None
        self.smplx_data = SMPLX()

        image_lst = [0, 1, 2]
        normal_F_lst = [0, 1, 2] if "image" not in self.in_geo else [3, 4, 5]
        normal_B_lst = [3, 4, 5] if "image" not in self.in_geo else [6, 7, 8]

        # only ICON or ICON-Keypoint use visibility

        if self.prior_type in ["icon", "keypoint"]:
            if "image" in self.in_geo:
                self.channels_filter = [
                    image_lst + normal_F_lst,
                    image_lst + normal_B_lst,
                ]
            else:
                self.channels_filter = [normal_F_lst, normal_B_lst]

        else:
            if "image" in self.in_geo:
                self.channels_filter = [image_lst + normal_F_lst + normal_B_lst]
            else:
                self.channels_filter = [normal_F_lst + normal_B_lst]

        use_vis = (self.prior_type in ["icon", "keypoint"]) and ("vis" in self.smpl_feats)
        if self.prior_type in ["pamir", "pifu"]:
            use_vis = 1

        if self.use_filter:
            channels_IF[0] = (self.hourglass_dim) * (2 - use_vis)
        else:
            channels_IF[0] = len(self.channels_filter[0]) * (2 - use_vis)

        if self.prior_type in ["icon", "keypoint"]:
            channels_IF[0] += self.smpl_dim
        elif self.prior_type == "pamir":
            channels_IF[0] += self.voxel_dim
            (
                smpl_vertex_code,
                smpl_face_code,
                smpl_faces,
                smpl_tetras,
            ) = read_smpl_constants(self.smplx_data.tedra_dir)
            self.voxelization = Voxelization(
                smpl_vertex_code,
                smpl_face_code,
                smpl_faces,
                smpl_tetras,
                volume_res=128,
                sigma=0.05,
                smooth_kernel_size=7,
                batch_size=cfg.batch_size,
                device=torch.device(f"cuda:{cfg.gpus[0]}"),
            )
            self.ve = VolumeEncoder(3, self.voxel_dim, self.opt.num_stack)

        elif self.prior_type == "pifu":
            channels_IF[0] += 1
        else:
            print(f"don't support {self.prior_type}!")

        self.base_keys = ["smpl_verts", "smpl_faces"]

        self.icon_keys = self.base_keys + [f"smpl_{feat_name}" for feat_name in self.smpl_feats]
        self.keypoint_keys = self.base_keys + [f"smpl_{feat_name}" for feat_name in self.smpl_feats]

        self.pamir_keys = ["voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"]
        self.pifu_keys = []

        self.if_regressor = MLP(
            filter_channels=channels_IF,
            name="if",
            res_layers=self.opt.res_layers,
            norm=self.opt.norm_mlp,
            last_op=nn.Sigmoid() if not cfg.test_mode else None,
        )

        self.sp_encoder = SpatialEncoder()

        # network
        if self.use_filter:
            if self.opt.gtype == "HGPIFuNet":
                self.F_filter = HGFilter(self.opt, self.opt.num_stack, len(self.channels_filter[0]))
            else:
                print(colored(f"Backbone {self.opt.gtype} is unimplemented", "green"))

        summary_log = (
            f"{self.prior_type.upper()}:\n" + f"w/ Global Image Encoder: {self.use_filter}\n" +
            f"Image Features used by MLP: {self.in_geo}\n"
        )

        if self.prior_type == "icon":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (ICON): {self.smpl_dim}\n"
        elif self.prior_type == "keypoint":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (Keypoint): {self.smpl_dim}\n"
        elif self.prior_type == "pamir":
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PaMIR): {self.voxel_dim}\n"
        else:
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PIFu): 1 (z-value)\n"

        summary_log += f"Dim of MLP's first layer: {channels_IF[0]}\n"

        print(colored(summary_log, "yellow"))

        self.normal_filter = NormalNet(cfg)

        init_net(self)

    def get_normal(self, in_tensor_dict):
        """
        Fuse multi-view normal maps using masks as confidence weights.
        Supports: in_tensor_dict["normal_list"], in_tensor_dict["mask_list"]
        """

        if not self.training and not self.overfit:
            with torch.no_grad():
                feat_lst = []

                if "image" in self.in_geo:
                    feat_lst.append(in_tensor_dict["image"])  # Optional RGB input

                # Multi-view input
                normals = in_tensor_dict.get("normal_list", [])
                masks = in_tensor_dict.get("mask_list", None)

                filtered = []
                weights = []

                for i, nml in enumerate(normals):
                    if nml.dim() == 3:
                        nml = nml.unsqueeze(0)  # [1,3,H,W]

                    # Use identity input as both F/B since we're using single-view maps
                    nmlF, _ = self.normal_filter({"normal_F": nml, "normal_B": nml})
                    filtered.append(nmlF)

                    if masks is not None:
                        msk = masks[i]
                        if msk.dim() == 2:
                            msk = msk.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                        elif msk.dim() == 3:
                            msk = msk.unsqueeze(0)  # [1,1,H,W]
                        weights.append(msk.expand_as(nmlF))  # match normal shape

                if len(filtered) == 0:
                    raise ValueError("No normal maps found in 'normal_list'.")

                filtered = torch.stack(filtered)  # [N, 1, 3, H, W]
                if masks is not None:
                    weights = torch.stack(weights)  # [N, 1, 3, H, W] broadcasted
                    weights = weights + 1e-5  # prevent division by 0
                    fused = (filtered * weights).sum(dim=0) / weights.sum(dim=0)
                else:
                    fused = filtered.mean(dim=0)  # fallback to uniform average

                feat_lst.append(fused)  # fused is [1,3,H,W]

            in_filter = torch.cat(feat_lst, dim=1)

        else:
            in_filter = torch.cat([in_tensor_dict[key] for key in self.in_geo], dim=1)

        return in_filter



    def get_mask(self, in_filter, size=128):

        mask = (
            F.interpolate(
                in_filter[:, self.channels_filter[0]],
                size=(size, size),
                mode="bilinear",
                align_corners=True,
            ).abs().sum(dim=1, keepdim=True) != 0.0
        )

        return mask

    def filter(self, in_tensor_dict, return_inter=False):
        """
        Filter the input images.
        This version assumes normal maps are pre-computed and provided in in_tensor_dict.
        """
        
        # --- Build the input feature tensor for the hourglass filter ---
        feat_lst = []
        if "image" in self.in_geo:
            feat_lst.append(in_tensor_dict["image"])
        
        # Directly use the provided normal maps
        if "normal_F" in self.in_geo and "normal_B" in self.in_geo:
            if "normal_F" not in in_tensor_dict or "normal_B" not in in_tensor_dict:
                 raise ValueError("'normal_F' or 'normal_B' not found in input tensor for filter.")
            feat_lst.append(in_tensor_dict["normal_F"])
            feat_lst.append(in_tensor_dict["normal_B"])

        in_filter = torch.cat(feat_lst, dim=1)
        
        # The rest of the function remains the same as ICON's original
        features_G = []

        if self.prior_type in ["icon", "keypoint"]:
            if self.use_filter:
                features_F = self.F_filter(
                    in_filter[:, self.channels_filter[0]]
                )
                features_B = self.F_filter(
                    in_filter[:, self.channels_filter[1]]
                )
            else:
                features_F = [in_filter[:, self.channels_filter[0]]]
                features_B = [in_filter[:, self.channels_filter[1]]]
            for idx in range(len(features_F)):
                features_G.append(torch.cat([features_F[idx], features_B[idx]], dim=1))
        else:
            if self.use_filter:
                features_G = self.F_filter(in_filter[:, self.channels_filter[0]])
            else:
                features_G = [in_filter[:, self.channels_filter[0]]]

        self.smpl_feat_dict = {
            k: in_tensor_dict[k] if k in in_tensor_dict.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        }

        if not self.training:
            features_out = [features_G[-1]]
        else:
            features_out = features_G

        if return_inter:
            return features_out, in_filter
        else:
            return features_out

    def query(self, features, points, calibs, transforms=None, regressor=None):

        xyz = self.projection(points, calibs, transforms)

        (xy, z) = xyz.split([2, 1], dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=True).detach().float()

        preds_list = []
        vol_feats = features

        if self.prior_type in ["icon", "keypoint"]:

            # smpl_verts [B, N_vert, 3]
            # smpl_faces [B, N_face, 3]
            # xyz [B, 3, N]  --> points [B, N, 3]

            point_feat_extractor = PointFeat(
                self.smpl_feat_dict["smpl_verts"], self.smpl_feat_dict["smpl_faces"]
            )

            point_feat_out = point_feat_extractor.query(
                xyz.permute(0, 2, 1).contiguous(), self.smpl_feat_dict
            )

            feat_lst = [
                point_feat_out[key] for key in self.smpl_feats if key in point_feat_out.keys()
            ]
            smpl_feat = torch.cat(feat_lst, dim=2).permute(0, 2, 1)

            if self.prior_type == "keypoint":
                kpt_feat = self.sp_encoder.forward(
                    cxyz=xyz.permute(0, 2, 1).contiguous(),
                    kptxyz=self.smpl_feat_dict["smpl_joint"],
                )

        elif self.prior_type == "pamir":

            voxel_verts = self.smpl_feat_dict["voxel_verts"][:, :-self.
                                                             smpl_feat_dict["pad_v_num"][0], :]
            voxel_faces = self.smpl_feat_dict["voxel_faces"][:, :-self.
                                                             smpl_feat_dict["pad_f_num"][0], :]

            self.voxelization.update_param(
                batch_size=voxel_faces.shape[0],
                smpl_tetra=voxel_faces[0].detach().cpu().numpy(),
            )
            vol = self.voxelization(voxel_verts)    # vol ~ [0,1]
            vol_feats = self.ve(vol, intermediate_output=self.training)

        for im_feat, vol_feat in zip(features, vol_feats):

            # normal feature choice by smpl_vis

            if self.prior_type == "icon":
                if "vis" in self.smpl_feats:
                    point_local_feat = feat_select(self.index(im_feat, xy), smpl_feat[:, [-1], :])
                    point_feat_list = [point_local_feat, smpl_feat[:, :-1, :]]
                else:
                    point_local_feat = self.index(im_feat, xy)
                    point_feat_list = [point_local_feat, smpl_feat[:, :, :]]

            if self.prior_type == "keypoint":

                if "vis" in self.smpl_feats:
                    point_local_feat = feat_select(self.index(im_feat, xy), smpl_feat[:, [-1], :])
                    point_feat_list = [point_local_feat, kpt_feat, smpl_feat[:, :-1, :]]
                else:
                    point_local_feat = self.index(im_feat, xy)
                    point_feat_list = [point_local_feat, kpt_feat, smpl_feat[:, :, :]]

            elif self.prior_type == "pamir":

                # im_feat [B, hg_dim, 128, 128]
                # vol_feat [B, vol_dim, 32, 32, 32]

                point_feat_list = [self.index(im_feat, xy), self.index(vol_feat, xyz)]

            elif self.prior_type == "pifu":
                point_feat_list = [self.index(im_feat, xy), z]

            point_feat = torch.cat(point_feat_list, 1)

            # out of image plane is always set to 0
            preds = regressor(point_feat)
            preds = in_cube * preds

            preds_list.append(preds)

        return preds_list

    def get_error(self, preds_if_list, labels):
        """calcaulate error

        Args:
            preds_list (list): list of torch.tensor(B, 3, N)
            labels (torch.tensor): (B, N_knn, N)

        Returns:
            torch.tensor: error
        """
        error_if = 0

        for pred_id in range(len(preds_if_list)):
            pred_if = preds_if_list[pred_id]
            error_if += self.error_term(pred_if, labels)

        error_if /= len(preds_if_list)

        return error_if

    def forward(self, in_tensor_dict):
        """
        sample_tensor [B, 3, N]
        calib_tensor [B, 4, 4]
        label_tensor [B, 1, N]
        smpl_feat_tensor [B, 59, N]
        """

        sample_tensor = in_tensor_dict["sample"]
        calib_tensor = in_tensor_dict["calib"]
        label_tensor = in_tensor_dict["label"]

        in_feat = self.filter(in_tensor_dict)

        preds_if_list = self.query(
            in_feat, sample_tensor, calib_tensor, regressor=self.if_regressor
        )

        error = self.get_error(preds_if_list, label_tensor)

        return preds_if_list[-1], error

    # Add this new method inside your HGPIFuNet class in apps/HGPIFuNet.py

    def query_multiview(self, features_list, points, calibs_list, transforms=None, regressor=None):
        """
        Queries features from multiple views, projects points for each view, and aggregates.
        This function is designed to be called by the reconstruction engine.
        
        Args:
            features_list (list): A list of feature maps, one for each view.
            points (torch.Tensor): [B, N, 3] query points in world space.
            calibs_list (list): A list of [B, 4, 4] calibration matrices.
            regressor (nn.Module): The MLP to predict SDF.

        Returns:
            torch.Tensor: The final SDF prediction.
        """
        
        all_view_features = []
        num_views = len(features_list)

        # Geometric prior (SMPL) features are calculated once in canonical space
        if self.prior_type in ["icon", "keypoint"]:
            point_feat_extractor = PointFeat(self.smpl_feat_dict["smpl_verts"], self.smpl_feat_dict["smpl_faces"])
            point_feat_out = point_feat_extractor.query(points.contiguous(), self.smpl_feat_dict)
            feat_lst = [point_feat_out[key] for key in self.smpl_feats if key in point_feat_out.keys()]
            smpl_feat = torch.cat(feat_lst, dim=2)

        # Loop through each view to sample image features
        for i in range(num_views):
            # Get the features and calibration for the current view
            im_feat = features_list[i][-1] # Use the last (highest-res) feature map
            calib = calibs_list[i]
            
            # Project world points into the current view's camera space
            xyz = self.projection(points.permute(0, 2, 1), calib, transforms)
            xy, z = xyz.split([2, 1], dim=1)

            # Sample image features from this view's feature map
            point_local_feat = self.index(im_feat, xy) # Shape: [B, C_img, N]
            
            # Combine image features with geometric prior
            if self.prior_type == "icon":
                if 'vis' in self.smpl_feats:
                    point_local_feat_vis = feat_select(point_local_feat, smpl_feat[..., -1:].permute(0, 2, 1))
                    point_feat_list = [point_local_feat_vis, smpl_feat[..., :-1].permute(0, 2, 1)]
                else:
                    point_feat_list = [point_local_feat, smpl_feat.permute(0, 2, 1)]
                view_feature = torch.cat(point_feat_list, dim=1)
                all_view_features.append(view_feature)

        # Aggregate features from all views
        if not all_view_features:
            raise ValueError("No features were sampled from any view.")
            
        stacked_features = torch.stack(all_view_features, dim=0)
        aggregated_features = torch.mean(stacked_features, dim=0)
        
        # Make the final prediction
        preds = regressor(aggregated_features)
        
        # Apply the in-cube mask
        in_cube = (points > -1.0) & (points < 1.0)
        in_cube = in_cube.all(dim=2).detach().float().unsqueeze(1)
        
        return preds * in_cube