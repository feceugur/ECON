#  -*- coding: utf-8 -*-

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
import os
import os.path as osp
import pickle
from collections import namedtuple
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logging.getLogger("smplx").setLevel(logging.ERROR)

from .lbs import find_dynamic_lmk_idx_and_bcoords, lbs, vertices2landmarks
from .utils import (
    Array,
    FLAMEOutput,
    MANOOutput,
    SMPLHOutput,
    SMPLOutput,
    SMPLXOutput,
    Struct,
    Tensor,
    find_joint_kin_chain,
    to_np,
    to_tensor,
)
from .vertex_ids import vertex_ids as VERTEX_IDS
from .vertex_joint_selector import VertexJointSelector

ModelOutput = namedtuple(
    "ModelOutput",
    [
        "vertices",
        "joints",
        "full_pose",
        "betas",
        "global_orient",
        "body_pose",
        "expression",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
    ],
)
ModelOutput.__new__.__defaults__ = (None, ) * len(ModelOutput._fields)


class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
        self,
        model_path: str,
        kid_template_path: str = "",
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        num_betas: int = 10,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        gender: str = "neutral",
        age: str = "adult",
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        v_personal: Optional[Union[Tensor, Array]] = None,
        **kwargs,
    ) -> None:
        """SMPL model constructor

        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        num_betas: int, optional
            Number of shape components to use
            (default = 10).
        create_betas: bool, optional
            Flag for creating a member variable for the shape space
            (default = True).
        betas: torch.tensor, optional, Bx10
            The default value for the shape member variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        joint_mapper: object, optional
            An object that re-maps the joints. Useful if one wants to
            re-order the SMPL joints to some other convention (e.g. MSCOCO)
            (default = None)
        gender: str, optional
            Which gender to load
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """

        self.gender = gender
        self.age = age

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = "SMPL_{}.{ext}".format(gender.upper(), ext="pkl")
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), "Path {} does not exist!".format(smpl_path)

            with open(smpl_path, "rb") as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file, encoding="latin1"))

        super(SMPL, self).__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if shapedirs.shape[-1] < self.SHAPE_SPACE_DIM:
            # print(f'WARNING: You are using a {self.name()} model, with only'
            #       ' 10 shape coefficients.')
            num_betas = min(num_betas, 10)
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        if self.age == "kid":
            v_template_smil = np.load(kid_template_path)
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(v_template_smil - data_struct.v_template, axis=2)
            shapedirs = np.concatenate((shapedirs[:, :, :num_betas], v_template_diff), axis=2)
            num_betas = num_betas + 1

        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer("shapedirs", to_tensor(to_np(shapedirs), dtype=dtype))

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS["smplh"]

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long),
        )

        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.num_betas], dtype=dtype)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter("betas", nn.Parameter(default_betas, requires_grad=True))

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(global_orient, dtype=dtype)

            global_orient = nn.Parameter(default_global_orient, requires_grad=True)
            self.register_parameter("global_orient", global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros([batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose, dtype=dtype)
            self.register_parameter(
                "body_pose", nn.Parameter(default_body_pose, requires_grad=True)
            )

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3], dtype=dtype, requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter("transl", nn.Parameter(default_transl, requires_grad=True))

        if v_template is None:
            v_template = data_struct.v_template

        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)

        if v_personal is not None:
            v_personal = to_tensor(to_np(v_personal), dtype=dtype)
            v_template += v_personal

        # The vertices of the template model
        self.register_buffer("v_template", v_template)

        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)
        self.register_buffer("J_regressor", j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        self.register_buffer("lbs_weights", to_tensor(to_np(data_struct.weights), dtype=dtype))

    @property
    def num_betas(self):
        return self._num_betas

    @property
    def num_expression_coeffs(self):
        return 0

    def create_mean_pose(self, data_struct) -> Tensor:
        pass

    def name(self) -> str:
        return "SMPL"

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self) -> int:
        return self.v_template.shape[0]

    def get_num_faces(self) -> int:
        return self.faces.shape[0]

    def extra_repr(self) -> str:
        msg = [
            f"Gender: {self.gender.upper()}",
            f"Number of joints: {self.J_regressor.shape[0]}",
            f"Betas: {self.num_betas}",
        ]
        return "\n".join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLOutput:
        """Forward pass for the SMPL model

        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3
            If given, ignore the member variable and use it as the global
            rotation of the body. Useful if someone wishes to predicts this
            with an external model. (default=None)
        betas: torch.tensor, optional, shape BxN_b
            If given, ignore the member variable `betas` and use it
            instead. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_verts: bool, optional
            Return the vertices. (default=True)
        return_full_pose: bool, optional
            Returns the full axis-angle pose vector (default=False)

        Returns
        -------
        """
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None and hasattr(self, "transl"):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
        )

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(
            vertices=vertices if return_verts else None,
            global_orient=global_orient,
            body_pose=body_pose,
            joints=joints,
            betas=betas,
            full_pose=full_pose if return_full_pose else None,
        )

        return output


class SMPLLayer(SMPL):
    def __init__(self, *args, **kwargs) -> None:
        # Just create a SMPL module without any member variables
        super(SMPLLayer, self).__init__(
            create_body_pose=False,
            create_betas=False,
            create_global_orient=False,
            create_transl=False,
            *args,
            **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLOutput:
        """Forward pass for the SMPL model

        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3x3
            Global rotation of the body.  Useful if someone wishes to
            predicts this with an external model. It is expected to be in
            rotation matrix format.  (default=None)
        betas: torch.tensor, optional, shape BxN_b
            Shape parameters. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape BxJx3x3
            Body pose. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            rotation matrix format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            Translation vector of the body.
            For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_verts: bool, optional
            Return the vertices. (default=True)
        return_full_pose: bool, optional
            Returns the full axis-angle pose vector (default=False)

        Returns
        -------
        """
        model_vars = [betas, global_orient, body_pose, transl]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            global_orient = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if body_pose is None:
            body_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3,
                                            3).expand(batch_size, self.NUM_BODY_JOINTS, -1,
                                                      -1).contiguous()
            )
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        full_pose = torch.cat(
            [
                global_orient.reshape(-1, 1, 3, 3),
                body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
            ],
            dim=1,
        )

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(
            vertices=vertices if return_verts else None,
            global_orient=global_orient,
            body_pose=body_pose,
            joints=joints,
            betas=betas,
            full_pose=full_pose if return_full_pose else None,
        )

        return output


class SMPLH(SMPL):

    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = SMPL.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS

    def __init__(
        self,
        model_path,
        kid_template_path: str = "",
        data_struct: Optional[Struct] = None,
        create_left_hand_pose: bool = True,
        left_hand_pose: Optional[Tensor] = None,
        create_right_hand_pose: bool = True,
        right_hand_pose: Optional[Tensor] = None,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        batch_size: int = 1,
        gender: str = "neutral",
        age: str = "adult",
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed: bool = True,
        ext: str = "pkl",
        **kwargs,
    ) -> None:
        """SMPLH model constructor

        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_left_hand_pose: bool, optional
            Flag for creating a member variable for the pose of the left
            hand. (default = True)
        left_hand_pose: torch.tensor, optional, BxP
            The default value for the left hand pose member variable.
            (default = None)
        create_right_hand_pose: bool, optional
            Flag for creating a member variable for the pose of the right
            hand. (default = True)
        right_hand_pose: torch.tensor, optional, BxP
            The default value for the right hand pose member variable.
            (default = None)
        num_pca_comps: int, optional
            The number of PCA components to use for each hand.
            (default = 6)
        flat_hand_mean: bool, optional
            If False, then the pose of the hand is initialized to False.
        batch_size: int, optional
            The batch size used for creating the member variables
        gender: str, optional
            Which gender to load
        dtype: torch.dtype, optional
            The data type for the created variables
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """

        self.num_pca_comps = num_pca_comps
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = "SMPLH_{}.{ext}".format(gender.upper(), ext=ext)
                smplh_path = os.path.join(model_path, model_fn)
            else:
                smplh_path = model_path
            assert osp.exists(smplh_path), "Path {} does not exist!".format(smplh_path)

            if ext == "pkl":
                with open(smplh_path, "rb") as smplh_file:
                    model_data = pickle.load(smplh_file, encoding="latin1")
            elif ext == "npz":
                model_data = np.load(smplh_path, allow_pickle=True)
            else:
                raise ValueError("Unknown extension: {}".format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS["smplh"]

        super(SMPLH, self).__init__(
            model_path=model_path,
            kid_template_path=kid_template_path,
            data_struct=data_struct,
            batch_size=batch_size,
            vertex_ids=vertex_ids,
            gender=gender,
            age=age,
            use_compressed=use_compressed,
            dtype=dtype,
            ext=ext,
            **kwargs,
        )

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean

        left_hand_components = data_struct.hands_componentsl[:num_pca_comps]
        right_hand_components = data_struct.hands_componentsr[:num_pca_comps]

        self.np_left_hand_components = left_hand_components
        self.np_right_hand_components = right_hand_components
        if self.use_pca:
            self.register_buffer(
                "left_hand_components", torch.tensor(left_hand_components, dtype=dtype)
            )
            self.register_buffer(
                "right_hand_components",
                torch.tensor(right_hand_components, dtype=dtype),
            )

        if self.flat_hand_mean:
            left_hand_mean = np.zeros_like(data_struct.hands_meanl)
        else:
            left_hand_mean = data_struct.hands_meanl

        if self.flat_hand_mean:
            right_hand_mean = np.zeros_like(data_struct.hands_meanr)
        else:
            right_hand_mean = data_struct.hands_meanr

        self.register_buffer("left_hand_mean", to_tensor(left_hand_mean, dtype=self.dtype))
        self.register_buffer("right_hand_mean", to_tensor(right_hand_mean, dtype=self.dtype))

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_left_hand_pose:
            if left_hand_pose is None:
                default_lhand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype)
            else:
                default_lhand_pose = torch.tensor(left_hand_pose, dtype=dtype)

            left_hand_pose_param = nn.Parameter(default_lhand_pose, requires_grad=True)
            self.register_parameter("left_hand_pose", left_hand_pose_param)

        if create_right_hand_pose:
            if right_hand_pose is None:
                default_rhand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype)
            else:
                default_rhand_pose = torch.tensor(right_hand_pose, dtype=dtype)

            right_hand_pose_param = nn.Parameter(default_rhand_pose, requires_grad=True)
            self.register_parameter("right_hand_pose", right_hand_pose_param)

        # Create the buffer for the mean pose.
        pose_mean_tensor = self.create_mean_pose(data_struct, flat_hand_mean=flat_hand_mean)
        if not torch.is_tensor(pose_mean_tensor):
            pose_mean_tensor = torch.tensor(pose_mean_tensor, dtype=dtype)
        self.register_buffer("pose_mean", pose_mean_tensor)

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3], dtype=self.dtype)

        pose_mean = torch.cat(
            [
                global_orient_mean,
                body_pose_mean,
                self.left_hand_mean,
                self.right_hand_mean,
            ],
            dim=0,
        )
        return pose_mean

    def name(self) -> str:
        return "SMPL+H"

    def extra_repr(self):
        msg = super(SMPLH, self).extra_repr()
        msg = [msg]
        if self.use_pca:
            msg.append(f"Number of PCA components: {self.num_pca_comps}")
        msg.append(f"Flat hand mean: {self.flat_hand_mean}")
        return "\n".join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLHOutput:
        """"""

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas
        left_hand_pose = (left_hand_pose if left_hand_pose is not None else self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else self.right_hand_pose)

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None:
            if hasattr(self, "transl"):
                transl = self.transl

        if self.use_pca:
            left_hand_pose = torch.einsum("bi,ij->bj", [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                "bi,ij->bj", [right_hand_pose, self.right_hand_components]
            )

        full_pose = torch.cat([global_orient, body_pose, left_hand_pose, right_hand_pose], dim=1)

        full_pose += self.pose_mean

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
        )

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLHOutput(
            vertices=vertices if return_verts else None,
            joints=joints,
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            full_pose=full_pose if return_full_pose else None,
        )

        return output


class SMPLHLayer(SMPLH):
    def __init__(self, *args, **kwargs) -> None:
        """SMPL+H as a layer model constructor"""
        super(SMPLHLayer, self).__init__(
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_betas=False,
            create_transl=False,
            *args,
            **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLHOutput:
        """Forward pass for the SMPL+H model

        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3x3
            Global rotation of the body. Useful if someone wishes to
            predicts this with an external model. It is expected to be in
            rotation matrix format. (default=None)
        betas: torch.tensor, optional, shape BxN_b
            Shape parameters. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape BxJx3x3
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            rotation matrix format. (default=None)
        left_hand_pose: torch.tensor, optional, shape Bx15x3x3
            If given, contains the pose of the left hand.
            It should be a tensor that contains joint rotations in
            rotation matrix format. (default=None)
        right_hand_pose: torch.tensor, optional, shape Bx15x3x3
            If given, contains the pose of the right hand.
            It should be a tensor that contains joint rotations in
            rotation matrix format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            Translation vector of the body.
            For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_verts: bool, optional
            Return the vertices. (default=True)
        return_full_pose: bool, optional
            Returns the full axis-angle pose vector (default=False)

        Returns
        -------
        """
        model_vars = [
            betas,
            global_orient,
            body_pose,
            transl,
            left_hand_pose,
            right_hand_pose,
        ]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            global_orient = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if body_pose is None:
            body_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 21, -1, -1).contiguous()
            )
        if left_hand_pose is None:
            left_hand_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
            )
        if right_hand_pose is None:
            right_hand_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
            )
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [
                global_orient.reshape(-1, 1, 3, 3),
                body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
                left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
                right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
            ],
            dim=1,
        )

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLHOutput(
            vertices=vertices if return_verts else None,
            joints=joints,
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            full_pose=full_pose if return_full_pose else None,
        )

        return output

# Assume necessary imports like torch, nn, Tensor, Optional, np, os, osp, pickle, etc.
# Assume SMPLH, lbs, find_dynamic_lmk_idx_and_bcoords, vertices2landmarks,
# Struct, to_tensor, to_np, find_joint_kin_chain, VERTEX_IDS, SMPLXOutput
# are defined elsewhere in the file or imported correctly.

class SMPLX(SMPLH):
    """
    SMPL-X (SMPL eXpressive) is a unified body model, with shape parameters
    trained jointly for the face, hands and body.
    SMPL-X uses standard vertex based linear blend skinning with learned
    corrective blend shapes, has N=10475 vertices and K=54 joints,
    which includes joints for the neck, jaw, eyeballs and fingers.
    """

    NUM_BODY_JOINTS = SMPLH.NUM_BODY_JOINTS    # 21
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3 # Jaw, left eye, right eye
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS # 21 + 30 + 3 = 54
    EXPRESSION_SPACE_DIM = 100 # Max possible expression coeffs
    NECK_IDX = 12
    # Assuming NUM_LANDMARKS might be needed by vertices2landmarks or joint concatenation logic
    # If landmarks are concatenated, the final joint count will be NUM_JOINTS + NUM_LANDMARKS
    # Let's define a standard number, though it might depend on the specific landmark set used.
    NUM_LANDMARKS = 68 # Standard face landmarks, adjust if different

    def __init__(
        self,
        model_path: str,
        kid_template_path: str = "",
        num_expression_coeffs: int = 10,
        create_expression: bool = True,
        expression: Optional[Tensor] = None,
        create_jaw_pose: bool = True,
        jaw_pose: Optional[Tensor] = None,
        create_leye_pose: bool = True,
        leye_pose: Optional[Tensor] = None,
        create_reye_pose=True,
        reye_pose: Optional[Tensor] = None,
        use_face_contour: bool = False,
        batch_size: int = 1,
        gender: str = "neutral",
        age: str = "adult",
        dtype=torch.float32,
        ext: str = "npz",
        **kwargs,
    ) -> None:
        """SMPLX model constructor (parameters as before)"""

        # Load the model
        if osp.isdir(model_path):
            model_fn = "SMPLX_{}.{ext}".format(gender.upper(), ext=ext)
            smplx_path = os.path.join(model_path, model_fn)
        else:
            smplx_path = model_path
        assert osp.exists(smplx_path), "Path {} does not exist!".format(smplx_path)

        if ext == "pkl":
            with open(smplx_path, "rb") as smplx_file:
                model_data = pickle.load(smplx_file, encoding="latin1")
        elif ext == "npz":
            model_data = np.load(smplx_path, allow_pickle=True)
        else:
            raise ValueError("Unknown extension: {}".format(ext))

        # Compatibility step for potentially sparse J_regressor
        if 'J_regressor' in model_data and not isinstance(model_data['J_regressor'], np.ndarray):
            model_data['J_regressor'] = model_data['J_regressor'].toarray()

        data_struct = Struct(**model_data)

        # Pass necessary kwargs to SMPLH/base class
        # Filter kwargs relevant to the base class constructor to avoid unexpected argument errors
        # (Assuming base class signature is known or handled appropriately)
        base_kwargs = {k: v for k, v in kwargs.items() if k in [
             'use_pca', 'num_pca_comps', 'num_betas', 'flat_hand_mean',
             'vertex_ids', 'joint_mapper', 'vertex_joint_selector', # Add others if needed by SMPLH
             'data_struct' # Pass data_struct if SMPLH needs it directly
             ]}
        # Ensure required args for SMPLH are present, potentially adding defaults if not in kwargs
        base_kwargs.setdefault('vertex_ids', VERTEX_IDS.get("smplx")) # Use .get for safety

        super(SMPLX, self).__init__(
            model_path=model_path, # Pass along for potential use in base class
            kid_template_path=kid_template_path,
            data_struct=data_struct, # Pass the loaded data struct
            dtype=dtype,
            batch_size=batch_size, # Pass batch_size
            gender=gender,
            age=age,
            ext=ext,
            **base_kwargs, # Pass filtered kwargs
        )

        # SMPL-X specific initializations
        # Ensure data from struct is converted correctly
        lmk_faces_idx = data_struct.lmk_faces_idx
        self.register_buffer("lmk_faces_idx", torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = data_struct.lmk_bary_coords
        self.register_buffer("lmk_bary_coords", torch.tensor(lmk_bary_coords, dtype=dtype))

        self.use_face_contour = use_face_contour
        if self.use_face_contour:
            # Ensure these exist in the model file if use_face_contour is True
            if not hasattr(data_struct, 'dynamic_lmk_faces_idx') or not hasattr(data_struct, 'dynamic_lmk_bary_coords'):
                 raise ValueError("use_face_contour=True but dynamic landmark data not found in model file.")

            dynamic_lmk_faces_idx = np.array(data_struct.dynamic_lmk_faces_idx)
            dynamic_lmk_faces_idx = torch.tensor(dynamic_lmk_faces_idx, dtype=torch.long)
            self.register_buffer("dynamic_lmk_faces_idx", dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = data_struct.dynamic_lmk_bary_coords
            dynamic_lmk_bary_coords = torch.tensor(np.array(dynamic_lmk_bary_coords), dtype=dtype)
            self.register_buffer("dynamic_lmk_bary_coords", dynamic_lmk_bary_coords)

            # Ensure parents buffer is already registered by base class or here
            if not hasattr(self, 'parents'):
                 # This should typically be handled by the base class init
                 self.register_buffer('parents', torch.tensor(data_struct.kintree_table[0], dtype=torch.long))

            neck_kin_chain = find_joint_kin_chain(self.NECK_IDX, self.parents)
            self.register_buffer("neck_kin_chain", torch.tensor(neck_kin_chain, dtype=torch.long))

        # Create parameters for face poses if requested
        if create_jaw_pose:
            default_jaw_pose = torch.zeros([batch_size, 3], dtype=dtype) if jaw_pose is None else torch.tensor(jaw_pose, dtype=dtype)
            self.register_parameter("jaw_pose", nn.Parameter(default_jaw_pose, requires_grad=True))

        if create_leye_pose:
            default_leye_pose = torch.zeros([batch_size, 3], dtype=dtype) if leye_pose is None else torch.tensor(leye_pose, dtype=dtype)
            self.register_parameter("leye_pose", nn.Parameter(default_leye_pose, requires_grad=True))

        if create_reye_pose:
            default_reye_pose = torch.zeros([batch_size, 3], dtype=dtype) if reye_pose is None else torch.tensor(reye_pose, dtype=dtype)
            self.register_parameter("reye_pose", nn.Parameter(default_reye_pose, requires_grad=True))

        # Handle expression blend shapes from shapedirs
        # Assumes self.shapedirs (shape part) is already registered by base class
        shapedirs_full = to_tensor(data_struct.shapedirs, dtype=dtype) # Load the full shapedirs
        # Ensure shapedirs is 3D [num_vertices, 3, num_shapes]
        if len(shapedirs_full.shape) < 3:
             shapedirs_full = shapedirs_full[:, :, None]

        total_shape_dims = shapedirs_full.shape[-1]
        # Assumes self.SHAPE_SPACE_DIM is correctly defined (e.g., 10 or 300)
        available_expr_dims = total_shape_dims - self.SHAPE_SPACE_DIM
        if available_expr_dims < 0:
             print(f"Warning: Model shapedirs have {total_shape_dims} modes, expected at least {self.SHAPE_SPACE_DIM} for shape.")
             available_expr_dims = 0

        # Determine the number of expression coefficients to actually use
        if num_expression_coeffs > self.EXPRESSION_SPACE_DIM:
             print(f"Warning: Requested {num_expression_coeffs} expression coeffs, exceeding max {self.EXPRESSION_SPACE_DIM}. Clamping.")
             num_expression_coeffs = self.EXPRESSION_SPACE_DIM

        if available_expr_dims < num_expression_coeffs:
            print(f"Warning: Requested {num_expression_coeffs} expression coeffs, but model only provides {available_expr_dims} beyond shape space. Using {available_expr_dims}.")
            self._num_expression_coeffs = available_expr_dims
        else:
            self._num_expression_coeffs = num_expression_coeffs

        # Register the expression blend shapes buffer if there are any to use
        if self._num_expression_coeffs > 0:
            expr_start_idx = self.SHAPE_SPACE_DIM
            expr_end_idx = self.SHAPE_SPACE_DIM + self._num_expression_coeffs
            expr_dirs = shapedirs_full[:, :, expr_start_idx:expr_end_idx]
            self.register_buffer("expr_dirs", expr_dirs) # Already a tensor
        else:
            # Register an empty buffer if no expression blend shapes are used/available
            # Ensure v_template exists from base class init
            if not hasattr(self, 'v_template'):
                 self.register_buffer('v_template', to_tensor(data_struct.v_template, dtype=dtype))
            self.register_buffer("expr_dirs", torch.empty((self.v_template.shape[0], 3, 0), dtype=dtype))


        # Create expression parameter if requested and possible
        if create_expression and self._num_expression_coeffs > 0:
            if expression is None:
                default_expression = torch.zeros([batch_size, self.num_expression_coeffs], dtype=dtype)
            else:
                # Ensure provided expression matches the number of coeffs being used
                expression = torch.tensor(expression, dtype=dtype)
                if expression.shape[-1] != self.num_expression_coeffs:
                     # Adjust shape if possible, or raise error
                     if expression.shape[-1] > self.num_expression_coeffs:
                         print(f"Warning: Provided expression has {expression.shape[-1]} coeffs, using first {self.num_expression_coeffs}.")
                         expression = expression[:, :self.num_expression_coeffs]
                     else: # expression.shape[-1] < self.num_expression_coeffs
                         raise ValueError(f"Provided expression shape {expression.shape} has fewer coeffs than requested {self.num_expression_coeffs}")
                default_expression = expression
            self.register_parameter("expression", nn.Parameter(default_expression, requires_grad=True))
        elif create_expression and self._num_expression_coeffs == 0:
             print("Warning: create_expression=True but num_expression_coeffs is 0. No expression parameter created.")


    def name(self) -> str:
        return "SMPL-X"

    @property
    def num_expression_coeffs(self):
        # Returns the *actual* number of expression coefficients being used
        return self._num_expression_coeffs

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Overwrite base class mean pose to include face joints
        # Ensure base class has run its __init__ first if it sets hand means
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3], dtype=self.dtype)
        jaw_pose_mean = torch.zeros([3], dtype=self.dtype)
        leye_pose_mean = torch.zeros([3], dtype=self.dtype)
        reye_pose_mean = torch.zeros([3], dtype=self.dtype)

        # Use hand means potentially set by base class __init__
        left_hand_mean = self.left_hand_mean if hasattr(self, 'left_hand_mean') else np.zeros(self.NUM_HAND_JOINTS * 3)
        right_hand_mean = self.right_hand_mean if hasattr(self, 'right_hand_mean') else np.zeros(self.NUM_HAND_JOINTS * 3)

        pose_mean = np.concatenate(
            [
                to_np(global_orient_mean),
                to_np(body_pose_mean),
                to_np(jaw_pose_mean),
                to_np(leye_pose_mean),
                to_np(reye_pose_mean),
                left_hand_mean,
                right_hand_mean,
            ],
            axis=0,
        )
        return pose_mean

    def extra_repr(self):
        # Assumes base class extra_repr exists
        msg_list = [super(SMPLX, self).extra_repr()]
        msg_list.append(f"Number of Expression Coefficients: {self.num_expression_coeffs}")
        if hasattr(self, 'expression'):
             msg_list.append(f"Expression Parameter Created: True")
        if hasattr(self, 'jaw_pose'):
             msg_list.append(f"Jaw Pose Parameter Created: True")
        if hasattr(self, 'leye_pose'):
             msg_list.append(f"Left Eye Pose Parameter Created: True")
        if hasattr(self, 'reye_pose'):
             msg_list.append(f"Right Eye Pose Parameter Created: True")
        msg_list.append(f"Using Face Contour Landmarks: {self.use_face_contour}")
        return "\n".join(msg_list)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        return_joint_transformation: bool = False, # Added back
        return_vertex_transformation: bool = False, # Added back
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLXOutput:
        """
        Forward pass for the SMPLX model (parameters mostly as before)

        Adds:
            return_joint_transformation: bool, optional
                If True, returns the transformations of the joints
                (default=False)
            return_vertex_transformation: bool, optional
                If True, returns the transformations of the vertices
                (default=False)
        """
        # Get device and dtype from a registered buffer (e.g., v_template)
        # Assumes v_template is registered by the base class or this class's init
        device = self.v_template.device
        dtype = self.v_template.dtype

        # --- Parameter Handling & Batch Size Determination ---
        # Store provided arguments
        provided_args = {
            'betas': betas, 'global_orient': global_orient, 'body_pose': body_pose,
            'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
            'transl': transl, 'expression': expression, 'jaw_pose': jaw_pose,
            'leye_pose': leye_pose, 'reye_pose': reye_pose
        }

        # Determine batch size robustly
        batch_size = -1
        inferred_from = None
        for name, arg in provided_args.items():
            if arg is not None:
                current_bs = arg.shape[0]
                if batch_size == -1:
                    batch_size = current_bs
                    inferred_from = name
                elif batch_size != current_bs:
                    raise ValueError(f"Inconsistent batch sizes found: {name} has {current_bs}, but {inferred_from} has {batch_size}")

        # If no inputs provided, try to infer from registered parameters
        if batch_size == -1:
             # Check parameters that are likely to exist and be batched
             param_check_order = ['betas', 'global_orient', 'body_pose', 'expression', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'transl']
             for name in param_check_order:
                 if hasattr(self, name):
                     param = getattr(self, name)
                     # Check if it's a Parameter or Tensor and has a batch dimension
                     if isinstance(param, (nn.Parameter, Tensor)) and param.dim() > 0:
                         batch_size = param.shape[0]
                         inferred_from = f"parameter self.{name}"
                         break
             if batch_size == -1: # Fallback if no parameters exist or are suitable
                 batch_size = 1 # Default to 1 if nothing else is available
                 inferred_from = "default"
        # print(f"Inferred batch size: {batch_size} from {inferred_from}") # Optional debug print

        # Helper to get parameters
        def get_param(name, default_shape):
            passed_value = provided_args.get(name)
            if passed_value is not None:
                # Ensure correct batch size if passed
                if passed_value.shape[0] != batch_size:
                     # This should ideally be caught earlier, but double-check
                     raise ValueError(f"Input {name} has wrong batch size {passed_value.shape[0]}, expected {batch_size}")
                return passed_value.to(device=device, dtype=dtype) # Ensure device/dtype
            elif hasattr(self, name):
                param = getattr(self, name)
                # Ensure param is on the correct device/dtype
                param = param.to(device=device, dtype=dtype)
                # Expand parameter batch dim if needed (e.g., if model batch_size=1 but input implies larger batch)
                if param.shape[0] == 1 and batch_size > 1:
                    param = param.expand(batch_size, *param.shape[1:])
                elif param.shape[0] != batch_size:
                     # This might happen if model was created with batch_size > 1 and input implies batch_size = 1
                     # Or if model batch size doesn't match inferred batch size from other inputs
                     print(f"Warning: Parameter self.{name} batch size {param.shape[0]} doesn't match inferred batch size {batch_size}. Using parameter's first entry expanded.")
                     # Take first slice along batch dim and expand
                     param = param[0:1].expand(batch_size, *param.shape[1:])
                return param
            else:
                # Create default zero tensor only if parameter doesn't exist
                # print(f"Creating default tensor for {name}") # Optional debug print
                return torch.zeros(default_shape, dtype=dtype, device=device)

        # --- Get All Parameters ---
        # Shape and Expression
        # Assumes self.num_betas is defined (likely by base class)
        betas = get_param("betas", (batch_size, self.num_betas))
        # Handle expression only if it's supposed to exist
        if self.num_expression_coeffs > 0:
            expression = get_param("expression", (batch_size, self.num_expression_coeffs))
            # Clamp expression values if needed (optional, depends on use case)
            # expression = torch.clamp(expression, -expression_range, expression_range)
        else:
            # Create a zero tensor placeholder if no expressions are used
            expression = torch.zeros((batch_size, 0), dtype=dtype, device=device)

        # Pose Parameters
        global_orient = get_param("global_orient", (batch_size, 3))
        body_pose = get_param("body_pose", (batch_size, self.NUM_BODY_JOINTS * 3))
        jaw_pose = get_param("jaw_pose", (batch_size, 3))
        leye_pose = get_param("leye_pose", (batch_size, 3))
        reye_pose = get_param("reye_pose", (batch_size, 3))

        # Hand Poses (Handle PCA)
        # Assumes self.use_pca and self.num_pca_comps are defined (likely by base class)
        hand_pose_dim = self.num_pca_comps if self.use_pca else self.NUM_HAND_JOINTS * 3
        left_hand_pose_input = get_param("left_hand_pose", (batch_size, hand_pose_dim))
        right_hand_pose_input = get_param("right_hand_pose", (batch_size, hand_pose_dim))

        # Apply PCA transformation if needed (assuming input is PCA coeffs)
        if self.use_pca:
            # Assumes self.left_hand_components and self.right_hand_components are registered buffers
            # Ensure components are on the correct device/dtype
            left_hand_components = self.left_hand_components.to(device=device, dtype=dtype)
            right_hand_components = self.right_hand_components.to(device=device, dtype=dtype)
            # PCA coeffs to axis-angle
            left_hand_pose = torch.einsum('bi,ij->bj', left_hand_pose_input, left_hand_components)
            right_hand_pose = torch.einsum('bi,ij->bj', right_hand_pose_input, right_hand_components)
            # Add hand mean pose if necessary (often PCA is relative to mean)
            # Check if base class handles mean pose addition or if it should be done here
            # if hasattr(self, 'left_hand_mean_tensor'): # Example check
            #     left_hand_pose = left_hand_pose + self.left_hand_mean_tensor.to(device=device, dtype=dtype)
            #     right_hand_pose = right_hand_pose + self.right_hand_mean_tensor.to(device=device, dtype=dtype)
        else:
            left_hand_pose = left_hand_pose_input
            right_hand_pose = right_hand_pose_input

        # Translation
        transl = get_param("transl", (batch_size, 3))
        # Determine if translation should be applied based on input or existence of self.transl
        apply_trans = provided_args['transl'] is not None or hasattr(self, "transl")

        # --- Prepare for LBS ---
        # Concatenate all pose parameters
        full_pose = torch.cat([
            global_orient, body_pose, jaw_pose,
            leye_pose, reye_pose, left_hand_pose, right_hand_pose
        ], dim=1)

        # Concatenate shape and expression components
        shape_components = torch.cat([betas, expression], dim=-1)
        # Concatenate shape and expression blend shapes
        # Assumes self.shapedirs (shape part) is registered buffer
        # Ensure expr_dirs buffer is on the correct device/dtype
        expr_dirs = self.expr_dirs.to(device=device, dtype=dtype)
        shapedirs = torch.cat([self.shapedirs, expr_dirs], dim=-1)

        # --- Call LBS ---
        # Decide if transformations are needed based on flags
        return_transformations = return_joint_transformation or return_vertex_transformation
        # Assumes lbs function is available and handles the 'return_transformation' argument
        lbs_output = lbs(
            betas=shape_components, # Pass combined shape+expression betas
            pose=full_pose,
            v_template=self.v_template,
            shapedirs=shapedirs, # Pass combined shape+expression dirs
            posedirs=self.posedirs,
            J_regressor=self.J_regressor,
            parents=self.parents,
            lbs_weights=self.lbs_weights,
            pose2rot=pose2rot,
            return_transformation=return_transformations,
        )

        # Unpack LBS output
        if return_transformations:
            vertices, joints, joint_transformation, vertex_transformation = lbs_output
        else:
            vertices, joints = lbs_output
            joint_transformation, vertex_transformation = None, None

        # --- Calculate Landmarks ---
        # Prepare landmark calculation inputs (ensure buffers are on correct device)
        lmk_faces_idx = self.lmk_faces_idx.to(device=device)
        lmk_bary_coords = self.lmk_bary_coords.to(device=device, dtype=dtype)

        # Expand batch dim for landmark calculation
        lmk_faces_idx_batch = lmk_faces_idx.unsqueeze(0).expand(batch_size, -1)
        lmk_bary_coords_batch = lmk_bary_coords.unsqueeze(0).expand(batch_size, -1, -1)

        # Calculate dynamic landmarks if requested
        if self.use_face_contour:
            # Ensure dynamic landmark buffers are on correct device
            dynamic_lmk_faces_idx = self.dynamic_lmk_faces_idx.to(device=device)
            dynamic_lmk_bary_coords = self.dynamic_lmk_bary_coords.to(device=device, dtype=dtype)
            neck_kin_chain = self.neck_kin_chain.to(device=device)

            # Assumes find_dynamic_lmk_idx_and_bcoords function is available
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
                vertices,
                full_pose, # Pass full pose for potential kinematic calculations
                dynamic_lmk_faces_idx,
                dynamic_lmk_bary_coords,
                neck_kin_chain,
                pose2rot=pose2rot, # Pass pose2rot flag
            )
            # Concatenate static and dynamic landmarks (batch-wise)
            lmk_faces_idx_batch = torch.cat([lmk_faces_idx_batch, dyn_lmk_faces_idx], dim=1)
            lmk_bary_coords_batch = torch.cat([lmk_bary_coords_batch, dyn_lmk_bary_coords], dim=1)

        # Ensure faces tensor is on the correct device
        # Assumes self.faces_tensor is registered buffer
        faces_tensor = self.faces_tensor.to(device=device)
        # Assumes vertices2landmarks function is available
        landmarks = vertices2landmarks(vertices, faces_tensor,
                                       lmk_faces_idx_batch, lmk_bary_coords_batch)

        # --- Finalize Joints ---
        # Apply vertex joint selector (if defined, e.g., in base class)
        # Assumes self.vertex_joint_selector is defined
        if hasattr(self, 'vertex_joint_selector') and self.vertex_joint_selector is not None:
             joints = self.vertex_joint_selector(vertices, joints)

        # Concatenate landmarks to joints
        # Ensure landmarks are calculated before this step
        joints = torch.cat([joints, landmarks], dim=1)

        # Apply joint mapper (if defined, e.g., in base class)
        # Assumes self.joint_mapper is defined
        if hasattr(self, 'joint_mapper') and self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        # --- Apply Translation ---
        if apply_trans:
            # Ensure transl is on the correct device/dtype before applying
            transl = transl.to(device=device, dtype=dtype)
            joints = joints + transl.unsqueeze(1)
            vertices = vertices + transl.unsqueeze(1)

        # --- Create Output ---
        # Assumes SMPLXOutput class is defined and accepts these arguments
        output = SMPLXOutput(
            vertices=vertices if return_verts else None,
            joints=joints,
            betas=betas,
            expression=expression if self.num_expression_coeffs > 0 else None, # Return expression only if used
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose, # Return axis-angle version
            right_hand_pose=right_hand_pose, # Return axis-angle version
            jaw_pose=jaw_pose,
            full_pose=full_pose if return_full_pose else None,
            joint_transformation=joint_transformation if return_joint_transformation else None, # Added back
            vertex_transformation=vertex_transformation if return_vertex_transformation else None, # Added back
        )

        return output

class SMPLXLayer(SMPLX):
    def __init__(self, *args, **kwargs) -> None:
        # Just create a SMPLX module without any member variables
        super(SMPLXLayer, self).__init__(
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_betas=False,
            create_expression=False,
            create_transl=False,
            *args,
            **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs,
    ) -> SMPLXOutput:
        """
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. It is expected to be in rotation matrix
                format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                Expression coefficients.
                For example, it can used if expression parameters
                `expression` are predicted from some external model.
            body_pose: torch.tensor, optional, shape BxJx3x3
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            left_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the left hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            right_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the right hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            jaw_pose: torch.tensor, optional, shape Bx3x3
                Jaw pose. It should either joint rotations in
                rotation matrix format.
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full pose vector (default=False)
            Returns
            -------
                output: ModelOutput
                A data class that contains the posed vertices and joints
        """
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [
            betas,
            global_orient,
            body_pose,
            transl,
            expression,
            left_hand_pose,
            right_hand_pose,
            jaw_pose,
        ]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if body_pose is None:
            body_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3,
                                            3).expand(batch_size, self.NUM_BODY_JOINTS, -1,
                                                      -1).contiguous()
            )
        if left_hand_pose is None:
            left_hand_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
            )
        if right_hand_pose is None:
            right_hand_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
            )
        if jaw_pose is None:
            jaw_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if leye_pose is None:
            leye_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if reye_pose is None:
            reye_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if expression is None:
            expression = torch.zeros([batch_size, self.num_expression_coeffs],
                                     dtype=dtype,
                                     device=device)
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [
                global_orient.reshape(-1, 1, 3, 3),
                body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
                jaw_pose.reshape(-1, 1, 3, 3),
                leye_pose.reshape(-1, 1, 3, 3),
                reye_pose.reshape(-1, 1, 3, 3),
                left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
                right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
            ],
            dim=1,
        )
        shape_components = torch.cat([betas, expression], dim=-1)

        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(
            shape_components,
            full_pose,
            self.v_template,
            shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )

        lmk_faces_idx = (self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1).contiguous())
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices,
                full_pose,
                self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=False,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords

            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([
                lmk_bary_coords.expand(batch_size, -1, -1), dyn_lmk_bary_coords
            ], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)
        # Map the joints to the current dataset

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLXOutput(
            vertices=vertices if return_verts else None,
            joints=joints,
            betas=betas,
            expression=expression,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            transl=transl,
            full_pose=full_pose if return_full_pose else None,
        )
        return output


class MANO(SMPL):
    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS

    def __init__(
        self,
        model_path: str,
        is_rhand: bool = True,
        data_struct: Optional[Struct] = None,
        create_hand_pose: bool = True,
        hand_pose: Optional[Tensor] = None,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        batch_size: int = 1,
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed: bool = True,
        ext: str = "pkl",
        **kwargs,
    ) -> None:
        """MANO model constructor

        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_hand_pose: bool, optional
            Flag for creating a member variable for the pose of the right
            hand. (default = True)
        hand_pose: torch.tensor, optional, BxP
            The default value for the right hand pose member variable.
            (default = None)
        num_pca_comps: int, optional
            The number of PCA components to use for each hand.
            (default = 6)
        flat_hand_mean: bool, optional
            If False, then the pose of the hand is initialized to False.
        batch_size: int, optional
            The batch size used for creating the member variables
        dtype: torch.dtype, optional
            The data type for the created variables
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """

        self.num_pca_comps = num_pca_comps
        self.is_rhand = is_rhand
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = "MANO_{}.{ext}".format("RIGHT" if is_rhand else "LEFT", ext=ext)
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = (True if "RIGHT" in os.path.basename(model_path) else False)
            assert osp.exists(mano_path), "Path {} does not exist!".format(mano_path)

            if ext == "pkl":
                with open(mano_path, "rb") as mano_file:
                    model_data = pickle.load(mano_file, encoding="latin1")
            elif ext == "npz":
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError("Unknown extension: {}".format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS["smplh"]

        super(MANO, self).__init__(
            model_path=model_path,
            data_struct=data_struct,
            batch_size=batch_size,
            vertex_ids=vertex_ids,
            use_compressed=use_compressed,
            dtype=dtype,
            ext=ext,
            **kwargs,
        )

        # add only MANO tips to the extra joints
        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            list(VERTEX_IDS["mano"].values()), dtype=torch.long
        )

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        if self.num_pca_comps == 45:
            self.use_pca = False
        self.flat_hand_mean = flat_hand_mean

        hand_components = data_struct.hands_components[:num_pca_comps]

        self.np_hand_components = hand_components

        if self.use_pca:
            self.register_buffer("hand_components", torch.tensor(hand_components, dtype=dtype))

        if self.flat_hand_mean:
            hand_mean = np.zeros_like(data_struct.hands_mean)
        else:
            hand_mean = data_struct.hands_mean

        self.register_buffer("hand_mean", to_tensor(hand_mean, dtype=self.dtype))

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_hand_pose:
            if hand_pose is None:
                default_hand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype)
            else:
                default_hand_pose = torch.tensor(hand_pose, dtype=dtype)

            hand_pose_param = nn.Parameter(default_hand_pose, requires_grad=True)
            self.register_parameter("hand_pose", hand_pose_param)

        # Create the buffer for the mean pose.
        pose_mean = self.create_mean_pose(data_struct, flat_hand_mean=flat_hand_mean)
        pose_mean_tensor = pose_mean.clone().to(dtype)
        # pose_mean_tensor = torch.tensor(pose_mean, dtype=dtype)
        self.register_buffer("pose_mean", pose_mean_tensor)

    def name(self) -> str:
        return "MANO"

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        pose_mean = torch.cat([global_orient_mean, self.hand_mean], dim=0)
        return pose_mean

    def extra_repr(self):
        msg = [super(MANO, self).extra_repr()]
        if self.use_pca:
            msg.append(f"Number of PCA components: {self.num_pca_comps}")
        msg.append(f"Flat hand mean: {self.flat_hand_mean}")
        return "\n".join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs,
    ) -> MANOOutput:
        """Forward pass for the MANO model"""
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        betas = betas if betas is not None else self.betas
        hand_pose = hand_pose if hand_pose is not None else self.hand_pose

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None:
            if hasattr(self, "transl"):
                transl = self.transl

        if self.use_pca:
            hand_pose = torch.einsum("bi,ij->bj", [hand_pose, self.hand_components])

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        full_pose += self.pose_mean

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=True,
        )

        # # Add pre-selected extra joints that might be needed
        # joints = self.vertex_joint_selector(vertices, joints)

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = MANOOutput(
            vertices=vertices if return_verts else None,
            joints=joints if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=full_pose if return_full_pose else None,
        )

        return output


class MANOLayer(MANO):
    def __init__(self, *args, **kwargs) -> None:
        """MANO as a layer model constructor"""
        super(MANOLayer, self).__init__(
            create_global_orient=False,
            create_hand_pose=False,
            create_betas=False,
            create_transl=False,
            *args,
            **kwargs,
        )

    def name(self) -> str:
        return "MANO"

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs,
    ) -> MANOOutput:
        """Forward pass for the MANO model"""
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            batch_size = 1
            global_orient = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        else:
            batch_size = global_orient.shape[0]
        if hand_pose is None:
            hand_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
            )
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = MANOOutput(
            vertices=vertices if return_verts else None,
            joints=joints if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=full_pose if return_full_pose else None,
        )

        return output


class FLAME(SMPL):
    NUM_JOINTS = 5
    SHAPE_SPACE_DIM = 300
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 0

    def __init__(
        self,
        model_path: str,
        data_struct=None,
        num_expression_coeffs=10,
        create_expression: bool = True,
        expression: Optional[Tensor] = None,
        create_neck_pose: bool = True,
        neck_pose: Optional[Tensor] = None,
        create_jaw_pose: bool = True,
        jaw_pose: Optional[Tensor] = None,
        create_leye_pose: bool = True,
        leye_pose: Optional[Tensor] = None,
        create_reye_pose=True,
        reye_pose: Optional[Tensor] = None,
        use_face_contour=False,
        batch_size: int = 1,
        gender: str = "neutral",
        dtype: torch.dtype = torch.float32,
        ext="pkl",
        **kwargs,
    ) -> None:
        """FLAME model constructor

        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        num_expression_coeffs: int, optional
            Number of expression components to use
            (default = 10).
        create_expression: bool, optional
            Flag for creating a member variable for the expression space
            (default = True).
        expression: torch.tensor, optional, Bx10
            The default value for the expression member variable.
            (default = None)
        create_neck_pose: bool, optional
            Flag for creating a member variable for the neck pose.
            (default = False)
        neck_pose: torch.tensor, optional, Bx3
            The default value for the neck pose variable.
            (default = None)
        create_jaw_pose: bool, optional
            Flag for creating a member variable for the jaw pose.
            (default = False)
        jaw_pose: torch.tensor, optional, Bx3
            The default value for the jaw pose variable.
            (default = None)
        create_leye_pose: bool, optional
            Flag for creating a member variable for the left eye pose.
            (default = False)
        leye_pose: torch.tensor, optional, Bx10
            The default value for the left eye pose variable.
            (default = None)
        create_reye_pose: bool, optional
            Flag for creating a member variable for the right eye pose.
            (default = False)
        reye_pose: torch.tensor, optional, Bx10
            The default value for the right eye pose variable.
            (default = None)
        use_face_contour: bool, optional
            Whether to compute the keypoints that form the facial contour
        batch_size: int, optional
            The batch size used for creating the member variables
        gender: str, optional
            Which gender to load
        dtype: torch.dtype
            The data type for the created variables
        """
        model_fn = f"FLAME_{gender.upper()}.{ext}"
        flame_path = os.path.join(model_path, model_fn)
        assert osp.exists(flame_path), "Path {} does not exist!".format(flame_path)
        if ext == "npz":
            file_data = np.load(flame_path, allow_pickle=True)
        elif ext == "pkl":
            with open(flame_path, "rb") as smpl_file:
                file_data = pickle.load(smpl_file, encoding="latin1")
        else:
            raise ValueError("Unknown extension: {}".format(ext))
        data_struct = Struct(**file_data)

        super(FLAME, self).__init__(
            model_path=model_path,
            data_struct=data_struct,
            dtype=dtype,
            batch_size=batch_size,
            gender=gender,
            ext=ext,
            **kwargs,
        )

        self.use_face_contour = use_face_contour

        self.vertex_joint_selector.extra_joints_idxs = to_tensor([], dtype=torch.long)

        if create_neck_pose:
            if neck_pose is None:
                default_neck_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_neck_pose = torch.tensor(neck_pose, dtype=dtype)
            neck_pose_param = nn.Parameter(default_neck_pose, requires_grad=True)
            self.register_parameter("neck_pose", neck_pose_param)

        if create_jaw_pose:
            if jaw_pose is None:
                default_jaw_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_jaw_pose = torch.tensor(jaw_pose, dtype=dtype)
            jaw_pose_param = nn.Parameter(default_jaw_pose, requires_grad=True)
            self.register_parameter("jaw_pose", jaw_pose_param)

        if create_leye_pose:
            if leye_pose is None:
                default_leye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_leye_pose = torch.tensor(leye_pose, dtype=dtype)
            leye_pose_param = nn.Parameter(default_leye_pose, requires_grad=True)
            self.register_parameter("leye_pose", leye_pose_param)

        if create_reye_pose:
            if reye_pose is None:
                default_reye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_reye_pose = torch.tensor(reye_pose, dtype=dtype)
            reye_pose_param = nn.Parameter(default_reye_pose, requires_grad=True)
            self.register_parameter("reye_pose", reye_pose_param)

        shapedirs = data_struct.shapedirs
        if len(shapedirs.shape) < 3:
            shapedirs = shapedirs[:, :, None]
        if shapedirs.shape[-1] < self.SHAPE_SPACE_DIM + self.EXPRESSION_SPACE_DIM:
            # print(f'WARNING: You are using a {self.name()} model, with only'
            #       ' 10 shape and 10 expression coefficients.')
            expr_start_idx = 10
            expr_end_idx = 20
            num_expression_coeffs = min(num_expression_coeffs, 10)
        else:
            expr_start_idx = self.SHAPE_SPACE_DIM
            expr_end_idx = self.SHAPE_SPACE_DIM + num_expression_coeffs
            num_expression_coeffs = min(num_expression_coeffs, self.EXPRESSION_SPACE_DIM)

        self._num_expression_coeffs = num_expression_coeffs

        expr_dirs = shapedirs[:, :, expr_start_idx:expr_end_idx]
        self.register_buffer("expr_dirs", to_tensor(to_np(expr_dirs), dtype=dtype))

        if create_expression:
            if expression is None:
                default_expression = torch.zeros([batch_size, self.num_expression_coeffs],
                                                 dtype=dtype)
            else:
                default_expression = torch.tensor(expression, dtype=dtype)
            expression_param = nn.Parameter(default_expression, requires_grad=True)
            self.register_parameter("expression", expression_param)

        # The pickle file that contains the barycentric coordinates for
        # regressing the landmarks
        landmark_bcoord_filename = osp.join(model_path, "flame_static_embedding.pkl")

        with open(landmark_bcoord_filename, "rb") as fp:
            landmarks_data = pickle.load(fp, encoding="latin1")

        lmk_faces_idx = landmarks_data["lmk_face_idx"].astype(np.int64)
        self.register_buffer("lmk_faces_idx", torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = landmarks_data["lmk_b_coords"]
        self.register_buffer("lmk_bary_coords", torch.tensor(lmk_bary_coords, dtype=dtype))
        if self.use_face_contour:
            face_contour_path = os.path.join(model_path, "flame_dynamic_embedding.npy")
            contour_embeddings = np.load(face_contour_path, allow_pickle=True,
                                         encoding="latin1")[()]

            dynamic_lmk_faces_idx = np.array(contour_embeddings["lmk_face_idx"], dtype=np.int64)
            dynamic_lmk_faces_idx = torch.tensor(dynamic_lmk_faces_idx, dtype=torch.long)
            self.register_buffer("dynamic_lmk_faces_idx", dynamic_lmk_faces_idx)

            dynamic_lmk_b_coords = torch.tensor(contour_embeddings["lmk_b_coords"], dtype=dtype)
            self.register_buffer("dynamic_lmk_bary_coords", dynamic_lmk_b_coords)

            neck_kin_chain = find_joint_kin_chain(self.NECK_IDX, self.parents)
            self.register_buffer("neck_kin_chain", torch.tensor(neck_kin_chain, dtype=torch.long))

    @property
    def num_expression_coeffs(self):
        return self._num_expression_coeffs

    def name(self) -> str:
        return "FLAME"

    def extra_repr(self):
        msg = [
            super(FLAME, self).extra_repr(),
            f"Number of Expression Coefficients: {self.num_expression_coeffs}",
            f"Use face contour: {self.use_face_contour}",
        ]
        return "\n".join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        neck_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> FLAMEOutput:
        """
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        """

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose

        leye_pose = leye_pose if leye_pose is not None else self.leye_pose
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose

        betas = betas if betas is not None else self.betas
        expression = expression if expression is not None else self.expression

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None:
            if hasattr(self, "transl"):
                transl = self.transl

        full_pose = torch.cat([global_orient, neck_pose, jaw_pose, leye_pose, reye_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0], jaw_pose.shape[0])
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / betas.shape[0])
        if scale > 1:
            betas = betas.expand(scale, -1)
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(
            shape_components,
            full_pose,
            self.v_template,
            shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
        )

        lmk_faces_idx = (self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1).contiguous())
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices,
                full_pose,
                self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=True,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords
            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([
                lmk_bary_coords.expand(batch_size, -1, -1), dyn_lmk_bary_coords
            ], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = FLAMEOutput(
            vertices=vertices if return_verts else None,
            joints=joints,
            betas=betas,
            expression=expression,
            global_orient=global_orient,
            neck_pose=neck_pose,
            jaw_pose=jaw_pose,
            full_pose=full_pose if return_full_pose else None,
        )
        return output


class FLAMELayer(FLAME):
    def __init__(self, *args, **kwargs) -> None:
        """ FLAME as a layer model constructor """
        super(FLAMELayer, self).__init__(
            create_betas=False,
            create_expression=False,
            create_global_orient=False,
            create_neck_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            *args,
            **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        neck_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> FLAMEOutput:
        """
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                Global rotation of the body. Useful if someone wishes to
                predicts this with an external model. It is expected to be in
                rotation matrix format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                Shape parameters. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            jaw_pose: torch.tensor, optional, shape Bx3x3
                Jaw pose. It should either joint rotations in
                rotation matrix format.
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        """
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            batch_size = 1
            global_orient = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        else:
            batch_size = global_orient.shape[0]
        if neck_pose is None:
            neck_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 1, -1, -1).contiguous()
            )
        if jaw_pose is None:
            jaw_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if leye_pose is None:
            leye_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if reye_pose is None:
            reye_pose = (
                torch.eye(3, device=device,
                          dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
            )
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas], dtype=dtype, device=device)
        if expression is None:
            expression = torch.zeros([batch_size, self.num_expression_coeffs],
                                     dtype=dtype,
                                     device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat([global_orient, neck_pose, jaw_pose, leye_pose, reye_pose], dim=1)

        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(
            shape_components,
            full_pose,
            self.v_template,
            shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )

        lmk_faces_idx = (self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1).contiguous())
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices,
                full_pose,
                self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=False,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords
            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([
                lmk_bary_coords.expand(batch_size, -1, -1), dyn_lmk_bary_coords
            ], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        joints += transl.unsqueeze(dim=1)
        vertices += transl.unsqueeze(dim=1)

        output = FLAMEOutput(
            vertices=vertices if return_verts else None,
            joints=joints,
            betas=betas,
            expression=expression,
            global_orient=global_orient,
            neck_pose=neck_pose,
            jaw_pose=jaw_pose,
            full_pose=full_pose if return_full_pose else None,
        )
        return output


def build_layer(model_path: str,
                model_type: str = "smpl",
                **kwargs) -> Union[SMPLLayer, SMPLHLayer, SMPLXLayer, MANOLayer, FLAMELayer]:
    """Method for creating a model from a path and a model type

    Parameters
    ----------
    model_path: str
        Either the path to the model you wish to load or a folder,
        where each subfolder contains the differents types, i.e.:
        model_path:
        |
        |-- smpl
            |-- SMPL_FEMALE
            |-- SMPL_NEUTRAL
            |-- SMPL_MALE
        |-- smplh
            |-- SMPLH_FEMALE
            |-- SMPLH_MALE
        |-- smplx
            |-- SMPLX_FEMALE
            |-- SMPLX_NEUTRAL
            |-- SMPLX_MALE
        |-- mano
            |-- MANO RIGHT
            |-- MANO LEFT
        |-- flame
            |-- FLAME_FEMALE
            |-- FLAME_MALE
            |-- FLAME_NEUTRAL

    model_type: str, optional
        When model_path is a folder, then this parameter specifies  the
        type of model to be loaded
    **kwargs: dict
        Keyword arguments

    Returns
    -------
        body_model: nn.Module
            The PyTorch module that implements the corresponding body model
    Raises
    ------
        ValueError: In case the model type is not one of SMPL, SMPLH,
        SMPLX, MANO or FLAME
    """

    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split("_")[0].lower()

    if model_type.lower() == "smpl":
        return SMPLLayer(model_path, **kwargs)
    elif model_type.lower() == "smplh":
        return SMPLHLayer(model_path, **kwargs)
    elif model_type.lower() == "smplx":
        return SMPLXLayer(model_path, **kwargs)
    elif "mano" in model_type.lower():
        return MANOLayer(model_path, **kwargs)
    elif "flame" in model_type.lower():
        return FLAMELayer(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model type {model_type}, exiting!")


def create(model_path: str,
           model_type: str = "smpl",
           **kwargs) -> Union[SMPL, SMPLH, SMPLX, MANO, FLAME]:
    """Method for creating a model from a path and a model type

    Parameters
    ----------
    model_path: str
        Either the path to the model you wish to load or a folder,
        where each subfolder contains the differents types, i.e.:
        model_path:
        |
        |-- smpl
            |-- SMPL_FEMALE
            |-- SMPL_NEUTRAL
            |-- SMPL_MALE
        |-- smplh
            |-- SMPLH_FEMALE
            |-- SMPLH_MALE
        |-- smplx
            |-- SMPLX_FEMALE
            |-- SMPLX_NEUTRAL
            |-- SMPLX_MALE
        |-- mano
            |-- MANO RIGHT
            |-- MANO LEFT

    model_type: str, optional
        When model_path is a folder, then this parameter specifies  the
        type of model to be loaded
    **kwargs: dict
        Keyword arguments

    Returns
    -------
        body_model: nn.Module
            The PyTorch module that implements the corresponding body model
    Raises
    ------
        ValueError: In case the model type is not one of SMPL, SMPLH,
        SMPLX, MANO or FLAME
    """
    
    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split("_")[0].lower()

    if model_type.lower() == "smpl":
        return SMPL(model_path, **kwargs)
    elif model_type.lower() == "smplh":
        return SMPLH(model_path, **kwargs)
    elif model_type.lower() == "smplx":
        return SMPLX(model_path, **kwargs)
    elif "mano" in model_type.lower():
        return MANO(model_path, **kwargs)
    elif "flame" in model_type.lower():
        return FLAME(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model type {model_type}, exiting!")
