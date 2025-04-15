import json
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

class CameraTransformManager:
    def __init__(self, json_path, target_frame, device="cpu", debug=False):
        self.device = device
        self.target_frame = target_frame
        self.debug = debug
        self.transforms = self._load_transforms(json_path)

    def _load_transforms(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return {entry["frame"]: entry for entry in data}

    def _get_pose_matrix(self, location, quaternion):
        """Build a 4x4 pose matrix from location and quaternion."""
        # Convert from [w, x, y, z] → [x, y, z, w]
        q = quaternion
        q_xyzw = [q[1], q[2], q[3], q[0]]
        rot = R.from_quat(q_xyzw).as_matrix()  # (3, 3)
        trans = np.array(location).reshape(3, 1)  # (3, 1)

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3:] = trans

        return pose

    def get_transform_to_target(self, frame):
        """
        Returns a 4x4 transformation matrix from `frame` to `target_frame` as torch.Tensor.
        Applies Blender-to-CV axis conversion.
        """
        if frame == self.target_frame:
            return torch.eye(4, dtype=torch.float32).to(self.device)

        pose_from = self._get_pose_matrix(
            self.transforms[frame]["location"],
            self.transforms[frame]["quaternion"]
        )
        pose_to = self._get_pose_matrix(
            self.transforms[self.target_frame]["location"],
            self.transforms[self.target_frame]["quaternion"]
        )

        # Compute transformation from 'frame' to 'target_frame'
        T = np.linalg.inv(pose_to) @ pose_from

        # Blender-to-CV conversion matrix
        """
        blender_to_cv = torch.tensor([
            [1,  0,  0, 0],
            [0,  0, -1, 0],
            [0, -1,  0, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32).to(self.device)
        """
        blender_to_cv = torch.tensor([
            [1,  0,  0, 0],
            [0,  1,  0, 0],
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32).to(self.device)

        T_tensor = torch.tensor(T, dtype=torch.float32).to(self.device)
        T_cv = blender_to_cv @ T_tensor @ torch.linalg.inv(blender_to_cv)

        if self.debug:
            print(f"[DEBUG] Transform from frame {frame} → {self.target_frame}:")
            print(T_cv)

        return T_cv
