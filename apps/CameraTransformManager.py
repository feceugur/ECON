import json
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CameraTransformManager:
    def __init__(self, json_path, target_frame, device="cpu", debug=False, use_blender_to_cv=True):
        self.device = device
        self.target_frame = target_frame
        self.debug = debug
        self.use_blender_to_cv = use_blender_to_cv
        self.transforms = self._load_transforms(json_path)

    def _load_transforms(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return {entry["frame"]: entry for entry in data}

    def _get_pose_matrix(self, location, quaternion):
        #Build a 4x4 pose matrix from location and quaternion.
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
        #Returns a 4x4 transformation matrix from `frame` to `target_frame` as torch.Tensor.
        #Applies Blender-to-CV axis conversion.
        
        if frame == self.target_frame:
            if self.debug:
                print(f"[DEBUG] Identity transform for frame {frame}")
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

        # Blender-to-CV conversion matrix (option to switch)
        if self.use_blender_to_cv:
            blender_to_cv = torch.tensor([
                [1,  0,  0, 0],
                [0,  0, -1, 0],
                [0, -1,  0, 0],
                [0,  0,  0, 1]
            ], dtype=torch.float32).to(self.device)
        else:
            blender_to_cv = torch.eye(4, dtype=torch.float32).to(self.device)

        T_tensor = torch.tensor(T, dtype=torch.float32).to(self.device)
        T_cv = blender_to_cv @ T_tensor @ torch.linalg.inv(blender_to_cv)

        if self.debug:
            print(f"[DEBUG] Transform from frame {frame} → {self.target_frame} (Blender-to-CV: {use_blender_to_cv}):")
            #print(T_cv)

        return T_cv

    def debug_visualize_cameras_and_smpl(self, smpl_vertices, save_path=None):
        """
        Visualize all camera frustums and the SMPL mesh in 3D for alignment checking.
        """

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Plot SMPL mesh
        smpl_np = smpl_vertices.detach().cpu().numpy()
        ax.scatter(smpl_np[:, 0], smpl_np[:, 1], smpl_np[:, 2], s=1, c='r', label='SMPL')
        # Plot camera centers
        for frame, entry in self.transforms.items():
            pose = self._get_pose_matrix(entry["location"], entry["quaternion"])
            cam_center = pose[:3, 3]
            ax.scatter(cam_center[0], cam_center[1], cam_center[2], c='b', marker='o', s=50)
            ax.text(cam_center[0], cam_center[1], cam_center[2], f"Cam {frame}", color='b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()