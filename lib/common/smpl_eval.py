import torch
import numpy as np
import trimesh
from collections import defaultdict
from lib.net.geometry import rot6d_to_rotmat

class SMPLMetricsCalculator:
    """
    A callable class to compute quantitative evaluation metrics for SMPL body fits
    against 2D multi-view evidence.

    It can operate in two modes:
    1. 'baseline': Evaluates each initial per-view SMPL fit against its own view.
    2. 'optimized': Evaluates a single, jointly optimized SMPL model against all views.
    """

    def __init__(self, dataset, SMPLX_object, device):
        """
        Initializes the calculator with necessary components.

        Args:
            dataset: The dataset object, which contains the SMPL model and rendering utilities.
            SMPLX_object: An object containing SMPL-X specific info like joint mappings.
            device: The torch device (e.g., 'cuda:0').
        """
        self.smpl_model = dataset.smpl_model
        self.dataset = dataset
        self.SMPLX_object = SMPLX_object
        self.device = device
        self.smpl_faces = dataset.smpl_model.faces_tensor.unsqueeze(0).to(device)

    def _calculate_landmark_error(self, smpl_joints_3d, view_data):
        """Calculates the 2D landmark reprojection error for a single view."""
        smpl_lmks_2d = smpl_joints_3d[:, self.SMPLX_object.ghum_smpl_pairs[:, 1], :2]
        ghum_lmks_2d = view_data["landmark"][:, self.SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(self.device)
        ghum_conf = view_data["landmark"][:, self.SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(self.device)

        valid_landmarks = ghum_conf > 0.5
        if not valid_landmarks.any():
            return torch.tensor(0.0, device=self.device)

        error = (torch.norm(ghum_lmks_2d - smpl_lmks_2d, dim=2) * ghum_conf * valid_landmarks.float()).sum()
        count = valid_landmarks.float().sum()

        return error / count if count > 0 else torch.tensor(0.0, device=self.device)

    def _calculate_silhouette_iou(self, smpl_verts, view_data):
        """Calculates the silhouette IoU for a single view."""
        # Render the silhouette mask for the given SMPL vertices
        # Note: The renderer in your dataset object might need specific coordinate system adjustments.
        # This implementation assumes the renderer handles vertices in the [-1, 1] normalized space.
        self.dataset.render.load_meshes(
            smpl_verts * torch.tensor([-1.0, -1.0, 1.0]).to(self.device), 
            self.smpl_faces
        )
        rendered_images = self.dataset.render.get_image(type="mask")
        rendered_mask = (rendered_images[0] > 0).bool()
        
        gt_mask = (view_data["img_mask"].to(self.device) > 0.5).bool()

        intersection = torch.sum(rendered_mask & gt_mask).float()
        union = torch.sum(rendered_mask | gt_mask).float()

        iou = intersection / (union + 1e-8)
        return iou

    def __call__(self, multi_view_data, transform_manager=None, mode='optimized', **kwargs):
        """
        Computes and aggregates metrics for a given set of SMPL parameters.

        Args:
            multi_view_data (list): A list of data dictionaries for each view.
            transform_manager (CameraTransformManager, optional): Required for 'optimized' mode.
            mode (str): 'optimized' or 'baseline'.
            **kwargs: For 'optimized' mode, provide optimed_betas, optimed_pose_mat, etc.

        Returns:
            dict: A dictionary containing the averaged metrics.
        """
        if mode not in ['optimized', 'baseline']:
            raise ValueError("Mode must be 'optimized' or 'baseline'")

        metrics = defaultdict(list)

        with torch.no_grad():
            if mode == 'optimized':
                # Generate one canonical model and project it into all views
                smpl_verts, _, smpl_joints = self.smpl_model(
                    shape_params=kwargs['optimed_betas'],
                    expression_params=kwargs['closest_exp'],
                    body_pose=kwargs['optimed_pose_mat'],
                    global_pose=kwargs['optimed_orient_mat'],
                    jaw_pose=kwargs['closest_jaw'],
                    left_hand_pose=kwargs['left_hand_pose'],
                    right_hand_pose=kwargs['right_hand_pose'],
                )
                smpl_verts = (smpl_verts + kwargs['optimed_trans']) * kwargs['scale']
                smpl_joints = (smpl_joints + kwargs['optimed_trans']) * kwargs['scale'] * torch.tensor([1.0, 1.0, -1.0]).to(self.device)

                for view_data in multi_view_data:
                    frame_id = int(view_data["name"].split("_")[1])
                    T_frame_to_target = transform_manager.get_transform_to_target(frame_id)
                    T_frame_to_target[:3, 3] = 0.0

                    # Apply transform to get view-specific vertices and joints
                    view_verts = apply_homogeneous_transform(smpl_verts, T_frame_to_target)
                    view_joints = apply_homogeneous_transform(smpl_joints, T_frame_to_target)
                    view_joints_3d_norm = (view_joints + 1.0) * 0.5

                    # Calculate metrics for this view
                    metrics['landmark_error_px'].append(self._calculate_landmark_error(view_joints_3d_norm, view_data).item() * 512) # Assuming 512px image
                    metrics['silhouette_iou'].append(self._calculate_silhouette_iou(view_verts, view_data).item())

            elif mode == 'baseline':
                # Evaluate each initial per-view fit independently
                for view_data in multi_view_data:
                    # Generate the 3D model using this specific view's initial parameters
                    # Note: We assume the initial parameters produce a model already in its view space.
                    # This logic mirrors the PIXIE output which is per-image.
                    view_pose_mat = rot6d_to_rotmat(view_data["body_pose"].view(-1, 6)).view(1, 21, 3, 3)
                    view_orient_mat = rot6d_to_rotmat(view_data["global_orient"].view(-1, 6)).view(1, 1, 3, 3)
                    
                    view_verts, _, view_joints = self.smpl_model(
                        shape_params=view_data['betas'],
                        expression_params=view_data['exp'],
                        body_pose=view_pose_mat,
                        global_pose=view_orient_mat,
                        jaw_pose=view_data['jaw_pose'],
                        left_hand_pose=view_data["left_hand_pose"],
                        right_hand_pose=view_data["right_hand_pose"],
                    )
                    
                    view_verts = (view_verts + view_data['trans']) * view_data['scale']
                    view_joints = (view_joints + view_data['trans']) * view_data['scale'] * torch.tensor([1.0, 1.0, -1.0]).to(self.device)
                    view_joints_3d_norm = (view_joints + 1.0) * 0.5
                    
                    # Calculate metrics for this view using its own data
                    metrics['landmark_error_px'].append(self._calculate_landmark_error(view_joints_3d_norm, view_data).item() * 512)
                    metrics['silhouette_iou'].append(self._calculate_silhouette_iou(view_verts, view_data).item())

        # Average the metrics
        averaged_metrics = {key: np.mean(val) for key, val in metrics.items()}
        return averaged_metrics

# Helper function (you likely have this already)
def apply_homogeneous_transform(points, T):
    points_h = torch.cat([points, torch.ones(points.shape[0], points.shape[1], 1, device=points.device)], dim=2)
    transformed_points_h = torch.matmul(points_h, T.transpose(1, 0))
    return transformed_points_h[:, :, :3]