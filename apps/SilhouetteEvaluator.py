import os
import sys
import torch
import numpy as np
import torchvision
from tqdm.auto import tqdm
import os.path as osp
from termcolor import colored

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

class SilhouetteEvaluator:
    """
    A class to calculate silhouette differences between SMPL bodies from different views
    and select the optimal canonical SMPL body with the least overall silhouette difference.
    """
    
    def __init__(self, dataset, device, transform_manager, output_dir, cfg):
        """
        Initialize the SilhouetteEvaluator.
        
        Args:
            dataset: Dataset object containing render functions
            device: Torch device
            transform_manager: CameraTransformManager for transformations between frames
            output_dir: Output directory for saving results
            cfg: Configuration object
        """
        self.dataset = dataset
        self.device = device
        self.transform_manager = transform_manager
        self.output_dir = output_dir
        self.cfg = cfg
        
        # Create output directories
        self.silhouette_dir = osp.join(output_dir, cfg.name, "silhouette_evaluation")
        os.makedirs(self.silhouette_dir, exist_ok=True)
        
        # Storage for candidate SMPL bodies and their evaluation results
        self.candidates = {}
        self.silhouette_diffs = {}
        self.avg_diffs = {}
        
    def apply_homogeneous_transform(self, x, T):
        """
        Applies a 4x4 homogeneous transformation matrix `T` to a [B, N, 3] tensor `x`.
        
        Args:
            x: Tensor of shape [B, N, 3] containing vertex positions
            T: 4x4 transformation matrix
            
        Returns:
            Transformed vertices of shape [B, N, 3]
        """
        B, N, _ = x.shape
        homo = torch.cat([x, torch.ones(B, N, 1).to(x.device)], dim=-1)  # [B, N, 4]
        return torch.matmul(homo, T[:3, :].T)  # [B, N, 3]
    
    def register_candidate(self, frame_id, smpl_verts, smpl_faces, scale=1.0):
        """
        Register a candidate SMPL body for evaluation.
        
        Args:
            frame_id: Frame ID of the candidate
            smpl_verts: SMPL vertices
            smpl_faces: SMPL faces
            scale: Scale factor
        """
        self.candidates[frame_id] = {
            'verts': smpl_verts.detach().clone(),
            'faces': smpl_faces.detach().clone(),
            'scale': scale
        }
        print(f"Registered candidate from frame {frame_id}")
        
    def calculate_silhouette_mask(self, frame_id, target_frame_id):
        """
        Calculate the silhouette mask of a candidate in the target frame's view.
        
        Args:
            frame_id: Frame ID of the candidate
            target_frame_id: Frame ID of the target view
            
        Returns:
            tuple: (front_mask, back_mask)
        """
        if frame_id not in self.candidates:
            raise ValueError(f"Candidate for frame {frame_id} not registered")
        
        # Get vertices and faces
        smpl_verts = self.candidates[frame_id]['verts']
        smpl_faces = self.candidates[frame_id]['faces']
        
        # Apply transformation from frame_id to target_frame_id
        if frame_id != target_frame_id:
            # Update the transform_manager's target frame
            current_target = self.transform_manager.target_frame
            self.transform_manager.target_frame = target_frame_id
            
            # Get the transformation
            T = self.transform_manager.get_transform_to_target(frame_id)
            
            # Zero out translation to keep model centered in frame
            # This is crucial for proper silhouette comparison
            T_modified = T.clone()
            T_modified[:3, 3] = 0.0
            
            # Restore original target frame
            self.transform_manager.target_frame = current_target
            
            # Apply transformation
            transformed_verts = self.apply_homogeneous_transform(smpl_verts, T_modified)
            
            # Save transformation matrix for debugging
            debug_dir = osp.join(self.silhouette_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            torch.save(
                {"T_original": T, "T_modified": T_modified, "frame_id": frame_id, "target_frame_id": target_frame_id},
                osp.join(debug_dir, f"transform_{frame_id}_to_{target_frame_id}.pt")
            )
        else:
            transformed_verts = smpl_verts
            
        # Calculate masks using renderer
        self.dataset.render.load_meshes(
            transformed_verts * torch.tensor([1.0, -1.0, -1.0]).to(self.device),
            smpl_faces
        )
        front_mask, back_mask = self.dataset.render.get_image(type="mask")
        
        # Save debug renders for analysis
        debug_dir = osp.join(self.silhouette_dir, "debug_renders")
        os.makedirs(debug_dir, exist_ok=True)
        torchvision.utils.save_image(
            front_mask.repeat(1, 3, 1, 1),
            osp.join(debug_dir, f"mask_cand{frame_id}_target{target_frame_id}_front.png")
        )
        torchvision.utils.save_image(
            back_mask.repeat(1, 3, 1, 1),
            osp.join(debug_dir, f"mask_cand{frame_id}_target{target_frame_id}_back.png")
        )
        
        return front_mask, back_mask
    
    def calculate_mask_difference(self, pred_mask, gt_mask):
        """
        Calculate the difference percentage between two masks.
        
        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            
        Returns:
            float: Difference percentage
        """
        # Ensure masks are binary
        pred_mask_bin = (pred_mask > 0.5).float()
        gt_mask_bin = (gt_mask > 0.5).float()
        
        # Calculate union and intersection
        union = torch.logical_or(pred_mask_bin, gt_mask_bin).sum()
        if union == 0:
            return 100.0  # Both masks are empty, maximum difference
            
        diff = torch.abs(pred_mask_bin - gt_mask_bin).sum()
        diff_percentage = (diff / union * 100).item()
        
        return diff_percentage
    
    def evaluate_all_candidates(self, gt_masks):
        """
        Evaluate all candidates against ground truth masks for each frame.
        
        Args:
            gt_masks: Dictionary mapping frame_ids to their ground truth masks
                      Format: {frame_id: {'front': mask_f, 'back': mask_b}}
        """
        print(colored("Evaluating all candidate SMPL bodies...", "green"))
        
        # Reset difference storage
        self.silhouette_diffs = {cand_id: {} for cand_id in self.candidates.keys()}
        self.avg_diffs = {}
        
        for cand_id in tqdm(self.candidates.keys()):
            total_diff = 0.0
            count = 0
            
            # Compare this candidate against all ground truth masks
            for target_id, masks in gt_masks.items():
                # Calculate silhouette of this candidate from target view
                pred_front, pred_back = self.calculate_silhouette_mask(cand_id, target_id)
                
                # Calculate difference with ground truth
                front_diff = self.calculate_mask_difference(pred_front, masks['front'])
                back_diff = self.calculate_mask_difference(pred_back, masks['back'])
                avg_diff = (front_diff + back_diff) / 2
                
                # Store differences
                self.silhouette_diffs[cand_id][target_id] = {
                    'front': front_diff,
                    'back': back_diff,
                    'avg': avg_diff
                }
                
                # Accumulate for average
                total_diff += avg_diff
                count += 1
                
                # Visualize differences
                self.visualize_difference(cand_id, target_id, pred_front, pred_back, 
                                         masks['front'], masks['back'])
            
            # Calculate average difference across all views
            self.avg_diffs[cand_id] = total_diff / count
            print(f"Candidate {cand_id} - Average difference: {self.avg_diffs[cand_id]:.2f}%")
    
    def visualize_difference(self, cand_id, target_id, pred_front, pred_back, 
                            gt_front, gt_back):
        """
        Visualize silhouette differences between predicted and ground truth masks.
        
        Args:
            cand_id: Candidate frame ID
            target_id: Target frame ID
            pred_front, pred_back: Predicted masks
            gt_front, gt_back: Ground truth masks
        """
        # Create visualization grid
        vis_grid = torch.cat([
            # Front view
            gt_front.repeat(1, 3, 1, 1),                # Ground truth
            pred_front.repeat(1, 3, 1, 1),              # Prediction
            torch.abs(gt_front - pred_front).repeat(1, 3, 1, 1),  # Difference
            # Back view
            gt_back.repeat(1, 3, 1, 1),                 # Ground truth
            pred_back.repeat(1, 3, 1, 1),               # Prediction
            torch.abs(gt_back - pred_back).repeat(1, 3, 1, 1),    # Difference
        ], dim=3)
        
        # Save visualization
        vis_path = osp.join(self.silhouette_dir, f"silhouette_diff_cand{cand_id}_target{target_id}.png")
        torchvision.utils.save_image(vis_grid, vis_path)
    
    def select_best_candidate(self):
        """
        Select the candidate with the least average silhouette difference.
        
        Returns:
            tuple: (best_frame_id, best_verts, best_faces, avg_diff)
        """
        if not self.avg_diffs:
            raise ValueError("No candidates have been evaluated yet")
        
        # Find candidate with minimum average difference
        best_frame_id = min(self.avg_diffs, key=self.avg_diffs.get)
        best_diff = self.avg_diffs[best_frame_id]
        
        print(colored(f"Best canonical SMPL body is from frame {best_frame_id} with average silhouette difference: {best_diff:.2f}%", "green"))
        
        # Return the best candidate
        return (
            best_frame_id, 
            self.candidates[best_frame_id]['verts'], 
            self.candidates[best_frame_id]['faces'],
            best_diff
        )
    
    def get_detailed_report(self):
        """
        Generate a detailed report of silhouette differences.
        
        Returns:
            str: Report text
        """
        report = "Silhouette Evaluation Report\n"
        report += "===========================\n\n"
        
        for cand_id, diffs in self.silhouette_diffs.items():
            report += f"Candidate from frame {cand_id}:\n"
            report += f"  Average difference: {self.avg_diffs[cand_id]:.2f}%\n"
            report += "  Per-frame differences:\n"
            
            for target_id, vals in diffs.items():
                report += f"    Frame {target_id}: Front {vals['front']:.2f}%, Back {vals['back']:.2f}%, Avg {vals['avg']:.2f}%\n"
            
            report += "\n"
        
        return report
    
    def save_report(self):
        """
        Save the detailed evaluation report to a file.
        """
        report = self.get_detailed_report()
        report_path = osp.join(self.silhouette_dir, "evaluation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Saved detailed evaluation report to {report_path}")
    
    def __call__(self, gt_masks):
        """
        Run the full evaluation process and return the best candidate.
        
        Args:
            gt_masks: Dictionary mapping frame_ids to their ground truth masks
        
        Returns:
            tuple: (best_frame_id, best_verts, best_faces, avg_diff)
        """
        self.evaluate_all_candidates(gt_masks)
        best_candidate = self.select_best_candidate()
        self.save_report()
        return best_candidate 