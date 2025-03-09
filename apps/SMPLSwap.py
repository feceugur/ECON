import torch

class SMPLSwap:
    """
    A callable class to swap left/right SMPL joints based on keypoint names.
    
    Usage:
        swapper = SMPLSwap(dataset.smpl_model, device)
        swapped_joints = swapper(smpl_joints_b)
    """
    def swap(self, smpl_model, device):
        self.device = device
        self.keypoint_names = smpl_model.keypoint_names  # List of 145 names
        self.name2idx = {name: idx for idx, name in enumerate(self.keypoint_names)}
        self.perm_tensor = self._generate_permutation()

    def _generate_permutation(self):
        # Automatically generate swap pairs: if a keypoint contains "left",
        # find the corresponding "right" version.
        swap_pairs = []
        for name in self.keypoint_names:
            if "left" in name:
                candidate = name.replace("left", "right")
                if candidate in self.name2idx:
                    left_idx = self.name2idx[name]
                    right_idx = self.name2idx[candidate]
                    # To avoid duplicates, add only if left_idx < right_idx.
                    if left_idx < right_idx:
                        swap_pairs.append((name, candidate))
        
        # Create an identity permutation for all keypoints.
        perm = list(range(len(self.keypoint_names)))
        for left_name, right_name in swap_pairs:
            left_idx = self.name2idx[left_name]
            right_idx = self.name2idx[right_name]
            # Swap the indices.
            perm[left_idx] = right_idx
            perm[right_idx] = left_idx
        
        # Convert the permutation list into a tensor on the given device.
        return torch.tensor(perm, dtype=torch.long).to(self.device)

    def __call__(self, smpl_joints_b):
        """
        Apply the left/right swap to the given SMPL joints tensor.
        
        Args:
            smpl_joints_b (torch.Tensor): Tensor of shape (batch, num_joints, 3)
        
        Returns:
            torch.Tensor: The tensor with left/right joints swapped.
        """
        return smpl_joints_b[:, self.perm_tensor, :]
