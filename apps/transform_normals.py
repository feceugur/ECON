import torch.nn.functional as F
import torch

def transform_normals(normal_B, T):
    """
    Applies a transformation matrix to the normal map.

    :param normal_B: Tensor (B, 3, H, W) containing normal vectors.
    :param T: Tensor (4, 4) transformation matrix or (3, 3) rotation matrix.
    :return: Transformed normal map.
    """
    B, C, H, W = normal_B.shape  # Extract dimensions
    
    # Extract 3x3 rotation matrix from 4x4 transformation matrix if needed
    if T.shape == (4, 4):
        R = T[:3, :3]  # Extract top-left 3x3 rotation matrix
    else:
        R = T  # Already a 3x3 matrix
    
    # Reshape normals to (B, 3, H*W) for matrix multiplication
    normal_B = normal_B.view(B, C, -1)  # Shape: (B, 3, N) where N = H * W

    # Apply rotation (batch-wise matrix multiplication)
    R = R.to(normal_B.device)  # Ensure same device (cuda)
    normal_B = torch.matmul(R, normal_B)  # (3,3) x (B,3,N) -> (B,3,N)

    # Reshape back to (B, 3, H, W)
    normal_B = normal_B.view(B, C, H, W)

    # Normalize the transformed normals
    normal_B = F.normalize(normal_B, p=2, dim=1)

    return normal_B