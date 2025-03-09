import torch.nn.functional as F
import torch

def transform_normals(normal_B, R_back):
    """
    Applies a rotation matrix to the normal map of the back view.

    :param normal_B: Tensor (B, 3, H, W) containing normal vectors.
    :param R_back: Tensor (3, 3) extracted from the transformation matrix.
    :return: Transformed normal map.
    """
    B, C, H, W = normal_B.shape  # Extract dimensions
    
    # Reshape normals to (B, 3, H*W) for matrix multiplication
    normal_B = normal_B.view(B, C, -1)  # Shape: (B, 3, N) where N = H * W

    # Apply rotation (batch-wise matrix multiplication)
    R_back = R_back.to(normal_B.device)  # Ensure same device (cuda)
    normal_B = torch.matmul(R_back, normal_B)  # (3,3) x (B,3,N) -> (B,3,N)

    # Reshape back to (B, 3, H, W)
    normal_B = normal_B.view(B, C, H, W)

    # Normalize the transformed normals
    normal_B = F.normalize(normal_B, p=2, dim=1)

    return normal_B