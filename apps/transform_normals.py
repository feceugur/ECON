import torch.nn.functional as F
import torch
import numpy as np

def transform_normals(normal_map, R_matrix):
    """
    Applies a rotation matrix to transform normal vectors from view space to world space.

    :param normal_map: numpy array (H, W, 3) containing normal vectors in [-1, 1] range
    :param R_matrix: Tensor (3, 3) rotation matrix for transformation
    :return: Transformed normal map as numpy array (H, W, 3)
    """
    
    
    # Handle input formats
    if isinstance(normal_map, np.ndarray):
        H, W, C = normal_map.shape  # Extract dimensions (H, W, 3)
        # Convert to tensor for processing
        normal_tensor = torch.from_numpy(normal_map).float()
    else:
        # If it's already a tensor, handle different shapes
        if len(normal_map.shape) == 4:  # (B, C, H, W)
            B, C, H, W = normal_map.shape
            normal_tensor = normal_map.squeeze(0).permute(1, 2, 0)  # Convert to (H, W, 3)
        elif len(normal_map.shape) == 3:  # (H, W, 3) or (3, H, W)
            if normal_map.shape[0] == 3:  # (3, H, W)
                normal_tensor = normal_map.permute(1, 2, 0)  # Convert to (H, W, 3)
            else:  # (H, W, 3)
                normal_tensor = normal_map
        else:
            raise ValueError(f"Unsupported normal map shape: {normal_map.shape}")
        
        H, W, C = normal_tensor.shape
    
    # Ensure R_matrix is on the same device as normal_tensor
    if isinstance(R_matrix, torch.Tensor):
        R_matrix = R_matrix.to(normal_tensor.device)
    else:
        R_matrix = torch.from_numpy(R_matrix).float().to(normal_tensor.device)
    
    # Reshape normals to (H*W, 3) for matrix multiplication
    normal_flat = normal_tensor.view(-1, 3)  # Shape: (H*W, 3)

    # Apply rotation: (H*W, 3) @ (3, 3).T -> (H*W, 3)
    normal_transformed = torch.matmul(normal_flat, R_matrix.T)
    
    # Optional: Flip Y-axis if coordinate systems are mismatched
    # Uncomment the next line if normals appear flipped vertically
    # normal_transformed[:, 1] = -normal_transformed[:, 1]

    # Reshape back to (H, W, 3)
    normal_transformed = normal_transformed.view(H, W, 3)

    # Normalize the transformed normals
    normal_transformed = F.normalize(normal_transformed, p=2, dim=2)
    
    # Clamp values to ensure they're in [-1, 1] range
    normal_transformed = torch.clamp(normal_transformed, -1.0, 1.0)

    # Convert back to numpy array
    return normal_transformed.detach().cpu().numpy()