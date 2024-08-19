import torch
import scipy
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class AffineRegistration:
    def __init__(self, in_tensor_b, in_tensor_f):
        self.in_tensor_b = in_tensor_b
        self.in_tensor_f = in_tensor_f
        self.T = None  # Transformation matrix
        self.t = None  # Translation vector
        self.new_normals = None  # Transformed normal tensor

    def affine_registration(self, P, Q):
        """
        Perform affine registration to align point cloud P with point cloud Q.
        """
        transposed = False
        if P.shape[0] < P.shape[1]:
            transposed = True
            P = P.T
            Q = Q.T

        (n, dim) = P.shape
        
        # Compute least squares
        p, res, rnk, s = scipy.linalg.lstsq(np.hstack((P, np.ones([n, 1]))), Q)
        
        # Extract translation and transformation matrix
        self.t = p[-1].T
        self.T = p[:-1].T
        
        # Compute transformed point cloud
        Pt = P @ self.T.T + self.t
        if transposed:
            Pt = Pt.T
            
        return Pt

    def fit(self):
        """
        Execute the full pipeline: from tensor extraction to applying affine transformation
        and returning the final transformed tensor.
        """
        # Extract the first item from the batch (assuming batch dimension is 1)
        batch_normals_b = self.in_tensor_b[0]  # (C, H, W)
        batch_normals_f = self.in_tensor_f[0]  # (C, H, W)
        
        # Convert to NumPy arrays and reshape
        height, width = batch_normals_b.shape[1:3]  # 512, 512
        normals_b = batch_normals_b.permute(1, 2, 0).cpu().numpy().reshape(-1, 3)  # (H*W, C)
        normals_f = batch_normals_f.permute(1, 2, 0).cpu().numpy().reshape(-1, 3)  # (H*W, C)

        # Perform affine registration using the normal vectors
        Pt_old = self.affine_registration(normals_b, normals_f)

        # Ensure the number of elements matches the original shape
        expected_size = height * width
        
        if Pt_old.shape[0] != expected_size:
            raise ValueError(f"Size mismatch: Pt_old has {Pt_old.shape[0]} elements, but expected {expected_size}.")
        
        # Reshape Pt_old back to (C, H, W)
        Pt = Pt_old.reshape(height, width, 3).transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Convert the transformed normals back to a PyTorch tensor
        self.new_normals = torch.from_numpy(Pt).unsqueeze(0)  # Add batch dimension

        # Handle NaNs (if any)
        self.new_normals[torch.isnan(self.new_normals)] = 0  # Replace NaNs with 0 or any other suitable value

        return self.new_normals

    def load_image_as_tensor(image_path, target_size=(512, 512), preserve_alpha=False):
	    # Open the image using PIL
	    image = Image.open(image_path)
    
	    if preserve_alpha and image.mode == 'RGBA':
	        # If we want to preserve alpha and the image has an alpha channel
	        channels = 4
	    else:
	        # Otherwise, convert to RGB
	        image = image.convert('RGB')
	        channels = 3
		    
	    # Define the transformation pipeline
	    transform = transforms.Compose([
	        transforms.Resize(target_size),  # Resize the image to 512x512
	        transforms.ToTensor(),  # Convert the image to a tensor
	    ])
	    
	    # Apply the transformations
	    tensor = transform(image)
	    
	    # Ensure the tensor has 3 channels (RGB)
	    if tensor.shape[0] == 1:  # If it's a grayscale image
	        tensor = tensor.repeat(3, 1, 1)
	    
	    return tensor