import cv2
import numpy as np

def preprocess_mask(input_path, output_path="mask_processed.png"):
    """
    Preprocess the mask as follows:
      1. Load the mask in grayscale.
      2. Turn all white (255) pixels to black (0).
      3. Then, for every pixel, if it is in the range [120, 140] it becomes white (255),
         and everything else stays black (0).
         
    This creates a single-channel mask where white indicates the areas to inpaint.
    """
    # Load the mask in grayscale (single-channel)
    mask_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise ValueError(f"Could not load mask image at {input_path}")

    # Turn all white (255) pixels to black (0)
    mask_gray[mask_gray == 255] = 0

    # Create a new mask: pixels in the range [120, 140] become white, everything else remains black.
    processed = np.zeros_like(mask_gray, dtype=np.uint8)
    gray_range = (mask_gray >= 110) & (mask_gray <= 140)
    processed[gray_range] = 255

    # Save the processed mask (single-channel)
    cv2.imwrite(output_path, processed)
    
    return output_path

def inpaint_multiscale(image_path, raw_mask_path, scale_factor=0.5, radius=25):
    """
    Multi-scale inpainting:
      1. Preprocess the raw mask.
      2. Load the original image and processed (single-channel) mask.
      3. Downscale both the image and mask by 'scale_factor'.
      4. Run inpainting on the downscaled image.
      5. Upscale the inpainted result back to the original dimensions.
    """
    # Preprocess the mask to produce a single-channel version
    processed_mask_path = preprocess_mask(raw_mask_path, output_path="mask_processed.png")
    
    # Load the processed mask in grayscale
    mask_full = cv2.imread(processed_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_full is None:
        raise ValueError("Processed mask could not be loaded.")
    
    # Load the original image in color
    image_full = cv2.imread(image_path)
    if image_full is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Get the original dimensions
    orig_h, orig_w = image_full.shape[:2]
    
    # Calculate new dimensions for downscaling
    new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
    
    # Downscale the image and mask
    image_small = cv2.resize(image_full, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_small  = cv2.resize(mask_full, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # (Optional) Debug overlay on the small image to verify mask alignment
    debug_small = image_small.copy()
    debug_small[mask_small == 255] = (0, 0, 255)
    cv2.imwrite("debug_overlay_small.png", debug_small)
    
    # Inpaint the downscaled image
    inpainted_small = cv2.inpaint(image_small, mask_small, radius, cv2.INPAINT_TELEA)
    
    # Upscale the inpainted image back to the original dimensions
    inpainted_full = cv2.resize(inpainted_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    return inpainted_full

if __name__ == "__main__":
    # Adjust the paths, scale factor, and radius as needed.
    inpainted_result = inpaint_multiscale(
        image_path="/home/ubuntu/projects/induxr/ECON/results/Fulden/IFN+_face_thresh_0.31/econ/cache/fulden_tpose_f1/texture.png",
        raw_mask_path="/home/ubuntu/projects/induxr/ECON/results/Fulden/IFN+_face_thresh_0.31/econ/cache/fulden_tpose_f1/mask.png",
        scale_factor=0.25,  # Downscale to 50% of original dimensions
        radius=50         # Inpainting radius (adjust as needed)
    )
    
    # Save the final inpainted result
    cv2.imwrite("inpainted_result_multiscale.png", inpainted_result)
    print("Multiscale inpainting complete, result saved as 'inpainted_result_multiscale.png'.")
