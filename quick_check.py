import cv2
import numpy as np
import os

def preprocess_mask(input_path, output_path="./inpainting/mask_processed.png", erosion_iterations=1):
    """
    Preprocess the mask as follows:
      1. Load the mask in grayscale.
      2. Turn all white (255) pixels to black (0).
      3. For every pixel, if it is in the range [120, 140] it becomes white (255),
         and everything else stays black (0).
      4. Apply erosion to slightly shrink the white regions.
         
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
    gray_range = (mask_gray >= 120) & (mask_gray <= 140)
    processed[gray_range] = 255

    # Apply erosion to shrink the inpainting regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.erode(processed, kernel, iterations=erosion_iterations)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the processed mask (single-channel)
    cv2.imwrite(output_path, processed)
    
    return output_path

def inpaint_single_scale(image_path, raw_mask_path, radius=25):
    """
    Single-scale inpainting:
      1. Preprocess the raw mask.
      2. Load the original image and the processed (single-channel) mask.
      3. Run inpainting on the full resolution image.
    """
    # Preprocess the mask to produce a single-channel version with erosion
    processed_mask_path = preprocess_mask(raw_mask_path, output_path="./inpainting/mask_processed.png")
    
    # Load the processed mask in grayscale
    mask = cv2.imread(processed_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Processed mask could not be loaded.")
    
    # Load the original image in color
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # (Optional) Create a debug overlay to verify mask alignment
    debug_overlay = image.copy()
    debug_overlay[mask == 255] = (0, 0, 255)
    cv2.imwrite("./inpainting/debug_overlay.png", debug_overlay)
    
    # Inpaint the full resolution image
    inpainted = cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
    
    return inpainted

if __name__ == "__main__":
    # Adjust the paths and inpainting radius as needed.
    inpainted_result = inpaint_single_scale(
        image_path="/home/ubuntu/projects/induxr/econ_s/ECON/results/Fulden/IFN+_face_thresh_0.31/econ/cache/fulden_tpose_f1/texture.png",
        raw_mask_path="/home/ubuntu/projects/induxr/econ_s/ECON/results/Fulden/IFN+_face_thresh_0.31/econ/cache/fulden_tpose_f1/mask.png",
        radius=10  # Inpainting radius (adjust as needed)
    )
    
    # Ensure the output directory exists
    os.makedirs("./inpainting", exist_ok=True)
    
    # Save the final inpainted result
    cv2.imwrite("./inpainting/inpainted_result.png", inpainted_result)
    print("Single-scale inpainting complete, result saved as 'inpainted_result.png'.")
