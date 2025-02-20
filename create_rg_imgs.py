import os
from PIL import Image

# Set the resolution for the high-resolution images
width, height = 1920, 1080  # You can change these values for a different resolution

# Create a completely green image (RGB: 0, 255, 0)
green_image = Image.new("RGB", (width, height), (0, 255, 0))

# Create a completely red image (RGB: 255, 0, 0)
red_image = Image.new("RGB", (width, height), (255, 0, 0))

# Define the output directory where images will be saved
output_dir = "./examples/jon"
os.makedirs(output_dir, exist_ok=True)

# Save the images in the output directory
green_image_path = os.path.join(output_dir, "green_image.png")
red_image_path = os.path.join(output_dir, "red_image.png")

green_image.save(green_image_path)
red_image.save(red_image_path)

print(f"Green image saved to: {green_image_path}")
print(f"Red image saved to: {red_image_path}")
