import cv2

img = cv2.imread("/home/ubuntu/projects/induxr/econ_s/ECON/results/Fulden/IFN+_face_thresh_0.31/econ/cache/fulden_tpose_f1/texture.png")
num_channels = img.shape[2] if len(img.shape) == 3 else 1
print(f"Number of channels: {num_channels}")
