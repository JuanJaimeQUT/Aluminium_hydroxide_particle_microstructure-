#SEGMENT ANYTHING SEGMENTATION BATCH 

import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load your image
#image_path = r"C:\Users\n11555581\OneDrive - Queensland University of Technology\SEM images\MORPHOLOGY CYCLE POS S2\31-12-2024 SOP S2\31-12-2024 SOP S2 SLIDE 3\SOP02010134.png"
#image = cv2.imread(image_path)
#if image is None:
#    raise FileNotFoundError("Image not found.")
#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the SAM model
checkpoint_path = r"C:\Users\n11555581\Downloads\sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

# Define the directory containing your images
image_dir = r"C:\Users\n11555581\OneDrive - Queensland University of Technology\Documents\SEGMENTANYTRIAL"
output_dir = r"C:\Users\n11555581\OneDrive - Queensland University of Technology\Documents\SEGMENTRESULTS"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# ---- Option 1: Automatic Mask Generation ----
#mask_generator = SamAutomaticMaskGenerator(sam) #GREAT FOR MOSAIC PARTICLES!!!!
#masks = mask_generator.generate(image)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=5,         # fewer sampling points FIND THE CONTOURS !!!!
    pred_iou_thresh=0.9,        # stricter quality requirement
    stability_score_thresh=0.95,
    min_mask_region_area=30   # remove tiny segments
)
# Process each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".png"):  # Adjust this if your images have a different extension
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Image '{filename}' not found.")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate masks for the current image
        masks = mask_generator.generate(image_rgb)

        # Merge overlapping masks
        merged_mask = np.zeros_like(masks[0]["segmentation"], dtype=bool)
        mask_list = [mask["segmentation"] for mask in masks]
        mask_list.sort(key=lambda m: np.sum(m), reverse=True)

        for m in mask_list:
            if np.any(np.logical_and(merged_mask, m)) or not np.any(merged_mask):
                merged_mask = np.logical_or(merged_mask, m)

        # Convert to 8-bit mask
        inverted_mask = np.logical_not(merged_mask)
        mask_uint8 = (inverted_mask * 255).astype(np.uint8)

        # Double check: is the mask selecting particles or background?
        num_white = np.sum(mask_uint8 == 255)
        num_black = np.sum(mask_uint8 == 0)

        if num_white < num_black:
            print(f"🔁 Inverting mask for '{filename}' — it was selecting background.")
            mask_uint8 = cv2.bitwise_not(mask_uint8)

        # Apply mask to keep only particle area (black background)
        cutout = np.zeros_like(image)
        cutout[inverted_mask] = image[inverted_mask]

        # Save the masked image
        output_filename = f"M_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cutout)
        print(f"✅ Saved masked cutout for '{filename}' to: {output_path}")

        # Optionally, save the mask as a separate image
        mask_output_path = os.path.join(output_dir, f"mask_{filename}")
        cv2.imwrite(mask_output_path, mask_uint8)
        print(f"✅ Saved mask for '{filename}' to: {mask_output_path}")

# Show masks
#plt.figure(figsize=(8, 8))
#plt.imshow(image)
#for mask in masks:
#    plt.contour(mask["segmentation"], colors='r', linewidths=1)
#plt.axis("off")
#plt.title("Automatic Segmentation")
#plt.show()

# ---- Option 2: Interactive Prompt (e.g., a point) ----
# from segment_anything import SamPredictor
# predictor = SamPredictor(sam)
# predictor.set_image(image)

# Example prompt: a point near the object of interest
# input_point = np.array([[350, 400]])  # x, y
# input_label = np.array([1])           # 1 = foreground, 0 = background

# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )

# plt.imshow(image)
# plt.imshow(masks[0], alpha=0.6)  # Show the first mask
# plt.title("Interactive Segmentation")
# plt.axis("off")
# plt.show()
