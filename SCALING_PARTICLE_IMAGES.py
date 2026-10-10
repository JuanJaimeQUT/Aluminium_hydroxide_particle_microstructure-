#SEGMENTA060226SCALE2

# SEGMENT ANYTHING FOR AGGLOMERATION FACTOR
# HPC BATCH VERSION — MATCHED FILES + FIXED CROP + UPSCALE

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("/mnt/hpccs01/home/n11555581/Code/31-12-2024 SOP S2 AFTER STEP 1 RESULTS IND")
FILTER_DIR = Path("/mnt/hpccs01/home/n11555581/Code/31-12-2024 SOP S2 AFTER STEP 1 RESULTS IND SA FILTERED")
OUTPUT_DIR = Path("/mnt/hpccs01/home/n11555581/Code/31-12-2024 SOP S2 AFTER STEP 1 RESULTS IND SA X2 NEW PAR SAM")

CHECKPOINT_PATH = Path(
    "/mnt/hpccs01/home/n11555581/Code/sam_vit_h_4b8939.pth"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]   

# ============================================================
# SETTINGS
# ============================================================

CROP_SIZE = 800
UPSCALE_FACTOR = 2  # 800 → 1600


# ============================================================
# LOAD SAM MODEL
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry["vit_h"](checkpoint=str(CHECKPOINT_PATH))
sam.to(device=device)

# =======================
# 🔥 TUNED MASK GENERATOR (ADDED)
# =======================

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=64,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.75,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=200
)


# ============================================================
# MATCH ORIGINAL IMAGES BASED ON FILTERED FOLDER
# ============================================================

original_images = {
    p.stem: p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in EXTS
}

matching_stems = []

for p in FILTER_DIR.iterdir():
    if p.suffix.lower() in EXTS:
        stem = p.stem

        if stem.endswith("_SAM_contours"):
            base_stem = stem.replace("_SAM_contours", "")
            if base_stem in original_images:
                matching_stems.append(base_stem)

matching_stems = sorted(set(matching_stems))

if len(matching_stems) == 0:
    raise RuntimeError("No matching images found between folders.")

print(f"Found {len(matching_stems)} matched images")


# ============================================================
# LOOP OVER MATCHED IMAGES
# ============================================================

for stem in matching_stems:
    img_path = original_images[stem]
    print(f"\nProcessing {img_path.name}")

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"⚠ Could not read {img_path.name}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]

    if h != 1024 or w != 1024:
        print(f"⚠ Warning: Expected 1024x1024 but got {h}x{w}")

    # --------------------------------------------------------
    # FIXED CENTER CROP (800x800)
    # --------------------------------------------------------

    start_x = (w - CROP_SIZE) // 2
    start_y = (h - CROP_SIZE) // 2

    cropped = image_rgb[
        start_y:start_y + CROP_SIZE,
        start_x:start_x + CROP_SIZE
    ]

    print(f"Cropped shape: {cropped.shape}")

    # --------------------------------------------------------
    # UPSCALE TO 1600x1600
    # --------------------------------------------------------

    upscaled = cv2.resize(
        cropped,
        (CROP_SIZE * UPSCALE_FACTOR,
         CROP_SIZE * UPSCALE_FACTOR),
        interpolation=cv2.INTER_LANCZOS4
    )

    print(f"Upscaled shape: {upscaled.shape}")

    # ========================================================
    # 🔥 PARTICLE MASKING (ADDED)
    # ========================================================

    gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
    _, particle_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    particle_mask = particle_mask.astype(bool)

    # Keep only largest connected component
    num_labels, labels_cc = cv2.connectedComponents(particle_mask.astype(np.uint8))

    if num_labels > 1:
        areas = []
        for i in range(1, num_labels):
            areas.append(np.sum(labels_cc == i))
        largest_label = np.argmax(areas) + 1
        particle_mask = labels_cc == largest_label

    masked_upscaled = upscaled.copy()
    masked_upscaled[~particle_mask] = 0

    # --------------------------------------------------------
    # GENERATE MASKS
    # --------------------------------------------------------

    masks = mask_generator.generate(masked_upscaled)

    print(f"Masks before filtering: {len(masks)}")

    # ========================================================
    # 🔥 FILTER MASKS OUTSIDE PARTICLE (ADDED)
    # ========================================================

    filtered_masks = []

    for mask in masks:
        seg = mask["segmentation"]
        overlap = np.sum(seg & particle_mask)
        if overlap > 0.9 * np.sum(seg):
            filtered_masks.append(mask)

    masks = filtered_masks

    print(f"Masks after filtering: {len(masks)}")

    # --------------------------------------------------------
    # SAVE SEGMENTATION RESULT
    # --------------------------------------------------------

    plt.figure(figsize=(8, 8))
    plt.imshow(upscaled)

    for mask in masks:
        plt.contour(mask["segmentation"], colors="r", linewidths=1)

    plt.axis("off")
    plt.title("SAM segmentation (800 crop → 2x upscale)")

    out_name = f"{stem}_SAM_crop800_up2x.png"
    out_path = OUTPUT_DIR / out_name

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✔ Saved {out_name}")

print("\n🎉 All matched images processed.")
