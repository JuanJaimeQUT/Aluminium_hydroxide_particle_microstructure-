# Isolate individual particles for cycle analysis 

import cv2
import numpy as np
import os

def isolate_particles_from_folder(input_folder, output_folder, target_size=512, min_area=500):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
            continue  # Skip non-image files

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Failed to read: {filename}")
            continue

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        base_filename = os.path.splitext(filename)[0]
        particle_count = 0

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            height, width = mask.shape
            if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
                continue  # Skip particles touching border

            # Create mask for individual particle
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

            # Isolate particle
            isolated = cv2.bitwise_and(image, image, mask=contour_mask)
            cropped = isolated[y:y+h, x:x+w]

            if h > target_size or w > target_size:
                continue  # Skip if too large

            # Center on blank canvas
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_offset = (target_size - h) // 2
            x_offset = (target_size - w) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

            output_name = f"{base_filename}_particle_{particle_count+1}.png"
            output_path = os.path.join(output_folder, output_name)
            cv2.imwrite(output_path, canvas)
            particle_count += 1

        print(f"✅ Processed {filename}: saved {particle_count} particles.")

# === USAGE ===
input_folder = r"C:\Users\n11555581\OneDrive - Queensland University of Technology\SEM images\MORPHOLOGY CYCLE POS S2\31-12-2024 SOP S2\31-12-2024 SOP S2 SEGMENTED PARTICLES"
output_folder = r"C:\Users\n11555581\OneDrive - Queensland University of Technology\SEM images\MORPHOLOGY CYCLE POS S2\31-12-2024 SOP S2\31-12-2024 SOP S2 ISOLATE PARTICLES AND RESIZE"

isolate_particles_from_folder(input_folder, output_folder)

