#Inference script for segmentation to separate images with particles

import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil

# === Paths (change as needed) ===
model_path = r"C:\Users\n11555581\OneDrive - Queensland University of Technology\SEM images\MORPHOLOGY CYCLE POS S2\CODE MORPHOLOGY CYCLE\STEP 1 CODE TO SEPARATE IMAGES WITH PARTICLES AND NO PARTICLES\vgg16_best_model_identifyimages.pth"
input_folder = r"C:\Users\n11555581\OneDrive - Queensland University of Technology\SEM images\MORPHOLOGY CYCLE POS S2\08-02-2025 SOP S2\SOP S2 08-02-2025 ALL"     # Folder with images to classify
output_folder = r"C:\Users\n11555581\OneDrive - Queensland University of Technology\SEM images\MORPHOLOGY CYCLE POS S2\08-02-2025 SOP S2\SOP S2 08-02-2025 ALL AFTER STEP 1"  # Folder where sorted images will go

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load the trained VGG16 model ===
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 1)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# === Image transform (match training: resize only) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Create output folders for class 0 and class 1 ===
class_dirs = [os.path.join(output_folder, "class_0"), os.path.join(output_folder, "class_1")]
for d in class_dirs:
    os.makedirs(d, exist_ok=True)

# === Inference and saving ===
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    try:
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor).squeeze()
            prob = torch.sigmoid(output).item()
            predicted_class = 1 if prob > 0.5 else 0

        # Save to corresponding folder
        save_path = os.path.join(output_folder, f"class_{predicted_class}", img_name)
        shutil.copy(img_path, save_path)
        print(f"✅ Saved {img_name} to class_{predicted_class}")

    except Exception as e:
        print(f"⚠️ Skipped {img_name}: {e}")

