#Segmentation to separate images with particles and images with no particles 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from PIL import Image

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Seed setup ===
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# === Data directories ===
data_dir = r"C:\Users\n11555581\OneDrive - Queensland University of Technology\SEM images\MORPHOLOGY CYCLE POS S2\CODE MORPHOLOGY CYCLE\TRAINING SEGMENTATION BACKGROUND"

# === Transform: Resize only, no crop ===
transform_resize_only = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

# === Dataset and split ===
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_resize_only)

def print_class_distribution(dataset, name):
    class_counts = [0] * len(dataset.dataset.classes)
    for idx in range(len(dataset)):
        try:
            _, label = dataset[idx]
            class_counts[label] += 1
        except Exception as e:
            print(f"⚠️ Skipped image at index {idx}: {e}")

    print(f"{name} class distribution:")
    for class_name, count in zip(dataset.dataset.classes, class_counts):
        print(f"{class_name}: {count}")


train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print_class_distribution(train_dataset, "Train")
print_class_distribution(val_dataset, "Validation")

# === DataLoaders ===
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Load and modify VGG16 ===
model = models.vgg16(pretrained=True)

# Freeze all features
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier's final layer for binary classification
model.classifier[6] = nn.Linear(4096, 1)

# Send to GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Loss and optimizer ===
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# === Save function ===
def save_model(model, optimizer, epoch, loss, best_acc, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_acc': best_acc
    }, filename)

# === Training setup ===
num_epochs = 10
checkpoint_interval = 2
best_val_acc = 0.0
best_model_path = "vgg16_best_model_identifyimages.pth"
checkpoint_path = "vgg16_checkpoint_identifyimages.pth"

train_accuracies = []
val_accuracies = []

# === Training loop ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    all_train_preds, all_train_labels = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prob = torch.sigmoid(outputs)
        predicted = (prob > 0.5).long()
        total_train += labels.size(0)
        correct_train += (predicted == labels.long()).sum().item()
        all_train_preds.extend(predicted.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())
        running_loss += loss.item()

    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)
    train_precision = precision_score(all_train_labels, all_train_preds)
    train_recall = recall_score(all_train_labels, all_train_preds)
    train_f1 = f1_score(all_train_labels, all_train_preds)

    print(f"\nEpoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, Precision: {train_precision:.2f}, "
          f"Recall: {train_recall:.2f}, F1: {train_f1:.2f}")

    # === Validation ===
    model.eval()
    correct = 0
    total = 0
    all_val_preds, all_val_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            prob = torch.sigmoid(outputs)
            predicted = (prob > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    val_precision = precision_score(all_val_labels, all_val_preds)
    val_recall = recall_score(all_val_labels, all_val_preds)
    val_f1 = f1_score(all_val_labels, all_val_preds)

    print(f"Validation Acc: {val_accuracy:.2f}%, Precision: {val_precision:.2f}, "
          f"Recall: {val_recall:.2f}, F1: {val_f1:.2f}")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        print("Validation accuracy improved, saving best model...")
        save_model(model, optimizer, epoch, running_loss, best_val_acc, best_model_path)

    if (epoch + 1) % checkpoint_interval == 0:
        print(f"Saving checkpoint at epoch {epoch+1}...")
        save_model(model, optimizer, epoch, running_loss, best_val_acc, checkpoint_path)

# === Plotting accuracy curves ===
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
