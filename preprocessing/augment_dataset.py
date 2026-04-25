"""
Dataset Augmentation Script

This script generates a fixed number of augmented images per class folder.

Features:
- Controlled augmentation (prevents dataset explosion)
- Uses only original numeric images as base
- Maintains consistent naming scheme
- Handles numerical folders differently (starts from index 22)

Usage:
    python augment_dataset.py
    python augment_dataset.py --data dataset
"""

import os
import random
import argparse
import numpy as np
from PIL import Image, ImageFilter
import cv2

# ================= CONFIG =================
IMG_SIZE = (128, 128)
TOTAL_AUGMENTS_PER_FOLDER = 100
IMG_EXTENSION = ".png"
# ==========================================


# ---------------- IMAGE FUNCTIONS ---------------- #

def load_image(path):
    """Load and resize image to grayscale"""
    img = Image.open(path).convert("L")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return img


def random_rotation(img):
    return img.rotate(random.uniform(-15, 15), fillcolor=255)


def random_shift(img):
    dx, dy = random.randint(-5, 5), random.randint(-5, 5)
    np_img = np.array(img)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        np_img,
        M,
        (np_img.shape[1], np_img.shape[0]),
        borderValue=255
    )
    return Image.fromarray(shifted)


def random_noise(img):
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 5, np_img.shape)
    noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def random_blur(img):
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1)))
    return img


def augment_image(img):
    """Apply random augmentations with probability"""
    if random.random() < 0.7:
        img = random_rotation(img)

    if random.random() < 0.7:
        img = random_shift(img)

    if random.random() < 0.5:
        img = random_noise(img)

    if random.random() < 0.5:
        img = random_blur(img)

    return img


# ---------------- HELPER FUNCTIONS ---------------- #

def get_existing_max_number(folder):
    """Find the highest numeric filename in a folder"""
    max_num = -1
    for f in os.listdir(folder):
        name, _ = os.path.splitext(f)
        if name.isdigit():
            max_num = max(max_num, int(name))
    return max_num


def is_numerical_folder(name):
    """Check if folder is numerical (0–9)"""
    return name.isdigit() and 0 <= int(name) <= 9


# ---------------- MAIN LOGIC ---------------- #

def process_folder(dataset_dir, folder):
    folder_path = os.path.join(dataset_dir, folder)

    if not os.path.isdir(folder_path):
        return

    print(f"\nProcessing: {folder}")

    # Only use original numeric images
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and os.path.splitext(f)[0].isdigit()
    ]

    if not image_files:
        print("  No valid base images found. Skipping.")
        return

    images = [load_image(os.path.join(folder_path, f)) for f in image_files]

    # Determine starting index
    if is_numerical_folder(folder):
        current_index = 22
    else:
        current_index = get_existing_max_number(folder_path) + 1

    print(f"  Starting index: {current_index}")

    count = 0

    while count < TOTAL_AUGMENTS_PER_FOLDER:
        img = random.choice(images)
        aug_img = augment_image(img)

        save_path = os.path.join(folder_path, f"{current_index}{IMG_EXTENSION}")

        # Safety: avoid overwrite
        if os.path.exists(save_path):
            current_index += 1
            continue

        aug_img.save(save_path)

        current_index += 1
        count += 1

    print(f"  Generated {count} images.")


def main():
    parser = argparse.ArgumentParser(description="Dataset Augmentation Script")
    parser.add_argument(
        "--data",
        default="dataset",
        help="Path to dataset directory"
    )
    args = parser.parse_args()

    dataset_dir = args.data

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    print("\nStarting augmentation...\n")

    for folder in os.listdir(dataset_dir):
        process_folder(dataset_dir, folder)

    print("\nAugmentation complete!\n")


if __name__ == "__main__":
    main()