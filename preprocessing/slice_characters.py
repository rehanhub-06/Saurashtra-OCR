"""
Interactive Character Slicer

This script allows you to:
- Click on a handwritten image
- Extract character regions
- Save them as 128x128 images
- Organize automatically into class folders

Usage:
    python slice_characters.py
"""

import cv2
import os

# ================= CONFIG =================
IMAGE_PATH = "input.png"          # change this
OUTPUT_DIR = "dataset/consonants"
DISPLAY_SIZE = (800, 1000)
CROP_SIZE = 180                  # half-size of crop box
FINAL_SIZE = (128, 128)
# ==========================================

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError(f"Image not found: {IMAGE_PATH}")

clone = img.copy()

display = cv2.resize(img, DISPLAY_SIZE)

scale_x = img.shape[1] / display.shape[1]
scale_y = img.shape[0] / display.shape[0]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_next_index(folder):
    existing = [f for f in os.listdir(folder) if f.endswith('.png')]
    indices = []
    for f in existing:
        try:
            indices.append(int(os.path.splitext(f)[0]))
        except:
            pass
    return max(indices) + 1 if indices else 0

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)

        x1, y1 = max(0, orig_x - CROP_SIZE), max(0, orig_y - CROP_SIZE)
        x2, y2 = orig_x + CROP_SIZE, orig_y + CROP_SIZE

        crop = clone[y1:y2, x1:x2]
        crop = cv2.resize(crop, FINAL_SIZE)

        label = input("Enter label (folder name): ").strip()

        if not label:
            print("Skipped (empty label)")
            return

        folder = os.path.join(OUTPUT_DIR, label)
        os.makedirs(folder, exist_ok=True)

        idx = get_next_index(folder)
        save_path = os.path.join(folder, f"{idx}.png")

        cv2.imwrite(save_path, crop)
        print(f"Saved → {save_path}")

print("\nInstructions:")
print("- Click on a character to crop it")
print("- Enter label name in terminal")
print("- Press ESC to exit\n")

cv2.imshow("Character Slicer", display)
cv2.setMouseCallback("Character Slicer", click_event)

while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()