"""
============================================================
  CNN MODEL TRAINING SCRIPT
  Character Recognition — Handwritten Characters
============================================================

EXPECTED FOLDER STRUCTURE:
    dataset/
    ├── అ/
    │   ├── image1.png
    │   ├── image1_aug_000.png
    │   └── ...
    ├── ఆ/
    │   └── ...
    └── ...

Run augment_dataset.py FIRST before running this script.

HOW TO RUN:
    python train_model.py

OUTPUT FILES:
    character_model.h5      ← Trained model (used by predict.py)
    label_map.json          ← Maps class index → character name
    training_history.png    ← Accuracy/loss plot

REQUIREMENTS:
    pip install tensorflow pillow numpy matplotlib scikit-learn
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# CONFIGURATION
DATASET_DIR   = "dataset"
IMG_SIZE      = (128, 128)   
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 0.001
VAL_SPLIT     = 0.2
MODEL_OUT     = "character_model.h5"
LABEL_MAP_OUT = "label_map.json"


def load_dataset(dataset_dir):
    images = []
    labels = []
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not class_names:
        print(f"ERROR: No class folders found in '{dataset_dir}'")
        sys.exit(1)

    print(f"\nFound {len(class_names)} classes:\n  {', '.join(class_names)}\n")

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_dir, class_name)
        files = [
            f for f in os.listdir(class_folder)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        print(f"  [{class_idx:3d}] '{class_name}' — {len(files)} images")

        for filename in files:
            img_path = os.path.join(class_folder, filename)
            try:
                img = Image.open(img_path).convert("L")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"    [SKIP] {filename}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    images = images[..., np.newaxis]

    return images, labels, class_names


def build_model(num_classes):
    model = models.Sequential([

        layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.40),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")

    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("  CHARACTER RECOGNITION — CNN TRAINING")
    print("=" * 60)

    print("\n[1/5] Loading dataset...")
    images, labels, class_names = load_dataset(DATASET_DIR)

    label_map = {str(idx): name for idx, name in enumerate(class_names)}
    with open(LABEL_MAP_OUT, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("\n[3/5] Splitting train/validation/test...")

    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.1, random_state=42, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT, random_state=42, stratify=y_temp
    )

    print("\n[4/5] Building model...")
    model = build_model(len(class_names))

    callbacks = [
        ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    print("\n[5/5] Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    plot_history(history)


if __name__ == "__main__":
    main()