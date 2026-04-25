"""
============================================================
  CHARACTER PREDICTION APP
  Draw a character in Paint → load image → see prediction
============================================================

HOW TO USE:
  1. Open MS Paint
  2. Set canvas to 200x200 pixels (or any square size)
  3. Draw your character with a thick black brush on WHITE background
  4. Save the image (File → Save As → PNG)
  5. Run this script:
       python predict.py
  6. A window will open → click "Load Image" → select your PNG
  7. The prediction and confidence will be shown on screen

REQUIREMENTS:
    pip install tensorflow pillow numpy
    (tkinter is built into Python — no extra install needed)

FILES NEEDED (from training step):
    character_model.h5
    label_map.json
"""

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, font as tkfont
import numpy as np
from PIL import Image, ImageTk, ImageOps

# Suppress TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

MODEL_PATH     = "character_model.h5"
LABEL_MAP_PATH = "label_map.json"
IMG_SIZE       = (128, 128)
TOP_N          = 3               # Show top-N predictions

# ─────────────────────────────────────────────────────────────


def load_model_and_labels():
    """Load the trained Keras model and label map."""
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found.\n"
            "Please run train_model.py first."
        )
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"Label map '{LABEL_MAP_PATH}' not found.\n"
            "Please run train_model.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)   # {"0": "అ", "1": "ఆ", ...}

    return model, label_map


def preprocess_image(img_path):
    """
    Load and preprocess an image for model inference.
    - Converts to grayscale
    - Inverts if background is dark (model trained on white background)
    - Resizes to IMG_SIZE
    - Normalizes to [0, 1]
    - Returns numpy array of shape (1, H, W, 1)
    """
    img = Image.open(img_path).convert("L")   # grayscale

    # Auto-invert: if the image is mostly dark, invert it
    # (model trained on white background + dark ink)
    img_arr = np.array(img)
    mean_val = img_arr.mean()
    if mean_val < 128:
        img = ImageOps.invert(img)

    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = img_arr[np.newaxis, ..., np.newaxis]   # (1, H, W, 1)
    return img_arr


def predict(model, label_map, img_path):
    """
    Run inference on an image.
    Returns list of (character, confidence%) tuples, sorted by confidence desc.
    """
    img_arr = preprocess_image(img_path)
    probs   = model.predict(img_arr, verbose=0)[0]   # shape: (num_classes,)
    top_indices = np.argsort(probs)[::-1][:TOP_N]
    
    results = [
        (label_map[str(i)], float(probs[i]) * 100)
        for i in top_indices
    ]
    print("Top predictions:", results)
    return results


# ─────────────────────────────────────────────────────────────
# GUI APPLICATION
# ─────────────────────────────────────────────────────────────

class PredictionApp:
    """Main prediction GUI window."""

    def __init__(self, root, model, label_map):
        self.root       = root
        self.model      = model
        self.label_map  = label_map
        self.current_image_path = None

        self.root.title("Character Recognition — CNN Predictor")
        self.root.geometry("680x600")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self._build_ui()

    def _build_ui(self):
        """Build all UI elements."""
        bg   = "#1e1e2e"
        card = "#2a2a3e"
        fg   = "#cdd6f4"
        acc  = "#89b4fa"   # blue accent
        grn  = "#a6e3a1"   # green for top prediction
        mute = "#6c7086"

        # Title
        title_font = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        tk.Label(self.root, text="✍  Handwritten Character Recognizer",
                 font=title_font, bg=bg, fg=acc).pack(pady=(20, 5))
        tk.Label(self.root,
                 text="Draw your character in Paint, save as PNG, then load it here.",
                 font=("Segoe UI", 10), bg=bg, fg=mute).pack(pady=(0, 16))

        # Image preview frame
        preview_frame = tk.Frame(self.root, bg=card, bd=0,
                                 highlightthickness=2,
                                 highlightbackground=acc)
        preview_frame.pack(pady=6)

        self.img_label = tk.Label(
            preview_frame,
            text="No image loaded",
            bg=card, fg=mute,
            font=("Segoe UI", 11),
            width=28, height=10
        )
        self.img_label.pack(padx=16, pady=16)

        # Load button
        btn_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        load_btn = tk.Button(
            self.root,
            text="📂  Load Image from Paint",
            font=btn_font,
            bg=acc, fg="#1e1e2e",
            activebackground="#74c7ec",
            relief="flat",
            cursor="hand2",
            padx=20, pady=10,
            command=self.load_and_predict
        )
        load_btn.pack(pady=12)

        # ── Results section ──────────────────────────────────
        results_frame = tk.Frame(self.root, bg=bg)
        results_frame.pack(fill="x", padx=40)

        tk.Label(results_frame, text="PREDICTION RESULTS",
                 font=("Segoe UI", 9, "bold"),
                 bg=bg, fg=mute).pack(anchor="w")

        tk.Frame(results_frame, bg=mute, height=1).pack(fill="x", pady=(2, 10))

        # Top prediction (big)
        self.top_pred_var = tk.StringVar(value="—")
        self.top_conf_var = tk.StringVar(value="")

        top_frame = tk.Frame(results_frame, bg=card, pady=14, padx=20)
        top_frame.pack(fill="x", pady=(0, 8))

        tk.Label(top_frame, text="Best Match",
                 font=("Segoe UI", 9), bg=card, fg=mute).pack(anchor="w")

        char_frame = tk.Frame(top_frame, bg=card)
        char_frame.pack(anchor="w", fill="x")

        self.char_display = tk.Label(
            char_frame,
            textvariable=self.top_pred_var,
            font=("Noto Sans Telugu", 48, "bold"),
            bg=card, fg=grn
        )
        self.char_display.pack(side="left", padx=(0, 16))

        self.conf_display = tk.Label(
            char_frame,
            textvariable=self.top_conf_var,
            font=("Segoe UI", 22),
            bg=card, fg=fg
        )
        self.conf_display.pack(side="left", anchor="s", pady=(0, 10))

        # Other candidates
        tk.Label(results_frame, text="Other Candidates",
                 font=("Segoe UI", 9), bg=bg, fg=mute).pack(anchor="w", pady=(4, 4))

        self.candidate_labels = []
        for i in range(1, TOP_N):
            row = tk.Frame(results_frame, bg=card, pady=8, padx=16)
            row.pack(fill="x", pady=3)
            lbl = tk.Label(row, text="—", font=("Noto Sans Telugu", 18),
                           bg=card, fg=fg, anchor="w")
            lbl.pack(side="left")
            conf_lbl = tk.Label(row, text="", font=("Segoe UI", 12),
                                bg=card, fg=mute, anchor="e")
            conf_lbl.pack(side="right")
            self.candidate_labels.append((lbl, conf_lbl))

        # Status bar
        self.status_var = tk.StringVar(value="Ready — load an image to predict.")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Segoe UI", 9), bg=bg, fg=mute).pack(side="bottom", pady=8)

    def load_and_predict(self):
        """Open file dialog, load image, and run prediction."""
        path = filedialog.askopenfilename(
            title="Select your handwritten character image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        self.current_image_path = path
        self.status_var.set(f"Loaded: {os.path.basename(path)}")

        # Show preview (zoom to 160x160 for display)
        try:
            pil_img = Image.open(path).convert("RGB")
            pil_img = pil_img.resize((160, 160), Image.LANCZOS)
            tk_img  = ImageTk.PhotoImage(pil_img)
            self.img_label.configure(image=tk_img, text="", width=160, height=160)
            self.img_label.image = tk_img   # hold reference
        except Exception as e:
            self.status_var.set(f"Preview failed: {e}")

        # Run prediction
        try:
            results = predict(self.model, self.label_map, path)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        # Display top result
        top_char, top_conf = results[0]
        self.top_pred_var.set(top_char)
        self.top_conf_var.set(f"{top_conf:.1f}%  confidence")

        # Colour-code confidence
        if top_conf >= 80:
            color = "#a6e3a1"   # green
        elif top_conf >= 50:
            color = "#f9e2af"   # yellow
        else:
            color = "#f38ba8"   # red

        self.char_display.configure(fg=color)

        # Display other candidates
        for i, (lbl, conf_lbl) in enumerate(self.candidate_labels):
            if i + 1 < len(results):
                ch, cn = results[i + 1]
                lbl.configure(text=ch)
                conf_lbl.configure(text=f"{cn:.1f}%")
            else:
                lbl.configure(text="—")
                conf_lbl.configure(text="")

        self.status_var.set(
            f"Predicted: '{top_char}'  with {top_conf:.1f}% confidence"
        )


def main():
    print("Loading model and labels...")
    try:
        model, label_map = load_model_and_labels()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"Model loaded. Classes: {len(label_map)}")
    print("Opening prediction window...\n")

    root = tk.Tk()
    app  = PredictionApp(root, model, label_map)
    root.mainloop()


if __name__ == "__main__":
    main()
