"""
visualise.py — research-paper style qualitative evaluation.

Produces a publication-quality figure with four columns per row:
    Input image | Ground truth | Predicted mask | Overlay

Usage:
    conda activate cuda_pt
    python visualise.py

Outputs:
    visualisations/qualitative_results.png   <- main figure
    visualisations/qualitative_results.pdf   <- vector version for LaTeX
"""

# %% Imports
import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import AttentionUNet

matplotlib.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 9,
    "axes.titlesize"  : 9,
    "axes.titleweight": "bold",
    "figure.dpi"      : 150,
})


# %% Config
DATASET_DIR     = "data_concrete_640"
MODEL_PATH      = "saved_models/best_attention_unet.pth"
OUTPUT_DIR      = "visualisations"
NUM_SAMPLES     = 8
IMAGE_SIZE      = 640
IN_CHANNELS     = 3
FEATURE_LIST    = [64, 128, 256, 512]
BOTTLENECK_SIZE = 1024

PRED_COLOR = (1.0, 0.2, 0.2, 0.45)   # semi-transparent red
GT_COLOR   = (0.2, 0.8, 0.2, 0.45)   # semi-transparent green

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# %% Helpers
def denormalise(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = AttentionUNet(
        in_channels     = IN_CHANNELS,
        feature_list    = FEATURE_LIST,
        bottleneck_size = BOTTLENECK_SIZE,
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def build_transforms():
    img_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.ToTensor(),
    ])
    return img_tf, mask_tf


def load_test_samples(dataset_dir: str, n: int):
    img_dir  = os.path.join(dataset_dir, "test", "images")
    mask_dir = os.path.join(dataset_dir, "test", "masks")

    images = sorted([f for f in os.listdir(img_dir)  if not f.startswith('.')])
    masks  = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
    assert len(images) == len(masks), "Image/mask count mismatch in test set"

    indices = sorted(np.random.choice(len(images), size=min(n, len(images)),
                                       replace=False))
    return [(Image.open(os.path.join(img_dir, images[i])).convert("RGB"),
             Image.open(os.path.join(mask_dir, masks[i])),
             images[i]) for i in indices]


@torch.no_grad()
def predict(model, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    logits = model(img_tensor.unsqueeze(0).to(device))
    prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def make_overlay(image_np, mask_np, color):
    overlay = image_np.copy()
    r, g, b, alpha = color
    for c, val in enumerate([r, g, b]):
        overlay[:, :, c] = np.where(
            mask_np == 1,
            overlay[:, :, c] * (1 - alpha) + val * alpha,
            overlay[:, :, c],
        )
    return np.clip(overlay, 0, 1)


# %% Main figure
def visualise(dataset_dir, model_path, output_dir, num_samples):
    os.makedirs(output_dir, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(model_path, device)
    img_tf, mask_tf = build_transforms()
    samples = load_test_samples(dataset_dir, num_samples)

    n_rows  = len(samples)
    pad_top = 0.35
    fig_w   = 4 * 2.4
    fig_h   = n_rows * 2.4 + pad_top

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    header_frac = pad_top / fig_h
    gs = fig.add_gridspec(
        n_rows, 4,
        top=1.0 - header_frac, bottom=0.03,
        left=0.01, right=0.99,
        hspace=0.06, wspace=0.04,
    )

    for col, label in enumerate(["Input image", "Ground truth",
                                   "Prediction", "Overlay"]):
        fig.text((col + 0.5) / 4, 1.0 - header_frac * 0.35,
                 label, ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    from torchmetrics.functional.classification import (
        binary_jaccard_index, binary_f1_score
    )

    for row, (pil_img, pil_mask, fname) in enumerate(samples):
        img_tensor  = img_tf(pil_img)
        mask_tensor = mask_tf(pil_mask)
        gt_mask     = (mask_tensor.squeeze().numpy() > 0.5).astype(np.uint8)
        pred_mask   = predict(model, img_tensor, device)
        image_np    = denormalise(img_tensor)

        pred_t = torch.from_numpy(pred_mask).float().unsqueeze(0)
        gt_t   = torch.from_numpy(gt_mask).float().unsqueeze(0)
        iou    = binary_jaccard_index(pred_t, gt_t).item()
        f1     = binary_f1_score(pred_t, gt_t).item()

        overlay = make_overlay(image_np, gt_mask,   GT_COLOR)
        overlay = make_overlay(overlay,  pred_mask, PRED_COLOR)

        panels = [
            (image_np, None),
            (image_np, gt_mask),
            (image_np, pred_mask),
            (overlay,  None),
        ]

        for col, (base, mask_to_show) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(base)

            if mask_to_show is not None:
                cmap = matplotlib.colors.ListedColormap(
                    ["none", "#00e676" if col == 1 else "#ff1744"]
                )
                ax.imshow(mask_to_show, cmap=cmap, vmin=0, vmax=1,
                          interpolation="nearest", alpha=0.6)

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(0.5)

            if col == 0:
                short = fname[:18] + "…" if len(fname) > 20 else fname
                ax.set_ylabel(short, fontsize=7, rotation=0,
                              labelpad=4, ha="right", va="center")
            if col == 3:
                ax.set_title(f"IoU {iou:.3f}  F1 {f1:.3f}",
                             fontsize=7.5, pad=3)

    fig.legend(
        handles=[
            mpatches.Patch(facecolor="#00e676", alpha=0.7, label="Ground truth"),
            mpatches.Patch(facecolor="#ff1744", alpha=0.7, label="Prediction"),
        ],
        loc="lower center", ncol=2, fontsize=8,
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.0),
    )

    png_path = os.path.join(output_dir, "qualitative_results.png")
    pdf_path = os.path.join(output_dir, "qualitative_results.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path,           bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}  (use this in LaTeX)")


# %% Entry point
if __name__ == "__main__":
    visualise(DATASET_DIR, MODEL_PATH, OUTPUT_DIR, NUM_SAMPLES)
