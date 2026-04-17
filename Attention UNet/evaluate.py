"""
evaluate.py — rigorous test set evaluation for binary crack segmentation.

Reports metrics separately for:
    - Crack-present images  (ground truth has at least one crack pixel)
    - Crack-absent images   (ground truth is all background)
    - Overall               (all test images combined)

Run after training:
    python evaluate.py
"""

# %% Imports
import os

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchmetrics.classification import (
    BinaryJaccardIndex, BinaryF1Score,
    BinaryPrecision, BinaryRecall
)
from tqdm import tqdm

from model import AttentionUNet


# %% Config
DATASET_DIR     = "data_concrete_640"
MODEL_PATH      = "saved_models/best_attention_unet.pth"
IMAGE_SIZE      = 640
IN_CHANNELS     = 3
FEATURE_LIST    = [64, 128, 256, 512]
BOTTLENECK_SIZE = 1024

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# %% Helpers
def build_transforms():
    img_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),
    ])
    return img_tf, mask_tf


def load_model(model_path, device):
    model = AttentionUNet(
        in_channels     = IN_CHANNELS,
        feature_list    = FEATURE_LIST,
        bottleneck_size = BOTTLENECK_SIZE,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def make_metric_set(device):
    return {
        "iou"       : BinaryJaccardIndex().to(device),
        "f1"        : BinaryF1Score().to(device),
        "precision" : BinaryPrecision().to(device),
        "recall"    : BinaryRecall().to(device),
    }


def compute_metrics(m):
    return {k: v.compute().item() for k, v in m.items()}


def print_metrics(label, m, n):
    print(f"\n  {label}  (n={n})")
    print(f"    IoU       : {m['iou']:.4f}")
    print(f"    F1 / Dice : {m['f1']:.4f}")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}")


# %% Evaluation
def evaluate(dataset_dir, model_path):
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model           = load_model(model_path, device)
    img_tf, mask_tf = build_transforms()

    img_dir  = os.path.join(dataset_dir, "test", "images")
    mask_dir = os.path.join(dataset_dir, "test", "masks")

    images = sorted([f for f in os.listdir(img_dir)  if not f.startswith('.')])
    masks  = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
    assert len(images) == len(masks), "Image/mask count mismatch in test set"

    m_all   = make_metric_set(device)
    m_crack = make_metric_set(device)

    n_all = n_crack = n_nocrack = 0
    fp_on_absent = 0

    for img_file, mask_file in tqdm(zip(images, masks), total=len(images),
                                     desc="Evaluating test set"):
        img  = Image.open(os.path.join(img_dir,  img_file)).convert("RGB")
        mask = Image.open(os.path.join(mask_dir, mask_file))

        img_t  = img_tf(img).unsqueeze(0).to(device)
        mask_t = mask_tf(mask).to(device)

        with torch.no_grad():
            logits = model(img_t)
            pred   = (torch.sigmoid(logits.squeeze(0)) > 0.5).float()

        gt_flat   = mask_t.squeeze()
        pred_flat = pred.squeeze()

        has_crack = gt_flat.sum().item() > 0

        for metric in m_all.values():
            metric.update(pred_flat.unsqueeze(0), gt_flat.unsqueeze(0))
        n_all += 1

        if has_crack:
            for metric in m_crack.values():
                metric.update(pred_flat.unsqueeze(0), gt_flat.unsqueeze(0))
            n_crack += 1
        else:
            if pred_flat.sum().item() > 0:
                fp_on_absent += 1
            n_nocrack += 1

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  TEST SET EVALUATION RESULTS")
    print("=" * 58)

    print_metrics("Crack-present images  <- primary metric",
                  compute_metrics(m_crack), n_crack)

    fpr = fp_on_absent / max(n_nocrack, 1) * 100
    print(f"\n  Crack-absent images  (n={n_nocrack})")
    print(f"    False positive rate : {fp_on_absent}/{n_nocrack}  ({fpr:.1f}%)")
    print(f"    True negative rate  : {100-fpr:.1f}%")

    print_metrics("Overall (all images)  <- suppressed by noncrack 0/0",
                  compute_metrics(m_all), n_all)

    print("\n" + "=" * 58)
    print("  Report crack-present IoU as the primary metric.")
    print("  Overall IoU is suppressed by crack-absent 0/0 scores.")
    print("=" * 58 + "\n")


# %% Entry point
if __name__ == "__main__":
    evaluate(DATASET_DIR, MODEL_PATH)
