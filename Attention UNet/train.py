"""
train.py — Attention U-Net binary crack segmentation training.

Usage:
    conda activate cuda_pt
    python train.py

The if __name__ == "__main__" guard is required for num_workers > 0 on Windows.
"""

# %% Imports
import copy
import csv
import os
import time

import torch
import torch.backends.cudnn
from torch.amp import GradScaler, autocast
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex
)
from tqdm import tqdm

from dataset import create_dataloaders, get_automated_weights
from loss import HybridLoss
from model import AttentionUNet


# %% Config
DATASET_DIR     = "data_640_512"
PRE_RESIZED     = True
SAVE_DIR        = "saved_models"
WEIGHTS_CACHE   = "weights_cache.pt"

LEARNING_RATE   = 1e-4
NUM_EPOCHS      = 100
IN_CHANNELS     = 3
FEATURE_LIST    = [64, 128, 256, 512]
BOTTLENECK_SIZE = 1024
IMAGE_SIZE      = 480
BATCH_SIZE      = 4
NUM_WORKERS     = 4


# %% Training function
def train(config: dict) -> torch.nn.Module:

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        # FIX: benchmark=False eliminates cuDNN's per-session algorithm search.
        # benchmark=True is faster per-batch but causes a ~10min hang on first
        # run of any new architecture. With our pre-sized dataset the default
        # algorithm is fast enough — no measurable difference in epoch time.
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

    os.makedirs(config["save_dir"], exist_ok=True)

    # ── Weights ───────────────────────────────────────────────────────────────
    train_mask_dir = os.path.join(config["dataset_dir"], "train", "masks")
    pos_weight = get_automated_weights(
        train_mask_dir,
        cache_path=config["weights_cache"],
    )

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir = config["dataset_dir"],
        batch_size  = config["batch_size"],
        image_size  = config["image_size"],
        num_workers = config["num_workers"],
        pre_resized = config["pre_resized"],
    )
    print(f"Train batches: {len(train_loader)} | "
          f"Val batches: {len(val_loader)} | "
          f"Test batches: {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AttentionUNet(
        in_channels     = config["in_channels"],
        feature_list    = config["feature_list"],
        bottleneck_size = config["bottleneck_size"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    # ── Optimizer / loss / scheduler ──────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn   = HybridLoss(alpha=0.5, pos_weight=pos_weight).to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6
    )
    scaler = GradScaler("cuda")

    # ── Metrics ───────────────────────────────────────────────────────────────
    train_acc_metric = BinaryAccuracy().to(device)
    val_acc_metric   = BinaryAccuracy().to(device)
    iou_metric       = BinaryJaccardIndex().to(device)
    f1_metric        = BinaryF1Score().to(device)

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = os.path.join(config["save_dir"], "metrics_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "val_iou", "val_f1",
            "lr", "epoch_time_s",
        ])

    best_val_iou     = 0.0
    best_epoch       = 0
    best_model_state = None
    training_start   = time.time()

    # ── Pre-warm DataLoader workers ───────────────────────────────────────────
    print("Spawning DataLoader workers...")
    _train_it = iter(train_loader); next(_train_it); del _train_it
    _val_it   = iter(val_loader);   next(_val_it);   del _val_it
    torch.cuda.empty_cache()
    print("Workers ready — starting training.")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(config["num_epochs"]):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_train_loss = 0.0
        train_acc_metric.reset()

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config['num_epochs']} [TRAIN]",
            leave=False,
        )

        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model(images)
                loss    = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_acc_metric.update(preds, masks)

            running_train_loss += loss.item() * images.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc  = train_acc_metric.compute()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        running_val_loss = 0.0
        val_acc_metric.reset()
        iou_metric.reset()
        f1_metric.reset()

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{config['num_epochs']} [VAL]",
            leave=False,
        )

        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)

                with autocast("cuda"):
                    outputs = model(images)
                    loss    = loss_fn(outputs, masks)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_acc_metric.update(preds, masks)
                iou_metric.update(preds, masks)
                f1_metric.update(preds, masks)

                running_val_loss += loss.item() * images.size(0)
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc  = val_acc_metric.compute()
        epoch_val_iou  = iou_metric.compute()
        epoch_val_f1   = f1_metric.compute()

        scheduler.step()

        # ── Timing + ETA ──────────────────────────────────────────────────────
        epoch_time    = time.time() - epoch_start
        elapsed_total = time.time() - training_start
        epochs_done   = epoch + 1
        epochs_left   = config["num_epochs"] - epochs_done
        eta_seconds   = (elapsed_total / epochs_done) * epochs_left
        eta_h         = int(eta_seconds // 3600)
        eta_m         = int((eta_seconds % 3600) // 60)
        current_lr    = optimizer.param_groups[0]["lr"]

        # ── Print ─────────────────────────────────────────────────────────────
        print(
            f"Epoch {epochs_done:>3}/{config['num_epochs']} | "
            f"Train Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.4f}  "
            f"IoU: {epoch_val_iou:.4f}  F1: {epoch_val_f1:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Epoch: {epoch_time/60:.1f}min | "
            f"ETA: {eta_h}h {eta_m}m"
        )

        # ── CSV log ───────────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epochs_done,
                f"{epoch_train_loss:.4f}", f"{epoch_train_acc:.4f}",
                f"{epoch_val_loss:.4f}",   f"{epoch_val_acc:.4f}",
                f"{epoch_val_iou:.4f}",    f"{epoch_val_f1:.4f}",
                f"{current_lr:.2e}",
                f"{epoch_time:.1f}",
            ])

        # ── Checkpoint on val IoU ─────────────────────────────────────────────
        if epoch_val_iou > best_val_iou:
            best_val_iou     = epoch_val_iou
            best_epoch       = epochs_done
            best_model_state = copy.deepcopy(model.state_dict())
            save_path        = os.path.join(config["save_dir"],
                                            "best_attention_unet.pth")
            torch.save(best_model_state, save_path)
            print(f"    ^ New best model  "
                  f"(epoch {best_epoch}, val iou {best_val_iou:.4f})"
                  f"  -> {save_path}")

    # ── End of training ───────────────────────────────────────────────────────
    total_time = time.time() - training_start
    print(f"\n--- Training complete ---")
    print(f"Total time : {total_time/3600:.2f}h")
    print(f"Best model : epoch {best_epoch}, val iou {best_val_iou:.4f}")
    print(f"Log saved  : {log_path}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: val IoU never improved. Returning last model.")

    return model


# %% Entry point
if __name__ == "__main__":

    config = dict(
        dataset_dir     = DATASET_DIR,
        pre_resized     = PRE_RESIZED,
        save_dir        = SAVE_DIR,
        weights_cache   = WEIGHTS_CACHE,
        learning_rate   = LEARNING_RATE,
        num_epochs      = NUM_EPOCHS,
        in_channels     = IN_CHANNELS,
        feature_list    = FEATURE_LIST,
        bottleneck_size = BOTTLENECK_SIZE,
        image_size      = IMAGE_SIZE,
        batch_size      = BATCH_SIZE,
        num_workers     = NUM_WORKERS,
    )

    trained_model = train(config)
