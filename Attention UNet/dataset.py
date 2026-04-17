# %% Imports
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# %% Transform helpers
# Named functions at module level — required for pickling with num_workers > 0

def mask_to_binary(x: torch.Tensor) -> torch.Tensor:
    """Snap all float mask values to exactly 0.0 or 1.0 after ToTensor."""
    return (x > 0.5).float()


def get_transforms(image_size: int, augment: bool, pre_resized: bool = False):
    """
    Returns (img_transform, mask_transform) for a given split.

    Args:
        image_size  : target spatial size (ignored if pre_resized=True)
        augment     : True for training split, False for val/test
        pre_resized : set True when images are already resized on disk
    """
    img_ops  = []
    mask_ops = []

    if not pre_resized:
        img_ops.append(transforms.Resize((image_size, image_size)))
        mask_ops.append(
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
        )

    if augment:
        img_ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomGrayscale(p=0.1),
        ]
        mask_ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(
                degrees=90,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=0,
            ),
        ]

    img_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    mask_ops += [
        transforms.ToTensor(),
        transforms.Lambda(mask_to_binary),
    ]

    return transforms.Compose(img_ops), transforms.Compose(mask_ops)


# %% SegDataset
class SegDataset(Dataset):
    """
    Paired image / mask segmentation dataset.

    Directory layout:
        root/
            images/   <- RGB images
            masks/    <- single-channel PNG masks (0=background, >0=foreground)

    Spatial augmentations are applied identically to image and mask via a
    shared random seed before each transform call.
    """

    def __init__(self, image_dir: str, mask_dir: str,
                 img_transform=None, mask_transform=None):
        self.image_dir      = image_dir
        self.mask_dir       = mask_dir
        self.img_transform  = img_transform
        self.mask_transform = mask_transform

        self.images = sorted([f for f in os.listdir(image_dir)
                               if not f.startswith('.')])
        self.masks  = sorted([f for f in os.listdir(mask_dir)
                               if not f.startswith('.')])

        assert len(self.images) == len(self.masks), (
            f"Image/mask count mismatch: "
            f"{len(self.images)} images vs {len(self.masks)} masks"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(
            os.path.join(self.image_dir, self.images[index])
        ).convert("RGB")
        mask = Image.open(
            os.path.join(self.mask_dir, self.masks[index])
        )

        seed = np.random.randint(2147483647)

        if self.img_transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.img_transform(image)

        if self.mask_transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        if isinstance(mask, Image.Image):
            mask = transforms.ToTensor()(mask)

        return image, mask


# %% DataLoader creation
def create_dataloaders(dataset_dir: str,
                       batch_size: int,
                       image_size: int,
                       num_workers: int = 4,
                       pre_resized: bool = False) -> tuple:
    """
    Builds train / val / test DataLoaders from:
        dataset_dir/train/images  dataset_dir/train/masks
        dataset_dir/val/images    dataset_dir/val/masks
        dataset_dir/test/images   dataset_dir/test/masks
    """
    train_img_tf, train_mask_tf = get_transforms(image_size, augment=True,
                                                  pre_resized=pre_resized)
    val_img_tf,   val_mask_tf   = get_transforms(image_size, augment=False,
                                                  pre_resized=pre_resized)

    train_dataset = SegDataset(
        image_dir      = os.path.join(dataset_dir, "train", "images"),
        mask_dir       = os.path.join(dataset_dir, "train", "masks"),
        img_transform  = train_img_tf,
        mask_transform = train_mask_tf,
    )
    val_dataset = SegDataset(
        image_dir      = os.path.join(dataset_dir, "val", "images"),
        mask_dir       = os.path.join(dataset_dir, "val", "masks"),
        img_transform  = val_img_tf,
        mask_transform = val_mask_tf,
    )
    test_dataset = SegDataset(
        image_dir      = os.path.join(dataset_dir, "test", "images"),
        mask_dir       = os.path.join(dataset_dir, "test", "masks"),
        img_transform  = val_img_tf,
        mask_transform = val_mask_tf,
    )

    loader_kwargs = dict(
        batch_size         = batch_size,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = (num_workers > 0),
        prefetch_factor    = 2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# %% Class weight calculation
def get_automated_weights(mask_dir: str,
                           cache_path: str = "weights_cache.pt") -> torch.Tensor:
    """
    Computes pos_weight = background_pixels / crack_pixels for BCEWithLogitsLoss.
    Cached to disk — delete cache if your dataset changes.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached weights from {cache_path}...")
        return torch.load(cache_path, weights_only=True)

    print(f"Calculating weights from: {mask_dir}  (runs once, then caches)...")

    counts = np.zeros(2)   # [background, crack]

    mask_files = [f for f in os.listdir(mask_dir) if not f.startswith('.')]
    for f in mask_files:
        mask = np.array(Image.open(os.path.join(mask_dir, f)))
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        unique, pixel_counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique, pixel_counts):
            if val < 2:
                counts[int(val)] += count

    pos_weight = counts[0] / (counts[1] + 1e-6)
    weight     = torch.tensor([pos_weight])

    torch.save(weight, cache_path)
    print(f"pos_weight: {pos_weight:.2f}  -> cached to {cache_path}")
    return weight


# %% One-time preprocessing
def preprocess_dataset(dataset_dir: str, image_size: int):
    """
    Resizes all images and masks once and saves to a new folder.
    Run once, then set DATASET_DIR to the new folder and PRE_RESIZED = True.
    Uses BILINEAR for images, NEAREST for masks.
    """
    splits = ['train', 'val', 'test']

    for split in splits:
        for kind in ['images', 'masks']:
            src_dir = os.path.join(dataset_dir, split, kind)
            dst_dir = os.path.join(f"{dataset_dir}_{image_size}", split, kind)

            if not os.path.exists(src_dir):
                print(f"Skipping {src_dir} (not found)")
                continue

            os.makedirs(dst_dir, exist_ok=True)
            files = [f for f in os.listdir(src_dir) if not f.startswith('.')]
            print(f"Processing {split}/{kind}: {len(files)} files...")

            for f in files:
                if kind == "images":
                    img = Image.open(os.path.join(src_dir, f)).convert("RGB")
                    img = img.resize((image_size, image_size), Image.BILINEAR)
                else:
                    img = Image.open(os.path.join(src_dir, f))
                    img = img.resize((image_size, image_size), Image.NEAREST)
                img.save(os.path.join(dst_dir, f))

    print(f"\nDone. New dataset at: {dataset_dir}_{image_size}/")
    print(f"Set DATASET_DIR = \"{dataset_dir}_{image_size}\" and PRE_RESIZED = True")


if __name__ == "__main__":
    preprocess_dataset("data_concrete", image_size=640)
