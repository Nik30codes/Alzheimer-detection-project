"""
PyTorch Dataset for the processed axial MRI slices.

Keras equivalent for reference (since that's what the old notebooks used):
this replaces ImageDataGenerator.flow_from_dataframe(). Same idea -- read a
manifest of filepaths, load images, apply augmentation to training data only.
"""

import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

CLASSES = ["CN", "AD", "EMCI", "LMCI"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Training augmentation: mild rotation/zoom/shift only -- MRI anatomy shouldn't be
# distorted aggressively, we just want the model to not memorize exact pixel positions.
TRAIN_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),  # scales 0-255 uint8 -> 0-1 float32
    T.Normalize(mean=[0.5], std=[0.5]),  # 0-1 -> -1..1, same range the old MobileNet notebook used
])

# Validation/test: no augmentation, just the same normalization.
EVAL_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# MobileNetV2/EfficientNet-B0 (Phase D) were pretrained on 3-channel ImageNet images with
# ImageNet's own normalization stats -- different from the 1-channel setup above, which was
# fine for a from-scratch model but would be wrong for pretrained weights. Grayscale(3)
# duplicates our single channel into 3 identical channels rather than actually adding color
# information, which is the standard trick for feeding grayscale medical images into
# RGB-pretrained networks.
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TRAIN_TRANSFORM_RGB = T.Compose([
    T.ToPILImage(),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.RandomHorizontalFlip(p=0.5),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

EVAL_TRANSFORM_RGB = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# --- 2.5D transforms -------------------------------------------------------
# Deliberately WITHOUT T.Grayscale(3): that op is what turns one slice into three
# identical channels, which is exactly what 2.5D is replacing. Feeding a genuine
# 3-slice stack through it would collapse the volumetric information back to a
# single slice and silently undo the whole idea.
# Normalization is the plain 0.5/0.5 used by the from-scratch models rather than
# ImageNet's per-channel constants -- with three anatomically different slices per
# sample, per-channel ImageNet statistics have no meaning here.
TRAIN_TRANSFORM_25D = T.Compose([
    T.ToPILImage(),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

EVAL_TRANSFORM_25D = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


class MRIDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, split: str, transform):
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"], cv2.IMREAD_GRAYSCALE)  # (224, 224) uint8
        if img is None:
            raise FileNotFoundError(row["filepath"])

        img = self.transform(img)  # -> (1, 224, 224) float32 tensor
        label = CLASS_TO_IDX[row["class"]]
        return img, label, row["subject_id"]


class MRI25DDataset(Dataset):
    """
    "2.5D" input: instead of one axial slice duplicated into 3 identical channels,
    each sample stacks THREE ADJACENT axial slices (depth-1, depth, depth+1) as the
    three channels of one image.

    Why this should help, given the models here are already 3-channel: the standard
    grayscale->fake-RGB trick feeds the network three copies of the same picture, so
    two thirds of the input carries no information. Stacking neighbours instead costs
    nothing (same tensor shape, same architecture, same parameter count) but gives the
    network real volumetric context. That matters for this task specifically -- a dark
    region on a single slice can be noise or a partial-volume artifact, whereas genuine
    hippocampal atrophy persists across consecutive slices. A 2D model literally cannot
    tell those apart; this one can.

    Slice files are named {subject}_{000..031}.png, so neighbours are found by index.
    At the top and bottom of the stack the missing neighbour is clamped to the edge
    slice (so the first slice becomes [0,0,1]) rather than zero-padded -- a black
    channel would be a strong artificial edge the model could key on.
    """

    def __init__(self, manifest: pd.DataFrame, split: str, transform):
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        self.transform = transform
        # index every available slice per subject so neighbours can be looked up
        self._by_subject = {}
        for subj, group in manifest.groupby("subject_id"):
            paths = sorted(group["filepath"], key=self._slice_index)
            self._by_subject[subj] = paths

    @staticmethod
    def _slice_index(path: str) -> int:
        return int(str(path).rsplit("_", 1)[-1].split(".")[0])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        siblings = self._by_subject[row["subject_id"]]
        pos = siblings.index(row["filepath"])

        channels = []
        for offset in (-1, 0, 1):
            neighbour = min(max(pos + offset, 0), len(siblings) - 1)  # clamp at the ends
            img = cv2.imread(siblings[neighbour], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(siblings[neighbour])
            channels.append(img)

        stacked = np.stack(channels, axis=-1)  # (224, 224, 3) uint8
        # One transform call on the stacked array, so RandomAffine/flip apply the SAME
        # geometric transform to all three slices -- augmenting them independently would
        # misalign the anatomy across channels and destroy the depth relationship.
        img = self.transform(stacked)
        label = CLASS_TO_IDX[row["class"]]
        return img, label, row["subject_id"]


def _slice_index(path: str) -> int:
    """Slice files are named {subject}_{000..031}.png -- pulls the trailing index."""
    return int(str(path).rsplit("_", 1)[-1].split(".")[0])


class VolumeTransform:
    """Callable transform for a whole (D, H, W) uint8 slice stack, applied as ONE
    consistent geometric transform across every slice -- not per-slice torchvision
    ops, which assume a single (H, W) or (H, W, C) image and would need one call per
    slice (and, per MRI25DDataset's docstring, applying augmentation independently
    per slice would misalign anatomy across depth and destroy the very cross-slice
    structure Simple3DCNN exists to use).

    Matches TRAIN_TRANSFORM's augmentation ranges (degrees=10, translate=0.05,
    scale=0.95-1.05, hflip p=0.5) and the 0.5/0.5 single-channel normalization used
    by the from-scratch 2D models, so the 3D result differs only in architecture.
    """

    def __init__(self, augment: bool):
        self.augment = augment

    def __call__(self, volume: np.ndarray) -> torch.Tensor:
        d, h, w = volume.shape
        if self.augment:
            angle = np.random.uniform(-10, 10)
            scale = np.random.uniform(0.95, 1.05)
            tx = np.random.uniform(-0.05, 0.05) * w
            ty = np.random.uniform(-0.05, 0.05) * h
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            matrix[0, 2] += tx
            matrix[1, 2] += ty
            flip = np.random.rand() < 0.5
            out = np.empty_like(volume)
            for i in range(d):
                sl = cv2.warpAffine(volume[i], matrix, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                out[i] = np.fliplr(sl) if flip else sl
            volume = out

        vol = volume.astype(np.float32) / 255.0
        vol = (vol - 0.5) / 0.5  # -> [-1, 1], same range as the 2D 0.5/0.5 normalization
        return torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)


VOLUME_TRAIN_TRANSFORM = VolumeTransform(augment=True)
VOLUME_EVAL_TRANSFORM = VolumeTransform(augment=False)


class MRI3DDataset(Dataset):
    """One sample = one SUBJECT's full 32-slice axial stack assembled into a
    (1, 32, 224, 224) volume, for Conv3d models (src.models.Simple3DCNN) -- unlike
    every other Dataset in this module, one row here is a subject, not a slice.

    Why this exists: decision 30 in CLAUDE.md found the 32 per-slice predictions
    scored independently and averaged (what MRIDataset feeds every other model here)
    are only ~1.3 effective independent measurements -- post-hoc averaging cannot
    recover cross-slice structure a true 3D convolution can see directly.

    classes lets this serve either the 4-way task (CLASSES) or a binary task
    (pass classes=["CN", "AD"]) without a subclass -- unlike MRIDataset's binary
    variants elsewhere (BinaryDataset/BinaryDS), which override __getitem__ instead.
    Defined at module level, not nested, so DataLoader worker processes can pickle it.
    """

    def __init__(self, manifest: pd.DataFrame, split: str, transform, classes=None):
        self.classes = classes if classes is not None else CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        df = manifest[manifest["split"] == split]
        df = df[df["class"].isin(self.classes)]
        self.transform = transform
        self.subjects = sorted(df["subject_id"].unique())
        self._paths, self._label = {}, {}
        for subj, group in df.groupby("subject_id"):
            self._paths[subj] = sorted(group["filepath"], key=_slice_index)
            self._label[subj] = group["class"].iloc[0]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        slices = []
        for p in self._paths[subj]:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(p)
            slices.append(img)
        volume = np.stack(slices, axis=0)  # (32, 224, 224) uint8
        volume = self.transform(volume)    # -> (1, 32, 224, 224) float32 tensor
        label = self.class_to_idx[self._label[subj]]
        return volume, label, subj


def build_dataloaders_3d(manifest: pd.DataFrame, batch_size: int = 4, num_workers: int = 2,
                         classes=None):
    """DataLoaders of whole-subject volumes for Simple3DCNN. batch_size defaults much
    lower than build_dataloaders' 32 -- each sample is a full (1, 32, 224, 224) volume
    instead of one (1, 224, 224) slice, on a 6GB GPU."""
    train_ds = MRI3DDataset(manifest, "train", VOLUME_TRAIN_TRANSFORM, classes=classes)
    val_ds = MRI3DDataset(manifest, "val", VOLUME_EVAL_TRANSFORM, classes=classes)
    test_ds = MRI3DDataset(manifest, "test", VOLUME_EVAL_TRANSFORM, classes=classes)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def build_dataloaders_25d(manifest: pd.DataFrame, batch_size: int = 32, num_workers: int = 2):
    """DataLoaders using the 2.5D 3-adjacent-slice stacking. Drop-in replacement for
    build_dataloaders(rgb=True) -- same shapes, same batch layout."""
    train_ds = MRI25DDataset(manifest, "train", TRAIN_TRANSFORM_25D)
    val_ds = MRI25DDataset(manifest, "val", EVAL_TRANSFORM_25D)
    test_ds = MRI25DDataset(manifest, "test", EVAL_TRANSFORM_25D)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def build_dataloaders(manifest: pd.DataFrame, batch_size: int = 32, num_workers: int = 2, rgb: bool = False):
    """One DataLoader per split. Train shuffles + augments; val/test don't.
    rgb=True switches to the 3-channel ImageNet-normalized transforms needed by
    the pretrained MobileNetV2/EfficientNet-B0 models (Phase D); rgb=False (default)
    keeps the 1-channel transforms the from-scratch SimpleCNN (Phase C) uses."""
    train_t, eval_t = (TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB) if rgb else (TRAIN_TRANSFORM, EVAL_TRANSFORM)
    train_ds = MRIDataset(manifest, "train", train_t)
    val_ds = MRIDataset(manifest, "val", eval_t)
    test_ds = MRIDataset(manifest, "test", eval_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def compute_class_weights(manifest: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency weights from the TRAIN split, for the loss function to
    pay more attention to the smaller classes (AD has the fewest images)."""
    train_df = manifest[manifest["split"] == "train"]
    counts = train_df["class"].value_counts()
    weights = [len(train_df) / counts[c] for c in CLASSES]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights / weights.sum() * len(CLASSES)  # normalize so weights average to 1
