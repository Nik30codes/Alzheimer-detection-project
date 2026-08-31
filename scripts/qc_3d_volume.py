"""
Smoke test for MRI3DDataset + Simple3DCNN, BEFORE any real training run.

Checks, cheaply, in order:
  1. Volume assembly is correct: an assembled (32, 224, 224) volume's slices match the
     source PNGs pixel-for-pixel (eval transform, no augmentation).
  2. VolumeTransform augmentation applies the SAME geometric warp to every slice in a
     volume (a rotated volume's slices should all be rotated by the identical amount --
     if it isn't, augmentation would misalign anatomy across depth and silently destroy
     the cross-slice signal Simple3DCNN exists to use).
  3. One forward+backward pass through Simple3DCNN fits on the 6GB RTX 3050 at the
     candidate batch size, and reports peak memory so the real training script can pick
     a safe batch size instead of guessing.

Not a training run -- no checkpoint is saved, no result JSON is written. This is the
"cheap sanity check before the big run" step called for in the plan.

Usage: python scripts/qc_3d_volume.py [batch_size]
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import MRI3DDataset, VOLUME_EVAL_TRANSFORM, VOLUME_TRAIN_TRANSFORM  # noqa: E402
from models import Simple3DCNN                                                    # noqa: E402


def check_volume_assembly(manifest):
    print("--- 1. volume assembly matches source PNGs ---")
    ds = MRI3DDataset(manifest, "train", VOLUME_EVAL_TRANSFORM, classes=["CN", "AD"])
    subj = ds.subjects[0]
    paths = ds._paths[subj]
    print(f"subject {subj}: {len(paths)} slices")
    assert len(paths) == 32, f"expected 32 slices, got {len(paths)}"

    volume, label, sid = ds[0]
    assert sid == subj
    assert volume.shape == (1, 32, 224, 224), volume.shape

    # de-normalize back to uint8 and compare against a directly-loaded slice
    for check_idx in (0, 15, 31):
        recovered = ((volume[0, check_idx].numpy() * 0.5 + 0.5) * 255).round().astype(np.uint8)
        direct = cv2.imread(paths[check_idx], cv2.IMREAD_GRAYSCALE)
        diff = np.abs(recovered.astype(int) - direct.astype(int)).max()
        print(f"  slice {check_idx:2d}: max abs pixel diff vs source PNG = {diff} "
              f"({'OK' if diff <= 1 else 'MISMATCH'})")
        assert diff <= 1, f"slice {check_idx} does not match source PNG (diff={diff})"
    print("  PASS: eval-transform volume matches source slices exactly.\n")


def check_augmentation_consistency(manifest):
    print("--- 2. augmentation applies the SAME warp to every slice ---")
    ds = MRI3DDataset(manifest, "train", VOLUME_TRAIN_TRANSFORM, classes=["CN", "AD"])
    np.random.seed(0)
    volume, _, subj = ds[0]
    vol = volume[0].numpy()  # (32, 224, 224)
    # A consistent rigid warp should leave the frame-to-frame difference between
    # ADJACENT slices close to the difference in the unaugmented data -- it will not be
    # zero (adjacent slices are genuinely different anatomy), but if augmentation were
    # applied independently per slice, every slice would ALSO pick up an independent
    # random shift, inflating adjacent-slice differences well beyond the anatomical one.
    eval_ds = MRI3DDataset(manifest, "train", VOLUME_EVAL_TRANSFORM, classes=["CN", "AD"])
    eval_vol = eval_ds[0][0][0].numpy()
    aug_adjacent_diff = np.abs(np.diff(vol, axis=0)).mean()
    eval_adjacent_diff = np.abs(np.diff(eval_vol, axis=0)).mean()
    ratio = aug_adjacent_diff / max(eval_adjacent_diff, 1e-6)
    print(f"  mean |adjacent-slice diff|: eval {eval_adjacent_diff:.4f}, "
          f"augmented {aug_adjacent_diff:.4f}  (ratio {ratio:.2f}x)")
    print("  (a per-slice-independent bug would blow this ratio up sharply; "
          "a shared rigid warp keeps it close to 1x)")
    assert ratio < 3.0, "augmentation looks like it's varying independently per slice"
    print("  PASS.\n")


def check_memory_and_shapes(manifest, batch_size):
    print(f"--- 3. forward+backward pass, batch_size={batch_size} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds = MRI3DDataset(manifest, "train", VOLUME_TRAIN_TRANSFORM, classes=["CN", "AD"])
    batch = [ds[i] for i in range(batch_size)]
    volumes = torch.stack([b[0] for b in batch]).to(device)
    labels = torch.tensor([b[1] for b in batch]).to(device)
    print(f"batch volumes shape: {tuple(volumes.shape)}")

    model = Simple3DCNN(num_classes=2, in_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Simple3DCNN params: {n_params:,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    out = model(volumes)
    assert out.shape == (batch_size, 2), out.shape
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    print(f"output shape: {tuple(out.shape)}   loss: {loss.item():.4f}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"peak GPU memory: {peak_mb:.0f} MB / {total_mb:.0f} MB "
              f"({peak_mb / total_mb:.1%})")
    print("  PASS.\n")


def main(batch_size=4):
    manifest = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3_adcn.csv"))
    check_volume_assembly(manifest)
    check_augmentation_consistency(manifest)
    check_memory_and_shapes(manifest, batch_size)
    print("=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
