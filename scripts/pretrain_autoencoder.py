"""Self-supervised pretraining: train the masked autoencoder on unlabelled slices,
then save its encoder for the classifier to start from.

Run this BEFORE the classifier runs that use --init from ssl_encoder.pt.

Data: every slice belonging to a TRAIN-SPLIT subject, across all 853 subjects and
both ADNI eras -- 597 subjects, ~19,100 slices. That is 38% more images than the
four-way task's own 432-subject training set, and it costs nothing because no labels
are involved.

Validation/test subjects are excluded. Their labels are never used here, but fitting
a model to images that later appear in evaluation is still leakage, and this project
has already quantified how badly that flatters results (+36.9 points).

Usage: python scripts/pretrain_autoencoder.py [epochs]
"""
import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from autoencoder import ConvAutoencoder, random_mask  # noqa: E402
from datasets import MRIDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM  # noqa: E402

BATCH_SIZE, LR, SEED = 48, 1e-3, 42
OUT_ENCODER = os.path.join(ROOT, "models", "checkpoints", "ssl_encoder.pt")


def main(epochs=25):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # All 853 subjects, both eras -- the more varied the images, the better the
    # features. Only the split matters here, never the class label.
    m = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3.csv"))
    train_m = m[m["split"] == "train"]
    val_m = m[m["split"] == "val"]
    print(f"pretraining slices : {len(train_m)} "
          f"({train_m.subject_id.nunique()} subjects, labels unused)")
    print(f"held-out for loss  : {len(val_m)} "
          f"({val_m.subject_id.nunique()} subjects)")
    print(f"by era: {train_m.groupby('era').size().to_dict()}")

    train_loader = DataLoader(MRIDataset(m, "train", TRAIN_TRANSFORM),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                              drop_last=True)
    val_loader = DataLoader(MRIDataset(m, "val", EVAL_TRANSFORM),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = ConvAutoencoder(in_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"autoencoder: {n_params:,} params "
          f"(encoder {sum(p.numel() for p in model.features.parameters()):,})")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Loss is measured ONLY on the blanked-out pixels. Averaging over the whole image
    # would let a large, easy, unmasked background dominate the gradient and the model
    # could score well while learning nothing about the hidden regions.
    crit = nn.MSELoss(reduction="none")

    history = {"train_loss": [], "val_loss": []}
    best = float("inf")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for imgs, _, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masked, mask = random_mask(imgs)
            opt.zero_grad()
            out = model(masked)
            loss = (crit(out, imgs) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
            tot += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = tot / n

        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for imgs, _, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masked, mask = random_mask(imgs)
                out = model(masked)
                loss = (crit(out, imgs) * mask).sum() / mask.sum().clamp(min=1)
                tot += loss.item() * imgs.size(0)
                n += imgs.size(0)
        val_loss = tot / n
        sched.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {ep+1:2d}/{epochs} - train {train_loss:.4f}  val {val_loss:.4f}"
              + ("  <- best, saving encoder" if val_loss < best else ""), flush=True)

        if val_loss < best:
            best = val_loss
            torch.save(model.encoder_state_dict(), OUT_ENCODER)
            torch.save(model.state_dict(),
                       os.path.join(ROOT, "models", "checkpoints", "ssl_autoencoder_full.pt"))

    mins = (time.time() - t0) / 60
    print(f"\ndone in {mins:.1f} min. best masked-reconstruction val MSE {best:.4f}")
    print(f"encoder -> {OUT_ENCODER}")

    # ---- visual proof it learned something, for the report -------------------
    model.load_state_dict(torch.load(
        os.path.join(ROOT, "models", "checkpoints", "ssl_autoencoder_full.pt")))
    model.eval()
    imgs, _, _ = next(iter(val_loader))
    imgs = imgs[:6].to(device)
    masked, _ = random_mask(imgs)
    with torch.no_grad():
        rec = model(masked)

    def to_img(t):
        return ((t.cpu().numpy()[0] + 1) / 2).clip(0, 1)

    fig, axes = plt.subplots(3, 6, figsize=(15, 7.5))
    for i in range(6):
        axes[0, i].imshow(to_img(imgs[i]), cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(to_img(masked[i]), cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(to_img(rec[i]), cmap="gray"); axes[2, i].axis("off")
    axes[0, 0].set_title("original", fontsize=9, loc="left")
    axes[1, 0].set_title("masked input", fontsize=9, loc="left")
    axes[2, 0].set_title("reconstruction", fontsize=9, loc="left")
    plt.suptitle("Masked autoencoder: the network never saw the blanked squares and "
                 "must infer them from surrounding anatomy", fontsize=11)
    plt.tight_layout()
    figpath = os.path.join(ROOT, "reports", "figures", "ssl_reconstruction.png")
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath, dpi=110)
    print("wrote", figpath)

    with open(os.path.join(ROOT, "reports", "ssl_pretrain_result.json"), "w") as f:
        json.dump({
            "n_pretrain_slices": int(len(train_m)),
            "n_pretrain_subjects": int(train_m.subject_id.nunique()),
            "epochs": epochs, "best_val_masked_mse": best,
            "encoder_params": sum(p.numel() for p in model.features.parameters()),
            "train_minutes": round(mins, 1),
            "history": history,
            "note": ("Trained on train-split subjects only; val/test subjects excluded "
                     "so that later evaluation stays clean."),
        }, f, indent=2)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
