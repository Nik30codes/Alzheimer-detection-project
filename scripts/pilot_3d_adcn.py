"""
FAST, NOT-FOR-REPORTING pilot: does Simple3DCNN actually train on AD vs CN?

Uses the existing single train/val/test split in manifest_v3_adcn.csv (not CV) and a
short epoch budget. This is a go/no-go gate before committing to the multi-hour 5-fold
CV run in cross_validate_3d_adcn.py -- checking for a sane loss curve (decreasing,
not stuck) and val accuracy clearing the majority baseline, not a number to quote.
Decision 33 in CLAUDE.md is the reason single-split numbers never get reported as
results in this project -- this script's output is explicitly exempt from that concern
because nothing here is meant to be reported at all.

Usage: python scripts/pilot_3d_adcn.py [epochs] [batch_size]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import build_dataloaders_3d       # noqa: E402
from models import Simple3DCNN                  # noqa: E402
from train import train_model                   # noqa: E402

BINARY = ["CN", "AD"]
SEED = 42


def main(epochs=10, batch_size=8):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  epochs: {epochs}  batch_size: {batch_size}")

    manifest = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3_adcn.csv"))
    manifest = manifest[manifest["class"].isin(BINARY)].reset_index(drop=True)
    subj = manifest.groupby("split")["subject_id"].nunique()
    print(f"subjects  train {subj.get('train', 0)}  val {subj.get('val', 0)}  "
          f"test {subj.get('test', 0)}")

    train_loader, val_loader, test_loader = build_dataloaders_3d(
        manifest, batch_size=batch_size, num_workers=2, classes=BINARY)

    train_subj = manifest[manifest.split == "train"].groupby("subject_id")["class"].first()
    counts = train_subj.value_counts()
    w = torch.tensor([len(train_subj) / counts[c] for c in BINARY], dtype=torch.float32)
    w = w / w.sum() * 2
    print(f"class weights (CN, AD): {w.tolist()}")

    model = Simple3DCNN(num_classes=2, in_channels=1)
    ckpt = os.path.join(ROOT, "models", "checkpoints", "_pilot_simple3dcnn_adcn.pt")

    t0 = time.time()
    hist = train_model(model, train_loader, val_loader, w, device,
                       epochs=epochs, lr=1e-3, patience=epochs, weight_decay=1e-4,
                       checkpoint_path=ckpt, select_by="val_loss")
    mins = (time.time() - t0) / 60
    print(f"\ntrained {len(hist['train_loss'])} epochs in {mins:.1f} min "
          f"({mins * 60 / len(hist['train_loss']):.0f} s/epoch)")
    print(f"best_epoch: {hist['best_epoch']}")
    print(f"train_acc at each epoch: {[round(a, 3) for a in hist['train_acc']]}")
    print(f"val_acc   at each epoch: {[round(a, 3) for a in hist['val_acc']]}")
    print(f"val_loss  at each epoch: {[round(l_, 3) for l_ in hist['val_loss']]}")

    # quick subject-level AUC on the untouched test split, still PILOT ONLY
    model.eval()
    rows = []
    with torch.no_grad():
        for vols, labels, sids in test_loader:
            p = torch.softmax(model(vols.to(device)), dim=1).cpu().numpy()
            for i in range(len(labels)):
                rows.append({"subject_id": sids[i], "true": BINARY[labels[i]],
                            "p_AD": float(p[i, 1])})
    df = pd.DataFrame(rows)
    auc = roc_auc_score((df["true"] == "AD").astype(int), df["p_AD"])
    acc_at_half = ((df["p_AD"] >= 0.5).map({True: "AD", False: "CN"}) == df["true"]).mean()
    baseline = df["true"].value_counts().max() / len(df)
    print(f"\n[PILOT ONLY -- not a reportable number, single split, default 0.5 threshold]")
    print(f"test subjects: {len(df)}   majority baseline: {baseline:.1%}")
    print(f"accuracy @ 0.5: {acc_at_half:.1%}   ROC AUC: {auc:.4f}")
    print(f"\nGo/no-go: {'proceed to full CV' if auc > 0.55 else 'investigate before CV -- AUC too close to chance'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10,
         int(sys.argv[2]) if len(sys.argv) > 2 else 8)
