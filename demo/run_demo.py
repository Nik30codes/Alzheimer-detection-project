"""
STANDALONE DEMO -- run this and watch it train real models on the real task.

    python demo/run_demo.py

Trains three architectures (custom CNN, MobileNetV2, EfficientNet-B0) FROM SCRATCH on
the project's headline task -- AD vs CN, 501 subjects spanning both ADNI scanner
generations (manifest_v3_adcn.csv) -- printing every epoch's loss/accuracy live, then
reports full performance for each: classification report, confusion matrix, accuracy,
macro F1, ROC AUC with a 95% confidence interval, and a few example predictions.
Confusion matrices and training curves are saved as PNGs in demo/results/, plus one
comparison table across all three models and a summary.json with everything in it.

WHY FROM SCRATCH (not fine-tuned from ImageNet): tested both ways earlier in this
project. ImageNet pretraining actively HURT here -- grayscale MRI duplicated into
fake-RGB channels is too different from natural photos, and the fine-tuned models
scored 15-20 points BELOW the same architectures trained from random initialisation.
See CLAUDE.md decision 7 for the full comparison.

HONEST CAVEAT, printed again at the end of the run: this demo trains on ONE
train/val/test split, for speed -- watching three models train is the point. The
project's own hard-earned rule (CLAUDE.md decision 33) is that a single split is not
trustworthy on its own; three different single-split "wins" in this project's history
evaporated once checked against 5-fold cross-validation. The real, cross-validated,
statistically-supported headline for this exact task (74.1% accuracy, ROC AUC 0.784,
95% CI [0.743, 0.826], over all 501 subjects) is read live from
reports/mobilenetv2_ADvsCN_cv_result.json and printed alongside this run's own number
so the two are never confused.

Expected runtime: roughly 15-45 minutes total for all three models on a single
consumer GPU (early stopping usually ends a model well before the 40-epoch budget).
Lower EPOCHS below for a faster, rougher run.
"""
import json
import math
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")  # no display needed -- just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DEMO_DIR)
RESULTS_DIR = os.path.join(DEMO_DIR, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import (MRIDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM,      # noqa: E402
                      TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB)
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0  # noqa: E402
from train import train_model                                          # noqa: E402

# ---- demo configuration -- change these if you want a faster/rougher run ----
ARCHS = ["custom_cnn", "mobilenetv2", "efficientnet_b0"]
BINARY = ["CN", "AD"]
EPOCHS, PATIENCE, BATCH_SIZE, LR, WD, SEED = 40, 7, 32, 1e-3, 1e-4, 42
ARCH_DISPLAY = {"custom_cnn": "Custom CNN (from scratch)",
                "mobilenetv2": "MobileNetV2 (from scratch)",
                "efficientnet_b0": "EfficientNet-B0 (from scratch)"}


def wilson_ci(correct, n, z=1.96):
    """95% Wilson interval for an accuracy estimate -- same formula this project
    uses everywhere else (scripts/train_binary_adni1.py), so numbers are comparable."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc_ci(auc, n_pos, n_neg, z=1.96):
    """Hanley-McNeil 95% interval for ROC AUC."""
    if n_pos == 0 or n_neg == 0:
        return (float("nan"), float("nan"))
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return (max(0.0, auc - z * se), min(1.0, auc + z * se))


class BinaryDataset(MRIDataset):
    """Same slices as MRIDataset, mapped to CN->0, AD->1 for the binary task."""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["filepath"])
        return self.transform(img), BINARY.index(row["class"]), row["subject_id"]


def build_model(arch):
    if arch == "custom_cnn":
        return SimpleCNN(num_classes=2, in_channels=1)
    if arch == "mobilenetv2":
        return build_mobilenetv2(2, pretrained=False)
    return build_efficientnet_b0(2, pretrained=False)


@torch.no_grad()
def subject_probs(model, loader, device):
    """Runs the model over every slice, then averages each subject's slice-level
    probabilities into one prediction per person (soft vote) -- the number that
    actually matters, since a diagnosis is made per person, not per image."""
    model.eval()
    rows = []
    for imgs, labels, sids in loader:
        p = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
        for i in range(len(labels)):
            rows.append({"subject_id": sids[i], "true": BINARY[labels[i]],
                        "p_AD": float(p[i, 1])})
    g = pd.DataFrame(rows).groupby("subject_id")
    return pd.DataFrame({"true": g["true"].first(), "p_AD": g["p_AD"].mean()})


def youden_threshold(val_df):
    """Pick the decision cut-point on VALIDATION subjects (never test) that maximises
    sensitivity + specificity - 1. At the default 0.5 these models can predict the
    majority class for everyone despite ranking correctly -- this fixes that without
    ever looking at the test set. See CLAUDE.md decision 35."""
    y = (val_df["true"] == "AD").astype(int).values
    best, best_j = 0.5, -1.0
    for t in np.unique(np.round(val_df["p_AD"].values, 4)):
        pred = (val_df["p_AD"].values >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j, best = j, float(t)
    return best


def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def plot_curves(history, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss per epoch"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy per epoch"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def run_one(arch, manifest, device):
    print(f"\n{'=' * 70}")
    print(f"  {ARCH_DISPLAY[arch]}")
    print(f"{'=' * 70}")

    rgb = arch != "custom_cnn"
    train_t, eval_t = (TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB) if rgb else (TRAIN_TRANSFORM, EVAL_TRANSFORM)
    loaders = {sp: DataLoader(BinaryDataset(manifest, sp, train_t if sp == "train" else eval_t),
                              batch_size=BATCH_SIZE, shuffle=(sp == "train"), num_workers=2)
              for sp in ("train", "val", "test")}

    counts = manifest[manifest.split == "train"]["class"].value_counts()
    w = torch.tensor([len(manifest[manifest.split == "train"]) / counts[c] for c in BINARY],
                     dtype=torch.float32)
    w = w / w.sum() * 2

    model = build_model(arch)
    ckpt = os.path.join(RESULTS_DIR, f"{arch}_demo.pt")
    hist_path = os.path.join(RESULTS_DIR, f"{arch}_demo_history.json")

    # RESUME-SAFETY: this machine has killed background training jobs mid-run more
    # than once (see the 3D CNN CV job's own resume-safety, same reason). If this
    # architecture already has a checkpoint + saved history, reuse them instead of
    # retraining -- the plotting/evaluation code below is what a kill during this
    # step previously destroyed, not the training itself.
    if os.path.exists(ckpt) and os.path.exists(hist_path):
        print(f"RESUMING {arch} from existing checkpoint + history (skipping training)")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        hist = json.load(open(hist_path))
        mins = hist.get("train_minutes", 0.0)
    else:
        t0 = time.time()
        hist = train_model(model, loaders["train"], loaders["val"], w, device,
                           epochs=EPOCHS, lr=LR, patience=PATIENCE, weight_decay=WD,
                           checkpoint_path=ckpt)
        mins = (time.time() - t0) / 60
        hist["train_minutes"] = mins
        with open(hist_path, "w") as f:
            json.dump(hist, f, indent=2)
        print(f"trained {len(hist['train_loss'])} epochs in {mins:.1f} min "
              f"(best epoch {hist['best_epoch']})")

    val_df = subject_probs(model, loaders["val"], device)
    thr = youden_threshold(val_df)
    test_df = subject_probs(model, loaders["test"], device)
    test_df["pred"] = np.where(test_df["p_AD"] >= thr, "AD", "CN")

    n = len(test_df)
    correct = int((test_df["true"] == test_df["pred"]).sum())
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    f1 = f1_score(test_df["true"], test_df["pred"], labels=BINARY, average="macro")
    n_pos = int((test_df["true"] == "AD").sum())
    auc = roc_auc_score((test_df["true"] == "AD").astype(int), test_df["p_AD"])
    alo, ahi = auc_ci(auc, n_pos, n - n_pos)
    cm = confusion_matrix(test_df["true"], test_df["pred"], labels=BINARY)

    print(f"\n--- test set performance ({n} subjects, threshold {thr:.3f} chosen on "
          f"validation) ---")
    print(classification_report(test_df["true"], test_df["pred"], labels=BINARY,
                                zero_division=0))
    print(f"confusion matrix (rows=true, cols=pred) {BINARY}:")
    print(pd.DataFrame(cm, index=[f"true_{c}" for c in BINARY],
                       columns=[f"pred_{c}" for c in BINARY]).to_string())
    print(f"\naccuracy   {acc:.1%}  ({correct}/{n})   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"macro F1   {f1:.3f}")
    print(f"ROC AUC    {auc:.3f}   95% CI [{alo:.3f}, {ahi:.3f}]")

    sample = test_df.sample(min(5, len(test_df)), random_state=SEED)
    print("\nsample predictions:")
    for sid, row in sample.iterrows():
        mark = "correct" if row["true"] == row["pred"] else "WRONG"
        print(f"  {sid}: true={row['true']:3s}  pred={row['pred']:3s}  "
              f"p(AD)={row['p_AD']:.3f}   [{mark}]")

    plot_confusion(cm, BINARY, f"{ARCH_DISPLAY[arch]} -- confusion matrix",
                   os.path.join(RESULTS_DIR, f"{arch}_confusion_matrix.png"))
    plot_curves(hist, f"{ARCH_DISPLAY[arch]} -- training curves",
               os.path.join(RESULTS_DIR, f"{arch}_training_curves.png"))

    return {
        "arch": arch, "n_test_subjects": n, "accuracy": acc, "accuracy_95CI": [lo, hi],
        "macro_f1": float(f1), "roc_auc": float(auc), "roc_auc_95CI": [alo, ahi],
        "decision_threshold": thr, "epochs_run": len(hist["train_loss"]),
        "best_epoch": hist["best_epoch"], "train_minutes": round(mins, 1),
        "confusion_matrix": cm.tolist(), "classes": BINARY,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3_adcn.csv"))
    manifest = manifest[manifest["class"].isin(BINARY)].reset_index(drop=True)

    print("Alzheimer's MRI classification -- live model demo")
    print(f"device: {device}")
    print(f"task: AD vs CN, real ADNI MRI scans, {manifest.subject_id.nunique()} subjects "
          f"spanning two scanner generations")
    subj = manifest.groupby("split")["subject_id"].nunique()
    print(f"split: train {subj.get('train', 0)}  val {subj.get('val', 0)}  "
          f"test {subj.get('test', 0)} subjects "
          f"(subject-wise -- no person appears in more than one split)")
    print(f"models to train: {', '.join(ARCH_DISPLAY[a] for a in ARCHS)}")

    t_start = time.time()
    results = [run_one(arch, manifest, device) for arch in ARCHS]

    print(f"\n{'=' * 70}")
    print("  COMPARISON ACROSS ALL MODELS (this run's single split)")
    print(f"{'=' * 70}")
    table = pd.DataFrame([{
        "model": ARCH_DISPLAY[r["arch"]], "accuracy": f"{r['accuracy']:.1%}",
        "macro F1": f"{r['macro_f1']:.3f}", "ROC AUC": f"{r['roc_auc']:.3f}",
        "epochs": f"{r['epochs_run']} (best {r['best_epoch']})",
        "minutes": r["train_minutes"],
    } for r in results])
    print(table.to_string(index=False))

    ref_path = os.path.join(ROOT, "reports", "mobilenetv2_ADvsCN_cv_result.json")
    print(f"\n{'=' * 70}")
    print("  HONEST REFERENCE POINT -- read this before quoting a number above")
    print(f"{'=' * 70}")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        print(f"This demo trains on ONE train/val/test split, for speed. This project's "
              f"own rule is that a single split is never trustworthy on its own -- three "
              f"separate single-split 'wins' in this project's history reversed once "
              f"checked against 5-fold cross-validation (over ALL 501 subjects, so every "
              f"person gets exactly one out-of-fold prediction).")
        print(f"\nThe real, cross-validated headline for this exact task:")
        print(f"  accuracy  {ref['accuracy']:.1%}   95% CI {ref['accuracy_95CI']}")
        print(f"  ROC AUC   {ref['roc_auc']:.4f}   95% CI {ref['roc_auc_95CI']}")
        print(f"  majority baseline {ref['majority_baseline']:.1%} "
              f"(a model that always guesses the larger class)")
    else:
        print("(reports/mobilenetv2_ADvsCN_cv_result.json not found -- run the full "
              "project pipeline to generate the cross-validated headline)")

    print(f"\ntotal demo runtime: {(time.time() - t_start) / 60:.1f} min")
    print(f"saved: confusion matrices + training curves per model, and "
          f"results/summary.json, in {RESULTS_DIR}")

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump({"results": results, "epochs_budget": EPOCHS, "patience": PATIENCE,
                   "n_subjects": int(manifest.subject_id.nunique())}, f, indent=2)


if __name__ == "__main__":
    main()
