"""
The honest core task: AD vs CN, restricted to the ADNI1 cohort.

Motivation (decision 10 in CLAUDE.md): in the 4-way task, class is perfectly confounded
with ADNI phase, so a model scores ~56% mostly by telling ADNI1 from ADNI-GO/2. Dropping
EMCI/LMCI removes that shortcut entirely -- CN and AD both come from ADNI1, same era,
same protocol -- so whatever accuracy remains here is actually about anatomy.

This is also the comparison the literature benchmarks, which makes the result
interpretable: properly subject-split studies report AD-vs-CN under 71% on ADNI-sized
data, under 59% on smaller sets. This dataset has 174 such subjects, so the low-to-mid
60s would be a credible result and anything near 90% would mean something is leaking.

The train/val/test assignment is inherited from the existing subject-wise split rather
than re-drawn, so no subject that was in test can drift into train.

Usage: python train_binary_adni1.py [arch] [manifest_key]
  arch         custom_cnn | efficientnet_b0 | mobilenetv2
  manifest_key v1 (original slices) | v2 (millimetre-anchored re-extraction) |
               v2crop (v2 + brain crop). Defaults to v2, since the v1 slices had a
               drifting axial band that sometimes missed the hippocampus entirely.
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import (MRIDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM,      # noqa: E402
                      TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB)
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0  # noqa: E402
from train import train_model                                          # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,   # noqa: E402
                             classification_report, confusion_matrix)

BINARY = ["CN", "AD"]
BATCH_SIZE, LR, WD, EPOCHS, PATIENCE, SEED = 32, 1e-3, 1e-4, 40, 7, 42


def auc_ci(auc, n_pos, n_neg, z=1.96):
    """Hanley-McNeil 95% interval for ROC AUC.

    Worth reporting alongside the point estimate: with ~11 positives and ~15 negatives the
    standard error is around 0.11, so an AUC of 0.67 has an interval that still touches
    0.5. Quoting the point estimate alone would imply far more certainty than 26 subjects
    can support.
    """
    if n_pos == 0 or n_neg == 0:
        return (float("nan"), float("nan"))
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return (max(0.0, auc - z * se), min(1.0, auc + z * se))


def wilson_ci(correct, n, z=1.96):
    """95% Wilson interval. With only ~26 test subjects the point estimate is very
    noisy, and reporting it without an interval would overstate precision."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class BinaryDataset(MRIDataset):
    """Same as MRIDataset but maps CN->0, AD->1 instead of the 4-class indices."""

    def __getitem__(self, idx):
        import cv2
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["filepath"])
        return self.transform(img), BINARY.index(row["class"]), row["subject_id"]


MANIFESTS = {"v1": "manifest.csv", "v2": "manifest_v2.csv",
             "v2crop": "manifest_v2_braincrop.csv",
             "v2mask": "manifest_v2_masked.csv",
             # v3adcn: AD vs CN over BOTH cohorts, 501 subjects instead of 174.
             # The expansion supplied ADNI-GO/2 CN and AD subjects, which leaves era
             # almost exactly uninformative about the label (best achievable from era
             # alone = 56.9% = the majority baseline), so restricting to one cohort is
             # no longer necessary to avoid the confound -- and keeping both nearly
             # triples the training set, which was the binding constraint before.
             "v3adcn": "manifest_v3_adcn.csv",
             # v4adcn: same 501 subjects and same splits as v3adcn, but rendered with
             # isotropic 1mm/px + a fixed 224mm physical window (decision 31), which
             # removes the acquisition-geometry confound of decision 27.
             "v4adcn": "manifest_v4_adcn.csv",
             # v3mask: v3 slices with skull/scalp/background zeroed, framing untouched.
             # Re-test of the masking experiment that was inconclusive at 26 test
             # subjects on v2; here the test set is 75.
             "v3mask": "manifest_v3_masked.csv"}

# The task label is only accurate for the single-cohort manifests; v3adcn spans both.
TASK_LABEL = {"v3adcn": "AD vs CN across both ADNI cohorts (era-balanced, 501 subjects)"}


def task_label(manifest_key):
    return TASK_LABEL.get(manifest_key, "AD vs CN within ADNI1 (no cohort confound)")


def main(arch="custom_cnn", manifest_key="v2"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(os.path.join(ROOT, "data", MANIFESTS[manifest_key]))
    print(f"manifest: {MANIFESTS[manifest_key]}")
    m = manifest[manifest["class"].isin(BINARY)].reset_index(drop=True)

    subj = m.groupby("split")["subject_id"].nunique()
    print(f"===== {task_label(manifest_key)} ({arch}) =====")
    if "era" in m.columns:
        print("subjects by class x era:")
        print(m.groupby(["class", "era"])["subject_id"].nunique().to_string())
    print(f"subjects  train {subj.get('train', 0)}  val {subj.get('val', 0)}  test {subj.get('test', 0)}")
    print(f"slices    train {(m.split=='train').sum()}  val {(m.split=='val').sum()}  test {(m.split=='test').sum()}")
    print("class balance (test subjects):",
          m[m.split == "test"].groupby("class")["subject_id"].nunique().to_dict())

    rgb = arch != "custom_cnn"
    train_t, eval_t = ((TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB) if rgb
                       else (TRAIN_TRANSFORM, EVAL_TRANSFORM))
    loaders = {sp: DataLoader(BinaryDataset(m, sp, train_t if sp == "train" else eval_t),
                              batch_size=BATCH_SIZE, shuffle=(sp == "train"), num_workers=2)
               for sp in ("train", "val", "test")}

    counts = m[m.split == "train"]["class"].value_counts()
    w = torch.tensor([len(m[m.split == "train"]) / counts[c] for c in BINARY], dtype=torch.float32)
    w = w / w.sum() * 2

    if arch == "custom_cnn":
        model = SimpleCNN(num_classes=2, in_channels=1)
    elif arch == "mobilenetv2":
        model = build_mobilenetv2(2, pretrained=False)
    else:
        model = build_efficientnet_b0(2, pretrained=False)

    # The manifest key MUST be in the checkpoint name. Without it every variant wrote
    # to {arch}_ADvsCN.pt, so running v3adcn silently destroyed the v2 weights -- those
    # are gone and cannot be re-evaluated without retraining. v1 keeps the bare name so
    # existing v1 checkpoints still resolve.
    ckpt_suffix = "" if manifest_key == "v1" else f"_{manifest_key}"
    ckpt = os.path.join(ROOT, "models", "checkpoints",
                        f"{arch}_ADvsCN{ckpt_suffix}.pt")
    t0 = time.time()
    hist = train_model(model, loaders["train"], loaders["val"], w, device,
                       epochs=EPOCHS, lr=LR, patience=PATIENCE,
                       checkpoint_path=ckpt, weight_decay=WD)
    mins = (time.time() - t0) / 60

    # ---- subject-level soft vote ----
    model.eval()

    def subject_probs(split):
        rows = []
        with torch.no_grad():
            for imgs, labels, sids in loaders[split]:
                p = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
                for i in range(len(labels)):
                    rows.append({"subject_id": sids[i], "true": BINARY[labels[i]],
                                 "p_AD": p[i, 1]})
        g = pd.DataFrame(rows).groupby("subject_id")
        return pd.DataFrame({"true": g["true"].first(), "p_AD": g["p_AD"].mean()})

    val_df = subject_probs("val")
    subj_df = subject_probs("test")

    # Pick the decision threshold on VALIDATION, never on test.
    #
    # Why this is needed: at the default 0.5 these models predicted CN for every single
    # subject, scoring exactly the majority baseline -- while their ROC AUC was ~0.67,
    # meaning the probabilities rank AD above CN perfectly well. The ranking is informative
    # but the probabilities are compressed below 0.5, because training on an imbalanced set
    # with a small validation set leaves the output poorly calibrated. Choosing the
    # threshold that maximises Youden's J (sensitivity + specificity - 1) on validation
    # converts that ranking ability into usable predictions without touching the test set.
    y_val = (val_df["true"] == "AD").astype(int).values
    best_thr, best_j = 0.5, -1.0
    for thr in np.unique(np.round(val_df["p_AD"].values, 4)):
        pred = (val_df["p_AD"].values >= thr).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum()); fn = int(((pred == 0) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum()); fp = int(((pred == 1) & (y_val == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens + spec - 1 > best_j:
            best_j, best_thr = sens + spec - 1, float(thr)
    print(f"\nthreshold chosen on validation: {best_thr:.3f} (Youden's J = {best_j:.3f}); "
          f"default 0.5 would have been used otherwise")

    subj_df["pred"] = np.where(subj_df["p_AD"] >= best_thr, "AD", "CN")
    acc_at_half = float((np.where(subj_df["p_AD"] >= 0.5, "AD", "CN") == subj_df["true"]).mean())

    n = len(subj_df)
    correct = int((subj_df["true"] == subj_df["pred"]).sum())
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    majority = subj_df["true"].value_counts().max() / n
    auc = roc_auc_score((subj_df["true"] == "AD").astype(int), subj_df["p_AD"])

    print(f"\ntrained {len(hist['train_loss'])} epochs in {mins:.1f} min")
    print("\n===== SUBJECT LEVEL, AD vs CN =====")
    print(classification_report(subj_df["true"], subj_df["pred"], labels=BINARY, zero_division=0))
    print(f"confusion matrix (rows=true, cols=pred) {BINARY}:")
    print(confusion_matrix(subj_df["true"], subj_df["pred"], labels=BINARY))
    print(f"\naccuracy    {acc:.1%}  ({correct}/{n})   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"majority baseline {majority:.1%}  <- must be beaten to mean anything")
    n_pos = int((subj_df["true"] == "AD").sum())
    alo, ahi = auc_ci(auc, n_pos, n - n_pos)
    print(f"ROC AUC     {auc:.3f}   95% CI [{alo:.3f}, {ahi:.3f}]"
          f"   (0.5 = no signal; robust to the class imbalance)")
    if alo > 0.5:
        print("            ^ interval excludes 0.5 -> genuine ranking signal")
    else:
        print("            ^ interval still includes 0.5 -> suggestive, not established")
    if hi < majority:
        verdict = "NO SIGNAL - significantly below the trivial baseline"
    elif lo > majority:
        verdict = "REAL SIGNAL - significantly above the trivial baseline"
    else:
        verdict = "INCONCLUSIVE - CI straddles the trivial baseline (test set is small)"
    print(f"verdict: {verdict}")

    out = {
        "task": task_label(manifest_key), "arch": arch,
        "manifest": MANIFESTS[manifest_key],
        "n_test_subjects": n, "accuracy": acc, "accuracy_95CI": [lo, hi],
        "decision_threshold": best_thr, "accuracy_at_default_0.5": acc_at_half,
        "majority_baseline": majority, "roc_auc": auc, "roc_auc_95CI": [alo, ahi],
        "verdict": verdict,
        "macro_f1": f1_score(subj_df["true"], subj_df["pred"], labels=BINARY, average="macro"),
        "epochs_run": len(hist["train_loss"]), "train_minutes": round(mins, 1),
    }
    suffix = "" if manifest_key == "v1" else f"_{manifest_key}"
    with open(os.path.join(ROOT, "reports", f"{arch}_ADvsCN{suffix}_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    subj_df.to_csv(os.path.join(ROOT, "reports", f"{arch}_ADvsCN{suffix}_subject_preds.csv"))
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "custom_cnn",
         sys.argv[2] if len(sys.argv) > 2 else "v2")
