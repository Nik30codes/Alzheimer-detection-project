"""Cross-era external validation for AD vs CN: train on one ADNI cohort, test on the
other. Impossible before the v3 expansion, which supplied the first CN/AD subjects
from ADNI-GO/2.

Why this is the strongest test the project can run. Every result so far splits
train/test inside one pool, so training and test images share scanner generation,
protocol and reconstruction pipeline. A model can therefore lean on cohort-specific
texture and still score well. Here the test set is an ENTIRELY DIFFERENT COHORT --
different scanners, years, matrices and vendors -- so any accuracy that survives has
to come from anatomy that is present in both. It is also the setting a real
deployment faces: scans from a machine the model never trained on.

Two directions are run, because they are not equivalent:
  adni1->go2   train on 235 ADNI1 subjects (2006-07), test on 266 GO/2 (2010+)
  go2->adni1   train on 266 GO/2 subjects,   test on 235 ADNI1

Test sets here are 3-4x larger than the 75-subject within-pool split, so the
confidence intervals are correspondingly tighter.

The decision threshold is chosen on a validation split held out from the TRAINING
era, never on the test cohort.

Usage: python scripts/train_cross_era.py <arch> <adni1->go2 | go2->adni1>
       arch: custom_cnn | mobilenetv2 | efficientnet_b0
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from datasets import (TRAIN_TRANSFORM, EVAL_TRANSFORM,                  # noqa: E402
                      TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB)
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0  # noqa: E402
from train import train_model                                          # noqa: E402
from train_binary_adni1 import BINARY, BinaryDataset, auc_ci, wilson_ci  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,   # noqa: E402
                             classification_report, confusion_matrix)

BATCH_SIZE, LR, WD, EPOCHS, PATIENCE, SEED = 32, 1e-3, 1e-4, 40, 7, 42
VAL_FRACTION = 0.15  # held out from the training era, for early stopping + threshold

DIRECTIONS = {"adni1->go2": ("ADNI1", "GO2"), "go2->adni1": ("GO2", "ADNI1")}


def build_split(m, source_era, target_era):
    """train/val from source_era subjects; test = every subject of target_era.

    Splitting is done on SUBJECTS, then mapped back to slices, so no person's slices
    straddle the train/val boundary.
    """
    rng = np.random.default_rng(SEED)
    src = m[m["era"] == source_era]
    subs = src.groupby("subject_id")["class"].first()

    val_ids = set()
    for cls in BINARY:  # stratify the validation split by class
        ids = np.array(sorted(subs[subs == cls].index))
        rng.shuffle(ids)
        n_val = max(1, int(round(VAL_FRACTION * len(ids))))
        val_ids.update(ids[:n_val])

    out = m.copy()
    out["split"] = np.where(
        out["era"] == target_era, "test",
        np.where(out["subject_id"].isin(val_ids), "val", "train"))
    return out


def main(arch="mobilenetv2", direction="adni1->go2", select_by="val_loss",
         manifest="manifest_v3_adcn.csv"):
    if direction not in DIRECTIONS:
        raise SystemExit(f"direction must be one of {list(DIRECTIONS)}")
    source_era, target_era = DIRECTIONS[direction]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # manifest is a parameter so the MASKED dataset can be put through the same
    # cross-era test. Decision 32 measured masking at +0.05 AUC under CV, but its
    # Grad-CAM attention got 14 points WORSE -- which is what a silhouette shortcut
    # would look like. A real anatomical gain should survive a change of scanner
    # generation; an artifact of zeroing the background should not.
    m = pd.read_csv(os.path.join(ROOT, "data", manifest))
    # Filter to the two binary classes. manifest_v3_adcn.csv is already AD/CN only, but
    # manifest_v3_masked.csv is the masked copy of the FULL 853-subject manifest and
    # still contains EMCI/LMCI -- which BinaryDataset cannot map, so training died with
    # "'EMCI' is not in list" partway through the first run.
    m = m[m["class"].isin(BINARY)].reset_index(drop=True)
    m = build_split(m, source_era, target_era)

    tag = f"{arch}_crossera_{direction.replace('->', '_to_')}"
    if "masked" in manifest:
        tag += "_masked"
    if select_by != "val_loss":
        tag += "_f1"
    print(f"===== {tag} on {device} (checkpoint selected by {select_by}) =====")
    print(f"train/val cohort: {source_era}     test cohort: {target_era}")
    subj = m.groupby("split")["subject_id"].nunique()
    print(f"subjects  train {subj.get('train',0)}  val {subj.get('val',0)}  "
          f"test {subj.get('test',0)}")
    print("test-cohort class balance:",
          m[m.split == "test"].groupby("class")["subject_id"].nunique().to_dict())
    # Sanity: the two cohorts must not share a subject, or this is not external at all.
    tr = set(m[m.split != "test"]["subject_id"])
    te = set(m[m.split == "test"]["subject_id"])
    assert not (tr & te), f"{len(tr & te)} subjects in both cohorts -- not external!"
    print(f"train/test subject overlap: {len(tr & te)} (must be 0)")

    rgb = arch != "custom_cnn"
    train_t, eval_t = ((TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB) if rgb
                       else (TRAIN_TRANSFORM, EVAL_TRANSFORM))
    loaders = {sp: DataLoader(BinaryDataset(m, sp, train_t if sp == "train" else eval_t),
                              batch_size=BATCH_SIZE, shuffle=(sp == "train"), num_workers=2)
               for sp in ("train", "val", "test")}

    counts = m[m.split == "train"]["class"].value_counts()
    w = torch.tensor([len(m[m.split == "train"]) / counts[c] for c in BINARY],
                     dtype=torch.float32)
    w = w / w.sum() * 2

    if arch == "custom_cnn":
        model = SimpleCNN(num_classes=2, in_channels=1)
    elif arch == "mobilenetv2":
        model = build_mobilenetv2(2, pretrained=False)
    else:
        model = build_efficientnet_b0(2, pretrained=False)

    ckpt = os.path.join(ROOT, "models", "checkpoints", f"{tag}.pt")
    t0 = time.time()
    hist = train_model(model, loaders["train"], loaders["val"], w, device,
                       epochs=EPOCHS, lr=LR, patience=PATIENCE,
                       checkpoint_path=ckpt, weight_decay=WD, select_by=select_by)
    mins = (time.time() - t0) / 60

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

    val_df, subj_df = subject_probs("val"), subject_probs("test")

    # Threshold from the TRAINING cohort's validation split only.
    y_val = (val_df["true"] == "AD").astype(int).values
    best_thr, best_j = 0.5, -1.0
    for thr in np.unique(np.round(val_df["p_AD"].values, 4)):
        pred = (val_df["p_AD"].values >= thr).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum()); fn = int(((pred == 0) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum()); fp = int(((pred == 1) & (y_val == 0)).sum())
        sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        if sens + spec - 1 > best_j:
            best_j, best_thr = sens + spec - 1, float(thr)
    print(f"\nthreshold {best_thr:.3f} chosen on the {source_era} validation split "
          f"(Youden's J {best_j:.3f}) -- the test cohort was never used for this")

    subj_df["pred"] = np.where(subj_df["p_AD"] >= best_thr, "AD", "CN")
    n = len(subj_df)
    correct = int((subj_df["true"] == subj_df["pred"]).sum())
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    majority = subj_df["true"].value_counts().max() / n
    auc = roc_auc_score((subj_df["true"] == "AD").astype(int), subj_df["p_AD"])
    n_pos = int((subj_df["true"] == "AD").sum())
    alo, ahi = auc_ci(auc, n_pos, n - n_pos)

    print(f"\ntrained {len(hist['train_loss'])} epochs "
          f"(best epoch {hist.get('best_epoch')}) in {mins:.1f} min")
    print(f"\n===== EXTERNAL COHORT ({target_era}), SUBJECT LEVEL =====")
    print(classification_report(subj_df["true"], subj_df["pred"], labels=BINARY,
                                zero_division=0))
    print(f"confusion matrix (rows=true, cols=pred) {BINARY}:")
    print(confusion_matrix(subj_df["true"], subj_df["pred"], labels=BINARY))
    print(f"\naccuracy    {acc:.1%}  ({correct}/{n})   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"majority baseline {majority:.1%}")
    print(f"ROC AUC     {auc:.3f}   95% CI [{alo:.3f}, {ahi:.3f}]")
    if alo > 0.5:
        print("            ^ interval excludes 0.5 -> the signal TRANSFERS across cohorts")
    else:
        print("            ^ interval includes 0.5 -> no demonstrated transfer")

    if hi < majority:
        verdict = "FAILS TO TRANSFER - below the trivial baseline on the new cohort"
    elif lo > majority:
        verdict = "TRANSFERS - significantly above baseline on a cohort never trained on"
    else:
        verdict = "INCONCLUSIVE - CI straddles the baseline"
    print(f"verdict: {verdict}")

    out = {
        "task": f"AD vs CN, trained on {source_era}, tested on {target_era} "
                f"(external cohort, zero subject overlap)",
        "arch": arch, "direction": direction, "select_by": select_by,
        "train_cohort": source_era, "test_cohort": target_era,
        "n_train_subjects": int(subj.get("train", 0)),
        "n_val_subjects": int(subj.get("val", 0)),
        "n_test_subjects": n,
        "accuracy": acc, "accuracy_95CI": [lo, hi], "majority_baseline": majority,
        "roc_auc": auc, "roc_auc_95CI": [alo, ahi],
        "decision_threshold": best_thr, "verdict": verdict,
        "macro_f1": f1_score(subj_df["true"], subj_df["pred"], labels=BINARY,
                             average="macro"),
        "epochs_run": len(hist["train_loss"]), "best_epoch": hist.get("best_epoch"),
        "train_minutes": round(mins, 1),
    }
    with open(os.path.join(ROOT, "reports", f"{tag}_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    subj_df.to_csv(os.path.join(ROOT, "reports", f"{tag}_subject_preds.csv"))
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mobilenetv2",
         sys.argv[2] if len(sys.argv) > 2 else "adni1->go2",
         sys.argv[3] if len(sys.argv) > 3 else "val_loss",
         sys.argv[4] if len(sys.argv) > 4 else "manifest_v3_adcn.csv")
