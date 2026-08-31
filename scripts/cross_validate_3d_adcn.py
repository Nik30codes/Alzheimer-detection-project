"""
5-fold subject-level cross-validation: Simple3DCNN (whole 32-slice volume) vs the
deployed 2D slice-averaged headline, on the same task and the same data.

Why this exists: this project's honest headline (AD vs CN, 74.1% acc / ROC AUC 0.784,
5-fold CV over 501 subjects, reports/mobilenetv2_ADvsCN_cv_result.json) scores each of a
subject's 32 axial slices independently and averages the probabilities. Decision 30 in
CLAUDE.md found those 32 predictions are only ~1.3 effective independent measurements
after averaging, and concluded cross-slice modelling has to go INSIDE the backbone.
This is that: a true Conv3d model over the whole stack at once, on IDENTICAL subjects
and folds as the 2D headline, so the two numbers are a fair comparison.

Deliberately reuses manifest_v3_adcn.csv (no new DICOM extraction) and val_loss
checkpoint selection (matches the deployed 2D configuration exactly, decision 33's
"cvheadline" procedure) so nothing except the architecture differs.

Never a single split (decision 33: three single-split numbers evaporated under CV in
this project already) -- every number reported here is pooled out-of-fold over all 501
subjects.

Usage: python scripts/cross_validate_3d_adcn.py [k] [epochs] [batch_size]
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

from datasets import MRI3DDataset, VOLUME_TRAIN_TRANSFORM, VOLUME_EVAL_TRANSFORM  # noqa: E402
from models import Simple3DCNN                                                    # noqa: E402
from train import train_model                                                     # noqa: E402
from sklearn.metrics import roc_auc_score                                         # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_binary_adni1 import wilson_ci, auc_ci  # noqa: E402

SEED, LR, WD, PATIENCE = 42, 1e-3, 1e-4, 7
BINARY = ["CN", "AD"]
TAG = "simple3dcnn_adcn_v3"


def folds_over(subj, k, seed=SEED):
    """Stratified k-fold over SUBJECTS. Same construction as cv_mask_adcn.folds_over --
    depends only on subject_id and class, both shared with manifest_v3_adcn.csv, so the
    fold assignment here is IDENTICAL to the one behind the 2D headline. That makes the
    two CV results directly comparable, not just similarly-shaped."""
    rng = np.random.default_rng(seed)
    f = [[] for _ in range(k)]
    for cls in BINARY:
        ids = np.array(sorted(subj[subj == cls].index))
        rng.shuffle(ids)
        for i, s in enumerate(ids):
            f[i % k].append(s)
    return [set(x) for x in f]


def _youden_threshold(val_df):
    """Pick the cut-point on VALIDATION subjects only -- never the test fold (decision 35:
    fitting a threshold on a single small split badly miscalibrated the deployed model)."""
    y = (val_df["true"] == "AD").astype(int).values
    best, bj = 0.5, -1.0
    for t in np.unique(np.round(val_df["p_AD"].values, 4)):
        pr = (val_df["p_AD"].values >= t).astype(int)
        tp = int(((pr == 1) & (y == 1)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
        tn = int(((pr == 0) & (y == 0)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > bj:
            bj, best = j, float(t)
    return best


def subject_probs(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for vols, labels, sids in loader:
            p = torch.softmax(model(vols.to(device)), dim=1).cpu().numpy()
            for i in range(len(labels)):
                rows.append({"subject_id": sids[i], "true": BINARY[labels[i]],
                            "p_AD": float(p[i, 1])})
    return pd.DataFrame(rows)  # one row per subject already -- one sample IS one subject


def main(k=5, epochs=40, batch_size=8):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3_adcn.csv"))
    manifest = manifest[manifest["class"].isin(BINARY)].reset_index(drop=True)
    subj = manifest.groupby("subject_id")["class"].first()
    all_ids = set(subj.index)
    folds = folds_over(subj, k)

    print(f"===== {TAG}, {k}-fold CV over {len(subj)} subjects =====")
    print(f"fold sizes {[len(f) for f in folds]}")
    print(f"class balance: {subj.value_counts().to_dict()}", flush=True)

    oof, per_fold = [], []
    t_start = time.time()
    markers_dir = os.path.join(ROOT, "reports", "run_logs")

    for i, test_ids in enumerate(folds):
        marker = os.path.join(markers_dir, f"{TAG}_fold{i+1}_done.json")
        oof_path = os.path.join(markers_dir, f"{TAG}_fold{i+1}_oof.csv")
        if os.path.exists(marker) and os.path.exists(oof_path):
            # RESTART-SAFETY: a prior run of this script was interrupted (the first
            # attempt was killed externally partway through fold 2, no Python error --
            # see CLAUDE.md infra notes on multi-hour queues needing per-job result
            # markers). Skip folds already trained instead of re-spending GPU time.
            with open(marker) as f:
                per_fold.append(json.load(f))
            fd = pd.read_csv(oof_path, index_col=0)
            oof.append(fd)
            print(f"\n--- fold {i+1}/{k}: RESUMED from {marker} "
                  f"(acc {per_fold[-1]['accuracy']:.1%}) ---", flush=True)
            continue

        print(f"\n--- fold {i+1}/{k}: {len(test_ids)} test subjects ---", flush=True)
        rest = sorted(all_ids - test_ids)
        rng = np.random.default_rng(SEED + i)
        rc = subj.loc[rest]
        val_ids = set()
        for c in BINARY:
            v = np.array(sorted(rc[rc == c].index))
            rng.shuffle(v)
            val_ids.update(v[:max(1, int(round(0.15 * len(v))))])
        assert not (test_ids & val_ids), "val leaked into test"

        mm = manifest.copy()
        mm["split"] = np.where(mm.subject_id.isin(test_ids), "test",
                               np.where(mm.subject_id.isin(val_ids), "val", "train"))

        # num_workers=0 deliberately: the host is currently under real memory pressure
        # (~3GB free of 15.5GB, 6 days uptime) and spawning extra worker processes per
        # loader adds to that. Single-process loading is a fine tradeoff here -- each
        # batch is GPU-compute-heavy (32x224x224 volumes through Conv3d), so I/O latency
        # hiding from worker parallelism matters less than it would for a lighter model.
        loaders = {
            "train": DataLoader(MRI3DDataset(mm, "train", VOLUME_TRAIN_TRANSFORM, classes=BINARY),
                                batch_size=batch_size, shuffle=True, num_workers=0),
            "val": DataLoader(MRI3DDataset(mm, "val", VOLUME_EVAL_TRANSFORM, classes=BINARY),
                              batch_size=batch_size, shuffle=False, num_workers=0),
            "test": DataLoader(MRI3DDataset(mm, "test", VOLUME_EVAL_TRANSFORM, classes=BINARY),
                               batch_size=batch_size, shuffle=False, num_workers=0),
        }

        train_subj = mm[mm.split == "train"].groupby("subject_id")["class"].first()
        counts = train_subj.value_counts()
        w = torch.tensor([len(train_subj) / counts[c] for c in BINARY], dtype=torch.float32)
        w = w / w.sum() * 2

        model = Simple3DCNN(num_classes=2, in_channels=1)
        ckpt = os.path.join(ROOT, "models", "checkpoints", f"{TAG}_fold{i+1}.pt")
        t0 = time.time()
        # select_by="val_loss" matches the DEPLOYED 2D configuration exactly (decision 33's
        # cvheadline procedure) so this is comparable to the 0.784 AUC headline, not to the
        # separately-CV'd val_macro_f1 masking numbers.
        hist = train_model(model, loaders["train"], loaders["val"], w, device,
                           epochs=epochs, lr=LR, patience=PATIENCE, weight_decay=WD,
                           checkpoint_path=ckpt, select_by="val_loss")

        val_df = subject_probs(model, loaders["val"], device)
        thr = _youden_threshold(val_df)
        fd = subject_probs(model, loaders["test"], device)
        fd["pred"] = np.where(fd["p_AD"] >= thr, "AD", "CN")
        fd["fold"] = i + 1
        oof.append(fd)

        acc = (fd.pred == fd.true).mean()
        train_acc_best = hist["train_acc"][hist["best_epoch"] - 1]
        val_acc_best = hist["val_acc"][hist["best_epoch"] - 1]
        mins = (time.time() - t0) / 60
        print(f"  fold {i+1}: acc {acc:.1%}  thr {thr:.3f}  "
              f"train_acc@best {train_acc_best:.3f}  val_acc@best {val_acc_best:.3f}  "
              f"(gap {train_acc_best - val_acc_best:+.3f})  "
              f"best_epoch {hist['best_epoch']}/{len(hist['train_loss'])}  {mins:.1f} min",
              flush=True)
        fold_record = {
            "fold": i + 1, "n_test": len(fd), "accuracy": float(acc), "threshold": thr,
            "train_acc_at_best": float(train_acc_best), "val_acc_at_best": float(val_acc_best),
            "overfit_gap": float(train_acc_best - val_acc_best),
            "best_epoch": hist["best_epoch"], "epochs_run": len(hist["train_loss"]),
            "minutes": round(mins, 1),
        }
        per_fold.append(fold_record)
        # Write the marker + oof predictions BEFORE moving to the next fold, so an
        # interruption during a later fold doesn't cost this one's ~15-25 min of GPU time.
        with open(marker, "w") as f:
            json.dump(fold_record, f, indent=2)
        fd.to_csv(oof_path)

    oof_df = pd.concat(oof)
    assert len(oof_df) == len(subj), f"{len(oof_df)} predictions for {len(subj)} subjects"
    n = len(oof_df)
    correct = int((oof_df["true"] == oof_df["pred"]).sum())
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    auc = roc_auc_score((oof_df["true"] == "AD").astype(int), oof_df["p_AD"])
    n_pos = int((oof_df["true"] == "AD").sum())
    alo, ahi = auc_ci(auc, n_pos, n - n_pos)
    majority = oof_df["true"].value_counts().max() / n
    fold_accs = [f["accuracy"] for f in per_fold]
    fold_gaps = [f["overfit_gap"] for f in per_fold]

    print(f"\n===== {TAG} POOLED OVER ALL {n} SUBJECTS =====")
    print(f"accuracy   {acc:.1%} ({correct}/{n})   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"ROC AUC    {auc:.4f}   95% CI [{alo:.4f}, {ahi:.4f}]")
    print(f"majority baseline {majority:.1%}")
    print(f"per-fold accuracy: {[f'{a:.1%}' for a in fold_accs]}  (sd {np.std(fold_accs):.3f})")
    print(f"per-fold train/val overfit gap: {[f'{g:+.3f}' for g in fold_gaps]}  "
          f"(mean {np.mean(fold_gaps):+.3f})")

    if alo > 0.5:
        auc_verdict = "interval excludes 0.5 -> genuine ranking signal"
    else:
        auc_verdict = "interval includes 0.5 -> not established"
    if lo > majority:
        acc_verdict = "REAL SIGNAL - significantly above majority baseline"
    elif hi < majority:
        acc_verdict = "BELOW BASELINE"
    else:
        acc_verdict = "INCONCLUSIVE - CI straddles the baseline"
    print(f"AUC verdict: {auc_verdict}")
    print(f"accuracy verdict: {acc_verdict}")

    ref_path = os.path.join(ROOT, "reports", "mobilenetv2_ADvsCN_cv_result.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        print(f"\n2D slice-averaged headline for comparison: "
              f"acc {ref['accuracy']:.1%}  AUC {ref['roc_auc']:.4f} "
              f"[{ref['roc_auc_95CI'][0]:.4f}, {ref['roc_auc_95CI'][1]:.4f}]")
        print(f"3D volume result:                          "
              f"acc {acc:.1%}  AUC {auc:.4f} [{alo:.4f}, {ahi:.4f}]")

    out = {
        "tag": TAG, "task": "AD vs CN, 5-fold subject-level CV, Simple3DCNN (whole-volume)",
        "comparison": "reports/mobilenetv2_ADvsCN_cv_result.json (2D slice-averaged headline)",
        "k": k, "epochs_budget": epochs, "batch_size": batch_size,
        "n_subjects": n, "accuracy": acc, "accuracy_95CI": [lo, hi],
        "roc_auc": auc, "roc_auc_95CI": [alo, ahi], "majority_baseline": majority,
        "auc_verdict": auc_verdict, "accuracy_verdict": acc_verdict,
        "per_fold": per_fold, "fold_accuracy_sd": float(np.std(fold_accs)),
        "mean_overfit_gap": float(np.mean(fold_gaps)),
        "total_minutes": round((time.time() - t_start) / 60, 1),
        "select_by": "val_loss",
        "note": ("select_by=val_loss to match the DEPLOYED 2D configuration exactly. "
                 "Threshold fit with Youden's J on each fold's own validation subjects, "
                 "never on that fold's test subjects."),
    }
    with open(os.path.join(ROOT, "reports", f"{TAG}_cv_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    oof_df.to_csv(os.path.join(ROOT, "reports", f"{TAG}_cv_oof_preds.csv"))
    print(f"\n{TAG} DONE in {out['total_minutes']} min -- "
          f"wrote reports/{TAG}_cv_result.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5,
         int(sys.argv[2]) if len(sys.argv) > 2 else 40,
         int(sys.argv[3]) if len(sys.argv) > 3 else 8)
