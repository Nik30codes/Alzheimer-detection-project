"""5-fold subject-level CV: does skull/scalp masking actually help AD vs CN?

Why this exists: on the single 75-subject test split, masking gained +2.7 accuracy points
on MobileNetV2 and +6.7 on the custom CNN -- but every confidence interval overlapped, and
this project has twice reported a single-split "win" that reversed under cross-validation
(decision 29: a +5.4-point gain became -3.5). Pooling out-of-fold predictions over all 501
AD/CN subjects narrows the interval from roughly +/-9 accuracy points to +/-4.

PAIRED BY CONSTRUCTION. Both arms use the IDENTICAL folds and the IDENTICAL subjects, and
the masked images are pixel-for-pixel the same slices with skull, scalp and background
zeroed. The arms therefore differ in exactly one variable, so the delta can be bootstrapped
per subject rather than compared as two independent intervals -- which is far more
sensitive, because the per-subject difficulty cancels out.

Usage: python scripts/cv_mask_adcn.py <arch> [k]
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
from models import SimpleCNN, build_mobilenetv2                         # noqa: E402
from train import train_model                                           # noqa: E402
from train_binary_adni1 import BinaryDataset, BINARY, wilson_ci, auc_ci  # noqa: E402
from sklearn.metrics import roc_auc_score                               # noqa: E402

SEED, BATCH, LR, WD, EPOCHS, PATIENCE = 42, 32, 1e-3, 1e-4, 40, 7
ARMS = {"plain": "manifest_v3_adcn.csv", "masked": "manifest_v3_masked.csv"}


def folds_over(subj, k, seed=SEED):
    """Stratified k-fold over SUBJECTS.

    Depends only on subject id and class, both of which the two manifests share, so the
    fold assignment is byte-identical between arms. That is what makes the comparison
    paired.
    """
    rng = np.random.default_rng(seed)
    f = [[] for _ in range(k)]
    for cls in BINARY:
        ids = np.array(sorted(subj[subj == cls].index))
        rng.shuffle(ids)
        for i, s in enumerate(ids):
            f[i % k].append(s)
    return [set(x) for x in f]


def build(arch):
    if arch == "custom_cnn":
        return SimpleCNN(num_classes=2, in_channels=1)
    return build_mobilenetv2(2, pretrained=False)


def _subject_probs(model, loader, dev):
    rows = []
    with torch.no_grad():
        for imgs, lab, sids in loader:
            p = torch.softmax(model(imgs.to(dev)), dim=1).cpu().numpy()
            for j in range(len(lab)):
                rows.append({"subject_id": sids[j], "true": BINARY[lab[j]],
                             "p_AD": float(p[j, 1])})
    g = pd.DataFrame(rows).groupby("subject_id")
    return pd.DataFrame({"true": g["true"].first(), "p_AD": g["p_AD"].mean()})


def _youden_threshold(vd):
    """Pick the cut-point on VALIDATION subjects only -- never the test fold."""
    y = (vd["true"] == "AD").astype(int).values
    best, bj = 0.5, -1.0
    for t in np.unique(np.round(vd["p_AD"].values, 4)):
        pr = (vd["p_AD"].values >= t).astype(int)
        tp = int(((pr == 1) & (y == 1)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
        tn = int(((pr == 0) & (y == 0)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > bj:
            bj, best = j, float(t)
    return best


def run_arm(arch, manifest_file, folds, tag, select_by="val_macro_f1"):
    m = pd.read_csv(os.path.join(ROOT, "data", manifest_file))
    m = m[m["class"].isin(BINARY)].reset_index(drop=True)
    subj = m.groupby("subject_id")["class"].first()
    all_ids = set(subj.index)
    rgb = arch != "custom_cnn"
    tr_t, ev_t = ((TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB) if rgb
                  else (TRAIN_TRANSFORM, EVAL_TRANSFORM))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof = []
    for i, test_ids in enumerate(folds):
        rest = sorted(all_ids - test_ids)
        rng = np.random.default_rng(SEED + i)
        rc = subj.loc[rest]
        val = set()
        for c in BINARY:
            v = np.array(sorted(rc[rc == c].index))
            rng.shuffle(v)
            val.update(v[:max(1, int(round(0.15 * len(v))))])

        mm = m.copy()
        mm["split"] = np.where(mm.subject_id.isin(test_ids), "test",
                               np.where(mm.subject_id.isin(val), "val", "train"))
        ld = {sp: DataLoader(BinaryDataset(mm, sp, tr_t if sp == "train" else ev_t),
                             batch_size=BATCH, shuffle=(sp == "train"), num_workers=2)
              for sp in ("train", "val", "test")}

        cnt = mm[mm.split == "train"]["class"].value_counts()
        w = torch.tensor([len(mm[mm.split == "train"]) / cnt[c] for c in BINARY],
                         dtype=torch.float32)
        w = w / w.sum() * 2

        model = build(arch)
        t0 = time.time()
        train_model(model, ld["train"], ld["val"], w, dev, epochs=EPOCHS, lr=LR,
                    patience=PATIENCE, weight_decay=WD, select_by=select_by,
                    checkpoint_path=os.path.join(ROOT, "models", "checkpoints",
                                                 f"cvmask_{tag}_f{i+1}.pt"))
        model.eval()
        fd = _subject_probs(model, ld["test"], dev)
        thr = _youden_threshold(_subject_probs(model, ld["val"], dev))
        fd["pred"] = np.where(fd["p_AD"] >= thr, "AD", "CN")
        fd["fold"] = i + 1
        oof.append(fd)
        print(f"    fold {i+1}: acc {(fd.pred == fd.true).mean():.1%}  thr {thr:.3f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.concat(oof)


def main(arch="mobilenetv2", k=5, select_by="val_macro_f1", arms=None):
    """arms=("plain",) with select_by="val_loss" reproduces the DEPLOYED configuration
    (train_binary_adni1.py passes no select_by, so it defaults to val_loss) under
    cross-validation -- which is how the 82.7% single-split headline gets checked."""
    active = {k2: v for k2, v in ARMS.items() if arms is None or k2 in arms}
    base = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3_adcn.csv"))
    base = base[base["class"].isin(BINARY)]
    subj = base.groupby("subject_id")["class"].first()
    folds = folds_over(subj, k)
    print(f"===== masked vs plain, {arch}, {k}-fold over {len(subj)} subjects =====")
    print(f"fold sizes {[len(f) for f in folds]}", flush=True)

    res = {}
    for arm, mf in active.items():
        print(f"\n  --- {arm} ({mf}) ---", flush=True)
        res[arm] = run_arm(arch, mf, folds, f"{arch}_{arm}_{select_by}", select_by)

    out = {"arch": arch, "k": k, "n_subjects": int(len(subj)),
           "select_by": select_by, "arms": list(active)}
    for arm, d in res.items():
        n = len(d)
        corr = int((d.true == d.pred).sum())
        lo, hi = wilson_ci(corr, n)
        auc = roc_auc_score((d.true == "AD").astype(int), d.p_AD)
        npos = int((d.true == "AD").sum())
        alo, ahi = auc_ci(auc, npos, n - npos)
        out[arm] = {"accuracy": corr / n, "accuracy_95CI": [lo, hi],
                    "roc_auc": auc, "roc_auc_95CI": [alo, ahi], "n": n}
        print(f"\n{arm:7s} acc {corr / n:.1%} [{lo:.1%},{hi:.1%}]  "
              f"AUC {auc:.4f} [{alo:.3f},{ahi:.3f}]", flush=True)

    if len(res) < 2:
        with open(os.path.join(ROOT, "reports",
                  f"cvheadline_{arch}_{select_by}.json"), "w") as f:
            json.dump(out, f, indent=2)
        list(res.values())[0].to_csv(os.path.join(
            ROOT, "reports", f"cvheadline_{arch}_{select_by}_oof.csv"))
        print("")
        print(f"wrote reports/cvheadline_{arch}_{select_by}.json")
        return

    # ---- paired bootstrap: both arms scored the SAME people ----
    a = res["plain"].sort_index()
    b = res["masked"].sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    ya = (a.true == "AD").astype(int).values
    rng = np.random.default_rng(0)
    d_auc, d_acc = [], []
    for _ in range(4000):
        idx = rng.integers(0, len(common), len(common))
        if len(set(ya[idx])) < 2:
            continue
        d_auc.append(roc_auc_score(ya[idx], b.p_AD.values[idx])
                     - roc_auc_score(ya[idx], a.p_AD.values[idx]))
        d_acc.append((b.pred.values[idx] == b.true.values[idx]).mean()
                     - (a.pred.values[idx] == a.true.values[idx]).mean())
    da = np.percentile(d_auc, [2.5, 97.5])
    dc = np.percentile(d_acc, [2.5, 97.5])
    sig = bool(da[0] > 0 or da[1] < 0)
    out["paired_delta"] = {
        "d_auc": float(np.mean(d_auc)), "d_auc_95CI": [float(da[0]), float(da[1])],
        "d_acc": float(np.mean(d_acc)), "d_acc_95CI": [float(dc[0]), float(dc[1])],
        "n_paired": int(len(common)),
    }
    out["verdict"] = "significant" if sig else "not significant"
    print(f"\nPAIRED delta (masked - plain) over {len(common)} subjects:")
    print(f"  AUC {np.mean(d_auc):+.4f}  95% CI [{da[0]:+.4f}, {da[1]:+.4f}]")
    print(f"  acc {np.mean(d_acc):+.1%}  95% CI [{dc[0]:+.1%}, {dc[1]:+.1%}]")
    print(f"  VERDICT: {'SIGNIFICANT' if sig else 'NOT significant (interval includes zero)'}")

    with open(os.path.join(ROOT, "reports", f"cvmask_{arch}_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    for arm, d in res.items():
        d.to_csv(os.path.join(ROOT, "reports", f"cvmask_{arch}_{arm}_oof.csv"))
    print(f"\nwrote reports/cvmask_{arch}_result.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mobilenetv2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 5,
         sys.argv[3] if len(sys.argv) > 3 else "val_macro_f1",
         tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else None)
