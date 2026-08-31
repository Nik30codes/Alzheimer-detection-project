"""Subject-level k-fold cross-validation for the four-way ADNI-GO/2 task.

Why this exists: every four-way conclusion so far rests on ONE 93-subject test set,
where the 95% confidence interval is about +/-10 accuracy points. That is wide enough
that "45.2% beats the 39.8% baseline" and "they are the same" are both consistent with
the data. Cross-validation gives every one of the 618 subjects an out-of-fold
prediction, so the pooled estimate is computed on 618 instead of 93 and the interval
narrows by roughly a factor of 2.6.

LEAKAGE, and why the pretrained configurations are handled the way they are.
The obvious implementation -- reuse the existing ssl_encoder.pt or the AD-vs-CN
checkpoint for every fold -- is WRONG. Those were fitted on the train split of the
original single split, and under cross-validation those same subjects rotate into test
folds. The encoder would then have already seen the very subjects being evaluated on.
The autoencoder uses no labels, but it still fits the model to those images, and this
project has quantified what that kind of contamination is worth (+36.9 points).

So pretraining is redone INSIDE each fold, on that fold's non-test subjects only:
  config=random  no pretraining
  config=ssl     masked autoencoder trained per fold, on every non-test subject of all
                 853 (both eras -- unlabelled data is free and more of it is better)
  config=adcn    AD-vs-CN binary model trained per fold on non-test AD/CN subjects,
                 then its backbone initialises the four-way model

Usage: python scripts/cross_validate.py <arch> <random|ssl|adcn> [k] [ssl_epochs]
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from autoencoder import ConvAutoencoder, random_mask                    # noqa: E402
from datasets import (CLASSES, MRIDataset, compute_class_weights,       # noqa: E402
                      TRAIN_TRANSFORM, EVAL_TRANSFORM,
                      TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB)
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0  # noqa: E402
from train import train_model                                          # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,  # noqa: E402
                             classification_report)

SEED = 42
BATCH, LR, WD = 32, 1e-3, 1e-4
EPOCHS, PATIENCE = 40, 7
BINARY = ["CN", "AD"]
ALL_SUBJECTS = set()   # filled in main(); what pretraining is allowed to see


def wilson_ci(correct, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def make_folds(subj, k, seed=SEED):
    """Stratified k-fold over SUBJECTS (never slices), so a person is wholly inside
    exactly one fold. subj is a Series indexed by subject_id giving the class."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in CLASSES:
        ids = np.array(sorted(subj[subj == cls].index))
        rng.shuffle(ids)
        for i, sid in enumerate(ids):       # deal round-robin -> even class balance
            folds[i % k].append(sid)
    return [set(f) for f in folds]


def build_loaders(m, arch, splits):
    rgb = arch != "custom_cnn"
    train_t, eval_t = ((TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB) if rgb
                       else (TRAIN_TRANSFORM, EVAL_TRANSFORM))
    mm = m.copy()
    mm["split"] = mm["subject_id"].map(splits)
    loaders = {sp: DataLoader(MRIDataset(mm, sp, train_t if sp == "train" else eval_t),
                              batch_size=BATCH, shuffle=(sp == "train"), num_workers=2)
               for sp in ("train", "val", "test")}
    return mm, loaders, (3 if rgb else 1)


class BinaryDS(MRIDataset):
    """CN->0, AD->1 instead of the 4-class indices.

    Defined at module level on purpose: Windows spawns DataLoader workers and pickles
    the dataset to them, and a class defined inside a function cannot be pickled
    ("Can't pickle local object"). That crashed the adcn config on its first run.
    """

    def __getitem__(self, idx):
        import cv2
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["filepath"])
        return self.transform(img), BINARY.index(row["class"]), row["subject_id"]


def build_model(arch, in_ch, n_out=4):
    if arch == "custom_cnn":
        return SimpleCNN(num_classes=n_out, in_channels=in_ch)
    if arch == "mobilenetv2":
        return build_mobilenetv2(n_out, pretrained=False)
    if arch == "efficientnet_b0":
        return build_efficientnet_b0(n_out, pretrained=False)
    raise ValueError(arch)


def pretrain_ssl(allowed_ids, device, epochs, fold_i):
    """Masked autoencoder on every non-test subject, both eras, labels unused."""
    full = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3.csv"))
    sub = full[full["subject_id"].isin(allowed_ids)].copy()
    sub["split"] = "train"
    print(f"    SSL pretrain on {sub.subject_id.nunique()} subjects / {len(sub)} slices "
          f"({epochs} epochs)", flush=True)
    loader = DataLoader(MRIDataset(sub, "train", TRAIN_TRANSFORM), batch_size=48,
                        shuffle=True, num_workers=4, drop_last=True)
    ae = ConvAutoencoder(in_channels=1).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss(reduction="none")
    for ep in range(epochs):
        ae.train()
        tot, n = 0.0, 0
        for imgs, _, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            masked, mask = random_mask(imgs)
            opt.zero_grad()
            loss = (crit(ae(masked), imgs) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
            tot += loss.item() * imgs.size(0)
            n += imgs.size(0)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"      ssl fold{fold_i} epoch {ep+1}/{epochs} loss {tot/n:.4f}",
                  flush=True)
    return ae.encoder_state_dict()


def pretrain_adcn(allowed_ids, arch, device, fold_i):
    """AD-vs-CN binary model on non-test AD/CN subjects; returns its state dict."""
    adcn = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3_adcn.csv"))
    sub = adcn[adcn["subject_id"].isin(allowed_ids)].copy()
    ids = sub.groupby("subject_id")["class"].first()
    rng = np.random.default_rng(SEED + fold_i)
    val_ids = set()
    for c in BINARY:
        v = np.array(sorted(ids[ids == c].index))
        rng.shuffle(v)
        val_ids.update(v[:max(1, int(0.15 * len(v)))])
    sub["split"] = np.where(sub["subject_id"].isin(val_ids), "val", "train")
    print(f"    AD-vs-CN pretrain on {sub.subject_id.nunique()} subjects", flush=True)

    rgb = arch != "custom_cnn"
    train_t, eval_t = ((TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB) if rgb
                       else (TRAIN_TRANSFORM, EVAL_TRANSFORM))
    ld = {sp: DataLoader(BinaryDS(sub, sp, train_t if sp == "train" else eval_t),
                         batch_size=BATCH, shuffle=(sp == "train"), num_workers=2)
          for sp in ("train", "val")}
    counts = sub[sub.split == "train"]["class"].value_counts()
    w = torch.tensor([len(sub[sub.split == "train"]) / counts[c] for c in BINARY],
                     dtype=torch.float32)
    w = w / w.sum() * 2
    model = build_model(arch, 3 if rgb else 1, n_out=2)
    ckpt = os.path.join(ROOT, "models", "checkpoints", f"cv_tmp_adcn_{arch}_{fold_i}.pt")
    train_model(model, ld["train"], ld["val"], w, device, epochs=25, lr=LR,
                patience=6, checkpoint_path=ckpt, weight_decay=WD,
                select_by="val_macro_f1")
    return model.state_dict()


def load_backbone(model, sd):
    own = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    print(f"    initialised {len(keep)}/{len(own)} tensors from pretrained weights",
          flush=True)
    return model


def main(arch="custom_cnn", config="random", k=5, ssl_epochs=15):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3_go2.csv"))
    subj = m.groupby("subject_id")["class"].first()
    folds = make_folds(subj, k)
    all_ids = set(subj.index)

    # Every subject in the project, including the ADNI1 ones outside the GO/2 four-way
    # pool. Used only to decide what pretraining may see (see the fold loop).
    global ALL_SUBJECTS
    ALL_SUBJECTS = set(pd.read_csv(
        os.path.join(ROOT, "data", "manifest_v3.csv"))["subject_id"].unique())
    print(f"pretraining pool: {len(ALL_SUBJECTS)} subjects total "
          f"({len(ALL_SUBJECTS) - len(all_ids)} outside the GO/2 four-way pool)")

    tag = f"cv_{arch}_{config}_k{k}"
    print(f"===== {tag} on {device} =====")
    print(f"{len(subj)} subjects, {k} folds "
          f"(sizes {[len(f) for f in folds]})")
    print(f"class balance: {subj.value_counts().to_dict()}")

    oof = []          # out-of-fold subject-level predictions
    per_fold = []
    t_start = time.time()

    for i, test_ids in enumerate(folds):
        print(f"\n--- fold {i+1}/{k}: {len(test_ids)} test subjects ---", flush=True)
        rest = sorted(all_ids - test_ids)
        rng = np.random.default_rng(SEED + i)
        rest_cls = subj.loc[rest]
        val_ids = set()
        for c in CLASSES:                      # stratified val carved from the rest
            v = np.array(sorted(rest_cls[rest_cls == c].index))
            rng.shuffle(v)
            val_ids.update(v[:max(1, int(round(0.15 * len(v))))])
        splits = {s: ("test" if s in test_ids else "val" if s in val_ids else "train")
                  for s in all_ids}
        assert not (test_ids & val_ids), "val leaked into test"

        # Everything except THIS fold's test subjects is fair game for pretraining.
        # That deliberately includes the 235 ADNI1 subjects, which are not in the GO/2
        # four-way pool at all and therefore can never appear in a test fold -- an
        # earlier version restricted pretraining to GO/2 non-test subjects only and so
        # threw away a third of the available unlabelled data for no safety benefit.
        non_test = set(ALL_SUBJECTS) - test_ids

        mm, loaders, in_ch = build_loaders(m, arch, splits)
        model = build_model(arch, in_ch)

        if config == "ssl":
            sd = pretrain_ssl(non_test, device, ssl_epochs, i + 1)
            model = load_backbone(model, sd)
        elif config == "adcn":
            sd = pretrain_adcn(non_test, arch, device, i + 1)
            model = load_backbone(model, sd)

        cw = compute_class_weights(mm)
        ckpt = os.path.join(ROOT, "models", "checkpoints", f"{tag}_fold{i+1}.pt")
        t0 = time.time()
        hist = train_model(model, loaders["train"], loaders["val"], cw, device,
                           epochs=EPOCHS, lr=LR, patience=PATIENCE,
                           checkpoint_path=ckpt, weight_decay=WD,
                           select_by="val_macro_f1")

        # subject-level soft vote on this fold's held-out subjects
        model.eval()
        rows = []
        with torch.no_grad():
            for imgs, labels, sids in loaders["test"]:
                p = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
                for j in range(len(labels)):
                    rows.append({"subject_id": sids[j], "true": CLASSES[labels[j]],
                                 **{f"p_{c}": p[j, ci] for ci, c in enumerate(CLASSES)}})
        g = pd.DataFrame(rows).groupby("subject_id")
        fold_df = pd.DataFrame({"true": g["true"].first(),
                                **{f"p_{c}": g[f"p_{c}"].mean() for c in CLASSES}})
        fold_df["pred"] = fold_df[[f"p_{c}" for c in CLASSES]].values.argmax(1)
        fold_df["pred"] = [CLASSES[i_] for i_ in fold_df["pred"]]
        fold_df["fold"] = i + 1
        oof.append(fold_df)

        acc = accuracy_score(fold_df["true"], fold_df["pred"])
        f1 = f1_score(fold_df["true"], fold_df["pred"], labels=CLASSES, average="macro")
        mins = (time.time() - t0) / 60
        print(f"  fold {i+1}: acc {acc:.1%}  macroF1 {f1:.3f}  "
              f"best_epoch {hist.get('best_epoch')}/{len(hist['train_loss'])}  "
              f"{mins:.1f} min", flush=True)
        per_fold.append({"fold": i + 1, "n_test": len(fold_df), "accuracy": acc,
                         "macro_f1": f1, "best_epoch": hist.get("best_epoch"),
                         "epochs_run": len(hist["train_loss"])})

    # ---- pooled out-of-fold result: every subject predicted exactly once ----
    oof_df = pd.concat(oof)
    assert len(oof_df) == len(subj), f"{len(oof_df)} predictions for {len(subj)} subjects"
    n = len(oof_df)
    correct = int((oof_df["true"] == oof_df["pred"]).sum())
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    macro_f1 = f1_score(oof_df["true"], oof_df["pred"], labels=CLASSES, average="macro")
    baseline = oof_df["true"].value_counts().max() / n
    cm = confusion_matrix(oof_df["true"], oof_df["pred"], labels=CLASSES)

    print(f"\n===== {tag} POOLED OVER ALL {n} SUBJECTS =====")
    print(classification_report(oof_df["true"], oof_df["pred"], labels=CLASSES,
                                zero_division=0))
    print(f"confusion matrix (rows=true, cols=pred) {CLASSES}:")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    print(f"\naccuracy   {acc:.1%} ({correct}/{n})   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"macro F1   {macro_f1:.3f}")
    print(f"baseline   {baseline:.1%}   chance 25.0%")
    fold_accs = [f["accuracy"] for f in per_fold]
    print(f"per-fold accuracy: {[f'{a:.1%}' for a in fold_accs]}  "
          f"(sd {np.std(fold_accs):.3f})")
    if lo > baseline:
        verdict = "ABOVE BASELINE, significant"
    elif hi < baseline:
        verdict = "BELOW BASELINE"
    else:
        verdict = "INCONCLUSIVE - CI straddles the baseline"
    print(f"verdict: {verdict}")

    out = {
        "tag": tag, "arch": arch, "config": config, "k": k,
        "ssl_epochs": ssl_epochs if config == "ssl" else None,
        "n_subjects": n, "accuracy": acc, "accuracy_95CI": [lo, hi],
        "macro_f1": macro_f1, "majority_baseline": baseline, "verdict": verdict,
        "per_fold": per_fold, "fold_accuracy_sd": float(np.std(fold_accs)),
        "confusion_matrix": cm.tolist(), "classes": CLASSES,
        "per_class_recall": {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
                             for i, c in enumerate(CLASSES)},
        "total_minutes": round((time.time() - t_start) / 60, 1),
        "note": ("Pretraining (ssl/adcn) is redone inside each fold on non-test "
                 "subjects only, so no fold's test subjects were ever seen during "
                 "pretraining."),
    }
    with open(os.path.join(ROOT, "reports", f"{tag}_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    oof_df.to_csv(os.path.join(ROOT, "reports", f"{tag}_oof_preds.csv"))
    print(f"\n{tag} DONE in {out['total_minutes']} min")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "custom_cnn",
         sys.argv[2] if len(sys.argv) > 2 else "random",
         int(sys.argv[3]) if len(sys.argv) > 3 else 5,
         int(sys.argv[4]) if len(sys.argv) > 4 else 15)
