"""
Systematic tuning of the TRAINING RECIPE for AD vs CN (manifest_v3_adcn.csv, 501 subjects).

Why this file exists
--------------------
Every result this project has ever produced used one recipe that was never tuned:
Adam, lr 1e-3, weight_decay 1e-4, batch 32, 40 epochs, patience 7, mild augmentation
(RandomAffine +/-10deg / 5% translate / 0.95-1.05 scale + horizontal flip), checkpoint
selected on val_loss. Meanwhile the models overfit visibly (train acc 80%+ while val loss
rises from early epochs). Untuned recipe + clear overfitting = the most likely remaining
source of free accuracy.

Deliberately self-contained: it does NOT call src/train.py, because the knobs being tested
(mixup, label smoothing, cosine+warmup, gradient clipping, AUC-based checkpoint selection)
would each need a new argument there, and other jobs import that module. `baseline` here is
a faithful re-implementation of what train_binary_adni1.py + train.train_model do.

Modes
-----
  python scripts/tune_recipe.py val  <config> [seed]   screening on the FIXED train/val split.
                                                       NEVER touches the test split.
  python scripts/tune_recipe.py cv   <config> [k]      5-fold subject-level CV over all 501
                                                       AD/CN subjects; every subject gets one
                                                       out-of-fold prediction.
  python scripts/tune_recipe.py compare <cfgA> <cfgB>  paired bootstrap of the CV difference
                                                       (same folds, same subjects, so the
                                                       comparison is paired).
  python scripts/tune_recipe.py list                   print the config registry.

Every run writes a per-run JSON marker in reports/ so a queue is restart-safe, and every
checkpoint is named tune_*.pt so nothing existing is overwritten.

Evaluation protocol is identical to train_binary_adni1.py so numbers stay comparable to the
recorded AUC 0.9055: subject-level score = mean of per-slice softmax P(AD); decision
threshold = the one maximising Youden's J on the VALIDATION split (never on test).
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
import torchvision.transforms as T
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import MRIDataset, IMAGENET_MEAN, IMAGENET_STD   # noqa: E402
from models import build_mobilenetv2, build_efficientnet_b0, SimpleCNN  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix   # noqa: E402

BINARY = ["CN", "AD"]
SEED = 42
MANIFEST = os.path.join(ROOT, "data", "manifest_v3_adcn.csv")
ARCH = "mobilenetv2"          # the arch that produced the recorded 0.9055 headline
NUM_WORKERS = 4
REPORTS = os.path.join(ROOT, "reports")
CKPT_DIR = os.path.join(ROOT, "models", "checkpoints")


# ---------------------------------------------------------------- configurations
# Every key that is absent falls back to BASE. A config is therefore a diff against
# the current (untuned) recipe, which keeps the comparison honest and readable.
BASE = dict(
    lr=1e-3, weight_decay=1e-4, optimizer="adam", batch_size=32,
    epochs=40, patience=7, scheduler="plateau", warmup_epochs=0,
    label_smoothing=0.0, mixup_alpha=0.0, head_dropout=None, grad_clip=None,
    sampler="none", class_weighted_loss=True, select_by="val_loss",
    # augmentation
    degrees=10, translate=0.05, scale=(0.95, 1.05), hflip=0.5,
    rrc_scale=None, jitter=None, erasing=0.0,
)

CONFIGS = {
    # --- 0. the recipe every existing result used -------------------------------
    "baseline":      {},

    # --- 1. regularisation against the observed overfitting ---------------------
    "wd1e-3":        dict(weight_decay=1e-3),
    "wd1e-2":        dict(weight_decay=1e-2),
    "dropout50":     dict(head_dropout=0.5),          # MobileNetV2 head default is 0.2
    "ls0.05":        dict(label_smoothing=0.05),
    "ls0.10":        dict(label_smoothing=0.10),
    "mixup0.2":      dict(mixup_alpha=0.2),
    "mixup0.4":      dict(mixup_alpha=0.4),

    # --- 2. stronger augmentation (no vertical flips: anatomically wrong) -------
    "aug_rot20":     dict(degrees=20, translate=0.08),
    "aug_rrc":       dict(rrc_scale=(0.85, 1.0)),
    "aug_jitter":    dict(jitter=(0.2, 0.2)),
    "aug_erase":     dict(erasing=0.25),
    "aug_all":       dict(degrees=15, translate=0.08, rrc_scale=(0.85, 1.0),
                          jitter=(0.2, 0.2), erasing=0.25),

    # --- 3. optimisation --------------------------------------------------------
    "adamw":         dict(optimizer="adamw", weight_decay=1e-2),
    "cosine":        dict(optimizer="adamw", weight_decay=1e-2, lr=3e-4,
                          scheduler="cosine", warmup_epochs=3, epochs=60, patience=12),
    "gradclip":      dict(grad_clip=1.0),

    # --- 4. class balance -------------------------------------------------------
    "balanced":      dict(sampler="balanced", class_weighted_loss=False),
    "balanced_both": dict(sampler="balanced", class_weighted_loss=True),

    # --- 5. checkpoint selection (decision 16: val_loss is spiky on this data) --
    "sel_f1":        dict(select_by="val_macro_f1"),
    "sel_auc":       dict(select_by="val_auc"),
}

# Stage-2 combinations of whatever stage 1 liked. Registered up front so the names are
# stable and reproducible; the driver decides which of them actually run.
CONFIGS.update({
    "combo_a": dict(optimizer="adamw", weight_decay=1e-2, lr=3e-4, scheduler="cosine",
                    warmup_epochs=3, epochs=60, patience=12, grad_clip=1.0,
                    select_by="val_auc"),
    "combo_b": dict(optimizer="adamw", weight_decay=1e-2, lr=3e-4, scheduler="cosine",
                    warmup_epochs=3, epochs=60, patience=12, grad_clip=1.0,
                    select_by="val_auc", label_smoothing=0.05,
                    degrees=15, translate=0.08, rrc_scale=(0.85, 1.0)),
    "combo_c": dict(optimizer="adamw", weight_decay=1e-2, lr=3e-4, scheduler="cosine",
                    warmup_epochs=3, epochs=60, patience=12, grad_clip=1.0,
                    select_by="val_auc", mixup_alpha=0.2),
    "combo_d": dict(grad_clip=1.0, select_by="val_auc", weight_decay=1e-3),
})


def cfg(name):
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}; known: {sorted(CONFIGS)}")
    c = dict(BASE)
    c.update(CONFIGS[name])
    c["name"] = name
    return c


# ---------------------------------------------------------------- statistics
def wilson_ci(correct, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc_ci(auc, n_pos, n_neg, z=1.96):
    """Hanley-McNeil, same as train_binary_adni1.py so the numbers line up."""
    if n_pos == 0 or n_neg == 0:
        return (float("nan"), float("nan"))
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return (max(0.0, auc - z * se), min(1.0, auc + z * se))


def bootstrap_ci(y, p, stat, n_boot=4000, seed=SEED):
    """Percentile bootstrap over SUBJECTS. Used for the pooled CV numbers, where
    Hanley-McNeil's parametric assumptions are not obviously right."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(stat(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def youden_threshold(y_true, probs):
    """Threshold maximising sensitivity+specificity-1, chosen on VALIDATION only.
    Identical procedure to train_binary_adni1.py."""
    best_thr, best_j = 0.5, -1.0
    for thr in np.unique(np.round(probs, 4)):
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr, best_j


# ---------------------------------------------------------------- data
class BinaryDataset(MRIDataset):
    """CN->0, AD->1. Module level so Windows can pickle it to DataLoader workers."""

    def __getitem__(self, idx):
        import cv2
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["filepath"])
        return self.transform(img), BINARY.index(row["class"]), row["subject_id"]


def build_transforms(c):
    """Training augmentation built from the config; eval transform is always the
    plain one (no augmentation), matching every other script in the project.

    Deliberately NO vertical flip: axial brains are near-symmetric left-right (so the
    horizontal flip already in use is anatomically defensible) but not top-bottom --
    anterior/posterior is a real, informative axis.
    """
    train = [T.ToPILImage()]
    if c["degrees"] or c["translate"] or c["scale"]:
        train.append(T.RandomAffine(degrees=c["degrees"],
                                    translate=(c["translate"], c["translate"]),
                                    scale=c["scale"]))
    if c["hflip"]:
        train.append(T.RandomHorizontalFlip(p=c["hflip"]))
    if c["rrc_scale"]:
        train.append(T.RandomResizedCrop(224, scale=tuple(c["rrc_scale"]),
                                         ratio=(0.95, 1.05), antialias=True))
    if c["jitter"]:
        b, ct = c["jitter"]
        train.append(T.ColorJitter(brightness=b, contrast=ct))
    train += [T.Grayscale(num_output_channels=3), T.ToTensor(),
              T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    if c["erasing"]:
        train.append(T.RandomErasing(p=c["erasing"], scale=(0.02, 0.15), value=0.0))
    evalt = T.Compose([T.ToPILImage(), T.Grayscale(num_output_channels=3), T.ToTensor(),
                       T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return T.Compose(train), evalt


def build_loaders(m, c, splits_present=("train", "val")):
    train_t, eval_t = build_transforms(c)
    loaders = {}
    for sp in splits_present:
        ds = BinaryDataset(m, sp, train_t if sp == "train" else eval_t)
        if sp == "train" and c["sampler"] == "balanced":
            counts = ds.df["class"].value_counts()
            w = ds.df["class"].map(lambda cl: 1.0 / counts[cl]).values
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                            num_samples=len(ds), replacement=True)
            loaders[sp] = DataLoader(ds, batch_size=c["batch_size"], sampler=sampler,
                                     num_workers=NUM_WORKERS, persistent_workers=True)
        else:
            loaders[sp] = DataLoader(ds, batch_size=c["batch_size"], shuffle=(sp == "train"),
                                     num_workers=NUM_WORKERS, persistent_workers=True)
    return loaders


def class_weights(m, c, device):
    if not c["class_weighted_loss"]:
        return torch.ones(2, dtype=torch.float32, device=device)
    tr = m[m.split == "train"]
    counts = tr["class"].value_counts()
    w = torch.tensor([len(tr) / counts[cl] for cl in BINARY], dtype=torch.float32)
    return (w / w.sum() * 2).to(device)


def build_model(c):
    if ARCH == "custom_cnn":
        model = SimpleCNN(num_classes=2, in_channels=1)
    elif ARCH == "efficientnet_b0":
        model = build_efficientnet_b0(2, pretrained=False)
    else:
        model = build_mobilenetv2(2, pretrained=False)
    if c["head_dropout"] is not None and ARCH != "custom_cnn":
        model.classifier[0] = nn.Dropout(p=c["head_dropout"])
    return model


# ---------------------------------------------------------------- training
def make_optimizer(model, c):
    params = [p for p in model.parameters() if p.requires_grad]
    if c["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=c["lr"], weight_decay=c["weight_decay"])
    return torch.optim.Adam(params, lr=c["lr"], weight_decay=c["weight_decay"])


def lr_at(c, step, steps_per_epoch, total_steps):
    """Linear warmup then cosine decay to 0, computed per STEP."""
    warm = c["warmup_epochs"] * steps_per_epoch
    if step < warm:
        return c["lr"] * (step + 1) / max(warm, 1)
    prog = (step - warm) / max(total_steps - warm, 1)
    return c["lr"] * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """One eval pass -> slice loss/acc plus subject-level mean P(AD)."""
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    ys, ps, sids = [], [], []
    for imgs, labels, sid in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            out = model(imgs)
            loss = criterion(out, labels)
        prob = torch.softmax(out.float(), dim=1)[:, 1]
        tot_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n += imgs.size(0)
        ys.append(labels.cpu().numpy()); ps.append(prob.cpu().numpy()); sids += list(sid)
    y = np.concatenate(ys); p = np.concatenate(ps)
    df = pd.DataFrame({"subject_id": sids, "y": y, "p": p})
    g = df.groupby("subject_id").agg(y=("y", "first"), p=("p", "mean"))
    subj_auc = roc_auc_score(g["y"].values, g["p"].values) if g["y"].nunique() > 1 else 0.5
    return dict(loss=tot_loss / n, acc=correct / n,
                macro_f1=f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0),
                subject_auc=float(subj_auc), subjects=g)


def train_one(c, m, device, ckpt_path, seed=SEED, log=print):
    """Train on m['split']=='train', early-stop/select on m['split']=='val'.
    Returns (model, history). The test split is never read here."""
    torch.manual_seed(seed); np.random.seed(seed)
    loaders = build_loaders(m, c)
    model = build_model(c).to(device)
    w = class_weights(m, c, device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=c["label_smoothing"])
    eval_criterion = nn.CrossEntropyLoss(weight=w)   # selection metric must not move with LS
    opt = make_optimizer(model, c)
    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))
    plateau = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
               if c["scheduler"] == "plateau" else None)
    spe = len(loaders["train"])
    total_steps = spe * c["epochs"]

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc",
                            "val_macro_f1", "val_subject_auc", "lr")}
    best_score, best_epoch, bad, step = -float("inf"), 0, 0, 0
    rng = np.random.default_rng(seed)

    for epoch in range(c["epochs"]):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for imgs, labels, _ in loaders["train"]:
            if c["scheduler"] == "cosine":
                for gp in opt.param_groups:
                    gp["lr"] = lr_at(c, step, spe, total_steps)
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            lam, perm = 1.0, None
            if c["mixup_alpha"] > 0:
                lam = float(rng.beta(c["mixup_alpha"], c["mixup_alpha"]))
                perm = torch.randperm(imgs.size(0), device=device)
                imgs = lam * imgs + (1 - lam) * imgs[perm]

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out = model(imgs)
                loss = (criterion(out, labels) if perm is None else
                        lam * criterion(out, labels) + (1 - lam) * criterion(out, labels[perm]))
            scaler.scale(loss).backward()
            if c["grad_clip"]:
                # unscale first, otherwise the clip threshold is applied to fp16-scaled
                # gradients and means nothing. This is also the documented fp16-overflow
                # spike guard (CLAUDE.md decision 16 / "spiky val loss").
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * imgs.size(0)
            tc += (out.argmax(1) == labels).sum().item()
            tn += imgs.size(0)
            step += 1

        va = evaluate(model, loaders["val"], eval_criterion, device)
        if plateau is not None:
            plateau.step(va["loss"])
        hist["train_loss"].append(tl / tn); hist["train_acc"].append(tc / tn)
        hist["val_loss"].append(va["loss"]); hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"]); hist["val_subject_auc"].append(va["subject_auc"])
        hist["lr"].append(opt.param_groups[0]["lr"])

        score = {"val_loss": -va["loss"], "val_macro_f1": va["macro_f1"],
                 "val_auc": va["subject_auc"]}[c["select_by"]]
        log(f"  ep {epoch+1:2d}/{c['epochs']} train_loss {tl/tn:.4f} acc {tc/tn:.4f} | "
            f"val_loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['macro_f1']:.4f} "
            f"subjAUC {va['subject_auc']:.4f} | lr {opt.param_groups[0]['lr']:.2e}")

        if score > best_score:
            best_score, best_epoch, bad = score, epoch + 1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= c["patience"]:
                log(f"  early stop at epoch {epoch+1} (no {c['select_by']} gain for {c['patience']})")
                break

    model.load_state_dict(torch.load(ckpt_path))
    hist["best_epoch"] = best_epoch
    hist["epochs_run"] = len(hist["train_loss"])
    log(f"  restored epoch {best_epoch} (selected by {c['select_by']})")
    return model, hist


def subject_scores(model, loader, device):
    """subject_id -> (true label 0/1, mean P(AD) over that subject's slices)."""
    model.eval()
    rows = []
    with torch.no_grad():
        for imgs, labels, sids in loader:
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out = model(imgs.to(device))
            p = torch.softmax(out.float(), 1)[:, 1].cpu().numpy()
            for i in range(len(labels)):
                rows.append({"subject_id": sids[i], "y": int(labels[i]), "p": float(p[i])})
    g = pd.DataFrame(rows).groupby("subject_id")
    return pd.DataFrame({"y": g["y"].first(), "p": g["p"].mean()})


# ---------------------------------------------------------------- mode: val
def run_val(config_name, seed=SEED):
    """Screening run on the project's fixed train/val split. Test is NEVER loaded."""
    c = cfg(config_name)
    tag = f"tune_val_{config_name}" + ("" if seed == SEED else f"_s{seed}")
    marker = os.path.join(REPORTS, f"{tag}_result.json")
    if os.path.exists(marker):
        print(f"SKIP {tag} (marker exists)"); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = pd.read_csv(MANIFEST)
    m = m[m["class"].isin(BINARY)].reset_index(drop=True)
    m = m[m["split"].isin(["train", "val"])].reset_index(drop=True)  # test not even loaded

    print(f"===== {tag} on {device} =====")
    print(f"config: {json.dumps({k: v for k, v in c.items() if k != 'name'})}")
    print(f"subjects train {m[m.split=='train'].subject_id.nunique()} "
          f"val {m[m.split=='val'].subject_id.nunique()}")
    t0 = time.time()
    ckpt = os.path.join(CKPT_DIR, f"{tag}.pt")
    model, hist = train_one(c, m, device, ckpt, seed=seed)
    mins = (time.time() - t0) / 60

    _, eval_t = build_transforms(c)
    val_loader = DataLoader(BinaryDataset(m, "val", eval_t), batch_size=c["batch_size"],
                            shuffle=False, num_workers=NUM_WORKERS)
    sdf = subject_scores(model, val_loader, device)
    y, p = sdf["y"].values, sdf["p"].values
    auc = roc_auc_score(y, p)
    thr, j = youden_threshold(y, p)
    pred = (p >= thr).astype(int)
    acc = float((pred == y).mean())
    lo, hi = auc_ci(auc, int(y.sum()), int((1 - y).sum()))
    # Overfit gap: how far train accuracy has run ahead of val accuracy at the chosen epoch.
    be = hist["best_epoch"] - 1
    gap = hist["train_acc"][be] - hist["val_acc"][be]

    out = dict(tag=tag, mode="val", config=config_name, seed=seed, arch=ARCH,
               config_full={k: v for k, v in c.items() if k != "name"},
               n_val_subjects=int(len(sdf)), val_subject_auc=float(auc),
               val_subject_auc_95CI=[lo, hi],
               val_subject_acc_at_youden=acc, youden_threshold=thr, youden_j=j,
               val_slice_acc=hist["val_acc"][be], val_macro_f1=hist["val_macro_f1"][be],
               train_acc_at_best=hist["train_acc"][be], overfit_gap=float(gap),
               best_epoch=hist["best_epoch"], epochs_run=hist["epochs_run"],
               minutes=round(mins, 1), history=hist)
    with open(marker, "w") as f:
        json.dump(out, f, indent=2)
    sdf.to_csv(os.path.join(REPORTS, f"{tag}_val_subject_preds.csv"))
    print(f"\n{tag}: val subject AUC {auc:.4f} [{lo:.3f}, {hi:.3f}]  "
          f"acc@Youden {acc:.3f}  best_epoch {hist['best_epoch']}/{hist['epochs_run']}  "
          f"train-val gap {gap:+.3f}  {mins:.1f} min")


# ---------------------------------------------------------------- mode: cv
def make_folds(subj, k, seed=SEED):
    """Stratified k-fold over SUBJECTS (never slices), round-robin dealt so class
    balance is even. Same construction as scripts/cross_validate.py."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cl in BINARY:
        ids = np.array(sorted(subj[subj == cl].index))
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            folds[i % k].append(sid)
    return [set(f) for f in folds]


def run_cv(config_name, k=5):
    c = cfg(config_name)
    tag = f"tune_cv_{config_name}_k{k}"
    marker = os.path.join(REPORTS, f"{tag}_result.json")
    if os.path.exists(marker):
        print(f"SKIP {tag} (marker exists)"); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = pd.read_csv(MANIFEST)
    full = full[full["class"].isin(BINARY)].reset_index(drop=True)
    subj = full.groupby("subject_id")["class"].first()
    folds = make_folds(subj, k)
    all_ids = set(subj.index)

    print(f"===== {tag} on {device} =====")
    print(f"{len(subj)} subjects, {k} folds (sizes {[len(f) for f in folds]}), "
          f"class balance {subj.value_counts().to_dict()}")
    print(f"config: {json.dumps({kk: v for kk, v in c.items() if kk != 'name'})}")

    oof, per_fold = [], []
    t_start = time.time()
    for i, test_ids in enumerate(folds):
        rest = sorted(all_ids - test_ids)
        rng = np.random.default_rng(SEED + i)
        rest_cls = subj.loc[rest]
        val_ids = set()
        for cl in BINARY:                       # stratified val carved out of the rest
            v = np.array(sorted(rest_cls[rest_cls == cl].index))
            rng.shuffle(v)
            val_ids.update(v[:max(1, int(round(0.15 * len(v))))])
        assert not (test_ids & val_ids), "val leaked into test"
        splits = {s: ("test" if s in test_ids else "val" if s in val_ids else "train")
                  for s in all_ids}
        mm = full.copy()
        mm["split"] = mm["subject_id"].map(splits)
        print(f"\n--- fold {i+1}/{k}: {len(test_ids)} test / {len(val_ids)} val subjects ---",
              flush=True)

        ckpt = os.path.join(CKPT_DIR, f"{tag}_fold{i+1}.pt")
        t0 = time.time()
        model, hist = train_one(c, mm, device, ckpt, seed=SEED + i)

        _, eval_t = build_transforms(c)
        vl = DataLoader(BinaryDataset(mm, "val", eval_t), batch_size=c["batch_size"],
                        shuffle=False, num_workers=NUM_WORKERS)
        tl = DataLoader(BinaryDataset(mm, "test", eval_t), batch_size=c["batch_size"],
                        shuffle=False, num_workers=NUM_WORKERS)
        vdf = subject_scores(model, vl, device)
        thr, _ = youden_threshold(vdf["y"].values, vdf["p"].values)   # threshold from VAL
        tdf = subject_scores(model, tl, device)
        tdf["pred"] = (tdf["p"] >= thr).astype(int)
        tdf["fold"] = i + 1
        tdf["threshold"] = thr
        oof.append(tdf)

        facc = float((tdf["pred"] == tdf["y"]).mean())
        fauc = roc_auc_score(tdf["y"].values, tdf["p"].values)
        mins = (time.time() - t0) / 60
        print(f"  fold {i+1}: acc {facc:.1%} AUC {fauc:.3f} thr {thr:.3f} "
              f"best_epoch {hist['best_epoch']}/{hist['epochs_run']} {mins:.1f} min", flush=True)
        per_fold.append(dict(fold=i + 1, n_test=int(len(tdf)), accuracy=facc, roc_auc=float(fauc),
                             threshold=thr, best_epoch=hist["best_epoch"],
                             epochs_run=hist["epochs_run"], minutes=round(mins, 1)))

    oof_df = pd.concat(oof)
    assert len(oof_df) == len(subj), f"{len(oof_df)} predictions for {len(subj)} subjects"
    y = oof_df["y"].values.astype(int); p = oof_df["p"].values
    pred = oof_df["pred"].values.astype(int)
    n = len(y); correct = int((pred == y).sum())
    acc = correct / n
    alo, ahi = wilson_ci(correct, n)
    auc = float(roc_auc_score(y, p))
    blo, bhi = bootstrap_ci(y, p, lambda yy, pp: roc_auc_score(yy, pp))
    hlo, hhi = auc_ci(auc, int(y.sum()), int(n - y.sum()))
    baseline = float(max((y == 1).mean(), (y == 0).mean()))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fold_accs = [f["accuracy"] for f in per_fold]
    fold_aucs = [f["roc_auc"] for f in per_fold]

    print(f"\n===== {tag} POOLED OVER {n} SUBJECTS =====")
    print(f"confusion matrix (rows=true CN,AD / cols=pred):\n{cm}")
    print(f"accuracy  {acc:.1%} ({correct}/{n})  95% CI [{alo:.1%}, {ahi:.1%}]  baseline {baseline:.1%}")
    print(f"ROC AUC   {auc:.4f}  bootstrap 95% CI [{blo:.4f}, {bhi:.4f}]  "
          f"(Hanley-McNeil [{hlo:.4f}, {hhi:.4f}])")
    print(f"per-fold acc {[f'{a:.1%}' for a in fold_accs]} (sd {np.std(fold_accs):.3f})")
    print(f"per-fold AUC {[f'{a:.3f}' for a in fold_aucs]} (sd {np.std(fold_aucs):.3f})")

    out = dict(tag=tag, mode="cv", config=config_name, k=k, arch=ARCH,
               config_full={kk: v for kk, v in c.items() if kk != "name"},
               n_subjects=n, accuracy=acc, accuracy_95CI=[alo, ahi],
               roc_auc=auc, roc_auc_95CI_bootstrap=[blo, bhi],
               roc_auc_95CI_hanley=[hlo, hhi], majority_baseline=baseline,
               macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
               confusion_matrix=cm.tolist(), per_fold=per_fold,
               fold_accuracy_sd=float(np.std(fold_accs)), fold_auc_sd=float(np.std(fold_aucs)),
               total_minutes=round((time.time() - t_start) / 60, 1))
    with open(marker, "w") as f:
        json.dump(out, f, indent=2)
    oof_df.to_csv(os.path.join(REPORTS, f"{tag}_oof_preds.csv"))
    print(f"\n{tag} DONE in {out['total_minutes']} min")


# ---------------------------------------------------------------- mode: compare
def run_compare(cfg_a, cfg_b, k=5, n_boot=4000):
    """Paired bootstrap over the SAME subjects and the SAME folds. Comparing two
    overlapping marginal CIs is a much weaker test than resampling the paired
    difference, and this project has repeatedly been burned by reading a difference
    off two noisy point estimates (decisions 21, 29)."""
    a = pd.read_csv(os.path.join(REPORTS, f"tune_cv_{cfg_a}_k{k}_oof_preds.csv"),
                    index_col=0).sort_index()
    b = pd.read_csv(os.path.join(REPORTS, f"tune_cv_{cfg_b}_k{k}_oof_preds.csv"),
                    index_col=0).sort_index()
    assert list(a.index) == list(b.index), "different subject sets"
    assert (a["y"].values == b["y"].values).all(), "label mismatch"
    assert (a["fold"].values == b["fold"].values).all(), "different fold assignment"
    y = a["y"].values.astype(int)
    d_auc = roc_auc_score(y, b["p"].values) - roc_auc_score(y, a["p"].values)
    d_acc = float((b["pred"].values == y).mean() - (a["pred"].values == y).mean())

    rng = np.random.default_rng(SEED)
    n = len(y); da, dc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        da.append(roc_auc_score(y[idx], b["p"].values[idx]) - roc_auc_score(y[idx], a["p"].values[idx]))
        dc.append((b["pred"].values[idx] == y[idx]).mean() - (a["pred"].values[idx] == y[idx]).mean())
    da, dc = np.array(da), np.array(dc)
    out = dict(baseline=cfg_a, candidate=cfg_b, k=k, n_subjects=n,
               delta_auc=float(d_auc),
               delta_auc_95CI=[float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
               delta_acc=d_acc,
               delta_acc_95CI=[float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
               p_candidate_better_auc=float((da > 0).mean()))
    out["separates"] = bool(out["delta_auc_95CI"][0] > 0 or out["delta_auc_95CI"][1] < 0)
    with open(os.path.join(REPORTS, f"tune_compare_{cfg_a}_vs_{cfg_b}_k{k}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- mode: queue
def run_queue(mode, names):
    """Sequential driver, one SUBPROCESS per job so a crash (OOM, a bad transform)
    kills only that job and the queue carries on. Restart-safe: each job is skipped
    if its result JSON already exists, the same pattern as scripts/run_cv.py."""
    import subprocess
    logs = os.path.join(REPORTS, "run_logs")
    os.makedirs(logs, exist_ok=True)
    t_start = time.time()
    for i, name in enumerate(names, 1):
        tag = f"tune_{mode}_{name}" + ("_k5" if mode == "cv" else "")
        marker = os.path.join(REPORTS, f"{tag}_result.json")
        if os.path.exists(marker):
            print(f"[{i}/{len(names)}] SKIP {tag}", flush=True)
            continue
        print(f"[{i}/{len(names)}] RUN  {tag}", flush=True)
        t0 = time.time()
        with open(os.path.join(logs, f"{tag}.log"), "w", encoding="utf-8") as lf:
            proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), mode, name],
                                  stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
        status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"[{i}/{len(names)}] {status} {tag} in {(time.time()-t0)/60:.1f} min", flush=True)
        if proc.returncode != 0:
            with open(os.path.join(logs, f"{tag}.log"), encoding="utf-8", errors="replace") as lf:
                for line in lf.read().splitlines()[-15:]:
                    print("    | " + line, flush=True)
    print(f"\nQUEUE ({mode}) DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        for name in CONFIGS:
            print(f"{name:16s} {CONFIGS[name]}")
        return
    mode = sys.argv[1]
    if mode == "val":
        run_val(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else SEED)
    elif mode == "cv":
        run_cv(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    elif mode == "compare":
        run_compare(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 5)
    elif mode == "queue":
        run_queue(sys.argv[2], [s for s in sys.argv[3].split(",") if s])
    else:
        raise SystemExit(f"unknown mode {mode!r} (val|cv|compare|queue|list)")


if __name__ == "__main__":
    main()
