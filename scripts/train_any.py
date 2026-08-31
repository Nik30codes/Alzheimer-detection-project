"""
One runner for every experiment in Phase D, so no result differs because of
incidental training-code differences.

Modes:
  honest2d   subject-wise split, single slice (duplicated to 3 channels for the
             torchvision architectures). The project's real evaluation setting.
  honest25d  subject-wise split, three ADJACENT slices stacked as channels.
  leaky      slice-wise random split, single slice. DELIBERATELY INVALID -- see below.

On `leaky`: each subject contributes 32 near-duplicate axial slices, so splitting on
slices puts the same person on both sides of the train/test boundary. The resulting
accuracy largely measures subject re-identification, not disease detection. It is
produced here only to quantify the size of that effect on this exact dataset, and is
written to disk with a WARNING field so it can never be mistaken for a real result.

Usage: python train_any.py <arch> <mode>
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = r"C:\Users\Nikunj\Documents\alzheimer-mri-project"
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import (CLASSES, MRIDataset, MRI25DDataset, compute_class_weights,  # noqa: E402
                      TRAIN_TRANSFORM, EVAL_TRANSFORM,
                      TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB,
                      TRAIN_TRANSFORM_25D, EVAL_TRANSFORM_25D)
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0  # noqa: E402
from train import train_model                                          # noqa: E402
from evaluate import (get_predictions, subject_level_report,           # noqa: E402
                      subject_level_soft_vote)
from sklearn.metrics import (accuracy_score, f1_score,                 # noqa: E402
                             classification_report, confusion_matrix)

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
# Leaky runs were first given 15 epochs / patience 4 on the assumption that an easier
# task converges faster. That backfired: validation loss is erratic in early epochs, so
# early stopping fired at epoch 5 and train_model restored the epoch-1 checkpoint --
# producing a model that predicted one class for almost everything (35% accuracy, worse
# than the honest split). The budget now matches the honest runs so that the split type
# is genuinely the only variable being changed.
EPOCHS = {"honest2d": 40, "honest25d": 40, "leaky": 30, "masked2d": 40,
          "braincrop2d": 40, "v2": 40, "v2crop": 40, "v2mask": 40,
          "v3": 40, "v3go2": 40, "v3go2hi": 40}
PATIENCE = {"honest2d": 7, "honest25d": 7, "leaky": 8, "masked2d": 7,
            "braincrop2d": 7, "v2": 7, "v2crop": 7, "v2mask": 7,
            "v3": 7, "v3go2": 7, "v3go2hi": 7}


def load_backbone_from(model, ckpt_path):
    """Initialise from another checkpoint of the same architecture, skipping any
    tensor whose shape differs (i.e. the classifier head).

    Motivation: decision 7 established that ImageNet weights transfer BADLY here --
    natural RGB photos are too far from grayscale MRI, and every fine-tuning strategy
    scored below a from-scratch model. That finding is about the SOURCE DOMAIN, not
    about transfer itself. The AD-vs-CN model is the same modality, same anatomy, same
    slice positions and same preprocessing, and it reaches ROC AUC 0.906 -- so its
    features are known to encode real atrophy. Reusing them as a starting point for
    the four-way task is in-domain transfer, which is a different proposition.

    The head is deliberately left random: it maps to 2 classes there and 4 here.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    own = model.state_dict()
    keep = {k: v for k, v in sd.items()
            if k in own and own[k].shape == v.shape}
    skipped = [k for k in sd if k not in keep]
    model.load_state_dict(keep, strict=False)
    print(f"initialised {len(keep)}/{len(own)} tensors from {os.path.basename(ckpt_path)}"
          f"; {len(skipped)} skipped (shape mismatch = classifier head): {skipped[:4]}")
    return model


def build_model(arch, in_channels):
    if arch == "custom_cnn":
        return SimpleCNN(num_classes=4, in_channels=in_channels)
    if arch == "mobilenetv2":
        return build_mobilenetv2(4, pretrained=False)
    if arch == "efficientnet_b0":
        return build_efficientnet_b0(4, pretrained=False)
    raise ValueError(arch)


def make_leaky_manifest(manifest, seed=SEED):
    """70/15/15 over INDIVIDUAL SLICES -- same proportions as the honest split, but the
    boundary no longer respects subject identity."""
    rng = np.random.default_rng(seed)
    leaky = manifest.copy()
    idx = rng.permutation(len(leaky))
    n_train, n_val = int(0.70 * len(leaky)), int(0.15 * len(leaky))
    split = np.empty(len(leaky), dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "val"
    split[idx[n_train + n_val:]] = "test"
    leaky["split"] = split
    return leaky


def main(arch, mode, select_by="val_loss", init_from=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # masked2d trains on the skull-stripped copies. Same subjects, same subject-wise
    # split, same everything else -- the only change is that scalp/skull/background are
    # zeroed, which removes a large part of the scanner-protocol cue behind decision 10.
    manifest_file = {
        "masked2d": "manifest_masked.csv",        # skull/scalp/background zeroed
        "braincrop2d": "manifest_braincrop.csv",  # ...plus cropped to brain and rescaled,
                                                  # which also normalises the head-size /
                                                  # field-of-view difference between the
                                                  # ADNI1 and ADNI-GO/2 cohorts
        # v2 = re-extracted with the millimetre-anchored axial band. Everything above was
        # trained on slices whose anatomical level drifted between subjects and cohorts,
        # so v2 results are the ones that mean anything.
        "v2": "manifest_v2.csv",
        "v2crop": "manifest_v2_braincrop.csv",    # v2 slices, then brain-cropped
        # v2mask: skull/scalp/background zeroed but framing untouched, so absolute brain
        # size survives. v2crop removes the scanner cue AND the brain-volume signal
        # together (AD-vs-CN AUC fell 0.667 -> 0.576 under the crop); this variant tries
        # to keep the second while still dropping the first.
        "v2mask": "manifest_v2_masked.csv",
        # v3 = the AlzheimerAdditional expansion merged in: 853 subjects instead of 439,
        # same millimetre-anchored band as v2.
        "v3": "manifest_v3.csv",
        # v3go2 is THE PRIMARY 4-WAY TASK. Every subject comes from ADNI-GO/2, so all
        # four classes share one scanner generation and cohort carries no label
        # information -- the first version of this task that decision 10 does not
        # invalidate. Only possible because the expansion supplied CN and AD subjects
        # from ADNI-GO/2, of which the original 439 had none.
        # Plain "v3" keeps EMCI/LMCI as GO/2-only, so era still leaks there; it is kept
        # for the size comparison, not as the headline.
        "v3go2": "manifest_v3_go2.csv",
        # v3go2hi: same subjects, same split, same slice positions as v3go2 -- the only
        # difference is that the 144px resolution-harmonization bottleneck is skipped.
        # That bottleneck equalizes ADNI1's 192x192 against GO/2's 256x256, but this
        # task is GO/2-only where every scan is natively 256 rows, so it removes ~1.7mm
        # of detail for nothing. See scripts/reextract_v3_hires.py.
        "v3go2hi": "manifest_v3_go2_hires.csv",
    }.get(mode, "manifest.csv")
    manifest = pd.read_csv(os.path.join(ROOT, "data", manifest_file))
    # Runs that select their checkpoint on macro F1 get their own tag, so they sit
    # alongside the val_loss runs for comparison instead of overwriting them.
    tag = f"{arch}_{mode}" if select_by == "val_loss" else f"{arch}_{mode}_f1"
    if init_from:
        # Name the source, so an AD-vs-CN-initialised run and a self-supervised one
        # do not collide on the same tag and silently overwrite each other's results.
        tag += "_init-" + os.path.splitext(os.path.basename(init_from))[0]
    print(f"===== {tag} on {device} (manifest: {manifest_file}, "
          f"checkpoint selected by {select_by}) =====")

    if mode == "leaky":
        manifest = make_leaky_manifest(manifest)
        tr_s = set(manifest[manifest.split == "train"]["subject_id"])
        te_s = set(manifest[manifest.split == "test"]["subject_id"])
        print(f"LEAKY split: {len(te_s)} test subjects, {len(tr_s & te_s)} of them also in train "
              f"({len(tr_s & te_s)/len(te_s):.1%} overlap). Honest split has 0% by construction.")

    if mode == "honest25d":
        in_ch = 3
        ds = lambda sp, t: MRI25DDataset(manifest, sp, t)  # noqa: E731
        train_t, eval_t = TRAIN_TRANSFORM_25D, EVAL_TRANSFORM_25D
    else:
        # honest2d, leaky and masked2d all use single-slice input; custom_cnn is a
        # 1-channel model while the torchvision ones need 3 channels
        rgb = arch != "custom_cnn"
        in_ch = 3 if rgb else 1
        ds = lambda sp, t: MRIDataset(manifest, sp, t)  # noqa: E731
        train_t, eval_t = ((TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB) if rgb
                           else (TRAIN_TRANSFORM, EVAL_TRANSFORM))

    train_loader = DataLoader(ds("train", train_t), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(ds("val", eval_t), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(ds("test", eval_t), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_weights = compute_class_weights(manifest)
    model = build_model(arch, in_ch)
    if init_from:
        ipath = init_from if os.path.isabs(init_from) else os.path.join(
            ROOT, "models", "checkpoints", init_from)
        model = load_backbone_from(model, ipath)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{arch}: {n_params:,} params, in_channels={in_ch}, mode={mode}")

    ckpt = os.path.join(ROOT, "models", "checkpoints", f"{tag}.pt")
    t0 = time.time()
    history = train_model(model, train_loader, val_loader, class_weights, device,
                          epochs=EPOCHS[mode], lr=LR, patience=PATIENCE[mode],
                          checkpoint_path=ckpt, weight_decay=WEIGHT_DECAY,
                          select_by=select_by)
    mins = (time.time() - t0) / 60

    preds = get_predictions(model, test_loader, device)
    preds.to_csv(os.path.join(ROOT, "reports", f"{tag}_test_preds.csv"), index=False)

    slice_acc = accuracy_score(preds["true"], preds["pred"])
    slice_f1 = f1_score(preds["true"], preds["pred"], labels=CLASSES, average="macro")
    print(f"\n--- {tag} SLICE LEVEL ---")
    print(classification_report(preds["true"], preds["pred"], labels=CLASSES, zero_division=0))
    print(f"slice confusion matrix (rows=true, cols=pred) {CLASSES}:")
    print(confusion_matrix(preds["true"], preds["pred"], labels=CLASSES))

    result = {
        "arch": arch, "mode": mode, "n_params": n_params,
        "select_by": select_by, "best_epoch": history.get("best_epoch"),
        "init_from": init_from,
        "epochs_run": len(history["train_loss"]), "train_minutes": round(mins, 1),
        "slice_level_accuracy": slice_acc, "slice_level_macro_f1": slice_f1,
    }

    # Subject-level voting only means something when a subject sits entirely on one
    # side of the split, which the leaky mode breaks by construction.
    if mode != "leaky":
        print(f"\n--- {tag} SUBJECT LEVEL (hard majority vote) ---")
        _, subj_hard = subject_level_report(preds)
        print(f"\n--- {tag} SUBJECT LEVEL (soft vote) ---")
        cm_soft, subj_soft = subject_level_soft_vote(preds)
        print(f"soft-vote confusion matrix (rows=true, cols=pred) {CLASSES}:")
        print(cm_soft)
        np.save(os.path.join(ROOT, "reports", f"{tag}_subject_cm_soft.npy"), cm_soft)
        subj_soft.to_csv(os.path.join(ROOT, "reports", f"{tag}_subject_preds.csv"))
        result.update({
            "subject_level_accuracy": accuracy_score(subj_hard["true"], subj_hard["pred"]),
            "subject_level_macro_f1": f1_score(subj_hard["true"], subj_hard["pred"],
                                               labels=CLASSES, average="macro"),
            "subject_level_accuracy_softvote": accuracy_score(subj_soft["true"], subj_soft["pred"]),
            "subject_level_macro_f1_softvote": f1_score(subj_soft["true"], subj_soft["pred"],
                                                        labels=CLASSES, average="macro"),
        })
    else:
        result["WARNING"] = ("INVALID FOR REPORTING. Slice-wise split leaks the same subject "
                             "into train and test; this number largely reflects subject "
                             "re-identification, not diagnostic accuracy.")

    with open(os.path.join(ROOT, "reports", f"{tag}_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(ROOT, "reports", f"{tag}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n=== {tag} SUMMARY ===")
    print(json.dumps(result, indent=2))
    print(f"=== {tag} DONE in {mins:.1f} min ===\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "val_loss",
         sys.argv[4] if len(sys.argv) > 4 else None)
