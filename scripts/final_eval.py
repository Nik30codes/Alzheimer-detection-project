"""
Re-scores every trained model from its checkpoint on the held-out test subjects,
builds the 3-model ensemble, and writes the consolidated comparison table.

Everything is re-scored (rather than reusing the saved *_test_preds.csv) because
those older CSVs were written before evaluate.get_predictions started emitting the
full softmax distribution, which soft voting and the ensemble both require.

Note the custom CNN uses the 1-channel transforms and the two pretrained-architecture
models use the 3-channel ImageNet-normalized ones, so they need separate DataLoaders.
Both are built with shuffle=False from the same manifest, so their rows still line up
one-to-one -- ensemble_predictions asserts this rather than assuming it.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = r"C:\Users\Nikunj\Documents\alzheimer-mri-project"
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import CLASSES, build_dataloaders, build_dataloaders_25d  # noqa: E402
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0  # noqa: E402
from evaluate import (get_predictions, subject_level_report,           # noqa: E402
                      subject_level_soft_vote, ensemble_predictions)
from sklearn.metrics import accuracy_score, f1_score, classification_report  # noqa: E402

CKPT = os.path.join(ROOT, "models", "checkpoints")
REPORTS = os.path.join(ROOT, "reports")

# (display name, builder, checkpoint filename, input mode)
# input mode: "gray1" = 1-channel single slice, "rgb3" = single slice duplicated to 3
# channels, "stack25d" = three adjacent slices stacked. Each needs its own loader.
# The EfficientNet honest run was launched from the earlier train_scratch.py, hence its
# "_scratch" filename while the queue's runs use the "{arch}_{mode}" convention.
MODELS = [
    ("custom_cnn",              lambda: SimpleCNN(4, in_channels=1),            "custom_cnn.pt",                 "gray1"),
    ("mobilenetv2",             lambda: build_mobilenetv2(4, pretrained=False), "mobilenetv2_honest2d.pt",       "rgb3"),
    ("efficientnet_b0",         lambda: build_efficientnet_b0(4, pretrained=False), "efficientnet_b0_scratch.pt", "rgb3"),
    ("custom_cnn_25d",          lambda: SimpleCNN(4, in_channels=3),            "custom_cnn_honest25d.pt",       "stack25d"),
    ("mobilenetv2_25d",         lambda: build_mobilenetv2(4, pretrained=False), "mobilenetv2_honest25d.pt",      "stack25d"),
    ("efficientnet_b0_25d",     lambda: build_efficientnet_b0(4, pretrained=False), "efficientnet_b0_honest25d.pt", "stack25d"),
]

# Models eligible for the headline ensemble: one per architecture, best input mode
# decided at runtime from measured soft-vote accuracy.
ARCH_GROUPS = {
    "custom_cnn": ["custom_cnn", "custom_cnn_25d"],
    "mobilenetv2": ["mobilenetv2", "mobilenetv2_25d"],
    "efficientnet_b0": ["efficientnet_b0", "efficientnet_b0_25d"],
}


def score(name, preds, verbose=True):
    """Slice-level, subject-level hard vote, and subject-level soft vote for one model."""
    slice_acc = accuracy_score(preds["true"], preds["pred"])
    slice_f1 = f1_score(preds["true"], preds["pred"], labels=CLASSES, average="macro")

    if verbose:
        print(f"\n----- {name}: slice level -----")
        print(classification_report(preds["true"], preds["pred"], labels=CLASSES, zero_division=0))

    if verbose:
        print(f"----- {name}: subject level, HARD majority vote -----")
    cm_hard, subj_hard = subject_level_report(preds)
    hard_acc = accuracy_score(subj_hard["true"], subj_hard["pred"])
    hard_f1 = f1_score(subj_hard["true"], subj_hard["pred"], labels=CLASSES, average="macro")

    if verbose:
        print(f"----- {name}: subject level, SOFT vote -----")
    cm_soft, subj_soft = subject_level_soft_vote(preds, verbose=verbose)
    soft_acc = accuracy_score(subj_soft["true"], subj_soft["pred"])
    soft_f1 = f1_score(subj_soft["true"], subj_soft["pred"], labels=CLASSES, average="macro")

    print(f"[{name}] slice {slice_acc:.3f} | subject hard {hard_acc:.3f} | subject soft {soft_acc:.3f}")
    print(f"  soft-vote confusion matrix (rows=true, cols=pred) order {CLASSES}:")
    print(cm_soft)

    return {
        "slice_level_accuracy": slice_acc,
        "slice_level_macro_f1": slice_f1,
        "subject_level_accuracy": hard_acc,
        "subject_level_macro_f1": hard_f1,
        "subject_level_accuracy_softvote": soft_acc,
        "subject_level_macro_f1_softvote": soft_f1,
    }, cm_soft, subj_soft


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(os.path.join(ROOT, "data", "manifest.csv"))

    # One test loader per input mode. All are built from the same manifest with
    # shuffle=False, so row i is the same underlying image in every one of them --
    # which is what makes averaging their probabilities in the ensemble valid.
    _, _, gray1 = build_dataloaders(manifest, batch_size=32, num_workers=0, rgb=False)
    _, _, rgb3 = build_dataloaders(manifest, batch_size=32, num_workers=0, rgb=True)
    _, _, stack25d = build_dataloaders_25d(manifest, batch_size=32, num_workers=0)
    loaders = {"gray1": gray1, "rgb3": rgb3, "stack25d": stack25d}

    all_metrics, all_preds = {}, {}

    for name, builder, ckpt_file, input_mode in MODELS:
        path = os.path.join(CKPT, ckpt_file)
        if not os.path.exists(path):
            print(f"SKIP {name}: no checkpoint at {path}")
            continue
        model = builder()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())

        preds = get_predictions(model, loaders[input_mode], device)
        preds.to_csv(os.path.join(REPORTS, f"{name}_test_preds.csv"), index=False)
        all_preds[name] = preds

        m, cm, subj = score(name, preds)
        m["n_params"] = n_params
        all_metrics[name] = m
        np.save(os.path.join(REPORTS, f"{name}_subject_cm_soft.npy"), cm)
        del model
        torch.cuda.empty_cache()

    # ---- ensemble: one model per architecture, whichever input mode scored better ----
    # Averaging every variant would let one architecture vote twice, which biases the
    # ensemble toward it rather than toward genuine model diversity.
    members = []
    for arch, candidates in ARCH_GROUPS.items():
        available = [c for c in candidates if c in all_preds]
        if not available:
            continue
        best = max(available, key=lambda c: all_metrics[c]["subject_level_accuracy_softvote"])
        members.append(best)
        if len(available) > 1:
            print(f"ensemble: for {arch} picked {best} "
                  f"({all_metrics[best]['subject_level_accuracy_softvote']:.3f} soft-vote) "
                  f"over {[c for c in available if c != best]}")

    if len(members) >= 2:
        names = members
        print(f"\n===== ENSEMBLE of {names} =====")
        ens = ensemble_predictions([all_preds[n] for n in names])
        ens.to_csv(os.path.join(REPORTS, "ensemble_test_preds.csv"), index=False)
        m, cm, subj = score("ensemble", ens)
        m["members"] = names
        all_metrics["ensemble"] = m
        np.save(os.path.join(REPORTS, "ensemble_subject_cm_soft.npy"), cm)
        subj.to_csv(os.path.join(REPORTS, "ensemble_subject_preds.csv"))

    # ---- keep the historical pretrained-MobileNetV2 numbers in the table ----
    # Read from the frozen snapshot, not from metrics.json: this script overwrites
    # metrics.json and now uses "mobilenetv2" for the from-scratch model, so reading
    # the live file would silently relabel the from-scratch result as the pretrained one.
    legacy_path = os.path.join(REPORTS, "metrics_legacy_pretrained.json")
    if os.path.exists(legacy_path):
        with open(legacy_path) as f:
            legacy = json.load(f)
        if "mobilenetv2" in legacy:
            all_metrics["mobilenetv2_pretrained_finetuned"] = dict(legacy["mobilenetv2"])
            all_metrics["mobilenetv2_pretrained_finetuned"]["note"] = (
                "ImageNet-pretrained, two-phase fine-tune. Kept for comparison: this is the "
                "negative-transfer result that motivated training from scratch instead.")

    # ---- fold in the deliberate-leakage runs, clearly flagged ----
    for arch in ARCH_GROUPS:
        leaky_path = os.path.join(REPORTS, f"{arch}_leaky_result.json")
        if os.path.exists(leaky_path):
            with open(leaky_path) as f:
                all_metrics[f"{arch}_LEAKY"] = json.load(f)

    old_path = os.path.join(REPORTS, "metrics.json")
    with open(old_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ---- printed comparison table ----
    def row(name, m):
        soft = m.get("subject_level_accuracy_softvote", float("nan"))
        softf1 = m.get("subject_level_macro_f1_softvote", float("nan"))
        return (f"{name:<34}{m.get('n_params', 0):>11,}"
                f"{m.get('slice_level_accuracy', float('nan')):>9.3f}"
                f"{m.get('subject_level_accuracy', float('nan')):>11.3f}"
                f"{soft:>11.3f}{softf1:>9.3f}")

    honest = {k: v for k, v in all_metrics.items() if not k.endswith("_LEAKY")}
    leaky = {k: v for k, v in all_metrics.items() if k.endswith("_LEAKY")}

    hdr = f"{'model':<34}{'params':>11}{'slice':>9}{'subj-hard':>11}{'subj-soft':>11}{'soft F1':>9}"
    print("\n\n=========== HONEST RESULTS - subject-wise split, 66 test subjects ===========")
    print(hdr)
    print("-" * len(hdr))
    for name, m in sorted(honest.items(),
                          key=lambda kv: kv[1].get("subject_level_accuracy_softvote",
                                                   kv[1].get("subject_level_accuracy", 0)),
                          reverse=True):
        print(row(name, m))

    if leaky:
        print("\n=========== LEAKY RESULTS - slice-wise split - NOT VALID RESULTS ===========")
        print("These share subjects between train and test. The accuracy below largely")
        print("measures subject re-identification, not Alzheimer's detection. Shown only")
        print("to quantify how much the methodological bug inflates the headline number.")
        print(hdr)
        print("-" * len(hdr))
        for name, m in sorted(leaky.items(),
                              key=lambda kv: kv[1].get("slice_level_accuracy", 0), reverse=True):
            print(row(name, m))
        best_h = max((v.get("subject_level_accuracy_softvote", 0) for v in honest.values()), default=0)
        best_l = max((v.get("slice_level_accuracy", 0) for v in leaky.values()), default=0)
        print(f"\nINFLATION: best honest {best_h:.1%} vs best leaky {best_l:.1%} "
              f"= +{(best_l - best_h) * 100:.1f} points of pure leakage.")

    print("\nwrote", old_path)


if __name__ == "__main__":
    main()
