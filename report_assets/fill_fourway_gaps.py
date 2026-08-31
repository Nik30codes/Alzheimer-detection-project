"""Fills the per-model visual-proof gap for the four-way (v3go2) task: real confusion
matrix, per-class classification report, and training-curve plot for each of the three
architectures, using already-trained checkpoints and already-saved history JSONs --
no retraining needed. Runs on CPU deliberately, to leave the GPU free for the binary
task's demo/run_demo.py running concurrently.
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Manifest is read BEFORE matplotlib/torch/sklearn/cv2 (via datasets.py) are imported,
# deliberately. Importing those first and calling pandas.read_csv afterward reliably
# segfaulted on this machine (confirmed via bisection -- a native-library load-order
# conflict between some combination of those packages' bundled DLLs, not a bug in this
# script's logic). Reading the CSV first with only pandas loaded avoids it entirely.
manifest = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3_go2.csv"))

import matplotlib          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                               # noqa: E402
import torch                                                                  # noqa: E402
from torch.utils.data import DataLoader                                       # noqa: E402
from datasets import CLASSES, MRIDataset, EVAL_TRANSFORM, EVAL_TRANSFORM_RGB  # noqa: E402
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0        # noqa: E402
from evaluate import get_predictions, subject_level_soft_vote                 # noqa: E402
from sklearn.metrics import classification_report                            # noqa: E402

device = torch.device("cpu")

# Model BUILDER functions, not built instances -- a dict literal evaluates every
# value eagerly at definition time, which was constructing all three torchvision
# architectures in one process even when only one was selected, and that combination
# is what segfaulted (confirmed: any one architecture in isolation runs cleanly).
# Deferring construction to inside the loop, after filtering by `only`, means a
# single-architecture run only ever builds that one model.
ARCHS = {
    "custom_cnn": (lambda: SimpleCNN(num_classes=4, in_channels=1), EVAL_TRANSFORM,
                   "custom_cnn_v3go2.pt", "custom_cnn_v3go2_history.json", "Custom CNN"),
    "mobilenetv2": (lambda: build_mobilenetv2(4, pretrained=False), EVAL_TRANSFORM_RGB,
                     "mobilenetv2_v3go2.pt", "mobilenetv2_v3go2_history.json", "MobileNetV2"),
    "efficientnet_b0": (lambda: build_efficientnet_b0(4, pretrained=False), EVAL_TRANSFORM_RGB,
                        "efficientnet_b0_v3go2.pt", "efficientnet_b0_v3go2_history.json", "EfficientNet-B0"),
}

only = sys.argv[1] if len(sys.argv) > 1 else None
items = {only: ARCHS[only]} if only else ARCHS
# Run ONE architecture per process invocation: looping over all three
# torchvision/SimpleCNN model constructions in a single process reliably
# segfaulted on this machine, while any one in isolation runs cleanly (isolated
# and confirmed by a standalone import/inference test). Safer to pay three
# process-startup costs than lose all three to one crash.

summary_path = os.path.join(OUT, "fourway_gaps_summary.json")
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
for key, (build_model, transform, ckpt_name, hist_name, display) in items.items():
    print(f"--- {display} ---")
    model = build_model()
    ckpt_path = os.path.join(ROOT, "models", "checkpoints", ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    test_loader = DataLoader(MRIDataset(manifest, "test", transform),
                             batch_size=32, shuffle=False, num_workers=0)
    preds_df = get_predictions(model, test_loader, device)
    cm, subj_df = subject_level_soft_vote(preds_df, verbose=False)
    report_txt = classification_report(subj_df["true"], subj_df["pred"], labels=CLASSES,
                                       zero_division=0)
    print(report_txt)

    # confusion matrix figure
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{display} -- four-stage confusion matrix (n={len(subj_df)})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"fourway_confusion_{key}.png"), dpi=140)
    plt.close(fig)

    # training curves from existing history JSON
    hist_path = os.path.join(ROOT, "reports", hist_name)
    if os.path.exists(hist_path):
        hist = json.load(open(hist_path))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].plot(hist["train_loss"], label="train")
        axes[0].plot(hist["val_loss"], label="val")
        axes[0].set_title("Loss per epoch"); axes[0].set_xlabel("epoch"); axes[0].legend()
        axes[1].plot(hist["train_acc"], label="train")
        axes[1].plot(hist["val_acc"], label="val")
        axes[1].set_title("Accuracy per epoch"); axes[1].set_xlabel("epoch"); axes[1].legend()
        fig.suptitle(f"{display} -- training curves (four-stage task)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"fourway_curves_{key}.png"), dpi=140)
        plt.close(fig)
        print(f"  training curve saved ({len(hist['train_loss'])} epochs)")
    else:
        print(f"  WARNING: no history file at {hist_path}")

    acc = (subj_df["true"] == subj_df["pred"]).mean()
    from sklearn.metrics import f1_score
    f1 = f1_score(subj_df["true"], subj_df["pred"], labels=CLASSES, average="macro", zero_division=0)
    summary[key] = {"display": display, "n_test": len(subj_df), "accuracy": float(acc),
                    "macro_f1": float(f1), "classification_report": report_txt,
                    "confusion_matrix": cm.tolist()}
    print(f"  soft-vote accuracy {acc:.1%}, macro F1 {f1:.3f}\n")

with open(os.path.join(OUT, "fourway_gaps_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("DONE -- wrote fourway_confusion_*.png, fourway_curves_*.png, fourway_gaps_summary.json")
