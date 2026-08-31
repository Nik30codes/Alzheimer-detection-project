"""
Phase E -- Grad-CAM explainability.

Grad-CAM answers "which pixels moved this prediction?" by weighting the final
convolutional feature maps by how strongly the predicted class's score responds to each
map, then collapsing them into a heatmap.

For this project it is not decoration, it is a test. Decision 10 established that the
models separate ADNI1 (CN/AD) from ADNI-GO/2 (EMCI/LMCI) with 95-100% accuracy, which
looks like a scanner-protocol cue rather than disease. If that is right, the heatmaps
should sit on the skull rim, the scalp and the image border. If the model were reading
atrophy they should sit on the medial temporal lobe / hippocampal region and the
ventricles.

So this script also computes a numeric summary, not just pictures: the fraction of each
heatmap's mass that falls OUTSIDE the brain mask. High values = the model is looking at
the skull. That turns a qualitative figure into a measurement, and gives a before/after
number for the skull-stripped models.

Usage:
  python gradcam.py <checkpoint_tag> [n_per_class]
    e.g. python gradcam.py mobilenetv2_honest2d
         python gradcam.py mobilenetv2_masked2d
"""

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from datasets import (CLASSES, EVAL_TRANSFORM, EVAL_TRANSFORM_RGB)      # noqa: E402
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0  # noqa: E402
from brain_mask import brain_mask                                       # noqa: E402


def load_model(tag, device):
    """Rebuilds the right architecture for a checkpoint tag and loads its weights.

    An "ADvsCN" tag is a two-class head, not four. Building a 4-class model for it
    would fail to load, and silently forcing four classes would make the heatmaps
    describe a model that does not exist.
    """
    ckpt = os.path.join(ROOT, "models", "checkpoints", f"{tag}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    binary = "ADvsCN" in tag
    n_out = 2 if binary else 4

    if tag.startswith("custom_cnn"):
        in_ch = 3 if "25d" in tag else 1
        model = SimpleCNN(n_out, in_channels=in_ch)
        target_layer = model.features[-2]      # last conv_block before the final pool
        rgb = in_ch == 3
    elif tag.startswith("mobilenetv2"):
        model = build_mobilenetv2(n_out, pretrained=False)
        target_layer = model.features[-1]      # final 1x1 conv + BN + ReLU6
        rgb = True
    elif tag.startswith("efficientnet_b0"):
        model = build_efficientnet_b0(n_out, pretrained=False)
        target_layer = model.features[-1]
        rgb = True
    else:
        raise ValueError(f"unrecognised tag {tag}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    # BINARY maps CN->0, AD->1, matching scripts/train_binary_adni1.py
    labels = ["CN", "AD"] if binary else CLASSES
    return model.to(device).eval(), target_layer, rgb, labels


class GradCAM:
    """Standard Grad-CAM: hook the target layer's activations and gradients, weight each
    channel by its mean gradient, sum, ReLU, normalise."""

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _i, output):
        self.acts = output.detach()

    def _save_grads(self, _m, _gi, grad_output):
        self.grads = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)      # global-average-pool the grads
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx, torch.softmax(logits, dim=1)[0].detach().cpu().numpy()


def outside_brain_fraction(cam, img):
    """Share of Grad-CAM mass lying outside the brain. The headline number here:
    a model reading anatomy should be near 0; one reading skull/protocol will be high."""
    masked = brain_mask(img)
    inside = masked > 0
    total = cam.sum()
    return float(cam[~inside].sum() / total) if total > 0 else float("nan")


def main(tag, n_per_class=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target_layer, rgb, labels = load_model(tag, device)
    cam_engine = GradCAM(model, target_layer)
    transform = EVAL_TRANSFORM_RGB if rgb else EVAL_TRANSFORM

    # The heatmaps must be computed on the SAME images the checkpoint was trained on,
    # otherwise the "attention outside the brain" fraction describes the wrong dataset.
    # Longest matching suffix wins, so v3go2 is not caught by the v3 entry.
    manifest_file = "manifest.csv"
    # "_f1" only records which validation metric chose the checkpoint; it says nothing
    # about the dataset, so strip it before matching or the lookup silently falls back
    # to manifest.csv and the heatmaps describe the wrong images.
    tag_for_manifest = tag[:-3] if tag.endswith("_f1") else tag
    for suffix, fname in sorted(
        {"_masked": "manifest_masked.csv",
         "_braincrop2d": "manifest_braincrop.csv",
         "_v2": "manifest_v2.csv",
         "_v2mask": "manifest_v2_masked.csv",
         "_v2crop": "manifest_v2_braincrop.csv",
         "_v3": "manifest_v3.csv",
         "_v3go2": "manifest_v3_go2.csv",
         "_ADvsCN": "manifest_v3_adcn.csv"}.items(),
        key=lambda kv: -len(kv[0]),
    ):
        if tag_for_manifest.endswith(suffix):
            manifest_file = fname
            break
    print(f"manifest: {manifest_file}")
    m = pd.read_csv(os.path.join(ROOT, "data", manifest_file))
    test = m[m["split"] == "test"]
    picks = pd.concat([test[test["class"] == c].sample(n_per_class, random_state=0)
                       for c in labels], ignore_index=True)

    rows = len(picks)
    fig, axes = plt.subplots(rows, 3, figsize=(9.5, 3.1 * rows))
    outside_by_class = {c: [] for c in labels}

    for ax_row, (_, r) in zip(axes, picks.iterrows()):
        img = cv2.imread(r["filepath"], cv2.IMREAD_GRAYSCALE)
        x = transform(img).unsqueeze(0).to(device)
        cam, pred_idx, probs = cam_engine(x)
        frac = outside_brain_fraction(cam, img)
        outside_by_class[r["class"]].append(frac)

        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]
        overlay = (0.55 * cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) + 0.45 * heat).astype(np.uint8)

        ax_row[0].imshow(img, cmap="gray")
        ax_row[0].set_title(f"true {r['class']}", fontsize=9)
        ax_row[1].imshow(cam, cmap="jet")
        ax_row[1].set_title(f"Grad-CAM ({frac:.0%} outside brain)", fontsize=9)
        ax_row[2].imshow(overlay)
        ax_row[2].set_title(f"pred {labels[pred_idx]} p={probs[pred_idx]:.2f}", fontsize=9)
        for a in ax_row:
            a.axis("off")

    plt.suptitle(f"Grad-CAM: {tag}", fontsize=12, y=1.001)
    plt.tight_layout()
    out_png = os.path.join(ROOT, "reports", "figures", f"gradcam_{tag}.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=110, bbox_inches="tight")

    all_fracs = [f for v in outside_by_class.values() for f in v]
    summary = {
        "tag": tag,
        "mean_gradcam_mass_outside_brain": float(np.mean(all_fracs)),
        "by_class": {c: float(np.mean(v)) for c, v in outside_by_class.items() if v},
        "interpretation": ("Fraction of Grad-CAM attention falling outside the brain mask. "
                           "Near 0 = the model is reading anatomy. Large = it is reading "
                           "skull/scalp/background, i.e. scanner-protocol cues."),
    }
    with open(os.path.join(ROOT, "reports", f"gradcam_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("wrote", out_png)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mobilenetv2_honest2d",
         int(sys.argv[2]) if len(sys.argv) > 2 else 3)
