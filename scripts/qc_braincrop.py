"""Side-by-side QC of the brain-cropped dataset against the originals.

The point of the crop is to make CN/AD (natively 192x192) and EMCI/LMCI (natively
256x256) indistinguishable by framing. So the check is not just "does it look like a
brain" but "do the four classes now fill the frame the same way" -- if ADNI1 and
ADNI-GO/2 still differ visibly in head size or border, the cue survives.
"""

import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["CN", "AD", "EMCI", "LMCI"]

orig = pd.read_csv(os.path.join(ROOT, "data", "manifest.csv"))
crop = pd.read_csv(os.path.join(ROOT, "data", "manifest_braincrop.csv"))

fig, axes = plt.subplots(2, 4, figsize=(13, 7))
fill_stats = {}
for j, c in enumerate(CLASSES):
    o = orig[orig["class"] == c].sample(1, random_state=1).iloc[0]
    axes[0, j].imshow(cv2.imread(o["filepath"], cv2.IMREAD_GRAYSCALE), cmap="gray")
    axes[0, j].set_title(f"{c} - original", fontsize=10)

    match = crop[crop["filepath"].str.endswith(os.path.basename(o["filepath"]))]
    cimg = cv2.imread(match.iloc[0]["filepath"], cv2.IMREAD_GRAYSCALE)
    axes[1, j].imshow(cimg, cmap="gray")
    axes[1, j].set_title(f"{c} - brain-cropped", fontsize=10)
    for i in (0, 1):
        axes[i, j].axis("off")

# quantify: what fraction of the frame is non-black, per class, before vs after?
for c in CLASSES:
    before, after = [], []
    for _, r in orig[orig["class"] == c].sample(40, random_state=2).iterrows():
        img = cv2.imread(r["filepath"], cv2.IMREAD_GRAYSCALE)
        before.append((img > 15).mean())
    for _, r in crop[crop["class"] == c].sample(40, random_state=2).iterrows():
        img = cv2.imread(r["filepath"], cv2.IMREAD_GRAYSCALE)
        after.append((img > 15).mean())
    fill_stats[c] = (float(np.mean(before)), float(np.mean(after)))

plt.tight_layout()
out = os.path.join(ROOT, "reports", "figures", "braincrop_qc.png")
plt.savefig(out, dpi=110)
print("wrote", out)

print("\nfraction of frame occupied by head/brain (the cue we are trying to erase):")
print(f"{'class':<6}{'before':>10}{'after':>10}")
for c in CLASSES:
    b, a = fill_stats[c]
    print(f"{c:<6}{b:>10.3f}{a:>10.3f}")

b_spread = max(fill_stats[c][0] for c in CLASSES) - min(fill_stats[c][0] for c in CLASSES)
a_spread = max(fill_stats[c][1] for c in CLASSES) - min(fill_stats[c][1] for c in CLASSES)
print(f"\nspread across classes: before {b_spread:.3f} -> after {a_spread:.3f}")
print("Smaller spread after = the framing cue has been reduced. A spread near zero means"
      "\nframing can no longer identify the cohort.")
