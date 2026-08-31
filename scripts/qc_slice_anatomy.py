"""
Checks whether the extracted axial slices actually capture consistent brain anatomy.

This is upstream of every accuracy number in the project. `data_prep.py` reslices each
subject's volume axially and keeps 32 slices from the band (0.28, 0.48) of *head height*.
That band is relative, so it only lands on the same anatomy in every subject if:
  - the head extent is detected consistently (a scan including a lot of neck, or cropped
    at the vertex, shifts the whole band), and
  - subjects' head proportions are similar enough that a fixed fractional band maps to a
    fixed anatomical level.

If slice index 15 means "mid-ventricle" in one subject and "high parietal cortex" in
another, the model receives incoherent input and no architecture can fix it.

Two checks:
  1. SAME INDEX ACROSS SUBJECTS - montage of slice i for many subjects. Rows should look
     anatomically alike. If they don't, the band is misaligned between subjects.
  2. FULL RANGE WITHIN A SUBJECT - all 32 slices for one subject, to see what the band
     actually covers and whether the hippocampal level is in it at all.

Usage: python qc_slice_anatomy.py
"""

import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
CLASSES = ["CN", "AD", "EMCI", "LMCI"]
os.makedirs(FIG, exist_ok=True)

m = pd.read_csv(os.path.join(ROOT, "data", "manifest.csv"))
m["idx"] = m["filepath"].str.extract(r"_(\d+)\.png$").astype(int)


def load(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# ---------- CHECK 1: same slice index, many subjects ----------
for probe_idx in (8, 16, 24):
    subs = []
    for c in CLASSES:
        pool = sorted(m[(m["class"] == c) & (m["idx"] == probe_idx)]["subject_id"].unique())
        subs += [(c, s) for s in pool[:6]]

    cols = 6
    rows = int(np.ceil(len(subs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.3 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (c, s) in zip(axes.flat, subs):
        row = m[(m["subject_id"] == s) & (m["idx"] == probe_idx)].iloc[0]
        ax.imshow(load(row["filepath"]), cmap="gray")
        ax.set_title(f"{c} {s[-4:]}", fontsize=7)
    plt.suptitle(f"Slice index {probe_idx} across subjects - should all be the same "
                 f"anatomical level", fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG, f"qc_anatomy_index{probe_idx}.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print("wrote", out)

# ---------- CHECK 2: full 32-slice range for one subject per class ----------
fig, axes = plt.subplots(4, 8, figsize=(16, 8.5))
for r, c in enumerate(CLASSES):
    subj = sorted(m[m["class"] == c]["subject_id"].unique())[0]
    rows = m[m["subject_id"] == subj].sort_values("idx")
    picks = rows.iloc[::4].head(8)          # every 4th slice: 0,4,8,...,28
    for a, (_, rr) in zip(axes[r], picks.iterrows()):
        a.imshow(load(rr["filepath"]), cmap="gray")
        a.set_title(f"{c} #{rr['idx']}", fontsize=8)
        a.axis("off")
plt.suptitle("Full extracted band per subject (every 4th of 32 slices) - "
             "does it cover the hippocampal level?", fontsize=12)
plt.tight_layout()
out = os.path.join(FIG, "qc_anatomy_range.png")
plt.savefig(out, dpi=100)
plt.close()
print("wrote", out)

# ---------- CHECK 3: numeric proxy for anatomical consistency ----------
# Ventricle-ish dark area inside the brain varies strongly and characteristically with
# axial level, so its spread across subjects at a FIXED index is a rough measure of
# misalignment. Large spread at a fixed index = subjects are not at the same level.
print("\ndark-interior fraction at a fixed slice index (proxy for axial level)")
print(f"{'index':>6}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
for probe_idx in range(0, 32, 4):
    vals = []
    sample = m[m["idx"] == probe_idx].sample(min(60, (m["idx"] == probe_idx).sum()),
                                             random_state=0)
    for _, r in sample.iterrows():
        img = load(r["filepath"])
        brain = img > 20
        if brain.sum() == 0:
            continue
        # dark voxels inside the brain region = CSF / ventricles
        vals.append(float(((img > 20) & (img < 70)).sum() / brain.sum()))
    if vals:
        print(f"{probe_idx:>6}{np.mean(vals):>9.3f}{np.std(vals):>9.3f}"
              f"{np.min(vals):>9.3f}{np.max(vals):>9.3f}")

print("""
How to read check 3: within one slice index, a LARGE std / wide min-max range means
different subjects show very different amounts of ventricle at that index, i.e. they are
not at the same anatomical level. Tight spread means the band is well aligned.
""")
