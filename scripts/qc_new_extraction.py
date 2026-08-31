"""
Validates the millimetre-anchored slice extraction on a few subjects BEFORE
reprocessing all 439 (re-extraction is expensive and overwrites the dataset).

Two things must be true for the fix to be real:
  1. The same slice index shows the same anatomy in every subject.
  2. The band reaches the medial temporal / hippocampal level in every subject,
     including the LMCI ones that previously stopped short.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data_prep as dp  # noqa: E402

CLASSES = ["CN", "AD", "EMCI", "LMCI"]
N_PER_CLASS = 3

sm = pd.read_csv(os.path.join(ROOT, "data", "subject_manifest.csv"))
picks = pd.concat([sm[sm["class"] == c].head(N_PER_CLASS) for c in CLASSES], ignore_index=True)

print(f"{'subject':<16}{'class':<7}{'spacing':>9}{'vertex':>8}{'rows':>7}{'band':>14}")
extracted = {}
for _, r in picks.iterrows():
    d = r["dicom_dir"]
    try:
        vol = dp._build_volume(d)
        spacing = dp._row_spacing_mm(d)
        vertex = dp._find_vertex_row(vol)
        lo = vertex + int(round(dp.AXIAL_MM_BELOW_VERTEX[0] / spacing))
        hi = min(vertex + int(round(dp.AXIAL_MM_BELOW_VERTEX[1] / spacing)), vol.shape[1] - 1)
        print(f"{r['subject_id']:<16}{r['class']:<7}{spacing:>9.2f}{vertex:>8}"
              f"{vol.shape[1]:>7}{f'{lo}-{hi}':>14}")
        extracted[(r["class"], r["subject_id"])] = dp.extract_slices(d)
    except Exception as e:  # noqa: BLE001 - report and continue, one bad scan shouldn't stop QC
        print(f"{r['subject_id']:<16}{r['class']:<7}  FAILED: {type(e).__name__}: {e}")

if not extracted:
    raise SystemExit("no subjects extracted")

# ---- same index across subjects ----
for probe in (0, 16, 31):
    keys = list(extracted)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.1 * len(keys), 2.6))
    for ax, k in zip(np.atleast_1d(axes), keys):
        ax.imshow(extracted[k][probe], cmap="gray")
        ax.set_title(f"{k[0]} {k[1][-4:]}", fontsize=7)
        ax.axis("off")
    plt.suptitle(f"NEW extraction - slice #{probe} across subjects", fontsize=10)
    plt.tight_layout()
    out = os.path.join(ROOT, "reports", "figures", f"qc_new_index{probe}.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print("wrote", out)

# ---- full band for one subject per class ----
rows = min(4, len(extracted))
fig, axes = plt.subplots(rows, 8, figsize=(16, 2.3 * rows))
axes = np.atleast_2d(axes)
seen = set()
ri = 0
for (cls, sid), sl in extracted.items():
    if cls in seen or ri >= rows:
        continue
    seen.add(cls)
    for a, i in zip(axes[ri], range(0, 32, 4)):
        a.imshow(sl[i], cmap="gray")
        a.set_title(f"{cls} #{i}", fontsize=8)
        a.axis("off")
    ri += 1
plt.suptitle("NEW extraction - full band (every 4th slice). Last columns should reach "
             "the temporal lobes / hippocampal level.", fontsize=11)
plt.tight_layout()
out = os.path.join(ROOT, "reports", "figures", "qc_new_range.png")
plt.savefig(out, dpi=100)
plt.close()
print("wrote", out)
