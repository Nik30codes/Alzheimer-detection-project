"""Phase 2 pre-flight: validate the mm-anchored extraction on the NEW subjects
before reprocessing all 853.

The expansion introduces series types that were never in the original pool
(MPRAGE_GRAPPA2, MPRAGE_SENSE2, ...), i.e. other vendors' parallel-imaging
variants. Two assumptions in data_prep.py were verified against the ORIGINAL
data only, and both would silently corrupt the dataset if they don't hold here:

  1. _build_volume() assumes the series is acquired SAGITTALLY with image rows
     running superior-inferior. If a vendor stores these axially or coronally,
     the "axial reslice" would cut through the wrong plane entirely.
  2. _row_spacing_mm() takes PixelSpacing[0] as millimetres-per-row along
     superior-inferior, which is what anchors the band in physical units.

This script reports ImageOrientationPatient, PixelSpacing and the detected
vertex per series type, then renders the same slice index across subjects so
anatomical consistency can be eyeballed.
"""
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
import pydicom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data_prep as dp  # noqa: E402

N_PER_GROUP = 2


def orientation_label(iop):
    """Map ImageOrientationPatient to a plane name.

    The plane is the axis of the cross product of the row and column direction
    cosines -- whichever patient axis it points along is the slice-normal.
    """
    if iop is None or len(iop) != 6:
        return "?"
    r = np.array(iop[:3], dtype=float)
    c = np.array(iop[3:], dtype=float)
    n = np.cross(r, c)
    axis = int(np.argmax(np.abs(n)))
    return ["SAGITTAL", "CORONAL", "AXIAL"][axis]


sm = pd.read_csv(os.path.join(ROOT, "data", "subject_manifest_v3.csv"))
sm["series"] = sm["dicom_dir"].map(lambda p: os.path.basename(os.path.dirname(os.path.dirname(p))))
# collapse "MPRAGE_SENSE2" / "MPRAGE_SENSE_repeat" etc into a family
sm["family"] = sm["series"].str.replace(r"_?(repeat|repe|2nd|Repeat|REPEAT|REPE)$", "", regex=True)

print("=== series families present (subjects) ===")
print(sm.groupby(["family", "era"]).size().unstack(fill_value=0).to_string())

# sample across family x era so every combination gets checked
picks = []
for (fam, era), grp in sm.groupby(["family", "era"]):
    picks.append(grp.head(N_PER_GROUP))
picks = pd.concat(picks, ignore_index=True)
print(f"\nchecking {len(picks)} subjects\n")

hdr = (f"{'subject':<16}{'class':<6}{'era':<7}{'family':<18}{'plane':<10}"
       f"{'spacing':>8}{'rows':>6}{'vertex':>8}{'band':>12}")
print(hdr)
print("-" * len(hdr))

extracted = {}
by_plane = defaultdict(int)
problems = []
for _, r in picks.iterrows():
    d = r["dicom_dir"]
    try:
        first = dp._read_sorted_dicom_files(d)[0]
        ds = pydicom.dcmread(first, stop_before_pixels=True)
        plane = orientation_label(ds.get("ImageOrientationPatient", None))
        by_plane[plane] += 1

        vol = dp._build_volume(d)
        spacing = dp._row_spacing_mm(d)
        vertex = dp._find_vertex_row(vol)
        lo = vertex + int(round(dp.AXIAL_MM_BELOW_VERTEX[0] / spacing))
        hi = min(vertex + int(round(dp.AXIAL_MM_BELOW_VERTEX[1] / spacing)), vol.shape[1] - 1)
        print(f"{r['subject_id']:<16}{r['class']:<6}{r['era']:<7}{r['family']:<18}"
              f"{plane:<10}{spacing:>8.2f}{vol.shape[1]:>6}{vertex:>8}{f'{lo}-{hi}':>12}")

        if plane != "SAGITTAL":
            problems.append(f"{r['subject_id']}: plane is {plane}, not SAGITTAL")
        if hi - lo < dp.N_SLICES_PER_SUBJECT:
            problems.append(f"{r['subject_id']}: band {lo}-{hi} thinner than "
                            f"{dp.N_SLICES_PER_SUBJECT} slices")
        if vertex == 0:
            problems.append(f"{r['subject_id']}: vertex detection returned row 0")

        extracted[(r["class"], r["era"], r["family"], r["subject_id"])] = dp.extract_slices(d)
    except Exception as e:  # noqa: BLE001 - one bad scan shouldn't stop QC
        print(f"{r['subject_id']:<16}{r['class']:<6}{r['era']:<7}{r['family']:<18}"
              f"FAILED: {type(e).__name__}: {e}")
        problems.append(f"{r['subject_id']}: {type(e).__name__}: {e}")

print(f"\nplanes seen: {dict(by_plane)}")
if problems:
    print("\n!!! PROBLEMS !!!")
    for p in problems:
        print("  -", p)
else:
    print("\nno problems detected")

if not extracted:
    raise SystemExit("nothing extracted")

figdir = os.path.join(ROOT, "reports", "figures")
os.makedirs(figdir, exist_ok=True)

# ---- same slice index across every family/era, to check anatomical alignment ----
keys = sorted(extracted, key=lambda k: (k[2], k[1], k[0]))
for probe in (0, 16, 28):
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 3.0))
    for ax, k in zip(np.atleast_1d(axes), keys):
        ax.imshow(extracted[k][probe], cmap="gray")
        ax.set_title(f"{k[0]}/{k[1]}\n{k[2][:14]}", fontsize=6)
        ax.axis("off")
    plt.suptitle(f"v3 extraction - slice #{probe} across series types "
                 f"(should show the SAME anatomy everywhere)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(figdir, f"qc_v3_index{probe}.png")
    plt.savefig(out, dpi=110)
    plt.close()
    print("wrote", out)

# ---- full band for one subject per family ----
seen, rows_keys = set(), []
for k in keys:
    if k[2] not in seen:
        seen.add(k[2])
        rows_keys.append(k)
fig, axes = plt.subplots(len(rows_keys), 8, figsize=(16, 2.3 * len(rows_keys)))
axes = np.atleast_2d(axes)
for ri, k in enumerate(rows_keys):
    for a, i in zip(axes[ri], range(0, 32, 4)):
        a.imshow(extracted[k][i], cmap="gray")
        a.set_title(f"{k[2][:12]} #{i}", fontsize=7)
        a.axis("off")
plt.suptitle("v3 extraction - full band per series family (every 4th slice). "
             "Last columns must reach temporal lobes / hippocampal level.", fontsize=11)
plt.tight_layout()
out = os.path.join(figdir, "qc_v3_range.png")
plt.savefig(out, dpi=110)
plt.close()
print("wrote", out)
