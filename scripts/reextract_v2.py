"""
Re-extracts all 439 subjects with the millimetre-anchored axial band.

Why this had to be redone: the original extraction took slices at a fixed FRACTION of
each scan's voxel height, never locating the head. Two things then went wrong at once:

  * ADNI1 (CN/AD) has 1.25mm row spacing over 192 rows; ADNI-GO/2 (EMCI/LMCI) has
    1.00mm over 256. The same fraction is a different physical depth in each cohort, so
    the anatomical offset was systematically correlated with the class label.
  * Scans differ in how far below the brain they extend and how much empty space sits
    above the vertex, so the band drifted between subjects within a cohort too.

QC (scripts/qc_slice_anatomy.py) showed slice #16 landing on the orbits in some subjects
and high in the centrum semiovale in others, and for several LMCI subjects the whole
32-slice band sat ABOVE the hippocampus -- the structure the task depends on was never
imaged. Every accuracy number produced before this fix was computed on that data.

Output goes to data/processed_v2/ + data/manifest_v2.csv. The original
data/processed/ and manifest.csv are left untouched so the before/after comparison
stays possible.

The subject-wise train/val/test split is inherited from subject_manifest.csv unchanged,
so v1 and v2 results are directly comparable and no subject moves across the boundary.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT / "src"))

import data_prep as dp  # noqa: E402

OUT_DIR = ROOT / "data" / "processed_v2"
OUT_MANIFEST = ROOT / "data" / "manifest_v2.csv"


def main():
    subj = pd.read_csv(ROOT / "data" / "subject_manifest.csv")
    print(f"re-extracting {len(subj)} subjects with "
          f"AXIAL_MM_BELOW_VERTEX={dp.AXIAL_MM_BELOW_VERTEX}")
    print(f"output: {OUT_DIR}")
    print(subj.groupby(["class", "split"]).size().unstack(), flush=True)

    t0 = time.time()
    # resume=True so an interrupted run picks up where it left off rather than
    # redoing hours of DICOM reading.
    slice_manifest = dp.process_all(subj, OUT_DIR, resume=True)
    slice_manifest.to_csv(OUT_MANIFEST, index=False)

    mins = (time.time() - t0) / 60
    print(f"\ndone in {mins:.1f} min")
    print(f"wrote {len(slice_manifest)} slices and {OUT_MANIFEST}")
    print(slice_manifest.groupby(["class", "split"]).size().unstack())

    # the split must be identical to v1, or the comparison is meaningless
    old = pd.read_csv(ROOT / "data" / "manifest.csv")
    old_split = old.groupby("subject_id")["split"].first()
    new_split = slice_manifest.groupby("subject_id")["split"].first()
    common = old_split.index.intersection(new_split.index)
    moved = int((old_split[common] != new_split[common]).sum())
    print(f"\nsubjects whose split changed vs v1: {moved} (must be 0)")


if __name__ == "__main__":
    main()
