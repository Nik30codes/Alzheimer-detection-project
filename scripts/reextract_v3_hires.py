"""Re-extract the ADNI-GO/2 subjects WITHOUT the 144px resolution bottleneck.

The bottleneck exists to equalize ADNI1's native 192x192 against ADNI-GO/2's 256x256
(decision 5), because resizing one group up and the other down left a sharpness
difference that tracked scanner protocol. That reasoning applies to a MIXED-era
dataset. The primary four-way task is GO/2 only, where every scan is natively 256
rows at 1.00-1.05mm spacing -- there is nothing to harmonize, and the bottleneck
low-passes each image to roughly 1.74mm/pixel while the anatomical differences
between disease stages are millimetre-scale.

This writes data/processed_v3_hires/ + manifest_v3_go2_hires.csv, leaving the
existing v3 dataset untouched so the two can be compared directly: identical
subjects, identical split, identical slice positions, one preprocessing difference.

Usage: python scripts/reextract_v3_hires.py
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

SUBJ = ROOT / "data" / "subject_manifest_v3.csv"
PROC = ROOT / "data" / "processed_v3_hires"
N_WORKERS = 6  # fewer than reextract_v3.py: a GPU job may be sharing this machine


def _expected_paths(split, cls, subject_id):
    out_dir = PROC / split / cls
    return [out_dir / f"{subject_id}_{i:03d}.png"
            for i in range(dp.N_SLICES_PER_SUBJECT)]


def extract_one(task):
    import cv2
    cv2.setNumThreads(1)
    split, cls, subject_id, dicom_dir = task
    paths = _expected_paths(split, cls, subject_id)
    if all(p.exists() for p in paths):
        return subject_id, len(paths), None
    try:
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        slices = dp.extract_slices(dicom_dir, bottleneck=None)  # <-- the whole point
        for img, p in zip(slices, paths):
            cv2.imwrite(str(p), img)
        return subject_id, len(slices), None
    except Exception as e:  # noqa: BLE001
        return subject_id, 0, f"{type(e).__name__}: {e}"


def main() -> None:
    sm = pd.read_csv(SUBJ)
    sm = sm[sm["era"] == "GO2"].reset_index(drop=True)
    print(f"{len(sm)} ADNI-GO/2 subjects -> {PROC}", flush=True)
    print(sm.groupby("class").size().to_string(), flush=True)

    tasks = [(r["split"], r["class"], r["subject_id"], r["dicom_dir"])
             for _, r in sm.iterrows()]

    t0, errors, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(extract_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="hi-res extract"):
            sid, _, err = fut.result()
            if err:
                errors.append((sid, err))
            else:
                done += 1
    print(f"\nextracted {done}/{len(tasks)} in {(time.time()-t0)/60:.1f} min", flush=True)
    for sid, err in errors[:20]:
        print(f"  FAILED {sid}: {err}")

    failed = {s for s, _ in errors}
    rows = []
    for _, r in sm.iterrows():
        if r["subject_id"] in failed:
            continue
        for p in _expected_paths(r["split"], r["class"], r["subject_id"]):
            if p.exists():
                rows.append({"subject_id": r["subject_id"], "class": r["class"],
                             "split": r["split"], "filepath": str(p.resolve()),
                             "era": r["era"]})
    out = pd.DataFrame(rows)
    dest = ROOT / "data" / "manifest_v3_go2_hires.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(out)} slices, {out.subject_id.nunique()} subjects)")

    # The comparison is only valid if the split is byte-identical to the v3go2 one.
    base = pd.read_csv(ROOT / "data" / "manifest_v3_go2.csv")
    a = base.groupby("subject_id")["split"].first()
    b = out.groupby("subject_id")["split"].first()
    shared = a.index.intersection(b.index)
    mismatch = int((a.loc[shared] != b.loc[shared]).sum())
    print(f"split agreement with manifest_v3_go2.csv: "
          f"{len(shared)} shared subjects, {mismatch} mismatches "
          f"({'OK' if mismatch == 0 else 'PROBLEM'})")


if __name__ == "__main__":
    main()
