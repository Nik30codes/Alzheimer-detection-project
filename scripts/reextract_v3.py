"""Phase 2: extract axial slices for all 853 subjects (original 439 + 414 from the
AlzheimerAdditional expansion) into data/processed_v3/ + manifest_v3.csv.

Uses the same millimetre-anchored band as v2 (decision 11). Validated on every new
series family first by scripts/qc_v3_extraction.py -- all are sagittally acquired,
so the axial reslice is correct for the Philips/Siemens parallel-imaging variants
that the original pool never contained.

Runs the per-subject work across processes. data_prep.process_all() is a serial loop,
which cost ~8.8 s/subject (~2 h for 853) while using one of this machine's twelve
cores; the work is embarrassingly parallel because each subject writes its own PNGs.
Each worker pins OpenCV to a single thread -- fastNlMeansDenoising is internally
multithreaded, so without that the pool oversubscribes the CPU and gets slower.

Resumable: a subject whose full set of PNGs already exists is skipped, so an
interrupted run can be restarted without repeating finished work.

Also writes the derived manifests the experiments actually train on:

  manifest_v3.csv       all 853 subjects (4-way, but EMCI/LMCI are ADNI-GO/2-only
                        so era still leaks -- report with that caveat)
  manifest_v3_go2.csv   THE PRIMARY TASK. 618 subjects, all four classes, all from
                        ADNI-GO/2, so cohort/scanner era carries no label information.
  manifest_v3_adcn.csv  AD vs CN over both eras, 501 subjects; era alone scores
                        exactly the majority baseline, so the confound is gone here too.
"""
import os
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
PROC = ROOT / "data" / "processed_v3"
N_WORKERS = 10  # of 12 logical cores, leaving headroom for the OS


def _expected_paths(split, cls, subject_id):
    out_dir = PROC / split / cls
    return [out_dir / f"{subject_id}_{i:03d}.png"
            for i in range(dp.N_SLICES_PER_SUBJECT)]


def extract_one(task):
    """Worker: extract and save one subject's slices. Returns (subject_id, n, error)."""
    import cv2
    cv2.setNumThreads(1)  # see module docstring - avoid pool oversubscription

    split, cls, subject_id, dicom_dir = task
    paths = _expected_paths(split, cls, subject_id)
    if all(p.exists() for p in paths):
        return subject_id, len(paths), None
    try:
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        slices = dp.extract_slices(dicom_dir)
        for img, p in zip(slices, paths):
            cv2.imwrite(str(p), img)
        return subject_id, len(slices), None
    except Exception as e:  # noqa: BLE001 - one bad scan must not kill the pool
        return subject_id, 0, f"{type(e).__name__}: {e}"


def main() -> None:
    sm = pd.read_csv(SUBJ)
    print(f"{len(sm)} subjects -> {PROC}", flush=True)
    print(sm.groupby(["class", "era"]).size().to_string(), flush=True)

    tasks = [(r["split"], r["class"], r["subject_id"], r["dicom_dir"])
             for _, r in sm.iterrows()]

    t0 = time.time()
    errors = []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(extract_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Extracting slices"):
            sid, n, err = fut.result()
            if err:
                errors.append((sid, err))
            else:
                done += 1
    print(f"\nextracted {done}/{len(tasks)} subjects in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    if errors:
        print(f"\n!!! {len(errors)} subjects FAILED:")
        for sid, err in errors[:20]:
            print(f"  {sid}: {err}")

    # Build the slice manifest from what is actually on disk, so a failed subject is
    # simply absent rather than silently pointing at missing files.
    failed = {sid for sid, _ in errors}
    rows = []
    for _, r in sm.iterrows():
        if r["subject_id"] in failed:
            continue
        for p in _expected_paths(r["split"], r["class"], r["subject_id"]):
            if p.exists():
                rows.append({"subject_id": r["subject_id"], "class": r["class"],
                             "split": r["split"], "filepath": str(p.resolve()),
                             "era": r["era"]})
    slices = pd.DataFrame(rows)

    out = ROOT / "data" / "manifest_v3.csv"
    slices.to_csv(out, index=False)
    print(f"wrote {out}  ({len(slices)} slices, {slices.subject_id.nunique()} subjects)")

    go2 = slices[slices["era"] == "GO2"]
    out_go2 = ROOT / "data" / "manifest_v3_go2.csv"
    go2.to_csv(out_go2, index=False)
    print(f"wrote {out_go2}  ({len(go2)} slices, {go2.subject_id.nunique()} subjects)")

    adcn = slices[slices["class"].isin(["AD", "CN"])]
    out_adcn = ROOT / "data" / "manifest_v3_adcn.csv"
    adcn.to_csv(out_adcn, index=False)
    print(f"wrote {out_adcn}  ({len(adcn)} slices, {adcn.subject_id.nunique()} subjects)")

    print("\nslices per split (primary GO2 4-way task):")
    print(go2.groupby(["split", "class"]).size().unstack(fill_value=0).to_string())
    print("\nsubjects per split (primary GO2 4-way task):")
    print(go2.groupby(["split", "class"])["subject_id"].nunique()
          .unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
