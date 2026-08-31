"""v4 extraction: physically-correct axial geometry (decision 27's fix).

WHAT IS DIFFERENT FROM v3, AND ONLY THIS
----------------------------------------
v3 (`data_prep.extract_slices`) cuts an axial plane out of the stacked sagittal
volume and resizes that plane -- whatever its shape -- into a square 224x224.
The plane is not square in physical terms: its rows are spaced by the
through-plane distance (1.2mm in all 853 series) and its columns by the in-plane
PixelSpacing (0.94-1.25mm, varying between subjects). The volume is therefore
anisotropic by slice_mm/row_mm, a factor that varies per subject and, because
protocol correlates with class here, per class. Squashing it into a square bakes
that per-subject stretch into the pixels. Decision 27 measured the consequence:
a RandomForest given ONLY acquisition metadata and no pixels at all scores 40.9%
on the 4-way GO/2 task against a 36.7% baseline.

v4 (`data_prep.extract_slices_isotropic`) instead:
  1. resamples the axial plane to isotropic 1.0 mm/px using the real
     PixelSpacing and the real through-plane spacing (taken from
     ImagePositionPatient; SpacingBetweenSlices is absent from most of these
     series, which is how it came to be dropped in the first place),
  2. crops a fixed 224mm x 224mm physical window centred on the head centroid,
  3. renders it at 224x224 -- i.e. exactly 1 mm per pixel, the SAME millimetre-to
     -pixel mapping for every subject.

Acquisition FOV, matrix size, anisotropy and head position in the bore all
disappear. Head size in millimetres does not: the scale factor is a constant, so
a small brain still renders small. That is the difference from decision 21's
v2crop, which normalised each brain's own bounding box and so deleted the
atrophy signal along with the confound.

Everything else is deliberately identical to v3: the same 48-92mm
vertex-anchored band, the same 32 slices, the same 144px resolution bottleneck,
the same NLM denoising parameters, the same 0.5/99.5 percentile normalisation.
Geometry is the only variable that moves between v3 and v4.

The one knock-on effect worth naming: the percentile normalisation is now
computed over the fixed physical window rather than over the whole acquired
plane, because the crop happens first. That is the more defensible of the two --
the statistics are taken over the same anatomical region in everybody instead of
over however much background the protocol happened to include.

SPLITS ARE INHERITED, NOT RECOMPUTED. This reads the same
`data/subject_manifest_v3.csv` that reextract_v3.py read, so v3 and v4 differ in
pixels and nothing else and are directly comparable. The run verifies against
`data/manifest_v3.csv` and prints the mismatch count (must be 0).

Outputs, mirroring v3 exactly:
  data/processed_v4/{split}/{class}/{subject}_{i:03d}.png
  data/manifest_v4.csv        all 853 subjects (still era-confounded; size only)
  data/manifest_v4_go2.csv    THE PRIMARY 4-WAY TASK, 618 subjects, GO/2 only
  data/manifest_v4_adcn.csv   AD vs CN across both eras, 501 subjects
  data/geometry_v4_qc.csv     per-subject measured geometry (spacings, head
                              extent in mm, head centre, fraction of head clipped
                              by the fixed FOV) -- so the claim "the window is
                              large enough" is checkable rather than asserted.

Parallel via ProcessPoolExecutor with cv2.setNumThreads(1) in each worker, as in
reextract_v3.py: fastNlMeansDenoising is internally multithreaded, so without
that pin the pool oversubscribes the CPU and gets slower. Capped at 6 workers
here because another job may be using the machine.

Resumable: a subject whose full set of PNGs already exists is skipped.
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

SUBJ = ROOT / "data" / "subject_manifest_v3.csv"   # READ-ONLY: v3's splits, inherited
PROC = ROOT / "data" / "processed_v4"
V3_MANIFEST = ROOT / "data" / "manifest_v3.csv"    # READ-ONLY: split cross-check
N_WORKERS = 6


def _expected_paths(split, cls, subject_id):
    out_dir = PROC / split / cls
    return [out_dir / f"{subject_id}_{i:03d}.png"
            for i in range(dp.N_SLICES_PER_SUBJECT)]


def extract_one(task):
    """Worker: extract and save one subject's slices. Returns (subject_id, n, qc, error)."""
    import cv2
    cv2.setNumThreads(1)  # see module docstring - avoid pool oversubscription

    split, cls, subject_id, dicom_dir = task
    paths = _expected_paths(split, cls, subject_id)
    try:
        if all(p.exists() for p in paths):
            # Still recompute the geometry record, which is header-only and cheap,
            # so a resumed run does not come back with a half-empty QC table.
            geo = dp._acquisition_geometry(dicom_dir)
            qc = {"row_mm": geo["row_mm"], "col_mm": geo["col_mm"],
                  "slice_mm": geo["slice_mm"], "rows": geo["rows"],
                  "cols": geo["cols"], "n_dicom": geo["n_dicom"], "resumed": 1}
            return subject_id, len(paths), qc, None
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        slices, qc = dp.extract_slices_isotropic(dicom_dir, return_qc=True)
        for img, p in zip(slices, paths):
            cv2.imwrite(str(p), img)
        qc["resumed"] = 0
        return subject_id, len(slices), qc, None
    except Exception as e:  # noqa: BLE001 - one bad scan must not kill the pool
        return subject_id, 0, None, f"{type(e).__name__}: {e}"


def main() -> None:
    sm = pd.read_csv(SUBJ)
    print(f"{len(sm)} subjects -> {PROC}", flush=True)
    print(f"geometry: isotropic {dp.ISO_MM_PER_PX} mm/px, "
          f"{dp.FOV_MM}mm FOV centred on the head, rendered at {dp.OUT_SIZE}px "
          f"(bottleneck {dp.BOTTLENECK_SIZE}px)", flush=True)
    print(sm.groupby(["class", "era"]).size().to_string(), flush=True)

    tasks = [(r["split"], r["class"], r["subject_id"], r["dicom_dir"])
             for _, r in sm.iterrows()]

    t0 = time.time()
    errors, qc_rows, done = [], [], 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(extract_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Extracting slices (v4)"):
            sid, n, qc, err = fut.result()
            if err:
                errors.append((sid, err))
            else:
                done += 1
                if qc is not None:
                    qc_rows.append({"subject_id": sid, **qc})
    elapsed = (time.time() - t0) / 60
    print(f"\nextracted {done}/{len(tasks)} subjects in {elapsed:.1f} min", flush=True)

    if errors:
        print(f"\n!!! {len(errors)} subjects FAILED:")
        for sid, err in errors[:20]:
            print(f"  {sid}: {err}")
    else:
        print("0 failures")

    # Build the slice manifest from what is actually on disk, so a failed subject
    # is simply absent rather than silently pointing at missing files.
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

    out = ROOT / "data" / "manifest_v4.csv"
    slices.to_csv(out, index=False)
    print(f"wrote {out}  ({len(slices)} slices, {slices.subject_id.nunique()} subjects)")

    go2 = slices[slices["era"] == "GO2"]
    out_go2 = ROOT / "data" / "manifest_v4_go2.csv"
    go2.to_csv(out_go2, index=False)
    print(f"wrote {out_go2}  ({len(go2)} slices, {go2.subject_id.nunique()} subjects)")

    adcn = slices[slices["class"].isin(["AD", "CN"])]
    out_adcn = ROOT / "data" / "manifest_v4_adcn.csv"
    adcn.to_csv(out_adcn, index=False)
    print(f"wrote {out_adcn}  ({len(adcn)} slices, {adcn.subject_id.nunique()} subjects)")

    if qc_rows:
        qc_df = pd.DataFrame(qc_rows).merge(sm[["subject_id", "class", "era"]],
                                            on="subject_id", how="left")
        out_qc = ROOT / "data" / "geometry_v4_qc.csv"
        qc_df.to_csv(out_qc, index=False)
        print(f"wrote {out_qc}  ({len(qc_df)} subjects)")
        if "clipped_frac_max" in qc_df:
            fresh = qc_df[qc_df["resumed"] == 0]
            if len(fresh):
                print("\nhead vs the fixed 224mm window (fresh extractions only):")
                print(fresh[["head_lr_mm", "head_ap_mm", "clipped_frac_max",
                             "clipped_frac_mean"]].describe().to_string())
                n_clip = int((fresh["clipped_frac_max"] > 0.01).sum())
                print(f"subjects losing >1% of head area at the worst slice: "
                      f"{n_clip}/{len(fresh)}")

    # --- split inheritance check: v4 must place every subject exactly where v3 did ---
    if V3_MANIFEST.exists():
        v3 = pd.read_csv(V3_MANIFEST)
        s3 = v3.groupby("subject_id")["split"].first()
        s4 = slices.groupby("subject_id")["split"].first()
        common = s3.index.intersection(s4.index)
        mismatches = int((s3.loc[common] != s4.loc[common]).sum())
        print(f"\nsplit inheritance vs manifest_v3.csv: {len(common)} shared subjects, "
              f"{mismatches} split mismatches")
        only3 = sorted(set(s3.index) - set(s4.index))
        only4 = sorted(set(s4.index) - set(s3.index))
        if only3:
            print(f"  in v3 but not v4 ({len(only3)}): {only3[:10]}")
        if only4:
            print(f"  in v4 but not v3 ({len(only4)}): {only4[:10]}")
    else:
        print("\n(manifest_v3.csv absent - split check skipped)")

    print("\nslices per split (primary GO2 4-way task):")
    print(go2.groupby(["split", "class"]).size().unstack(fill_value=0).to_string())
    print("\nsubjects per split (primary GO2 4-way task):")
    print(go2.groupby(["split", "class"])["subject_id"].nunique()
          .unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
