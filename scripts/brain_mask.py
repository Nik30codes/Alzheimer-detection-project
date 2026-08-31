"""
Removes skull, scalp and background from the processed axial slices.

Why this is the best lever available without new data (decision 10): the models currently
separate ADNI1 (CN/AD) from ADNI-GO/2 (EMCI/LMCI) with 95-100% accuracy, which is a
scanner/protocol cue rather than disease. Much of that cue lives OUTSIDE the brain --
skull marrow brightness, scalp fat thickness, how much neck is in frame, and background
noise texture all vary by scanner and sequence while carrying no information about
atrophy. Masking to brain-only deletes that shortcut while keeping the hippocampal region
the task actually depends on.

Method (a classic threshold-based strip, run on the 224x224 PNGs rather than re-deriving
from DICOM, which keeps it cheap and reversible):
  1. Otsu threshold to separate head from background.
  2. Morphological opening with a large kernel: the skull is a THIN bright ring, so
     opening breaks its connection to the brain while the brain, being thick, survives.
  3. Keep the largest connected component -- the brain.
  4. Close and fill holes so ventricles and dark interior structures are not punched out.
  5. Slight erosion to shave any residual inner skull table.

Usage:
  python brain_mask.py qc      -> writes a before/after grid to reports/figures for review
  python brain_mask.py run     -> masks every slice into data/processed_masked/ + manifest
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_MANIFEST = os.path.join(ROOT, "data", "manifest.csv")
OUT_ROOT = os.path.join(ROOT, "data", "processed_masked")
OUT_MANIFEST = os.path.join(ROOT, "data", "manifest_masked.csv")
CROP_ROOT = os.path.join(ROOT, "data", "processed_braincrop")
CROP_MANIFEST = os.path.join(ROOT, "data", "manifest_braincrop.csv")


SKULL_THICKNESS_PX = 9  # scalp + skull at 224x224; measured off the QC grid


def brain_mask(img: np.ndarray) -> np.ndarray:
    """Returns the masked image (uint8), brain kept, skull/scalp/background zeroed.

    Deliberately NOT Otsu-thresholded. Otsu on a T1 slice splits bright white matter from
    everything darker, so the mid-grey cortical ribbon lands on the background side and
    gets deleted -- which destroys exactly the tissue atrophy shows up in (a first attempt
    did this and retained only ~21% of the head).

    Instead: take the whole head with a LOW threshold (background is near zero), fill it
    solid, then erode inward by roughly the scalp+skull thickness. Erosion costs a thin
    outer rim of cortex, which is an acceptable trade for removing the skull marrow and
    scalp fat that carry the scanner-protocol cue.
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Low threshold = head vs background, rather than bright tissue vs the rest.
    # Otsu on the nonzero pixels only would still sit too high, so use a fixed low cut
    # relative to the slice's own dynamic range.
    thresh = max(10, int(0.12 * float(blur.max())))
    head = (blur > thresh).astype(np.uint8) * 255

    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(head, connectivity=8)
    if n <= 1:
        return img
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    head = (labels == largest).astype(np.uint8) * 255

    # Fill so interior dark structures (ventricles, CSF) stay inside the head mask.
    contours, _ = cv2.findContours(head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(head)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # Step inward past scalp + skull.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * SKULL_THICKNESS_PX + 1, 2 * SKULL_THICKNESS_PX + 1))
    brain = cv2.erode(filled, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(brain, connectivity=8)
    if n <= 1:
        return img
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    brain = (labels == largest).astype(np.uint8) * 255

    return cv2.bitwise_and(img, img, mask=brain)


def brain_crop(img: np.ndarray, out_size: int = 224, pad: int = 4) -> np.ndarray:
    """Mask to brain, crop tightly to it, then rescale to a fixed square size.

    Masking alone is not enough. Grad-CAM on the unmasked model showed its attention
    sitting on the LEFT AND RIGHT IMAGE MARGINS -- empty background -- and 91-95% of the
    attention for EMCI/LMCI fell outside the brain entirely. The reason is that CN/AD
    were acquired at native 192x192 and EMCI/LMCI at 256x256, so after a common resize
    the two cohorts differ in how much of the frame the head fills. That border geometry
    is a perfect label for scanner era.

    Zeroing the background removes its texture but leaves the head-size difference
    intact. Cropping to the brain's bounding box and rescaling normalises both: every
    subject ends up with the brain filling the same fraction of the frame regardless of
    the acquisition matrix.

    Note this deliberately discards absolute head size. Brain volume does shrink in
    Alzheimer's, so some real signal is lost -- but absolute size here is dominated by
    the acquisition matrix and by normal head-size variation between people, so it was
    never a usable cue in this dataset anyway.
    """
    masked = brain_mask(img)
    ys, xs = np.where(masked > 0)
    if len(ys) == 0:
        return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    y0, y1 = max(0, ys.min() - pad), min(masked.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(masked.shape[1], xs.max() + pad + 1)
    crop = masked[y0:y1, x0:x1]

    # Pad to a square first so the rescale does not distort anatomy's aspect ratio.
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=crop.dtype)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    return cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_AREA)


def coverage(img, masked):
    """Fraction of originally-bright pixels retained -- a sanity number per slice.
    Very low values mean the mask ate the brain; very high means it kept the skull."""
    orig = (img > 20).sum()
    kept = (masked > 20).sum()
    return kept / orig if orig else 0.0


def qc(n_per_class=3):
    """Before/after grid across all four classes, for eyeballing before committing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = pd.read_csv(SRC_MANIFEST)
    picks = pd.concat([m[m["class"] == c].sample(n_per_class, random_state=0)
                       for c in ["CN", "AD", "EMCI", "LMCI"]], ignore_index=True)

    rows = len(picks)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    covs = []
    for ax_row, (_, r) in zip(axes, picks.iterrows()):
        img = cv2.imread(r["filepath"], cv2.IMREAD_GRAYSCALE)
        masked = brain_mask(img)
        c = coverage(img, masked)
        covs.append(c)
        ax_row[0].imshow(img, cmap="gray"); ax_row[0].set_title(f"{r['class']} original", fontsize=9)
        ax_row[1].imshow(masked, cmap="gray"); ax_row[1].set_title(f"masked (kept {c:.0%})", fontsize=9)
        ax_row[2].imshow(img.astype(int) - masked.astype(int), cmap="magma")
        ax_row[2].set_title("removed", fontsize=9)
        for a in ax_row:
            a.axis("off")
    plt.tight_layout()
    out = os.path.join(ROOT, "reports", "figures", "brain_mask_qc.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=110)
    print("wrote", out)
    print(f"retained-brightness fraction: min {min(covs):.0%}, "
          f"median {np.median(covs):.0%}, max {max(covs):.0%}")
    print("Sanity: expect roughly 50-85%. Much lower means the mask is eating brain; "
          "much higher means skull is surviving.")


def run(crop=False, source="v1"):
    """crop=False -> mask only (data/processed_masked).
    crop=True  -> mask + tight brain crop + rescale (data/processed_braincrop), which
    additionally normalises the head-size/field-of-view difference between cohorts.

    source="v2" reads the millimetre-anchored re-extraction (manifest_v2.csv) instead of
    the original slices, and writes data/processed_v2_braincrop/ + manifest_v2_braincrop.csv.
    """
    if source in ("v3", "v4"):
        # v3 = the 853-subject era-balanced dataset; v4 = v3 with isotropic geometry
        # (decision 31). Masking was last tested on v2, where the AD-vs-CN test set was
        # 26 subjects and every confidence interval spanned [0.25, 0.89] -- genuinely
        # uninformative. Re-testing on 501 era-balanced subjects with a 75-subject test
        # set is the point of adding these.
        src_manifest = os.path.join(ROOT, "data", f"manifest_{source}.csv")
        src_dir = os.path.join(ROOT, "data", f"processed_{source}")
        suffix = "braincrop" if crop else "masked"
        out_root = os.path.join(ROOT, "data", f"processed_{source}_{suffix}")
        out_manifest = os.path.join(ROOT, "data", f"manifest_{source}_{suffix}.csv")
    elif source == "v2":
        src_manifest = os.path.join(ROOT, "data", "manifest_v2.csv")
        src_dir = os.path.join(ROOT, "data", "processed_v2")
        # Mask-only keeps the original framing, so absolute brain size is preserved.
        # That matters: cropping+rescaling every brain to fill the frame normalises away
        # brain volume, and global atrophy is a genuine Alzheimer's marker -- measured
        # effect was AD-vs-CN AUC dropping 0.667 -> 0.576 when the crop was applied.
        # Mask-only aims to remove the scanner cue (skull, scalp, background) while
        # keeping the volumetric signal.
        suffix = "braincrop" if crop else "masked"
        out_root = os.path.join(ROOT, "data", f"processed_v2_{suffix}")
        out_manifest = os.path.join(ROOT, "data", f"manifest_v2_{suffix}.csv")
    else:
        src_manifest = SRC_MANIFEST
        src_dir = os.path.join(ROOT, "data", "processed")
        out_root = CROP_ROOT if crop else OUT_ROOT
        out_manifest = CROP_MANIFEST if crop else OUT_MANIFEST

    # "class" is a Python keyword, so itertuples would rename it; rename it up front
    # and keep the loop readable instead.
    m = pd.read_csv(src_manifest).rename(columns={"class": "label"})
    out_rows, bad = [], 0
    for i, r in enumerate(m.itertuples(index=False), 1):
        img = cv2.imread(r.filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(r.filepath)
        masked = brain_mask(img)
        c = coverage(img, masked)
        if c < 0.25 or c > 0.98:
            bad += 1  # mask failed badly; keep the original so we never blank a slice
            out_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA) if crop else img
        else:
            out_img = brain_crop(img) if crop else masked

        rel = os.path.relpath(r.filepath, src_dir)
        dest = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cv2.imwrite(dest, out_img)
        out_rows.append({"subject_id": r.subject_id, "class": r.label,
                         "split": r.split, "filepath": dest})
        if i % 2000 == 0:
            print(f"  {i}/{len(m)} slices", flush=True)

    pd.DataFrame(out_rows).to_csv(out_manifest, index=False)
    print(f"wrote {len(out_rows)} slices to {out_root}")
    print(f"manifest: {out_manifest}")
    print(f"fell back to the original image on {bad} slices ({bad/len(m):.2%}) where the "
          f"mask looked implausible")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "qc"
    if mode == "qc":
        qc()
    elif mode == "run":
        run(crop=False)
    elif mode == "runcrop":
        run(crop=True)
    elif mode == "runcrop_v2":
        run(crop=True, source="v2")
    elif mode == "runmask_v3":
        run(crop=False, source="v3")
    elif mode == "runcrop_v3":
        run(crop=True, source="v3")
    elif mode == "runmask_v4":
        run(crop=False, source="v4")
    elif mode == "runmask_v2":
        run(crop=False, source="v2")
    else:
        raise SystemExit(f"unknown mode {mode!r}; "
                         "use qc | run | runcrop | runcrop_v2 | runmask_v2")
