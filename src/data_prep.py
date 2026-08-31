"""
Turns raw ADNI DICOM folders into a labeled, split, ready-to-train image dataset.

Pipeline (see functions below, in order of use):
  1. build_manifest   -- find each subject's MRI scan, one row per subject
  2. split_subjects    -- decide train/val/test PER SUBJECT, before any images exist
  3. extract_slices     -- read one subject's 3D scan, pick central slices, preprocess
  4. process_all         -- run step 3 for every subject, save PNGs, write final manifest

Why subject-level, not slice/image-level: each subject has ~160-180 DICOM slices
that are all part of the SAME brain scan. If we split by image, near-identical
slices from one person end up in both train and test -- the model partly
"memorizes" that person instead of learning the disease pattern, and test
accuracy looks better than it really is. Splitting by subject_id first closes
that loophole: every slice from a given person stays on one side of the split.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import cv2
from sklearn.model_selection import train_test_split

CLASSES = ["CN", "AD", "EMCI", "LMCI"]
MIN_SLICES_PER_SESSION = 100  # below this it's a localizer/scout, not a full brain volume
N_SLICES_PER_SUBJECT = 32      # how many 2D training images each subject contributes
AXIAL_MM_BELOW_VERTEX = (48.0, 92.0)
# Axial band expressed as MILLIMETRES BELOW THE TOP OF THE HEAD, which is what actually
# fixes slice alignment between subjects.
#
# What was wrong before: AXIAL_ROW_RANGE below took a fixed fraction of the scan's own
# voxel height. Nothing in the code ever located the head, despite the comment saying
# "fraction of head height". ADNI protocols differ in how far below the brain they
# extend (some include a lot of neck) and in how much empty space sits above the vertex,
# so the same fraction landed on completely different anatomy per subject. QC
# (scripts/qc_slice_anatomy.py) showed slice #16 sitting at the orbits in some subjects
# and high in the centrum semiovale in others, and for some LMCI subjects the ENTIRE
# 32-slice band sat above the hippocampus -- meaning the most important structure for
# Alzheimer's was never imaged. Worse, the drift correlated with cohort, so it acted as
# yet another non-diagnostic shortcut on top of decision 10.
#
# Why millimetres from the vertex work: adult brain height is far more consistent than
# scan field of view, so a fixed physical offset from the top of the skull hits the same
# structures in everyone regardless of protocol. 48mm is roughly the body of the lateral
# ventricles; 92mm reaches the medial temporal lobe / hippocampal level. The band
# therefore spans ventricles down through the hippocampus -- the two structures that
# actually carry Alzheimer's signal.
AXIAL_ROW_RANGE = (0.28, 0.48)  # DEPRECATED, kept only for reference; see above.
                                # fraction of head height (0=top of skull, 1=bottom of scan).
                                   # The DICOM series is acquired SAGITTALLY, but we reconstruct a
                                   # 3D volume per subject and reslice it AXIALLY (top-down view --
                                   # the standard view for this task, showing the ventricles as the
                                   # classic "butterfly" shape). Originally capped at 0.42 to avoid
                                   # the noisier sinus/orbit region -- but the custom CNN baseline
                                   # confused AD with CN far more than expected (AD is usually the
                                   # *easiest* class to separate from CN), and 0.42 stops short of
                                   # the hippocampus / medial temporal lobe, which sits a bit lower
                                   # and is the single most established atrophy marker for AD.
                                   # Extended to 0.48 to include that region, accepting slightly more
                                   # noise (now mitigated by the Non-Local Means denoising below) in
                                   # exchange for not omitting the most disease-relevant anatomy.
OUT_SIZE = 224                    # matches MobileNetV2 / EfficientNet-B0 input size
BOTTLENECK_SIZE = 144              # CN/AD scans are natively 192x192, but EMCI/LMCI scans (a later
                                      # ADNI protocol) are natively 256x256. Resizing straight to
                                      # OUT_SIZE would upsample one group and downsample the other --
                                      # two different operations with different frequency effects --
                                      # which measurably left EMCI/LMCI images sharper/grainier than
                                      # CN/AD in a way that tracked scanner protocol, not anatomy (a
                                      # real confound: a model could partly tell classes apart by scan
                                      # protocol instead of disease). Routing every image through this
                                      # smaller common resolution first, before the final resize up to
                                      # OUT_SIZE, equalizes both groups onto the same frequency ceiling.
RANDOM_SEED = 42


def build_manifest(raw_dir: Path) -> pd.DataFrame:
    """
    Walk data/raw/{CLASS}/ADNI/{subject_id}/MPRAGE/{session}/{image_id}/*.dcm
    and return one row per subject: which session directory to use, and how
    many slices it has. If a subject has more than one session, keep the
    largest (most complete scan).
    """
    manifest_rows = []
    for cls in CLASSES:
        class_dir = raw_dir / cls / "ADNI"
        if not class_dir.exists():
            continue
        for subject_dir in sorted(class_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name
            session_dirs = {p.parent for p in subject_dir.rglob("*.dcm")}
            best_session, best_count = None, 0
            for session_dir in session_dirs:
                n = len(list(session_dir.glob("*.dcm")))
                if n > best_count:
                    best_session, best_count = session_dir, n
            if best_session is not None and best_count >= MIN_SLICES_PER_SESSION:
                manifest_rows.append({
                    "class": cls,
                    "subject_id": subject_id,
                    "dicom_dir": str(best_session.resolve()),
                    "n_slices_raw": best_count,
                })

    return pd.DataFrame(manifest_rows)


def split_subjects(manifest: pd.DataFrame) -> pd.DataFrame:
    """Stratified 70/15/15 train/val/test split, one decision per subject_id."""
    train_val, test = train_test_split(
        manifest, test_size=0.15, stratify=manifest["class"], random_state=RANDOM_SEED
    )
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, stratify=train_val["class"], random_state=RANDOM_SEED
    )
    manifest = manifest.copy()
    manifest.loc[train.index, "split"] = "train"
    manifest.loc[val.index, "split"] = "val"
    manifest.loc[test.index, "split"] = "test"
    return manifest


def _read_sorted_dicom_files(dicom_dir: str) -> list[Path]:
    """Fast header-only read (no pixel data) just to sort slices into anatomical order."""
    files = list(Path(dicom_dir).glob("*.dcm"))
    tagged = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        instance_num = int(ds.get("InstanceNumber", 0))
        tagged.append((instance_num, f))
    tagged.sort(key=lambda t: t[0])
    return [f for _, f in tagged]


def _build_volume(dicom_dir: str) -> np.ndarray:
    """
    Stack this subject's sorted sagittal DICOM slices into one 3D array.
    Resulting axes: [0]=left-right (sagittal slice order), [1]=superior-inferior
    (image rows), [2]=anterior-posterior (image columns) -- confirmed from this
    dataset's ImageOrientationPatient tag, not assumed.
    """
    sorted_files = _read_sorted_dicom_files(dicom_dir)
    return np.stack([pydicom.dcmread(f).pixel_array for f in sorted_files], axis=0)


def _row_spacing_mm(dicom_dir: str) -> float:
    """Millimetres per row along the superior-inferior axis.

    The volume is stacked from sagittal images whose ROWS run superior-inferior, so the
    first element of PixelSpacing (row spacing) is the millimetres-per-row we need.
    Falls back to 1.0mm if the tag is missing, which is the usual ADNI T1 value.
    """
    first = _read_sorted_dicom_files(dicom_dir)[0]
    ds = pydicom.dcmread(first, stop_before_pixels=True)
    spacing = ds.get("PixelSpacing", None)
    return float(spacing[0]) if spacing is not None else 1.0


def _find_vertex_row(volume: np.ndarray) -> int:
    """Index of the topmost axial row that contains real head tissue (the vertex).

    This is the anatomical anchor the old code was missing. Scans differ in how much
    empty space sits above the head and how far below the brain they extend (some ADNI
    protocols include a lot of neck), so any band defined as a fraction of the scan's
    own height lands on different anatomy in different subjects.

    Robustness: the threshold is taken from the volume's own intensity distribution, and
    a row only counts as "head" once its tissue area exceeds 25% of the maximum tissue
    area over all rows. An earlier 8% cutoff fired ~10 rows too early on some subjects,
    latching onto scalp or noise floating above the true skull, which pushed the whole
    band upward and left LMCI subjects short of the hippocampus. The area profile rises
    steeply at the vertex, so a higher cutoff is both safer and barely less precise.

    The profile is smoothed first so a single noisy row cannot define the anchor.
    """
    thresh = np.percentile(volume, 60)
    areas = (volume > thresh).sum(axis=(0, 2)).astype(np.float64)
    if areas.max() <= 0:
        return 0
    kernel = np.ones(5) / 5.0
    areas = np.convolve(areas, kernel, mode="same")
    head_rows = np.flatnonzero(areas > 0.25 * areas.max())
    return int(head_rows[0]) if len(head_rows) else 0


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """MRI intensities are arbitrary units (not standardized like CT), so we
    clip extreme outliers per-image and rescale into a normal 0-255 image."""
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)


def extract_slices(dicom_dir: str, n_slices: int = N_SLICES_PER_SUBJECT,
                   bottleneck: int | None = BOTTLENECK_SIZE) -> list[np.ndarray]:
    """
    Reconstruct one subject's 3D scan and return n_slices axial (top-down) images,
    evenly spaced across the millimetre-anchored band, each resized to OUT_SIZE.

    bottleneck=None skips the resolution-harmonization step and resizes straight to
    OUT_SIZE. Only correct when every scan in the dataset shares a native matrix size.

    Why that option exists: BOTTLENECK_SIZE equalizes ADNI1's 192x192 against
    ADNI-GO/2's 256x256 (see BOTTLENECK_SIZE). But the primary four-way task is
    restricted to ADNI-GO/2, where every scan is natively 256 rows at 1.00-1.05mm, so
    there is nothing to harmonize -- and routing through 144px low-passes the image to
    roughly 1.74mm per pixel, discarding detail at exactly the scale that separates
    disease stages. Passing None keeps the full 224px detail for that dataset.
    """
    volume = _build_volume(dicom_dir)  # (left_right, superior_inferior, anterior_posterior)
    n_rows = volume.shape[1]

    # Anchor the band on the top of the head and step down a fixed number of
    # MILLIMETRES, rather than taking a fraction of the scan's own voxel height.
    # See AXIAL_MM_BELOW_VERTEX for why the old fractional version was broken.
    spacing = _row_spacing_mm(dicom_dir)
    vertex = _find_vertex_row(volume)
    row_lo = vertex + int(round(AXIAL_MM_BELOW_VERTEX[0] / spacing))
    row_hi = vertex + int(round(AXIAL_MM_BELOW_VERTEX[1] / spacing))

    # If a scan simply does not extend far enough down, clamp into range and keep the
    # requested thickness where possible, so we never index off the end of the volume.
    row_hi = min(row_hi, n_rows - 1)
    row_lo = min(row_lo, max(row_hi - n_slices, 0))
    row_idxs = np.linspace(row_lo, row_hi, n_slices).round().astype(int)

    slices = []
    for r in row_idxs:
        axial = volume[:, r, :].astype(np.float32)  # a horizontal cross-section at height r
        axial = _normalize_to_uint8(axial)

        # common bottleneck resolution first (see BOTTLENECK_SIZE comment), THEN resize up to OUT_SIZE
        if bottleneck is not None:
            axial = cv2.resize(axial, (bottleneck, bottleneck), interpolation=cv2.INTER_AREA)
            axial = cv2.resize(axial, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)
        else:
            # INTER_AREA is the right filter going 256 -> 224 (a downsample); INTER_CUBIC
            # would alias.
            axial = cv2.resize(axial, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

        # These axial views are reconstructed by cutting across ~160 independently-acquired
        # sagittal DICOM slices, so unlike a natively-acquired image, neighboring pixels along
        # that axis come from separate acquisitions with uncorrelated noise -- this shows up as
        # grain. Non-Local Means denoising (a standard technique, not something bespoke) cleans
        # this up while preserving edges (ventricle borders, cortical folds) much better than a
        # plain blur would.
        axial = cv2.fastNlMeansDenoising(axial, h=8, templateWindowSize=7, searchWindowSize=21)

        slices.append(axial)

    return slices


# ---------------------------------------------------------------------------
# v4: physically-correct (isotropic, fixed-field-of-view) axial rendering.
#
# Everything below is ADDITIVE. extract_slices() above is unchanged and still
# backs v1/v2/v3; v4 uses extract_slices_isotropic().
#
# THE BUG IT FIXES (decision 27). _build_volume() np.stack()s the sagittally
# acquired DICOMs and throws the through-plane spacing away, so the volume's
# axis 0 (left-right) is implicitly treated as if it had the same pitch as the
# in-plane axes. It does not: through-plane spacing is 1.2mm in every one of the
# 853 series, while in-plane spacing varies 0.94-1.25mm between subjects. The
# reconstructed axial plane is therefore anisotropic by slice_mm/row_mm, a
# SUBJECT-VARYING factor, and extract_slices() then squashes that whole
# variable-size plane into a square 224x224 -- baking a per-subject geometric
# stretch into the pixels. Because protocol correlates with class in this
# dataset, a RandomForest trained on acquisition metadata alone (no pixels)
# beats the majority baseline on the 4-way GO/2 task.
#
# THE FIX, in three steps:
#   1. resample the axial plane to ISOTROPIC voxels at a fixed ISO_MM_PER_PX,
#      using the real PixelSpacing (in-plane) and the real through-plane spacing
#      (from ImagePositionPatient, falling back to SpacingBetweenSlices /
#      SliceThickness). After this, one pixel means the same physical distance in
#      every subject and in both image directions.
#   2. crop a FIXED PHYSICAL field of view (FOV_MM square) centred on the head
#      centroid. This is what removes the protocol variables -- acquisition FOV,
#      matrix size, head position in the bore -- because every subject is now
#      rendered through the same physical window.
#   3. render that window to OUT_SIZE pixels. The scale factor is the SAME for
#      every subject, so HEAD SIZE IN MILLIMETRES SURVIVES.
#
# Step 3 is the whole point and the reason this is not decision 21's v2crop.
# v2crop normalised each brain's own bounding box to fill the frame, which
# rescales every subject by a different factor and therefore deletes brain
# volume -- the actual atrophy signal -- along with the confound. Here the
# millimetre-to-pixel mapping is a constant, so a small brain still renders
# small. Only the nuisance geometry is normalised.
#
# The 48-92mm axial band, the NLM denoising and the 0.5/99.5 percentile
# normalisation are deliberately IDENTICAL to v3. Geometry is the only variable
# that changes between v3 and v4.
# ---------------------------------------------------------------------------

ISO_MM_PER_PX = 1.0   # millimetres per pixel after resampling; fixed for every subject.
FOV_MM = 224.0        # physical size of the square window rendered to OUT_SIZE pixels.
# 224mm at 1.0mm/px lands exactly on OUT_SIZE, so the resample and the crop are a
# single operation with no second rescale.
#
# Why not the 180mm that was first suggested: measured on a 30-subject sample
# spanning both eras and all four classes, the head's own extent inside the
# 48-92mm band is 146-196mm left-right (mean 163) and 185-233mm
# anterior-posterior (mean 205) -- the A-P figure includes nose and occiput at
# the lower slices. A 180mm window would clip the head of essentially every
# subject, and clipping is a head-size normalisation applied only to big heads:
# exactly the v2crop failure mode, in a subtler form. 224mm contains the head for
# all but the most extreme subject and costs nothing, because the source data is
# 0.94-1.25mm/px so a 1.0mm/px grid is already at native resolution -- a larger
# physical window here buys margin without discarding detail.
_HEAD_MASK_FRAC = 0.12  # of the slice's 99.5th percentile; low on purpose (see below)


def _build_volume_from_files(sorted_files: list[Path]) -> np.ndarray:
    """_build_volume() split so callers that already sorted the headers don't re-read them."""
    return np.stack([pydicom.dcmread(f).pixel_array for f in sorted_files], axis=0)


def _acquisition_geometry(dicom_dir: str) -> dict:
    """Every physical dimension of one series, read once.

    Returns the sorted file list alongside the spacings so a caller can build the
    volume without a second header pass.

      row_mm   millimetres per image ROW    -> superior-inferior (volume axis 1)
      col_mm   millimetres per image COLUMN -> anterior-posterior (volume axis 2)
      slice_mm millimetres between SLICES   -> left-right (volume axis 0)

    slice_mm is taken from ImagePositionPatient (first to last, divided by the
    number of gaps), which is the only source that is always present and always
    describes the actual reconstructed pitch. SpacingBetweenSlices is absent from
    roughly two thirds of these series -- which is how it came to be ignored --
    and SliceThickness describes the excited slab, not the pitch, so both are
    used only as fallbacks.
    """
    files = _read_sorted_dicom_files(dicom_dir)
    ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)
    spacing = ds0.get("PixelSpacing", None)
    row_mm = float(spacing[0]) if spacing is not None else 1.0
    col_mm = float(spacing[1]) if spacing is not None else row_mm

    slice_mm = None
    if len(files) > 1:
        try:
            p0 = np.asarray(ds0.ImagePositionPatient, dtype=np.float64)
            pn = np.asarray(pydicom.dcmread(files[-1], stop_before_pixels=True)
                            .ImagePositionPatient, dtype=np.float64)
            d = float(np.linalg.norm(pn - p0)) / (len(files) - 1)
            if 0.2 < d < 10.0:  # sanity: reject a malformed or duplicated position tag
                slice_mm = d
        except Exception:  # noqa: BLE001 - fall through to the tag-based fallbacks
            slice_mm = None
    if slice_mm is None:
        sbs = ds0.get("SpacingBetweenSlices", None)
        thk = ds0.get("SliceThickness", None)
        slice_mm = float(sbs) if sbs else (float(thk) if thk else 1.2)

    return {
        "files": files,
        "row_mm": row_mm,
        "col_mm": col_mm,
        "slice_mm": slice_mm,
        "rows": int(ds0.Rows),
        "cols": int(ds0.Columns),
        "n_dicom": len(files),
    }


def _head_mask_2d(axial: np.ndarray) -> np.ndarray:
    """Boolean mask of the whole head (skull + scalp, not just brain) in one axial slice.

    Deliberately NOT Otsu -- decision 10's note records that Otsu deletes the
    entire cortical ribbon on T1. Threshold low (12% of the 99.5th percentile) to
    catch scalp and dark grey matter alike, open away the salt-and-pepper noise
    floor, keep the largest connected component so a stray bright artefact in the
    corner cannot drag the centroid, then fill interior holes.

    Only used to locate the head; it never touches the pixels that get saved.
    """
    hi = np.percentile(axial, 99.5)
    m = (axial > _HEAD_MASK_FRAC * max(hi, 1e-6)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n_lab <= 1:
        return m.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == biggest).astype(np.uint8)
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    return (m | (1 - flood)).astype(bool)


def _iso_resample_maps(slice_mm: float, col_mm: float, centre_lr_mm: float,
                       centre_ap_mm: float, mm_per_px: float, fov_mm: float):
    """cv2.remap coordinate maps for one subject's fixed-physical-FOV crop.

    The maps depend only on the subject's spacings and head centre, so they are
    built once and reused for all 32 slices.

    Index/millimetre convention: source voxel k covers [k*s, (k+1)*s) and its
    centre sits at (k+0.5)*s, so a physical position p maps to index p/s - 0.5.
    Output pixel i covers a mm_per_px-wide cell whose centre is
    fov_origin + (i+0.5)*mm_per_px. Getting this half-pixel wrong would shift
    every subject by up to half a voxel in a spacing-dependent direction, which
    is precisely the kind of protocol-correlated offset being removed.
    """
    out_px = int(round(fov_mm / mm_per_px))
    lr0 = centre_lr_mm - fov_mm / 2.0
    ap0 = centre_ap_mm - fov_mm / 2.0
    src_rows = (lr0 + (np.arange(out_px) + 0.5) * mm_per_px) / slice_mm - 0.5
    src_cols = (ap0 + (np.arange(out_px) + 0.5) * mm_per_px) / col_mm - 0.5
    map_y = np.repeat(src_rows.astype(np.float32)[:, None], out_px, axis=1)
    map_x = np.tile(src_cols.astype(np.float32), (out_px, 1))
    return map_x, map_y, out_px


def extract_slices_isotropic(dicom_dir: str, n_slices: int = N_SLICES_PER_SUBJECT,
                             bottleneck: int | None = BOTTLENECK_SIZE,
                             mm_per_px: float = ISO_MM_PER_PX,
                             fov_mm: float = FOV_MM,
                             return_qc: bool = False):
    """v4 extraction: same slices as extract_slices(), physically correct geometry.

    Identical to extract_slices() in every respect except the rendering geometry:
    same 48-92mm vertex-anchored band, same 32 slices, same 0.5/99.5 percentile
    normalisation, same 144px resolution bottleneck, same NLM denoising with the
    same parameters. What changes is that each axial plane is resampled to
    isotropic `mm_per_px` voxels and cropped to a fixed `fov_mm` physical window
    centred on the head, instead of being stretched to a square.

    A note on the bottleneck, which is now doing its job properly for the first
    time: BOTTLENECK_SIZE exists to give every subject the same spatial-frequency
    ceiling (decision 5). Under v3 that ceiling was FOV/144 millimetres per pixel
    -- 1.33mm for a 192mm ADNI1 acquisition but 1.76mm for a 253mm GO/2 one, so
    the "harmonizer" itself varied with protocol. With a fixed physical FOV it is
    a constant 1.56mm/px for everybody.

    return_qc=True additionally returns a dict of measured geometry (spacings,
    head extent in mm, head centre, fraction of head clipped by the FOV) so the
    extraction can be audited without re-reading the DICOMs.
    """
    geo = _acquisition_geometry(dicom_dir)
    volume = _build_volume_from_files(geo["files"])  # (left_right, sup_inf, ant_post)
    n_rows = volume.shape[1]

    # --- band selection: byte-for-byte the same logic as extract_slices() ---
    spacing = geo["row_mm"]
    vertex = _find_vertex_row(volume)
    row_lo = vertex + int(round(AXIAL_MM_BELOW_VERTEX[0] / spacing))
    row_hi = vertex + int(round(AXIAL_MM_BELOW_VERTEX[1] / spacing))
    row_hi = min(row_hi, n_rows - 1)
    row_lo = min(row_lo, max(row_hi - n_slices, 0))
    row_idxs = np.linspace(row_lo, row_hi, n_slices).round().astype(int)

    # --- head centre, in millimetres, from the band itself ---
    # Per-slice centroids and then the MEDIAN: one slice whose mask caught a
    # shoulder or a wrap-around artefact cannot move the crop. Falling back to
    # the geometric centre of the acquisition only matters for a slice with no
    # tissue at all, which does not occur in this band.
    planes = [volume[:, r, :].astype(np.float32) for r in row_idxs]
    masks = [_head_mask_2d(p) for p in planes]
    c_lr, c_ap, ext_lr, ext_ap, clipped = [], [], [], [], []
    for m in masks:
        if m.sum() < 50:
            continue
        w_lr = m.sum(axis=1).astype(np.float64)
        w_ap = m.sum(axis=0).astype(np.float64)
        c_lr.append((np.average(np.arange(m.shape[0]), weights=w_lr) + 0.5) * geo["slice_mm"])
        c_ap.append((np.average(np.arange(m.shape[1]), weights=w_ap) + 0.5) * geo["col_mm"])
        i0 = np.flatnonzero(w_lr > 0)
        i1 = np.flatnonzero(w_ap > 0)
        ext_lr.append((i0[-1] - i0[0] + 1) * geo["slice_mm"])
        ext_ap.append((i1[-1] - i1[0] + 1) * geo["col_mm"])
    if c_lr:
        centre_lr_mm, centre_ap_mm = float(np.median(c_lr)), float(np.median(c_ap))
    else:
        centre_lr_mm = volume.shape[0] * geo["slice_mm"] / 2.0
        centre_ap_mm = volume.shape[2] * geo["col_mm"] / 2.0

    map_x, map_y, out_px = _iso_resample_maps(
        geo["slice_mm"], geo["col_mm"], centre_lr_mm, centre_ap_mm, mm_per_px, fov_mm)

    slices = []
    for plane in planes:
        # 1+2. resample to isotropic mm_per_px AND crop the fixed physical window in
        # one interpolation. INTER_LINEAR is right here: the in-plane rescale is
        # 0.94-1.05x and the through-plane one is 1.2x upsampling, so nothing is
        # being decimated and there is no aliasing to guard against. Outside the
        # acquisition the border is zero, i.e. background, which is what a scanner
        # would have recorded there anyway.
        patch = cv2.remap(plane, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        # From here on: identical to extract_slices().
        axial = _normalize_to_uint8(patch)
        if bottleneck is not None:
            axial = cv2.resize(axial, (bottleneck, bottleneck), interpolation=cv2.INTER_AREA)
            axial = cv2.resize(axial, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)
        elif out_px != OUT_SIZE:
            axial = cv2.resize(axial, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
        axial = cv2.fastNlMeansDenoising(axial, h=8, templateWindowSize=7, searchWindowSize=21)
        slices.append(axial)

    if not return_qc:
        return slices

    # How much head, if any, the fixed window cut off -- the one thing that could
    # reintroduce a size normalisation, so it is measured rather than assumed.
    lo_lr, hi_lr = centre_lr_mm - fov_mm / 2, centre_lr_mm + fov_mm / 2
    lo_ap, hi_ap = centre_ap_mm - fov_mm / 2, centre_ap_mm + fov_mm / 2
    for m in masks:
        tot = int(m.sum())
        if tot < 50:
            continue
        rr = (np.arange(m.shape[0]) + 0.5) * geo["slice_mm"]
        cc = (np.arange(m.shape[1]) + 0.5) * geo["col_mm"]
        inside = m & ((rr >= lo_lr) & (rr < hi_lr))[:, None] & \
            ((cc >= lo_ap) & (cc < hi_ap))[None, :]
        clipped.append(1.0 - inside.sum() / tot)

    qc = {
        "row_mm": geo["row_mm"], "col_mm": geo["col_mm"], "slice_mm": geo["slice_mm"],
        "rows": geo["rows"], "cols": geo["cols"], "n_dicom": geo["n_dicom"],
        "vertex_row": int(vertex), "row_lo": int(row_lo), "row_hi": int(row_hi),
        "centre_lr_mm": centre_lr_mm, "centre_ap_mm": centre_ap_mm,
        "head_lr_mm": float(np.max(ext_lr)) if ext_lr else float("nan"),
        "head_ap_mm": float(np.max(ext_ap)) if ext_ap else float("nan"),
        "clipped_frac_max": float(np.max(clipped)) if clipped else 0.0,
        "clipped_frac_mean": float(np.mean(clipped)) if clipped else 0.0,
        "mm_per_px": mm_per_px, "fov_mm": fov_mm, "out_px": out_px,
    }
    return slices, qc


def process_all(manifest: pd.DataFrame, processed_dir: Path, resume: bool = True) -> pd.DataFrame:
    """
    Run extract_slices() for every subject in the manifest, save each slice as a PNG
    under processed_dir/{split}/{class}/{subject_id}_{i:03d}.png, and return a new
    slice-level manifest (one row per saved image) for use by the PyTorch Dataset.

    resume=True skips any subject whose full set of N_SLICES_PER_SUBJECT files already
    exists on disk -- lets this be safely re-run after an interruption without redoing
    already-completed subjects.
    """
    from tqdm import tqdm

    slice_rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting slices"):
        out_dir = processed_dir / row["split"] / row["class"]
        out_dir.mkdir(parents=True, exist_ok=True)

        expected = [out_dir / f"{row['subject_id']}_{i:03d}.png" for i in range(N_SLICES_PER_SUBJECT)]
        if resume and all(p.exists() for p in expected):
            for fpath in expected:
                slice_rows.append({
                    "subject_id": row["subject_id"], "class": row["class"],
                    "split": row["split"], "filepath": str(fpath.resolve()),
                })
            continue

        slices = extract_slices(row["dicom_dir"])
        for i, img in enumerate(slices):
            fpath = out_dir / f"{row['subject_id']}_{i:03d}.png"
            cv2.imwrite(str(fpath), img)
            slice_rows.append({
                "subject_id": row["subject_id"],
                "class": row["class"],
                "split": row["split"],
                "filepath": str(fpath.resolve()),
            })

    return pd.DataFrame(slice_rows)
