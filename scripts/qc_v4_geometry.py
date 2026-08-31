"""QC figure: the same subject, the same slice, rendered by v3 and by v4.

Six subjects spanning both scanner eras and all four classes, chosen to include
the most extreme acquisition geometries in the pool, so the aspect-ratio
correction is visible rather than argued. Top row is v3 (variable field of view
squashed into a square); bottom row is v4 (isotropic 1.0 mm/px, fixed 224mm
window centred on the head). Both rows read from the PNGs that were actually
written, not from a re-extraction, so what is shown is what will be trained on.

Writes reports/figures/qc_v4_geometry.png. Reads only; trains nothing.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import cv2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QC = ROOT / "data" / "geometry_v4_qc.csv"
SLICE_IDX = 16          # middle of the 48-92mm band
OUT = ROOT / "reports" / "figures" / "qc_v4_geometry.png"


def png_path(version, split, cls, sid):
    return ROOT / "data" / f"processed_{version}" / split / cls / f"{sid}_{SLICE_IDX:03d}.png"


def main() -> None:
    if not QC.exists():
        sys.exit(f"missing {QC} - run scripts/reextract_v4.py first")
    qc = pd.read_csv(QC)
    sm = pd.read_csv(ROOT / "data" / "subject_manifest_v3.csv")
    qc = qc.merge(sm[["subject_id", "split"]], on="subject_id", how="left")
    qc["aniso"] = qc["slice_mm"] / qc["row_mm"]

    # Six subjects: every (class, era) cell that exists, each represented by the
    # subject whose v3 anisotropy is furthest from 1.0 -- i.e. the ones v3
    # distorts most. EMCI/LMCI have no ADNI1 members, so this gives 2 + 4.
    picks = []
    for (cls, era), g in qc.groupby(["class", "era"]):
        g = g.dropna(subset=["aniso"])
        g = g.assign(dev=(g["aniso"] - 1.0).abs()).sort_values("dev", ascending=False)
        for _, r in g.iterrows():
            if png_path("v3", r["split"], cls, r["subject_id"]).exists() and \
               png_path("v4", r["split"], cls, r["subject_id"]).exists():
                picks.append(r)
                break
    order = {("AD", "ADNI1"): 0, ("CN", "ADNI1"): 1, ("AD", "GO2"): 2,
             ("CN", "GO2"): 3, ("EMCI", "GO2"): 4, ("LMCI", "GO2"): 5}
    picks.sort(key=lambda r: order.get((r["class"], r["era"]), 9))
    picks = picks[:6]
    if not picks:
        sys.exit("no subject has both a v3 and a v4 PNG on disk")

    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 8.2), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, r in enumerate(picks):
        sid, cls, era, split = r["subject_id"], r["class"], r["era"], r["split"]
        img3 = cv2.imread(str(png_path("v3", split, cls, sid)), cv2.IMREAD_GRAYSCALE)
        img4 = cv2.imread(str(png_path("v4", split, cls, sid)), cv2.IMREAD_GRAYSCALE)

        fov_lr = r["n_dicom"] * r["slice_mm"]
        fov_ap = r["cols"] * r["col_mm"]
        axes[0, j].imshow(img3, cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(f"{cls} · {era}\n{sid}", fontsize=9)
        axes[0, j].set_xlabel(
            f"{int(r['n_dicom'])}×{int(r['cols'])} px\n"
            f"{fov_lr/224:.2f} / {fov_ap/224:.2f} mm/px\n"
            f"aniso {r['aniso']:.3f}", fontsize=7.5)

        axes[1, j].imshow(img4, cmap="gray", vmin=0, vmax=255)
        head = ""
        if not pd.isna(r.get("head_lr_mm", np.nan)):
            head = f"\nhead {r['head_lr_mm']:.0f}×{r['head_ap_mm']:.0f} mm"
        axes[1, j].set_xlabel(f"224×224 px\n1.00 / 1.00 mm/px\naniso 1.000{head}",
                              fontsize=7.5)
        # 50mm scale bar: identical length in every v4 panel, because the
        # millimetre-to-pixel mapping is now a constant across subjects.
        axes[1, j].plot([12, 62], [212, 212], color="#ffcc33", lw=2.5)
        axes[1, j].text(37, 206, "50 mm", color="#ffcc33", fontsize=7,
                        ha="center", va="bottom")

        for i in (0, 1):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    axes[0, 0].set_ylabel("v3\nsquashed to square", fontsize=10)
    axes[1, 0].set_ylabel("v4\n1 mm/px, 224 mm FOV", fontsize=10)
    fig.suptitle(
        "v3 vs v4 acquisition geometry — same subject, same slice (#%d of the 48–92 mm band)\n"
        "v3 stretches each subject by slice_mm/row_mm before squashing a variable FOV into a "
        "square; v4 resamples to isotropic 1 mm/px and crops a fixed 224 mm window on the head "
        "centroid,\nso protocol geometry is constant while head size in millimetres is preserved."
        % SLICE_IDX, fontsize=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    for r in picks:
        print(f"  {r['class']:5s} {r['era']:6s} {r['subject_id']:12s} "
              f"aniso={r['aniso']:.3f} matrix={int(r['n_dicom'])}x{int(r['cols'])} "
              f"row_mm={r['row_mm']:.3f}")


if __name__ == "__main__":
    main()
