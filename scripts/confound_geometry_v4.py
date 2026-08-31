"""Does the v4 geometry fix actually kill decision 27's acquisition-geometry confound?

The test from decision 27, re-run on v4. Train a RandomForest on ACQUISITION
GEOMETRY ONLY -- no pixels, no image, nothing but numbers describing how the
picture was made -- and see how far above the majority baseline it scores. If
geometry carries label information, that classifier beats the baseline, and
anything a pixel model learns is suspect by the same amount.

Three feature sets, run over the same subjects, the same folds and the same
model, so the numbers are directly comparable:

  v3_render   what v3 ACTUALLY renders: the source matrix and spacings, the
              anisotropy slice_mm/row_mm, the physical field of view in each
              direction, and the millimetres-per-pixel that the squash-to-square
              produces. This reproduces the "before" measurement.

  v4_render   what v4 ACTUALLY renders. Every one of those quantities is now a
              constant by construction -- 1.0 mm/px in both directions, a 224mm
              window, 224 output pixels, aspect 1.0, anisotropy 1.0 -- so the
              classifier has nothing to key on. That is the point, but it is also
              close to a tautology, which is why the third set exists.

  v4_residual v4's constants PLUS the source spacings, kept as the resampling
              factors they have been demoted to. These no longer change the
              geometry of the output, but they do change how much interpolation
              blur each subject receives, so they are the honest upper bound on
              what protocol could still leave in a v4 image. If THIS set also
              lands at baseline, the confound is gone rather than hidden.

Reads data/geometry_v4_qc.csv (written by scripts/reextract_v4.py) so no second
pass over 140k DICOM headers is needed. CPU only; trains no image model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
QC = ROOT / "data" / "geometry_v4_qc.csv"
OUT_SIZE = 224
N_SLICES = 32
SEED = 0


def feature_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(df)
    lr_px = df["n_dicom"].astype(float)      # sagittal files = left-right pixels
    ap_px = df["cols"].astype(float)         # image columns  = anterior-posterior pixels
    si_px = df["rows"].astype(float)         # image rows     = superior-inferior pixels
    row_mm, col_mm, slice_mm = df["row_mm"], df["col_mm"], df["slice_mm"]

    fov_lr = lr_px * slice_mm
    fov_ap = ap_px * col_mm

    v3 = pd.DataFrame({
        "rows": si_px, "cols": ap_px, "n_dicom": lr_px,
        "row_mm": row_mm, "col_mm": col_mm, "slice_mm": slice_mm,
        "n_slices": np.full(n, N_SLICES, float),
        # the plane's pixel aspect ratio before it is squashed into a square
        "aspect_px": lr_px / ap_px,
        # decision 27's anisotropy factor
        "aniso": slice_mm / row_mm,
        "fov_lr_mm": fov_lr, "fov_ap_mm": fov_ap,
        # what one rendered pixel ends up meaning, in millimetres, per direction
        "mmpx_lr": fov_lr / OUT_SIZE, "mmpx_ap": fov_ap / OUT_SIZE,
        "mmpx_ratio": (fov_lr / OUT_SIZE) / (fov_ap / OUT_SIZE),
    })

    v4 = pd.DataFrame({
        "rows": np.full(n, OUT_SIZE, float), "cols": np.full(n, OUT_SIZE, float),
        "n_dicom": np.full(n, OUT_SIZE, float),
        "row_mm": np.full(n, 1.0), "col_mm": np.full(n, 1.0),
        "slice_mm": np.full(n, 1.0),
        "n_slices": np.full(n, N_SLICES, float),
        "aspect_px": np.ones(n), "aniso": np.ones(n),
        "fov_lr_mm": np.full(n, 224.0), "fov_ap_mm": np.full(n, 224.0),
        "mmpx_lr": np.ones(n), "mmpx_ap": np.ones(n), "mmpx_ratio": np.ones(n),
    })

    # Residual set 1 -- the defensible upper bound. Only quantities that STILL
    # change a v4 pixel: the resampling factors (how much interpolation blur the
    # subject received) and how much head the fixed window clipped. Note
    # resample_lr is constant, because the through-plane spacing is 1.2mm in
    # every series; the in-plane one is the only survivor.
    v4rp = v4.copy()
    v4rp["resample_lr"] = slice_mm.values / 1.0   # >1 = upsampled
    v4rp["resample_ap"] = col_mm.values / 1.0
    if "clipped_frac_max" in df:
        v4rp["clipped_frac_max"] = df["clipped_frac_max"].fillna(0.0).values

    # Residual set 2 -- deliberately unfair. Adds the source matrix sizes, which
    # under a fixed physical FOV cannot change the output at all. Included to
    # show what the number looks like when the classifier is handed protocol
    # identity outright; it is an over-statement, not a measurement of v4.
    v4rm = v4rp.copy()
    v4rm["src_row_mm"] = row_mm.values
    v4rm["src_col_mm"] = col_mm.values
    v4rm["src_slice_mm"] = slice_mm.values
    v4rm["src_matrix_ap"] = ap_px.values
    v4rm["src_n_dicom"] = lr_px.values

    return {"v3_render": v3, "v4_render": v4,
            "v4_res_pixels": v4rp, "v4_res_meta": v4rm}


def score(X: pd.DataFrame, y: np.ndarray) -> dict:
    """5-fold subject-level accuracy of a RandomForest on geometry alone."""
    counts = pd.Series(y).value_counts()
    baseline = float(counts.iloc[0] / len(y))
    n_var = int((X.nunique() > 1).sum())
    if n_var == 0:
        # No feature varies at all: the forest can only predict the majority class.
        return {"acc": baseline, "baseline": baseline, "gain": 0.0,
                "n": int(len(y)), "n_varying_features": 0, "degenerate": True}
    clf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(clf, X.values, y, cv=cv)
    acc = float((pred == y).mean())
    return {"acc": acc, "baseline": baseline, "gain": acc - baseline,
            "n": int(len(y)), "n_varying_features": n_var, "degenerate": False}


def sharpness_check(df: pd.DataFrame) -> dict:
    """Does the one surviving protocol variable actually imprint on v4's pixels?

    v4_residual is an UPPER BOUND, not a measurement: it asks what a classifier
    could do if it could read the source spacing perfectly off the image. In v4
    the only spacing that still varies is the in-plane one (0.94-1.06mm; the
    through-plane 1.2mm is identical in all 853 series), and it survives only as
    a slightly different amount of interpolation blur. So measure that directly
    -- variance of the Laplacian on the rendered PNGs -- and see how strongly it
    still tracks the source spacing and the class label.

    Descriptive only: correlations and group means, no model, no fitting.
    """
    from scipy.stats import spearmanr

    sm = pd.read_csv(ROOT / "data" / "subject_manifest_v3.csv")[["subject_id", "split"]]
    d = df.merge(sm, on="subject_id", how="left")

    out = {}
    for version in ("v3", "v4"):
        vals = []
        for _, r in d.iterrows():
            p = (ROOT / "data" / f"processed_{version}" / r["split"] / r["class"] /
                 f"{r['subject_id']}_016.png")
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
            vals.append(float(cv2.Laplacian(img, cv2.CV_64F).var()) if img is not None
                        else np.nan)
        d[f"sharp_{version}"] = vals
        ok = d[f"sharp_{version}"].notna()
        rho = spearmanr(d.loc[ok, "col_mm"], d.loc[ok, f"sharp_{version}"]).statistic
        by_cls = d.loc[ok].groupby("class")[f"sharp_{version}"].mean()
        out[version] = {
            "n": int(ok.sum()),
            "spearman_sharpness_vs_source_col_mm": float(rho),
            "class_means": {k: float(v) for k, v in by_cls.items()},
            "between_class_spread_over_mean": float(
                (by_cls.max() - by_cls.min()) / by_cls.mean()),
        }

    print("\npixel-level check -- image sharpness (variance of Laplacian, slice 16)")
    print(f"{'':<10}{'rho vs source col_mm':>22}{'between-class spread':>22}")
    for version in ("v3", "v4"):
        o = out[version]
        print(f"{version:<10}{o['spearman_sharpness_vs_source_col_mm']:>22.3f}"
              f"{o['between_class_spread_over_mean']*100:>21.1f}%")
    print("class-mean sharpness:")
    print(pd.DataFrame({v: out[v]["class_means"] for v in ("v3", "v4")}).round(1).to_string())
    return out


def main() -> None:
    if not QC.exists():
        sys.exit(f"missing {QC} - run scripts/reextract_v4.py first")
    df = pd.read_csv(QC)
    print(f"{len(df)} subjects from {QC.name}\n")

    go2 = df[df["era"] == "GO2"].reset_index(drop=True)
    adcn = df[df["class"].isin(["AD", "CN"])].reset_index(drop=True)

    # np.asarray(..., str): the CSV may load class as an arrow-backed extension
    # array, which sklearn's fold indexing cannot slice.
    tasks = {
        "4way_go2": (go2, np.asarray(go2["class"], dtype="<U6")),
        "adcn_vs_mci_go2": (go2, np.where(go2["class"].isin(["AD", "CN"]),
                                          "ADCN", "MCI")),
        "ad_vs_cn": (adcn, np.asarray(adcn["class"], dtype="<U6")),
    }

    # Decision 27's anisotropy table, and what v4 replaces it with.
    print("anisotropy (slice_mm / row_mm) as v3 renders it:")
    aniso = df.assign(aniso=df["slice_mm"] / df["row_mm"])
    print(aniso.groupby("class")["aniso"].agg(["mean", "std"]).round(3).to_string())
    print("\nsame quantity as v4 renders it: 1.000 +/- 0.000 for every class "
          "(isotropic by construction)\n")

    results = {}
    header = f"{'task':<18}{'features':<14}{'n':>5}{'acc':>9}{'baseline':>10}{'gain':>9}"
    print(header)
    print("-" * len(header))
    for tname, (sub, y) in tasks.items():
        fs = feature_sets(sub)
        results[tname] = {}
        for fname, X in fs.items():
            r = score(X, y)
            results[tname][fname] = r
            flag = "  (all features constant)" if r["degenerate"] else ""
            print(f"{tname:<18}{fname:<14}{r['n']:>5}{r['acc']*100:>8.1f}%"
                  f"{r['baseline']*100:>9.1f}%{r['gain']*100:>+8.1f}%{flag}")
        print()

    results["pixel_sharpness"] = sharpness_check(df)

    out = ROOT / "reports" / "geometry_confound_v4.json"
    out.write_text(json.dumps({
        "note": "metadata-only (no pixels) RandomForest, 5-fold, subject level",
        "reference_decision_27": {
            "4way_go2": {"acc": 0.409, "baseline": 0.367},
            "adcn_vs_mci_go2": {"acc": 0.717, "baseline": 0.570},
            "ad_vs_cn": {"acc": 0.523, "baseline": 0.569},
        },
        "results": results,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
