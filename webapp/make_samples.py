"""Copy held-out test slices into the frontend as built-in demo samples.

Why this exists: the page's whole interaction is gated behind an MRI upload, which a
visitor almost never has. One-click samples make the demo usable by anyone.

Samples are drawn from the TEST split only, so they are images the model never trained
on -- a sample it happens to get right is therefore a fair demonstration rather than a
memorised answer.

Two kinds of sample are exported:

  * SINGLE slices (`cn_1.png`, ...) — one mid-band slice, which is the harder, noisier
    task and runs several points below the published accuracy.
  * FULL SUBJECTS (`subject_cn/`, `subject_ad/`) — all 32 axial slices for one subject.
    This is the input the reported metrics were actually computed on: the 32 per-slice
    probabilities get averaged into one subject-level prediction. Without these, the demo
    could not reproduce its own headline number.

Subject choice is deliberately NOT the most confidently-classified subject. Among the
test subjects the recorded run predicted correctly, this picks the one with the MEDIAN
probability, i.e. a typical correct case rather than a cherry-picked easy one. If the
recorded predictions are missing it falls back to the first subject alphabetically and
makes no claim about whether the model gets it right.

Usage: python webapp/make_samples.py
"""
import json
import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "webapp", "frontend", "public", "samples")
PREDS = os.path.join(ROOT, "reports", "mobilenetv2_ADvsCN_v3adcn_subject_preds.csv")
N_PER_CLASS = 2
SLICE_INDEX = 16          # mid-band: ventricles / basal ganglia level


def _typical_correct_subject(cls, candidates):
    """A representative correctly-classified test subject, or None.

    Median-of-correct rather than best-of-correct: the point of the demo is to show what
    the model typically does, and picking the single most confident subject would flatter
    it. Returns None when the recorded predictions are unavailable, so the caller can
    fall back without pretending to know the outcome.
    """
    if not os.path.exists(PREDS):
        return None
    d = pd.read_csv(PREDS)
    d = d[(d["true"] == cls) & (d["pred"] == cls) & (d["subject_id"].isin(candidates))]
    if d.empty:
        return None
    d = d.sort_values("p_AD").reset_index(drop=True)
    return str(d.loc[len(d) // 2, "subject_id"])


def export_subject(test, cls, sid):
    """Write every axial slice for one subject into its own folder."""
    rows = test[test["subject_id"] == sid].sort_values("filepath")
    folder = f"subject_{cls.lower()}"
    dest = os.path.join(OUT, folder)
    shutil.rmtree(dest, ignore_errors=True)      # stale slices would silently linger
    os.makedirs(dest, exist_ok=True)

    files = []
    for i, src in enumerate(rows["filepath"]):
        name = f"slice_{i:02d}.png"
        shutil.copyfile(src, os.path.join(dest, name))
        files.append(f"/samples/{folder}/{name}")
    print(f"  {folder}/  <- {sid}  ({len(files)} slices)")
    return files


def _score_slices(test):
    """Run the mid-band slice of every test subject through the deployed model.

    Uses the webapp's own inference path so the numbers match exactly what a visitor
    clicking the button will see -- scoring with a slightly different transform would
    reintroduce the mismatch this function exists to prevent.
    """
    import sys
    sys.path.insert(0, os.path.join(ROOT, "webapp", "backend"))
    import cv2
    import inference as inf

    thr = json.load(open(os.path.join(
        ROOT, "reports", "mobilenetv2_ADvsCN_v3adcn_result.json")))["decision_threshold"]
    out = []
    for sid, grp in test.groupby("subject_id"):
        rows = grp.sort_values("filepath")
        if len(rows) <= SLICE_INDEX:
            continue
        src = rows.iloc[SLICE_INDEX]["filepath"]
        img = inf.harmonise(cv2.imread(src, cv2.IMREAD_GRAYSCALE))
        probs, _, task, _ = inf.predict(img, "ad_vs_cn")
        p_ad = float(probs[task["classes"].index("AD")])
        out.append({"subject_id": sid, "class": rows["class"].iloc[0], "src": src,
                    "p_ad": p_ad, "pred": "AD" if p_ad >= thr else "CN"})
    return pd.DataFrame(out)


def _typical_correct_slices(scored, cls, n):
    """Median-confidence correctly-classified slices for one class."""
    ok = scored[(scored["class"] == cls) & (scored["pred"] == cls)]
    if ok.empty:
        print(f"  !! no correctly-classified {cls} slice; falling back to first by id")
        fb = scored[scored["class"] == cls].sort_values("subject_id").head(n)
        return [(r.subject_id, r.src, r.p_ad) for r in fb.itertuples()]
    ok = ok.assign(dist=(ok["p_ad"] - ok["p_ad"].median()).abs()).sort_values("dist")
    return [(r.subject_id, r.src, r.p_ad) for r in ok.head(n).itertuples()]


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(os.path.join(ROOT, "data", "manifest_v3_adcn.csv"))
    test = m[m["split"] == "test"]

    manifest = []

    # ---- single slices -----------------------------------------------------
    # Selected by SCORING THE ACTUAL SLICE, not alphabetically.
    #
    # The first version took the first two subjects per class by subject_id. That
    # shipped `ad_1.png`, a slice the model calls CN (p_AD 0.376 against a 0.422
    # threshold) -- so half the AD demo buttons produced a wrong answer on click, which
    # reads as a broken model rather than as the known ~74% single-slice accuracy.
    #
    # A single slice is genuinely a ~74%-accurate task, so a wrong one is not a bug. But
    # a DEMO BUTTON should show the typical case, and the typical case is correct. This
    # picks, among slices the model gets right, the one with MEDIAN confidence -- not the
    # most confident, which would flatter it. If nothing scores correctly for a class it
    # falls back to the old behaviour and says so.
    scored = _score_slices(test)
    for cls in ("CN", "AD"):
        picks = _typical_correct_slices(scored, cls, N_PER_CLASS)
        for i, (sid, src, p_ad) in enumerate(picks, 1):
            name = f"{cls.lower()}_{i}.png"
            shutil.copyfile(src, os.path.join(OUT, name))
            manifest.append({
                "kind": "slice",
                "multi": False,
                "n_slices": 1,
                "file": f"/samples/{name}",
                "files": [f"/samples/{name}"],
                "true_class": cls,
                "label": f"Single slice — true diagnosis {cls}",
                "p_ad": round(float(p_ad), 3),
            })
            print(f"  {name}  <- {sid} {os.path.basename(src)}  p_AD={p_ad:.3f}")

    # ---- full 32-slice subjects -------------------------------------------
    for cls in ("CN", "AD"):
        candidates = sorted(test[test["class"] == cls]["subject_id"].unique())
        sid = _typical_correct_subject(cls, candidates)
        known = sid is not None
        if sid is None:
            sid = candidates[0]
        files = export_subject(test, cls, sid)
        if not files:
            continue
        manifest.append({
            "kind": "subject",
            "multi": True,
            "n_slices": len(files),
            "file": files[len(files) // 2],       # thumbnail: a mid-band slice
            "files": files,
            "true_class": cls,
            "label": f"Full subject — {len(files)} slices, true diagnosis {cls}",
            "note": ("A typical correctly-classified held-out subject (median "
                     "probability among the correct ones, not the most confident)."
                     if known else
                     "A held-out test subject. The recorded predictions were not "
                     "available, so no claim is made about the outcome."),
        })

    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    n_multi = sum(1 for s in manifest if s["multi"])
    print(f"\nwrote {len(manifest)} samples ({n_multi} full subjects) + index.json "
          f"to {OUT}")


if __name__ == "__main__":
    main()
