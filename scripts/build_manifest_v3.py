"""Phase 1: merge the AlzheimerAdditional download into data/raw/ and build the
combined subject manifest with an ERA column and an era-aware split.

Why an era column exists at all: in the pre-expansion dataset, diagnosis and ADNI
cohort were the same variable (decision 10) -- every CN/AD came from ADNI1, every
EMCI/LMCI from ADNI-GO/2 -- so "4-way accuracy" was mostly scanner detection. The
expansion adds CN and AD subjects from ADNI-GO/2, which makes a genuinely
era-matched 4-way task possible for the first time:

    ADNI-GO/2 only: CN, AD, EMCI and LMCI all present -> no cohort shortcut.

The split is stratified by class x era so that subset is itself correctly split
70/15/15 per class, and so the AD-vs-CN and cross-era experiments can all be cut
from this one manifest without re-splitting anything.

Usage:
    python scripts/build_manifest_v3.py --merge     # move raw_new/ into raw/, then build
    python scripts/build_manifest_v3.py             # build from whatever is in raw/
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data_prep  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
RAW = PROJ / "data" / "raw"
STAGE = PROJ / "data" / "raw_new"
OUT = PROJ / "data" / "subject_manifest_v3.csv"

# The date folder sits at .../{subject}/{description}/{YYYY-MM-DD_hh_mm_ss.0}/{imageuid}
DATE_IN_PATH = re.compile(r"[\\/](\d{4})-\d{2}-\d{2}_")

# ADNI1 ran ~2005-2009; ADNI-GO/2 from 2010. The boundary is a clean gap in this
# dataset (scans are 2006-2007 or 2010-2013), so no subject sits near it.
ERA_SPLIT_YEAR = 2010


def era_of(dicom_dir: str) -> str:
    m = DATE_IN_PATH.search(str(dicom_dir))
    if not m:
        return "UNKNOWN"
    return "ADNI1" if int(m.group(1)) < ERA_SPLIT_YEAR else "GO2"


def merge_staging() -> int:
    """Move each staged subject directory into data/raw/{CLASS}/ADNI/.

    Reversible: the moved subject IDs are exactly the ones listed in
    data/raw_new/_extracted_subjects.txt, and none of them existed in data/raw/
    beforehand (verified: zero overlap with subject_manifest.csv).
    """
    if not STAGE.exists():
        print(f"nothing to merge, {STAGE} does not exist")
        return 0
    moved = 0
    for cls_dir in sorted(STAGE.iterdir()):
        if not cls_dir.is_dir():
            continue
        src_root = cls_dir / "ADNI"
        if not src_root.exists():
            continue
        dst_root = RAW / cls_dir.name / "ADNI"
        dst_root.mkdir(parents=True, exist_ok=True)
        for subj_dir in sorted(src_root.iterdir()):
            if not subj_dir.is_dir():
                continue
            dst = dst_root / subj_dir.name
            if dst.exists():
                print(f"  SKIP (already present): {cls_dir.name}/{subj_dir.name}")
                continue
            shutil.move(str(subj_dir), str(dst))
            moved += 1
    print(f"merged {moved} subject directories into {RAW}")
    return moved


def split_by_class_and_era(manifest: pd.DataFrame) -> pd.DataFrame:
    """70/15/15 subject-wise split, stratified on class AND era.

    Stratifying on the pair keeps every split era-balanced within each class, which
    is what makes the ADNI-GO/2-only 4-way subset a valid dataset on its own.
    """
    manifest = manifest.copy().reset_index(drop=True)
    strata = manifest["class"] + "|" + manifest["era"]

    # Strata with <3 members cannot be split three ways; park them in train.
    counts = strata.value_counts()
    small = set(counts[counts < 3].index)
    if small:
        print(f"  note: strata too small to split, assigned to train: {sorted(small)}")
    splittable = manifest[~strata.isin(small)]
    strat_s = strata[~strata.isin(small)]

    tv, test = train_test_split(
        splittable, test_size=0.15, stratify=strat_s,
        random_state=data_prep.RANDOM_SEED,
    )
    train, val = train_test_split(
        tv, test_size=0.15 / 0.85, stratify=strat_s.loc[tv.index],
        random_state=data_prep.RANDOM_SEED,
    )
    manifest["split"] = "train"
    manifest.loc[val.index, "split"] = "val"
    manifest.loc[test.index, "split"] = "test"
    return manifest


def report(m: pd.DataFrame) -> None:
    print("\n=== subjects by class x era ===")
    print(pd.crosstab(m["class"], m["era"], margins=True))

    print("\n=== subjects by class x split ===")
    print(pd.crosstab(m["class"], m["split"], margins=True))

    print("\n=== PRIMARY TASK: 4-way within ADNI-GO/2 (era-matched, no cohort shortcut) ===")
    go2 = m[m["era"] == "GO2"]
    print(pd.crosstab(go2["class"], go2["split"], margins=True))
    if len(go2):
        base = go2["class"].value_counts().iloc[0] / len(go2)
        print(f"  majority-class baseline: {base:.1%}   chance: 25.0%")

    print("\n=== SECONDARY: AD vs CN across both eras (era carries no label info) ===")
    adcn = m[m["class"].isin(["AD", "CN"])]
    print(pd.crosstab(adcn["class"], adcn["era"], margins=True))
    if len(adcn):
        best_era = sum(
            adcn[adcn["era"] == e]["class"].value_counts().iloc[0]
            for e in adcn["era"].unique()
        ) / len(adcn)
        maj = adcn["class"].value_counts().iloc[0] / len(adcn)
        print(f"  majority baseline {maj:.1%} | best possible from ERA ALONE {best_era:.1%} "
              f"(gain {best_era - maj:+.1%})")

    print("\n=== slices per subject available (raw DICOM count) ===")
    print(m.groupby("era")["n_slices_raw"].describe()[["count", "min", "50%", "max"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true",
                    help="move data/raw_new/ into data/raw/ before building")
    args = ap.parse_args()

    if args.merge:
        merge_staging()

    print("\nscanning data/raw/ ...")
    m = data_prep.build_manifest(RAW)
    print(f"  {len(m)} subjects with a usable session "
          f"(>= {data_prep.MIN_SLICES_PER_SESSION} slices)")

    m["era"] = m["dicom_dir"].map(era_of)
    unknown = (m["era"] == "UNKNOWN").sum()
    if unknown:
        print(f"  WARNING: {unknown} subjects with unparseable scan date")

    m = split_by_class_and_era(m)
    report(m)

    m.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(m)} subjects)")


if __name__ == "__main__":
    main()
