"""
Decomposes the 4-way accuracy into the part that is acquisition-cohort separation and
the part that is actual diagnosis.

In this dataset class is perfectly confounded with ADNI phase:
    CN, AD      -> ADNI1     (~2005-07, native 192x192)
    EMCI, LMCI  -> ADNI-GO/2 (~2011+,   native 256x256)

So a model can score well on the 4-way task by first splitting the two cohorts (an easy
scanner/protocol cue that has nothing to do with disease) and only then guessing within
the cohort. This script measures those two components separately:

  era accuracy         - can it tell ADNI1 from ADNI-GO/2? (should be ~chance if it is
                         reading anatomy; near-perfect means a protocol cue survives)
  AD vs CN             - the clinically important comparison, restricted to ADNI1
  EMCI vs LMCI         - the other within-cohort comparison
"""

import os

import numpy as np
import pandas as pd

ROOT = r"C:\Users\Nikunj\Documents\alzheimer-mri-project"
REPORTS = os.path.join(ROOT, "reports")
ERA = {"CN": "ADNI1", "AD": "ADNI1", "EMCI": "ADNI-GO/2", "LMCI": "ADNI-GO/2"}
PROB_COLS = ["p_CN", "p_AD", "p_EMCI", "p_LMCI"]


def subject_table(preds):
    """Collapse slices to subjects, soft vote if probabilities are available."""
    if all(c in preds.columns for c in PROB_COLS):
        g = preds.groupby("subject_id")
        mean_p = g[PROB_COLS].mean()
        pred = [c[2:] for c in mean_p.columns[mean_p.values.argmax(axis=1)]]
        return pd.DataFrame({"true": g["true"].first().values, "pred": pred},
                            index=mean_p.index)
    g = preds.groupby("subject_id")
    return pd.DataFrame({
        "true": g["true"].first(),
        "pred": g["pred"].agg(lambda s: s.value_counts().idxmax()),
    })


def analyse(name, preds):
    s = subject_table(preds)
    s["true_era"] = s["true"].map(ERA)
    s["pred_era"] = s["pred"].map(ERA)

    era_acc = (s["true_era"] == s["pred_era"]).mean()
    overall = (s["true"] == s["pred"]).mean()

    print(f"\n================ {name} ================")
    print(f"4-way subject accuracy            : {overall:.1%}  (chance 25%)")
    print(f"acquisition-era accuracy (2-way)  : {era_acc:.1%}  (chance 50%)")

    for era, (a, b) in [("ADNI1", ("CN", "AD")), ("ADNI-GO/2", ("EMCI", "LMCI"))]:
        sub = s[s["true_era"] == era]
        # only count subjects the model kept inside the right era, then ask whether it
        # got the class right -- this isolates diagnosis from cohort separation
        inside = sub[sub["pred_era"] == era]
        if len(inside) == 0:
            continue
        acc = (inside["true"] == inside["pred"]).mean()
        n_a = (inside["true"] == a).sum()
        n_b = (inside["true"] == b).sum()
        majority = max(n_a, n_b) / len(inside)
        verdict = "NO real signal" if acc <= majority + 0.02 else "some signal"
        print(f"  {a} vs {b:<5} within {era:<10}: {acc:.1%} on {len(inside)} subjects "
              f"| always-guess-majority would score {majority:.1%} -> {verdict}")

    return {"model": name, "four_way": overall, "era": era_acc}


rows = []
for fname, label in [
    ("efficientnet_b0_scratch_test_preds.csv", "EfficientNet-B0 (from scratch)"),
    ("custom_cnn_test_preds.csv", "custom CNN (Phase C)"),
    ("mobilenetv2_honest2d_test_preds.csv", "MobileNetV2 (from scratch)"),
]:
    path = os.path.join(REPORTS, fname)
    if not os.path.exists(path):
        print(f"(not yet available: {fname})")
        continue
    rows.append(analyse(label, pd.read_csv(path)))

if rows:
    print("\n\n================ WHAT THE HEADLINE NUMBER IS MADE OF ================")
    df = pd.DataFrame(rows).set_index("model")
    print((df * 100).round(1))
    print("""
Read this way: the 4-way accuracy is largely the era split (easy, and not diagnosis)
combined with near-chance guessing inside each era. The clinically meaningful question --
AD vs CN -- is where the models have little to no real signal, which is exactly the
comparison a diagnostic tool would need to get right.

Root cause is the dataset, not the models: every CN and AD subject comes from ADNI1 and
every EMCI and LMCI subject from ADNI-GO/2, so "class" and "scanner era" are the same
variable. No architecture can separate them.

Fix, and it is worth doing during the planned LONI expansion: pull CN/AD subjects from
ADNI-GO/2 and EMCI/LMCI subjects from ADNI1 as well, so each class spans both eras. Then
protocol cues stop being predictive of the label and the accuracy that remains is real.
""")
