"""Compute a confusion matrix + macro-F1 for the AD-vs-CN CV headline, for the webapp.

The served headline (reports/mobilenetv2_ADvsCN_cv_result.json: accuracy 0.741, ROC AUC
0.7845) has accuracy/AUC/CI but no confusion matrix or F1 -- those were never written out
for this specific run. reports/cvheadline_mobilenetv2_val_loss_oof.csv holds the exact
out-of-fold predictions (p_AD per subject) behind that headline, so this recomputes a
confusion matrix from the SAME predictions rather than retraining or approximating.

Writes reports/mobilenetv2_ADvsCN_cv_confusion.json (a new file -- never overwrites the
existing result JSON, same convention the rest of this project's scripts follow).

Sanity gate: recomputed accuracy must equal the recorded 0.741 to 3 decimals, exactly
like docs/slice_attention.md and docs/slice_informativeness.md check their own numbers
before trusting them.
"""
import json
import os

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")

OOF_CSV = os.path.join(REPORTS, "cvheadline_mobilenetv2_val_loss_oof.csv")
RESULT_JSON = os.path.join(REPORTS, "mobilenetv2_ADvsCN_cv_result.json")
OUT_JSON = os.path.join(REPORTS, "mobilenetv2_ADvsCN_cv_confusion.json")

CLASSES = ["CN", "AD"]


def main():
    with open(RESULT_JSON) as f:
        headline = json.load(f)
    threshold = headline["decision_threshold"]
    recorded_accuracy = headline["accuracy"]

    df = pd.read_csv(OOF_CSV)
    # Recompute pred from p_AD + the recorded threshold rather than trusting the CSV's
    # own `pred` column, in case it predates the threshold recalibration (decision 35).
    df["pred_recomputed"] = df["p_AD"].apply(lambda p: "AD" if p >= threshold else "CN")

    cm = confusion_matrix(df["true"], df["pred_recomputed"], labels=CLASSES)
    macro_f1 = f1_score(df["true"], df["pred_recomputed"], labels=CLASSES, average="macro")
    accuracy = (df["true"] == df["pred_recomputed"]).mean()

    print(f"n subjects: {len(df)}")
    print(f"threshold used: {threshold}")
    print(f"recomputed accuracy: {accuracy:.4f}  (recorded headline: {recorded_accuracy:.4f})")
    print(f"macro F1: {macro_f1:.4f}")
    print(f"confusion matrix {CLASSES}:\n{cm}")

    if round(accuracy, 3) != round(recorded_accuracy, 3):
        raise SystemExit(
            f"SANITY GATE FAILED: recomputed accuracy {accuracy:.4f} does not match "
            f"the recorded headline {recorded_accuracy:.4f} to 3 decimals. Not writing "
            f"output -- investigate the threshold/column before trusting this matrix."
        )

    out = {
        "source": "reports/cvheadline_mobilenetv2_val_loss_oof.csv",
        "note": (
            "Confusion matrix and macro-F1 recomputed from the exact out-of-fold "
            "predictions behind the 5-fold CV headline in "
            "mobilenetv2_ADvsCN_cv_result.json, using the same decision threshold. "
            "Not a separate run, not a separate model."
        ),
        "classes": CLASSES,
        "confusion_matrix": cm.tolist(),
        "macro_f1": round(float(macro_f1), 4),
        "accuracy_check": round(float(accuracy), 4),
        "decision_threshold": threshold,
        "n_subjects": len(df),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
