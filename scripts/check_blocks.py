"""
Tests whether the models are separating DISEASE or ACQUISITION ERA.

CN/AD were acquired in ADNI1 (~2005-07, native 192x192); EMCI/LMCI in ADNI-GO/2
(~2011+, native 256x256). If a model were reading atrophy, its confusions would follow
clinical severity -- above all AD<->LMCI, since late MCI is the stage that converts to
Alzheimer's. If instead it is reading scanner/protocol era, confusions will stay strictly
inside the {CN,AD} and {EMCI,LMCI} blocks and essentially never cross.

"cross-block error rate" below is the fraction of all errors that cross the era boundary.
Near 0% is the red flag.
"""

import os

import numpy as np
import pandas as pd

ROOT = r"C:\Users\Nikunj\Documents\alzheimer-mri-project"
CLASSES = ["CN", "AD", "EMCI", "LMCI"]
ERA = {"CN": "ADNI1", "AD": "ADNI1", "EMCI": "ADNI-GO/2", "LMCI": "ADNI-GO/2"}


def analyse(name, cm):
    cm = np.asarray(cm)
    total_err = cm.sum() - np.trace(cm)
    cross = 0
    for i, ti in enumerate(CLASSES):
        for j, pj in enumerate(CLASSES):
            if i != j and ERA[ti] != ERA[pj]:
                cross += cm[i, j]
    within = total_err - cross
    print(f"\n--- {name} ---")
    print(pd.DataFrame(cm, index=[f"true {c}" for c in CLASSES],
                       columns=[f"pred {c}" for c in CLASSES]))
    print(f"total errors {total_err} | within-era {within} | CROSS-era {cross} "
          f"({cross/total_err:.1%} of errors)" if total_err else "no errors")
    # the specific pair that should dominate if this were clinical
    ad_lmci = cm[CLASSES.index("AD"), CLASSES.index("LMCI")] + cm[CLASSES.index("LMCI"), CLASSES.index("AD")]
    ad_cn = cm[CLASSES.index("AD"), CLASSES.index("CN")] + cm[CLASSES.index("CN"), CLASSES.index("AD")]
    print(f"AD<->LMCI confusions (clinically the HARDEST pair): {ad_lmci}")
    print(f"AD<->CN   confusions (clinically the EASIEST pair): {ad_cn}")
    if ad_cn > ad_lmci:
        print("  ^ model confuses AD with HEALTHY more than with late MCI -- "
              "inconsistent with reading atrophy")
    return cross, total_err


found = False
for fname, label in [
    ("efficientnet_b0_scratch_subject_cm.npy", "EfficientNet-B0 (soft vote)"),
    ("custom_cnn_subject_cm.npy", "custom CNN (hard vote, from Phase C)"),
    ("custom_cnn_slice_cm.npy", "custom CNN (slice level, from Phase C)"),
]:
    path = os.path.join(ROOT, "reports", fname)
    if os.path.exists(path):
        analyse(label, np.load(path))
        found = True
    else:
        print(f"(missing {fname})")

if found:
    print("""
INTERPRETATION
If cross-era errors are ~0% across independent architectures, the models are almost
certainly keying on acquisition-protocol differences between the ADNI1 (CN/AD) and
ADNI-GO/2 (EMCI/LMCI) cohorts rather than on disease. The 4-way accuracy would then be
partly a scanner-era classifier wearing a diagnosis label -- a confound of the same
family as decision 5 in CLAUDE.md, but not fixed by the resolution harmonization.
""")
