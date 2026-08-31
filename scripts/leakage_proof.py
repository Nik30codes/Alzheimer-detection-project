"""
Proves what leakage does, using one single model and two test sets.

The question this answers: "if the model is leaky, can it still predict?"
Yes -- on people it has already seen. Not on anyone new. That distinction is invisible
in a normal leaky experiment because every subject is leaked, so there is nobody new
left to test on.

Design:
  * SEEN subjects   - 80% of subjects. Their 32 slices are split across train and test,
                      so the model trains on some slices of each of these people and is
                      tested on OTHER slices of the SAME people. This is the leak.
  * UNSEEN subjects - the remaining 20%. Held out entirely; not one slice is trained on.
                      These stand in for new patients arriving at a hospital.

One model is trained, then evaluated on both. Same weights, same preprocessing, same
day. The only difference is whether the test person was in the training set. The gap
between the two accuracies IS the leakage, quantified.

Usage: python leakage_proof.py [mobilenetv2|custom_cnn|efficientnet_b0]
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import (CLASSES, MRIDataset, compute_class_weights,      # noqa: E402
                      TRAIN_TRANSFORM, EVAL_TRANSFORM,
                      TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB)
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0  # noqa: E402
from train import train_model                                          # noqa: E402
from evaluate import get_predictions, subject_level_soft_vote          # noqa: E402
from sklearn.metrics import accuracy_score, classification_report      # noqa: E402

BATCH, LR, WD, EPOCHS, PATIENCE, SEED = 32, 1e-3, 1e-4, 30, 8, 42
UNSEEN_FRACTION = 0.20


def build(arch):
    if arch == "custom_cnn":
        return SimpleCNN(4, in_channels=1), False
    if arch == "mobilenetv2":
        return build_mobilenetv2(4, pretrained=False), True
    return build_efficientnet_b0(4, pretrained=False), True


def main(arch="mobilenetv2"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)

    m = pd.read_csv(os.path.join(ROOT, "data", "manifest.csv"))

    # ---- carve subjects into SEEN and UNSEEN, stratified by class ----
    unseen = []
    for cls in CLASSES:
        subs = np.array(sorted(m[m["class"] == cls]["subject_id"].unique()))
        rng.shuffle(subs)
        unseen.extend(subs[:max(1, int(round(UNSEEN_FRACTION * len(subs))))])
    unseen = set(unseen)
    seen = set(m["subject_id"].unique()) - unseen

    is_unseen = m["subject_id"].isin(unseen)

    # SEEN subjects: shuffle their slices into train/test -> this is the leak
    seen_rows = m[~is_unseen].copy()
    idx = rng.permutation(len(seen_rows))
    n_train = int(0.80 * len(seen_rows))
    split = np.empty(len(seen_rows), dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:]] = "test"
    seen_rows["split"] = split

    # UNSEEN subjects: every slice is test, none is trained on
    unseen_rows = m[is_unseen].copy()
    unseen_rows["split"] = "test"

    train_df = seen_rows[seen_rows["split"] == "train"].copy()
    val_df = train_df.sample(frac=0.12, random_state=SEED)      # small val for early stop
    train_df = train_df.drop(val_df.index)
    val_df["split"] = "val"

    print(f"===== LEAKAGE PROOF ({arch}) =====")
    print(f"SEEN subjects   : {len(seen)}  (slices split across train and test -> LEAKED)")
    print(f"UNSEEN subjects : {len(unseen)} (never trained on -> stands in for new patients)")
    print(f"train slices {len(train_df)}, val {len(val_df)}, "
          f"test-seen {int((seen_rows.split=='test').sum())}, test-unseen {len(unseen_rows)}")
    overlap = set(train_df["subject_id"]) & set(seen_rows[seen_rows.split == "test"]["subject_id"])
    print(f"subjects in BOTH train and test-seen: {len(overlap)} <- the leak")
    print(f"subjects in BOTH train and test-unseen: "
          f"{len(set(train_df['subject_id']) & set(unseen_rows['subject_id']))} <- must be 0\n")

    model, rgb = build(arch)
    train_t, eval_t = ((TRAIN_TRANSFORM_RGB, EVAL_TRANSFORM_RGB) if rgb
                       else (TRAIN_TRANSFORM, EVAL_TRANSFORM))

    combined = pd.concat([train_df, val_df], ignore_index=True)
    train_loader = DataLoader(MRIDataset(combined, "train", train_t), batch_size=BATCH,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(MRIDataset(combined, "val", eval_t), batch_size=BATCH,
                            shuffle=False, num_workers=2)

    ckpt = os.path.join(ROOT, "models", "checkpoints", f"{arch}_leakproof.pt")
    t0 = time.time()
    train_model(model, train_loader, val_loader, compute_class_weights(combined), device,
                epochs=EPOCHS, lr=LR, patience=PATIENCE, checkpoint_path=ckpt, weight_decay=WD)
    mins = (time.time() - t0) / 60

    # ---- the two test sets ----
    results = {}
    for name, df in [("SEEN (leaked)", seen_rows[seen_rows["split"] == "test"]),
                     ("UNSEEN (new patients)", unseen_rows)]:
        d = df.copy()
        d["split"] = "test"
        loader = DataLoader(MRIDataset(d, "test", eval_t), batch_size=BATCH,
                            shuffle=False, num_workers=2)
        preds = get_predictions(model, loader, device)
        slice_acc = accuracy_score(preds["true"], preds["pred"])
        _, subj = subject_level_soft_vote(preds, verbose=False)
        subj_acc = accuracy_score(subj["true"], subj["pred"])
        results[name] = {"slice_accuracy": slice_acc, "subject_accuracy": subj_acc,
                         "n_subjects": int(preds["subject_id"].nunique())}
        print(f"\n----- {name} -----")
        print(classification_report(preds["true"], preds["pred"], labels=CLASSES, zero_division=0))
        print(f"slice {slice_acc:.1%} | subject {subj_acc:.1%} "
              f"on {preds['subject_id'].nunique()} subjects")

    a = results["SEEN (leaked)"]["slice_accuracy"]
    b = results["UNSEEN (new patients)"]["slice_accuracy"]
    print("\n\n=================== THE ANSWER ===================")
    print(f"Same model, same weights, evaluated twice:")
    print(f"  on people it trained on (leaked)  : {a:.1%}")
    print(f"  on people it has never seen       : {b:.1%}")
    print(f"  -> leakage is worth {(a-b)*100:+.1f} accuracy points of pure illusion")
    print("""
So a leaky model absolutely still predicts -- it just only works on people already in
its training set. A real patient has never been in the training set, so the honest
number is the one on the right. This is why the project splits by subject.
""")

    out = {"arch": arch, "train_minutes": round(mins, 1),
           "n_seen_subjects": len(seen), "n_unseen_subjects": len(unseen),
           "results": results, "leakage_inflation_points": (a - b) * 100}
    with open(os.path.join(ROOT, "reports", f"{arch}_leakage_proof.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.join(ROOT, "reports", f"{arch}_leakage_proof.json"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mobilenetv2")
