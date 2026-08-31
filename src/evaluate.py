"""
Evaluation utilities. Two levels of accuracy are reported everywhere here:

- slice-level: treats every one of the 32 images per subject as an independent
  prediction. Easy to compute, but not what a clinician would actually use.
- subject-level: majority vote across a subject's 32 slice predictions -> one
  prediction per person. This is the number that actually matters, since in
  practice you're diagnosing a person, not a single 2D image.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from datasets import CLASSES


PROB_COLS = [f"p_{c}" for c in CLASSES]


@torch.no_grad()
def get_predictions(model, loader, device) -> pd.DataFrame:
    """Runs the model over a DataLoader and returns one row per image:
    subject_id, true class, predicted class, the model's confidence, and the
    full softmax distribution in columns p_CN/p_AD/p_EMCI/p_LMCI.

    The full distribution is kept (not just the winning class's confidence)
    because soft voting and the multi-model ensemble both need to average
    probabilities, which is impossible to reconstruct from the argmax alone."""
    model.eval()
    rows = []
    for imgs, labels, subject_ids in loader:
        imgs = imgs.to(device)
        probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for i in range(len(labels)):
            row = {
                "subject_id": subject_ids[i],
                "true": CLASSES[labels[i]],
                "pred": CLASSES[preds[i]],
                "confidence": probs[i, preds[i]],
            }
            row.update(dict(zip(PROB_COLS, probs[i])))
            rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def get_predictions_tta(model, loader, device) -> pd.DataFrame:
    """
    Test-time augmentation: predict on each image AND its horizontal mirror, then
    average the two softmax distributions.

    Why a horizontal flip is the right (and only) augmentation to use here: an axial
    brain slice is close to left-right symmetric, so the mirrored image is still a
    plausible brain, and the training transform already included RandomHorizontalFlip --
    so the model has seen both orientations and neither is out of distribution. Rotations
    or crops would be riskier, since they can move the hippocampal region relative to the
    frame. Averaging two views cancels some of the per-slice prediction noise, which is
    worth more here than usual because individual slices sit near chance.

    Costs one extra forward pass per image and no retraining.
    """
    model.eval()
    rows = []
    for imgs, labels, subject_ids in loader:
        imgs = imgs.to(device)
        probs = torch.softmax(model(imgs), dim=1)
        probs = probs + torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
        probs = (probs / 2).cpu().numpy()
        preds = probs.argmax(axis=1)
        for i in range(len(labels)):
            row = {
                "subject_id": subject_ids[i],
                "true": CLASSES[labels[i]],
                "pred": CLASSES[preds[i]],
                "confidence": probs[i, preds[i]],
            }
            row.update(dict(zip(PROB_COLS, probs[i])))
            rows.append(row)
    return pd.DataFrame(rows)


def slice_level_report(preds_df: pd.DataFrame):
    print(classification_report(preds_df["true"], preds_df["pred"], labels=CLASSES))
    return confusion_matrix(preds_df["true"], preds_df["pred"], labels=CLASSES)


def subject_level_report(preds_df: pd.DataFrame):
    """Majority vote: each subject's final prediction is whichever class most
    of their 32 slices voted for."""
    majority = (
        preds_df.groupby("subject_id")
        .agg(true=("true", "first"), pred=("pred", lambda s: s.value_counts().idxmax()))
    )
    print(classification_report(majority["true"], majority["pred"], labels=CLASSES))
    return confusion_matrix(majority["true"], majority["pred"], labels=CLASSES), majority


def subject_level_soft_vote(preds_df: pd.DataFrame, verbose: bool = True):
    """
    Soft vote: average the full softmax distribution across a subject's 32 slices,
    then take the argmax of that average.

    Why this can beat the hard majority vote in `subject_level_report`: majority
    voting discards confidence. A subject whose 17 slices weakly prefer EMCI
    (0.30 each) and whose 15 slices strongly say AD (0.85 each) is called EMCI by
    majority vote, even though the averaged evidence clearly points to AD.
    Averaging probabilities keeps that magnitude information. It also degrades more
    gracefully when slices are individually near chance, which is the regime this
    dataset is actually in (slice-level accuracy is only ~54%).

    Returns (confusion_matrix, per-subject dataframe with true/pred + mean probs).
    """
    grouped = preds_df.groupby("subject_id")
    mean_probs = grouped[PROB_COLS].mean()
    subjects = pd.DataFrame({
        "true": grouped["true"].first(),
        "pred": [CLASSES[i] for i in mean_probs.values.argmax(axis=1)],
    }, index=mean_probs.index).join(mean_probs)

    if verbose:
        print(classification_report(subjects["true"], subjects["pred"], labels=CLASSES))
    return confusion_matrix(subjects["true"], subjects["pred"], labels=CLASSES), subjects


def ensemble_predictions(preds_dfs: list, weights: list = None) -> pd.DataFrame:
    """
    Combines several models' slice-level prediction frames into one by averaging
    their softmax distributions per (subject, slice-position) row.

    All frames must come from the same test loader in the same order (shuffle=False
    in build_dataloaders guarantees this), so row i refers to the same image in every
    frame. `weights` optionally weights the models; defaults to equal weighting.

    The point of averaging probabilities rather than voting on labels: the models fail
    on different subjects, and averaging lets a confidently-correct model outvote two
    unconfidently-wrong ones.
    """
    if weights is None:
        weights = [1.0] * len(preds_dfs)
    total = sum(weights)

    base = preds_dfs[0]
    for df in preds_dfs[1:]:
        if not base["subject_id"].equals(df["subject_id"]) or not base["true"].equals(df["true"]):
            raise ValueError("prediction frames are not row-aligned -- were they all built "
                             "from the same test loader with shuffle=False?")

    avg = sum(w * df[PROB_COLS].values for w, df in zip(weights, preds_dfs)) / total
    out = base[["subject_id", "true"]].copy()
    out[PROB_COLS] = avg
    out["pred"] = [CLASSES[i] for i in avg.argmax(axis=1)]
    out["confidence"] = avg.max(axis=1)
    return out
