"""
Learned slice aggregation (attention-MIL) vs the plain mean, for AD vs CN.

WHY THIS EXISTS
---------------
Every subject-level number in this project comes from `groupby("subject_id")["p_AD"].mean()`
-- 32 axial slices are classified independently and their softmax probabilities are
averaged. The MIL literature is explicit that this throws information away: "classifying
each slice independently and then using majority voting eliminates the opportunity to
learn patterns and features that span across slices", and attention-based pooling
(Ilse et al., ICML 2018) is the standard fix.

This project has its own measured reason to suspect the mean: `docs/slice_informativeness.md`
found per-slice AUC ranging 0.777-0.919 across the 32 indices (decision 26). The slices
genuinely differ in usefulness, yet the mean weights all 32 equally. A learned head could
in principle up-weight the inferior slices (24-31, where AUC peaks at 0.919).

METHOD, and why it is cheap
---------------------------
The backbone is NOT retrained. `models/checkpoints/mobilenetv2_ADvsCN.pt` is run ONCE over
all 16,032 slices and the 1280-d pooled feature vector (the input to MobileNetV2's
classifier) plus the 2-d logits are cached to an .npz. Every aggregation head is then
fitted on those cached features on CPU in seconds. One short GPU burst, then nothing.

  mean               plain average of the 32 per-slice p_AD  (the incumbent)
  max                max-pool over p_AD  (the trivial MIL baseline)
  logistic_on_stats  logistic regression on order statistics of the 32 p_AD
                     (mean/std/min/max/quartiles) -- cheap, and often matches attention
  gated_attention    Ilse et al. 2018 gated attention MIL:
                       a_i = softmax(w^T (tanh(V h_i) * sigm(U h_i)))
                       z   = sum_i a_i h_i          (subject embedding)
                       y   = Linear(z)
                     hidden dim 128, on the 1280-d features

*** LEAKAGE CAVEAT -- READ BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT ***
The cached features come from a backbone that was TRAINED on the train split of this very
manifest (351 of the 501 subjects). A 5-fold CV over all 501 subjects therefore evaluates
heads on subjects the *backbone* has already seen. The absolute accuracies/AUCs printed
below are consequently OPTIMISTIC and are NOT estimates of deployment performance.
They are valid for ONE thing: the RELATIVE comparison between aggregation heads, because
every head consumes byte-identical features and is exposed to exactly the same
contamination. The honest deployment number remains the single-split one from
`reports/mobilenetv2_ADvsCN_v3adcn_result.json` (0.8267 / 0.9055 on 75 unseen subjects).

Usage:
  python scripts/slice_attention.py            # cache (if needed) + single split + 5-fold CV
  python scripts/slice_attention.py cache      # feature caching pass only
  python scripts/slice_attention.py --cpu      # force the caching pass onto CPU
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets import EVAL_TRANSFORM_RGB                      # noqa: E402
from models import build_mobilenetv2                         # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import roc_auc_score, confusion_matrix  # noqa: E402

BINARY = ["CN", "AD"]          # CN = 0, AD = 1, the checkpoint's class order
N_SLICES = 32
SEED = 42
BATCH = 32                     # hard constraint: another agent may be on the GPU
CKPT = os.path.join(ROOT, "models", "checkpoints", "mobilenetv2_ADvsCN.pt")
MANIFEST = os.path.join(ROOT, "data", "manifest_v3_adcn.csv")
# The cache is a 16,032 x 1280 float32 array (~82 MB). It is a derived artifact, fully
# regenerable from the checkpoint in one pass, so it lives outside the repo by default.
CACHE = os.path.join(
    os.environ.get("SLICE_ATTN_CACHE_DIR",
                   r"C:\Users\Nikunj\AppData\Local\Temp\claude\C--Windows-system32"
                   r"\bc5824cb-b812-405b-93e9-5653a619c071\scratchpad"),
    "slice_features_ADvsCN.npz")
HEADS = ["mean", "max", "logistic_on_stats", "gated_attention"]

# The published single-split numbers this must reproduce before anything else is believed.
REFERENCE = {"accuracy": 0.8266666666666667, "roc_auc": 0.9055232558139534}


# --------------------------------------------------------------------------- stats
def wilson_ci(correct, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc_ci(auc, n_pos, n_neg, z=1.96):
    """Hanley-McNeil 95% interval, the same estimator used elsewhere in this project."""
    if n_pos == 0 or n_neg == 0:
        return (float("nan"), float("nan"))
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return (max(0.0, auc - z * math.sqrt(max(var, 0.0))),
            min(1.0, auc + z * math.sqrt(max(var, 0.0))))


def youden_threshold(p, y):
    """Pick the decision threshold maximising sensitivity+specificity-1 on VALIDATION.

    Necessary here for the same reason as in train_binary_adni1.py: the heads emit
    differently-calibrated scores (a max-pool score is systematically higher than a mean),
    so comparing them all at a fixed 0.5 would measure calibration, not ranking.
    """
    best_thr, best_j = 0.5, -1.0
    for thr in np.unique(np.round(p, 4)):
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr


def paired_bootstrap_delta(y, p_a, p_b, n_boot=4000, seed=SEED):
    """95% CI on AUC(a) - AUC(b), resampling SUBJECTS and scoring both heads on the
    same resample. Paired, because the two heads see identical subjects -- an unpaired
    interval would be far wider and would hide a real difference (or invent one)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        deltas.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
    deltas = np.array(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ----------------------------------------------------------------- feature caching
class SliceDS(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import cv2
        row = self.df.iloc[i]
        img = cv2.imread(row["filepath"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["filepath"])
        return EVAL_TRANSFORM_RGB(img), i


def slice_index(path):
    """Slice files are named {subject}_{000..031}.png."""
    return int(str(path).rsplit("_", 1)[-1].split(".")[0])


def cache_features(force_cpu=False):
    """ONE pass over all 16,032 slices through the frozen backbone.

    MobileNetV2's forward is features -> global avg pool -> flatten -> classifier, so the
    1280-d vector taken here is exactly the classifier's input; the logits recomputed from
    it are bit-identical to model(x). Dropout is inert in eval mode, so nothing is random.
    """
    if os.path.exists(CACHE):
        print(f"cache already present: {CACHE}")
        return
    m = pd.read_csv(MANIFEST)
    m["slice"] = m["filepath"].map(slice_index)
    m = m.sort_values(["subject_id", "slice"]).reset_index(drop=True)

    device = torch.device("cpu")
    if not force_cpu and torch.cuda.is_available():
        free_used = torch.cuda.mem_get_info()
        used_mib = (free_used[1] - free_used[0]) / 1024 ** 2
        if used_mib > 4500:
            print(f"GPU already using {used_mib:.0f} MiB (>4500) -> caching on CPU instead")
        else:
            device = torch.device("cuda")
            print(f"GPU in use: {used_mib:.0f} MiB -> caching on GPU, batch {BATCH}")

    model = build_mobilenetv2(2, pretrained=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval().to(device)

    loader = torch.utils.data.DataLoader(SliceDS(m), batch_size=BATCH, shuffle=False,
                                         num_workers=4)
    feats = np.zeros((len(m), 1280), dtype=np.float32)
    logits = np.zeros((len(m), 2), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for imgs, idx in loader:
            h = model.features(imgs.to(device))
            h = torch.flatten(F.adaptive_avg_pool2d(h, (1, 1)), 1)   # (B, 1280)
            lg = model.classifier(h)                                 # (B, 2)
            feats[idx.numpy()] = h.cpu().numpy()
            logits[idx.numpy()] = lg.cpu().numpy()
    print(f"cached {len(m)} slices in {(time.time()-t0)/60:.1f} min on {device}")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    # np.array(list(...)) rather than .values.astype(str): under numpy 2.x a pandas
    # object-dtype column stays object through .astype(str), and an object array can only
    # be read back with allow_pickle=True. Forcing a real unicode dtype keeps the cache
    # loadable without pickle.
    def as_str(col):
        return np.array(m[col].tolist(), dtype="U32")

    np.savez_compressed(CACHE, features=feats, logits=logits,
                        subject_id=as_str("subject_id"),
                        slice=m["slice"].values.astype(np.int16),
                        label=(m["class"] == "AD").astype(np.int8).values,
                        split=as_str("split"),
                        era=as_str("era") if "era" in m else
                            np.array(["?"] * len(m), dtype="U32"))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"wrote {CACHE} ({os.path.getsize(CACHE)/1e6:.0f} MB)")


def load_bags():
    """Reshape the flat slice cache into per-subject bags: (n_subj, 32, 1280)."""
    z = np.load(CACHE, allow_pickle=False)
    sid, sl = z["subject_id"], z["slice"]
    order = np.lexsort((sl, sid))
    sid, sl = sid[order], sl[order]
    feats, logits, lab, spl = (z["features"][order], z["logits"][order],
                               z["label"][order], z["split"][order])
    subjects = sid[::N_SLICES]
    assert (sl.reshape(-1, N_SLICES) == np.arange(N_SLICES)).all(), "slice order broken"
    assert (sid.reshape(-1, N_SLICES) == subjects[:, None]).all(), "a bag mixes subjects"
    H = feats.reshape(-1, N_SLICES, 1280)
    L = logits.reshape(-1, N_SLICES, 2)
    P = torch.softmax(torch.from_numpy(L), dim=2).numpy()[:, :, 1]     # p_AD per slice
    y = lab.reshape(-1, N_SLICES)[:, 0].astype(int)
    split = spl.reshape(-1, N_SLICES)[:, 0]
    return subjects, H, P, y, split


# ------------------------------------------------------------------------ the heads
class GatedAttentionMIL(nn.Module):
    """Ilse et al. 2018, eq. 9. The gate (sigmoid branch) lets the head suppress a slice
    multiplicatively, which plain tanh attention cannot do because tanh is near-linear
    over its useful range."""

    def __init__(self, dim=1280, hidden=128, dropout=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.head = nn.Linear(dim, 1)

    def attention(self, H):                       # H: (B, S, D)
        Hd = self.drop(H)
        a = self.w(torch.tanh(self.V(Hd)) * torch.sigmoid(self.U(Hd)))   # (B, S, 1)
        return torch.softmax(a, dim=1)

    def forward(self, H):
        a = self.attention(H)
        z = (a * H).sum(dim=1)                    # (B, D) subject embedding
        return self.head(z).squeeze(-1), a.squeeze(-1)


def prob_stats(P):
    """Order statistics of a subject's 32 slice probabilities."""
    return np.column_stack([P.mean(1), P.std(1), P.min(1), P.max(1),
                            np.percentile(P, 25, axis=1),
                            np.percentile(P, 50, axis=1),
                            np.percentile(P, 75, axis=1)])


def fit_head(head, H, P, y, tr, va, seed=SEED, save_path=None):
    """Fit `head` on the `tr` subjects, select/tune on `va`.

    Returns score_fn(idx) -> subject scores, and an optional (n_subj, 32) attention map.
    Everything that touches data -- feature standardisation, hyperparameters, the number
    of epochs, the decision threshold -- is fitted on tr/va only.
    """
    if head == "mean":
        return (lambda idx: P[idx].mean(1)), None
    if head == "max":
        return (lambda idx: P[idx].max(1)), None

    if head == "logistic_on_stats":
        S = prob_stats(P)
        mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-8
        best, best_auc = None, -1.0
        for C in (0.01, 0.1, 1.0, 10.0):
            lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
            lr.fit((S[tr] - mu) / sd, y[tr])
            a = roc_auc_score(y[va], lr.predict_proba((S[va] - mu) / sd)[:, 1])
            if a > best_auc:
                best_auc, best = a, lr
        return (lambda idx: best.predict_proba((S[idx] - mu) / sd)[:, 1]), None

    if head == "gated_attention":
        torch.manual_seed(seed)
        # Standardise features on the TRAIN subjects only. MobileNetV2's post-ReLU
        # features have wildly different per-channel scales; without this the tanh/sigmoid
        # gates saturate immediately and every slice gets weight 1/32 (i.e. the mean).
        flat = H[tr].reshape(-1, 1280)
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        Hs = torch.from_numpy(((H - mu) / sd).astype(np.float32))
        Y = torch.from_numpy(y.astype(np.float32))

        model = GatedAttentionMIL()
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
        # class_weight: AD is the minority class in every fold
        pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        Htr, Ytr = Hs[tr], Y[tr]
        rng = np.random.default_rng(seed)
        best_auc, best_state, bad = -1.0, None, 0
        for ep in range(300):
            model.train()
            perm = rng.permutation(len(tr))
            for b in range(0, len(perm), 16):
                sel = perm[b:b + 16]
                opt.zero_grad()
                out, _ = model(Htr[sel])
                crit(out, Ytr[sel]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                out, _ = model(Hs[va])
            a = roc_auc_score(y[va], out.numpy())
            if a > best_auc + 1e-5:
                best_auc, bad = a, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 40:                      # early stop on val AUC
                    break
        model.load_state_dict(best_state)
        model.eval()
        if save_path:
            # The feature scaler is part of the head -- without it the saved weights are
            # unusable, since the head was fitted on standardised MobileNetV2 features.
            torch.save({"state_dict": best_state, "feat_mean": mu, "feat_std": sd,
                        "hidden": 128, "dim": 1280, "val_auc": best_auc,
                        "note": "gated attention MIL head over frozen "
                                "mobilenetv2_ADvsCN 1280-d pooled features"}, save_path)
        with torch.no_grad():
            out, att = model(Hs)
        p_all = torch.sigmoid(out).numpy()
        return (lambda idx: p_all[idx]), att.numpy()

    raise ValueError(head)


# --------------------------------------------------------------- evaluation harness
def score_split(y_te, p_te, thr):
    pred = (p_te >= thr).astype(int)
    n = len(y_te)
    correct = int((pred == y_te).sum())
    auc = roc_auc_score(y_te, p_te)
    lo, hi = wilson_ci(correct, n)
    alo, ahi = auc_ci(auc, int(y_te.sum()), int(n - y_te.sum()))
    return {"n": n, "correct": correct, "accuracy": correct / n, "accuracy_95CI": [lo, hi],
            "roc_auc": auc, "roc_auc_95CI": [alo, ahi], "threshold": thr}


def make_folds(y, subjects, k=5, seed=SEED):
    """Stratified k-fold over SUBJECTS, never over slices -- same construction as
    scripts/cross_validate.py (round-robin dealing of a shuffled per-class list)."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        ids = np.where(y == cls)[0]
        ids = ids[np.argsort(subjects[ids])]
        rng.shuffle(ids)
        for i, s in enumerate(ids):
            folds[i % k].append(s)
    return [np.array(sorted(f)) for f in folds]


def main(argv):
    force_cpu = "--cpu" in argv
    cache_features(force_cpu)
    if "cache" in argv:
        return

    subjects, H, P, y, split = load_bags()
    print(f"\n{len(subjects)} subjects x {N_SLICES} slices, features {H.shape}, "
          f"AD {int(y.sum())} / CN {int((y == 0).sum())}")

    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    te = np.where(split == "test")[0]

    # ---------------------------------------------------------------- sanity gate
    p_mean_va, p_mean_te = P[va].mean(1), P[te].mean(1)
    thr = youden_threshold(p_mean_va, y[va])
    ref = score_split(y[te], p_mean_te, thr)
    print("\n===== SANITY GATE: does the cached-feature path reproduce the headline? =====")
    print(f"  accuracy  {ref['accuracy']:.4f}   expected {REFERENCE['accuracy']:.4f}")
    print(f"  ROC AUC   {ref['roc_auc']:.4f}   expected {REFERENCE['roc_auc']:.4f}")
    print(f"  threshold {thr:.3f} (val-chosen; result JSON recorded 0.422)")
    ok = (abs(ref["accuracy"] - REFERENCE["accuracy"]) < 1e-4
          and abs(ref["roc_auc"] - REFERENCE["roc_auc"]) < 1e-4)
    if not ok:
        print("\n*** MISMATCH -- the cached features do not reproduce the published "
              "baseline. STOPPING; something in the inference path is wrong. ***")
        sys.exit(1)
    print("  exact match -> the cache is faithful, continue.")

    results = {"reference": REFERENCE, "single_split": {}, "cv": {}}

    # ------------------------------------------------- how much headroom is there AT ALL?
    # Before comparing heads, bound what any reweighting could possibly buy. The 32 slice
    # probabilities are not 32 independent opinions -- they come from one backbone reading
    # 32 overlapping views of one head. If they are strongly correlated, the mean is
    # already close to the best linear combination and there is nothing for attention to
    # win. The oracle below fits weights on ALL 501 subjects' true labels and is scored
    # in-sample: it CANNOT be beaten by any honest linear head, so it is a hard ceiling.
    C = np.corrcoef(P.T)
    iu = np.triu_indices(N_SLICES, 1)
    w_unif = np.ones(N_SLICES) / N_SLICES
    var_ratio = float(w_unif @ C @ w_unif)
    oracle = LogisticRegression(max_iter=5000, C=1e6).fit(P, y)
    auc_oracle = roc_auc_score(y, oracle.decision_function(P))
    auc_mean_all = roc_auc_score(y, P.mean(1))
    results["headroom"] = {
        "cross_slice_corr_mean": float(C[iu].mean()),
        "cross_slice_corr_min": float(C[iu].min()),
        "cross_slice_corr_max": float(C[iu].max()),
        "adjacent_slice_corr_mean": float(np.mean([C[i, i + 1] for i in range(N_SLICES - 1)])),
        "effective_independent_slices": 1.0 / var_ratio,
        "auc_mean_head_all501": float(auc_mean_all),
        "auc_oracle_insample_reweight": float(auc_oracle),
        "max_possible_gain": float(auc_oracle - auc_mean_all),
    }
    print("\n===== HEADROOM: what could ANY reweighting buy? =====")
    print(f"  mean pairwise correlation between slice probabilities : {C[iu].mean():.3f}"
          f"  (adjacent slices {np.mean([C[i, i+1] for i in range(N_SLICES-1)]):.3f})")
    print(f"  effective number of INDEPENDENT slices                 : "
          f"{1/var_ratio:.1f} of 32")
    print(f"  plain mean, all 501 subjects                           : AUC {auc_mean_all:.4f}")
    print(f"  ORACLE linear reweight (weights fitted on the true      : AUC {auc_oracle:.4f}")
    print(f"    labels of all 501 and scored in-sample -- cheating)")
    print(f"  => absolute ceiling for any linear slice reweighting    : "
          f"{auc_oracle - auc_mean_all:+.4f} AUC")

    # ------------------------------------------------- part 1: the existing single split
    print("\n===== SINGLE SPLIT (fit on 351 train, tune on 75 val, score 75 test) =====")
    print(f"{'head':<20}{'acc':>8}{'95% CI':>18}{'AUC':>8}{'95% CI':>18}")
    for head in HEADS:
        save = (os.path.join(ROOT, "models", "checkpoints", "agg_gated_attention.pt")
                if head == "gated_attention" else None)
        score_fn, _ = fit_head(head, H, P, y, tr, va, save_path=save)
        t = youden_threshold(score_fn(va), y[va])
        r = score_split(y[te], score_fn(te), t)
        results["single_split"][head] = r
        aci = "[{:.3f}, {:.3f}]".format(*r["accuracy_95CI"])
        rci = "[{:.3f}, {:.3f}]".format(*r["roc_auc_95CI"])
        print(f"{head:<20}{r['accuracy']:>8.3f}{aci:>18}{r['roc_auc']:>8.3f}{rci:>18}")

    # ------------------------------------------------------- part 2: 5-fold subject CV
    print("\n===== 5-FOLD SUBJECT-LEVEL CV OVER ALL 501 SUBJECTS =====")
    print("(interval computed on 501 subjects, not 75 -- but see the leakage caveat: the")
    print(" BACKBONE saw 351 of these during its own training, so absolute numbers are")
    print(" optimistic. Only the head-vs-head comparison is defensible.)")
    folds = make_folds(y, subjects, k=5)
    print(f"fold sizes {[len(f) for f in folds]}, "
          f"AD per fold {[int(y[f].sum()) for f in folds]}")
    for i, f in enumerate(folds):
        for j, g in enumerate(folds):
            assert i == j or not set(f) & set(g), "folds overlap"

    oof = {h: np.zeros(len(subjects)) for h in HEADS}
    oof_pred = {h: np.zeros(len(subjects), dtype=int) for h in HEADS}
    per_fold = {h: [] for h in HEADS}
    att_sum, att_n = np.zeros(N_SLICES), 0
    att_by_class = {0: np.zeros(N_SLICES), 1: np.zeros(N_SLICES)}
    att_cnt = {0: 0, 1: 0}

    for i, test_ids in enumerate(folds):
        rest = np.array(sorted(set(range(len(subjects))) - set(test_ids.tolist())))
        rng = np.random.default_rng(SEED + i)
        val_ids = []
        for cls in (0, 1):                       # stratified val carved from the rest
            v = rest[y[rest] == cls]
            v = v[np.argsort(subjects[v])].copy()
            rng.shuffle(v)
            val_ids.extend(v[:max(1, int(round(0.15 * len(v))))].tolist())
        val_ids = np.array(sorted(val_ids))
        train_ids = np.array(sorted(set(rest.tolist()) - set(val_ids.tolist())))
        assert not set(test_ids.tolist()) & set(val_ids.tolist())
        assert not set(test_ids.tolist()) & set(train_ids.tolist())

        line = [f"  fold {i+1}: train {len(train_ids)} val {len(val_ids)} test {len(test_ids)}"]
        for head in HEADS:
            score_fn, att = fit_head(head, H, P, y, train_ids, val_ids, seed=SEED + i)
            t = youden_threshold(score_fn(val_ids), y[val_ids])
            s = score_fn(test_ids)
            oof[head][test_ids] = s
            oof_pred[head][test_ids] = (s >= t).astype(int)
            fa = roc_auc_score(y[test_ids], s)
            per_fold[head].append({"fold": i + 1, "n": len(test_ids),
                                   "accuracy": float((oof_pred[head][test_ids] == y[test_ids]).mean()),
                                   "roc_auc": float(fa), "threshold": t})
            line.append(f"{head[:4]} {fa:.3f}")
            if head == "gated_attention" and att is not None:
                # attention weights of THIS fold's held-out subjects only
                a = att[test_ids]
                att_sum += a.sum(0); att_n += len(a)
                for cls in (0, 1):
                    sel = a[y[test_ids] == cls]
                    att_by_class[cls] += sel.sum(0); att_cnt[cls] += len(sel)
        print("  ".join(line), flush=True)

    print(f"\n{'head':<20}{'acc':>8}{'95% CI':>18}{'AUC':>8}{'95% CI':>18}"
          f"{'dAUC vs mean':>16}{'95% CI':>20}")
    for head in HEADS:
        n = len(y)
        correct = int((oof_pred[head] == y).sum())
        acc = correct / n
        alo_, ahi_ = wilson_ci(correct, n)
        auc = roc_auc_score(y, oof[head])
        blo, bhi = auc_ci(auc, int(y.sum()), int(n - y.sum()))
        d, dlo, dhi = ((0.0, 0.0, 0.0) if head == "mean" else
                       paired_bootstrap_delta(y, oof[head], oof["mean"]))
        results["cv"][head] = {
            "n": n, "correct": correct, "accuracy": acc, "accuracy_95CI": [alo_, ahi_],
            "roc_auc": auc, "roc_auc_95CI": [blo, bhi],
            "delta_auc_vs_mean": d, "delta_auc_95CI": [dlo, dhi],
            "per_fold": per_fold[head],
            "fold_auc_sd": float(np.std([f["roc_auc"] for f in per_fold[head]])),
        }
        print(f"{head:<20}{acc:>8.3f}{f'[{alo_:.3f}, {ahi_:.3f}]':>18}"
              f"{auc:>8.3f}{f'[{blo:.3f}, {bhi:.3f}]':>18}"
              f"{d:>+16.4f}{f'[{dlo:+.4f}, {dhi:+.4f}]':>20}")

    baseline = max((y == 0).mean(), y.mean())
    print(f"\nmajority baseline {baseline:.3f}")
    for head in HEADS:
        r = results["cv"][head]
        if head == "mean":
            continue
        verdict = ("BEATS the mean" if r["delta_auc_95CI"][0] > 0 else
                   "LOSES to the mean" if r["delta_auc_95CI"][1] < 0 else
                   "indistinguishable from the mean")
        print(f"  {head:<20} {verdict}  (paired bootstrap on dAUC)")

    # ----------------------------------------------------- part 3: attention weights
    att_mean = att_sum / max(att_n, 1)
    results["attention"] = {
        "mean_weight_per_slice": att_mean.tolist(),
        "mean_weight_CN": (att_by_class[0] / max(att_cnt[0], 1)).tolist(),
        "mean_weight_AD": (att_by_class[1] / max(att_cnt[1], 1)).tolist(),
        "uniform_weight": 1.0 / N_SLICES,
        "n_subjects": att_n,
    }
    per_slice_auc = np.array([roc_auc_score(y, P[:, s]) for s in range(N_SLICES)])
    results["attention"]["per_slice_auc_all501"] = per_slice_auc.tolist()
    rho = pd.Series(att_mean).corr(pd.Series(per_slice_auc), method="spearman")
    results["attention"]["spearman_attention_vs_slice_auc"] = float(rho)
    print(f"\nattention weights: min {att_mean.min():.4f}  max {att_mean.max():.4f}  "
          f"uniform would be {1/N_SLICES:.4f}  (ratio max/min {att_mean.max()/att_mean.min():.2f})")
    print(f"Spearman(attention weight, per-slice AUC) = {rho:+.3f}")

    plot_attention(att_mean, att_by_class, att_cnt, per_slice_auc, results)

    # ------------------------------------------- part 4: is the near-tie seed-stable?
    print("\n===== SEED ROBUSTNESS (gated_attention, folds fixed) =====")
    results["seed_robustness"] = seed_robustness(H, P, y, subjects, n_seeds=5)

    with open(os.path.join(ROOT, "reports", "slice_attention_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote reports/slice_attention_result.json")
    print("*** REMINDER: CV numbers are head-vs-head only. The backbone trained on 351 "
          "of these 501 subjects. ***")


def seed_robustness(H, P, y, subjects, n_seeds=5):
    """Re-run the whole 5-fold CV for gated_attention under n_seeds different
    initialisations (folds held fixed, so only the head's init/shuffling changes).

    Why this is not optional: the attention head's CV AUC came out +0.005 above the mean,
    inside its own interval. A single seed cannot distinguish "a real but tiny effect"
    from "this seed got lucky". If the seed-to-seed spread is comparable to the gap, the
    gap is not a finding.
    """
    folds = make_folds(y, subjects, k=5)
    rows = []
    for s in range(n_seeds):
        oof = np.zeros(len(subjects))
        for i, test_ids in enumerate(folds):
            rest = np.array(sorted(set(range(len(subjects))) - set(test_ids.tolist())))
            rng = np.random.default_rng(SEED + i)      # same val split every seed
            val_ids = []
            for cls in (0, 1):
                v = rest[y[rest] == cls]
                v = v[np.argsort(subjects[v])].copy()
                rng.shuffle(v)
                val_ids.extend(v[:max(1, int(round(0.15 * len(v))))].tolist())
            val_ids = np.array(sorted(val_ids))
            train_ids = np.array(sorted(set(rest.tolist()) - set(val_ids.tolist())))
            fn, _ = fit_head("gated_attention", H, P, y, train_ids, val_ids,
                             seed=1000 * (s + 1) + i)
            oof[test_ids] = fn(test_ids)
        auc = roc_auc_score(y, oof)
        rows.append(auc)
        print(f"  seed set {s+1}: gated_attention CV AUC {auc:.4f}", flush=True)
    mean_auc = roc_auc_score(y, np.array([P[j].mean() for j in range(len(y))]))
    print(f"\ngated_attention over {n_seeds} seeds: mean {np.mean(rows):.4f} "
          f"sd {np.std(rows):.4f}  range [{min(rows):.4f}, {max(rows):.4f}]")
    print(f"plain mean head (deterministic, no seed): {mean_auc:.4f}")
    return {"seed_aucs": rows, "mean": float(np.mean(rows)), "sd": float(np.std(rows)),
            "mean_head_auc": float(mean_auc)}


def plot_attention(att_mean, att_by_class, att_cnt, per_slice_auc, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(N_SLICES)

    ax = axes[0]
    ax.bar(x, att_mean, color="#4C72B0", label="learned attention weight")
    ax.axhline(1 / N_SLICES, color="#C44E52", ls="--", lw=1.6,
               label=f"uniform (the mean) = {1/N_SLICES:.4f}")
    ax.plot(x, att_by_class[1] / max(att_cnt[1], 1), color="#DD8452", lw=1.2,
            marker="o", ms=3, label="AD subjects")
    ax.plot(x, att_by_class[0] / max(att_cnt[0], 1), color="#55A868", lw=1.2,
            marker="s", ms=3, label="CN subjects")
    ax.set_xlabel("axial slice index (0 = 48 mm below vertex, 31 = 92 mm)")
    ax.set_ylabel("mean attention weight")
    ax.set_title("Gated-attention MIL: learned per-slice weights\n"
                 "(out-of-fold, 501 subjects)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(att_mean.max() * 1.35, 1.6 / N_SLICES))

    ax = axes[1]
    ax.scatter(per_slice_auc, att_mean, c=x, cmap="viridis", s=45)
    for s in (0, 11, 20, 31):
        ax.annotate(f"{s:03d}", (per_slice_auc[s], att_mean[s]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(1 / N_SLICES, color="#C44E52", ls="--", lw=1.2)
    rho = results["attention"]["spearman_attention_vs_slice_auc"]
    ax.set_xlabel("per-slice ROC AUC (that slice alone, 501 subjects)")
    ax.set_ylabel("mean attention weight")
    ax.set_title(f"Does attention find the informative slices?\nSpearman rho = {rho:+.3f}")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("slice index")

    fig.tight_layout()
    out = os.path.join(ROOT, "reports", "figures", "slice_attention_weights.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
