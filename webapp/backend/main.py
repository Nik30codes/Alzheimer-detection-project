"""MultiModel Alzheimer Detection — FastAPI backend.

Implements docs/SRS_webapp.md. Serves a JSON API to the React frontend, and in
production also serves the built static files so the whole thing is one container.

Key behaviours:
  * Only tasks marked enabled in tasks.py are reachable (FR-13) -- an unvalidated
    model cannot be served even by a crafted request.
  * Uploads are read into memory and never written to disk or logged (NFR-1).
  * Progress is streamed as newline-delimited JSON, one event per stage as it actually
    completes (FR-9) -- not a fixed animation.
  * Displayed metrics come from the recorded result JSONs (FR-7).

Run:  uvicorn backend.main:app --reload --port 8000     (from webapp/)
"""
import json
import os
import re
import sys
import time
from typing import List

import numpy as np

# uvicorn imports this as `backend.main`, which does not put this directory on the
# path, so the sibling modules below would not resolve without this.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from inference import (InvalidInput, decode_image, gradcam_overlay, harmonise,
                       load_task_model, predict, to_data_uri, validate_slice, DEVICE)
from tasks import CLASS_LABELS, enabled_tasks, get_task, load_metrics

BRAND = "MultiModel Alzheimer Detection"
MAX_BYTES = 8 * 1024 * 1024              # per file (NFR-3)
MAX_TOTAL_BYTES = 48 * 1024 * 1024       # whole request; 32 slice PNGs are ~1.5 MB
MAX_FILES = 64                           # a subject contributes 32 slices

DISCLAIMER = (
    "Research demonstration only. This is NOT a medical device and must not be used "
    "for diagnosis. It was trained on research-grade ADNI scans; performance on images "
    "from other scanners or clinical settings is lower and unverified."
)
DICOM_NOTE = (
    "DICOM (.dcm) is supported, and its acquisition plane is read from the file header. "
    "ADNI's T1 scans are acquired SAGITTALLY (side-on) while this model reads AXIAL "
    "slices (top-down), so a raw ADNI DICOM is rejected with an explanation — that is "
    "correct behaviour, not a bug. PNG and JPEG carry no header, so orientation cannot "
    "be verified: please upload an axial slice."
)
SINGLE_SLICE_NOTE = (
    "The accuracy figures are subject-level: each subject contributes 32 slices whose "
    "probabilities are averaged into one prediction. Uploading a single slice reproduces "
    "only part of that — a one-slice answer is measurably noisier and less accurate than "
    "the headline number. Select all of a subject's slices at once to reproduce the "
    "published figure faithfully."
)
MULTI_SLICE_NOTE = (
    "Select multiple slices (or a whole folder) and each one is run through the model "
    "separately, then the per-slice probabilities are averaged into one subject-level "
    "prediction — exactly the aggregation used to compute the reported accuracy and ROC "
    "AUC. The decision threshold is applied once, to the averaged probability, never per "
    "slice."
)

app = FastAPI(title=BRAND, version="1.0",
              description="Alzheimer's classification from structural MRI. "
                          "Research demonstration; not a medical device.")

# The React dev server runs on a different port during development, and a deployed
# frontend hosted as a separate static site is a different origin entirely -- add its
# URL via the CORS_ORIGINS env var (comma-separated) rather than hardcoding it here.
_EXTRA_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                                  *_EXTRA_ORIGINS],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/config")
def config():
    """Everything the UI needs to render itself: tasks, metrics, and the notices."""
    return {
        "brand": BRAND,
        "disclaimer": DISCLAIMER,
        "dicom_note": DICOM_NOTE,
        "single_slice_note": SINGLE_SLICE_NOTE,
        "multi_slice_note": MULTI_SLICE_NOTE,
        "max_files": MAX_FILES,
        "device": str(DEVICE),
        "tasks": [{"id": tid, **{k: v for k, v in t.items() if k != "checkpoint"},
                   "metrics": load_metrics(t)}
                  for tid, t in enabled_tasks().items()],
    }


@app.get("/api/health")
def health():
    return {"ok": True, "brand": BRAND, "tasks": list(enabled_tasks()),
            "device": str(DEVICE)}



SUBJECT_RE = re.compile(r"^(\d{3}_S_\d{4})_\d+\.(png|jpg|jpeg)$", re.I)


def subject_ids_in(filenames):
    """Distinct ADNI subject ids parsable from the uploaded filenames.

    Slices are named like `002_S_5178_016.png`, so a mixed upload is detectable without
    opening a single file. This matters because the whole point of a multi-slice upload
    is that the per-slice probabilities get AVERAGED INTO ONE PERSON'S prediction --
    average across two people and the number is meaningless. A real user hit exactly
    this: they selected the first 45 files from a folder holding 1,376 slices belonging
    to 43 different subjects.

    Returns (ids, n_unparsed). An empty set means the names carry no subject id, in
    which case nothing can be inferred and no warning is shown.
    """
    ids, unparsed = set(), 0
    for fn in filenames:
        m = SUBJECT_RE.match(os.path.basename(fn or ""))
        if m:
            ids.add(m.group(1))
        else:
            unparsed += 1
    return ids, unparsed


def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _analyse(uploads, task_id):
    """Yield one NDJSON event per completed stage (FR-9).

    `uploads` is a list of (bytes, filename). One entry reproduces the original
    single-slice behaviour; 32 entries reproduce the published metric.

    WHY THIS AGGREGATES: the reported accuracy and ROC AUC are SUBJECT-level. In
    `scripts/train_binary_adni1.py` each subject's 32 axial slices are scored
    independently and their softmax probabilities are averaged
    (`groupby("subject_id")["p_AD"].mean()`) into one number, and the validation-chosen
    decision threshold is applied to THAT average. Scoring a single slice is a different,
    harder task -- it runs several points below the headline figure -- so a demo that only
    accepted one slice systematically under-performed its own published number. This
    function reproduces the training-time aggregation exactly: mean of per-slice softmax,
    threshold applied once at the end.
    """
    def ev(**kw):
        return json.dumps(kw) + "\n"

    n_in = len(uploads)
    multi = n_in > 1
    try:
        t0 = time.time()

        # ---- stage 1: decode -------------------------------------------------
        yield ev(stage=1, pct=4,
                 label=f"Reading {_plural(n_in, 'file')}" if multi else "Reading file")
        decoded, skipped = [], []       # decoded: (img, kind, filename)
        first_reason = None
        for i, (data, filename) in enumerate(uploads, 1):
            try:
                img, kind = decode_image(data, filename)
                decoded.append((img, kind, filename))
            except InvalidInput as e:
                if first_reason is None:
                    first_reason = str(e)
                skipped.append({"file": filename, "reason": str(e)})
            if multi:
                yield ev(stage=1, pct=4 + int(12 * i / n_in),
                         label=f"Reading file {i} of {n_in}")
        if not decoded:
            # Every file failed to decode -- there is nothing to fall back on, so this is
            # a hard error exactly as it was before multi-file support existed.
            raise InvalidInput(first_reason or "No readable image in the upload.")
        shape = decoded[0][0].shape
        yield ev(stage=1, pct=18, done_stage=True,
                 label=(f"Decoded {_plural(len(decoded), 'slice')} "
                        f"({shape[1]}x{shape[0]})" if multi else
                        f"Decoded {decoded[0][1]} ({shape[1]}x{shape[0]})"))

        # ---- stage 2: validate ----------------------------------------------
        yield ev(stage=2, pct=24,
                 label=("Validating axial brain slices" if multi else
                        "Validating axial brain slice"))
        valid = []
        for i, (img, kind, filename) in enumerate(decoded, 1):
            try:
                validate_slice(img)
                valid.append((img, kind, filename))
            except InvalidInput as e:
                if first_reason is None:
                    first_reason = str(e)
                skipped.append({"file": filename, "reason": str(e)})
            if multi:
                yield ev(stage=2, pct=24 + int(14 * i / len(decoded)),
                         label=f"Validating slice {i} of {len(decoded)}")
        if not valid:
            # ALL invalid -> error, same as the single-slice behaviour. Reporting a
            # prediction here would mean predicting from nothing.
            raise InvalidInput(first_reason or "No usable axial slice in the upload.")
        n_used, n_skipped = len(valid), len(skipped)
        yield ev(stage=2, pct=40, done_stage=True,
                 label=(f"{n_used} of {n_in} slices look like axial MRI"
                        f"{f' ({n_skipped} skipped)' if n_skipped else ''}" if multi
                        else "Input looks like an axial slice"))

        # ---- stage 3: harmonise ---------------------------------------------
        yield ev(stage=3, pct=46, label="Harmonising resolution (144px to 224px)")
        images = [harmonise(img) for img, _, _ in valid]
        kinds = sorted({k for _, k, _ in valid})
        yield ev(stage=3, pct=52, done_stage=True,
                 label="Resolution matched to training pipeline")

        # ---- stage 4: inference, one slice at a time ------------------------
        task = get_task(task_id)
        yield ev(stage=4, pct=56,
                 label=f"Running {task['arch']} + mirrored test-time augmentation")
        per_slice = []
        state = None
        for i, img in enumerate(images, 1):
            probs, state, task, _ = predict(img, task_id)
            per_slice.append(probs)
            # Stage 4 owns 56 -> 80% of the bar, so progress moves once per real forward
            # pass rather than on a timer.
            yield ev(stage=4, pct=56 + int(24 * i / n_used),
                     label=(f"Running model on slice {i} of {n_used}" if multi
                            else f"Running {task['arch']} + mirrored test-time augmentation"))
        stack = np.stack(per_slice)          # (n_slices, n_classes)
        probs = stack.mean(axis=0)           # <-- the subject-level soft vote
        yield ev(stage=4, pct=80, done_stage=True,
                 label=(f"Averaged probabilities over {_plural(n_used, 'slice')}" if multi
                        else "Inference complete"))

        classes = task["classes"]
        order = probs.argsort()[::-1]
        metrics = load_metrics(task)

        # The decision. For a task with a positive class the threshold comes from the
        # recorded result JSON and is applied to the AVERAGED probability -- applying it
        # per slice and voting would be a different (and unvalidated) estimator.
        pos_idx = (classes.index(task["positive_class"])
                   if task["positive_class"] in classes else None)
        # A single cut-point only defines a decision for a two-class head; a four-class
        # task falls back to argmax.
        binary = pos_idx is not None and len(classes) == 2
        thr = metrics.get("threshold") if binary else None
        if binary and thr is not None:
            pred_idx = pos_idx if float(probs[pos_idx]) >= thr else 1 - pos_idx
        else:
            pred_idx = int(order[0])

        # ---- stage 5: Grad-CAM on the most representative slice -------------
        # Averaging destroys the link between the answer and any one image, so showing the
        # attention map of an arbitrary slice would be misleading. The slice whose own
        # probability vector is closest to the average is the one that best stands in for
        # the aggregate.
        yield ev(stage=5, pct=86,
                 label=("Computing Grad-CAM on the most representative slice" if multi
                        else "Computing Grad-CAM attention map"))
        rep = int(np.abs(stack - probs).sum(axis=1).argmin())
        rep_img = images[rep]
        _, state, task, x_rep = predict(rep_img, task_id)
        overlay = gradcam_overlay(state, x_rep, rep_img, pred_idx)
        yield ev(stage=5, pct=95, done_stage=True, label="Attention map ready")

        # Detect a mixed-subject upload before reporting anything. This is a hard
        # correctness issue rather than a nicety: averaging slices across two people
        # produces a confident number that means nothing.
        _sids, _unparsed = subject_ids_in([fn for _, fn in uploads])
        result = {
            "prediction": {"code": classes[pred_idx],
                           "label": CLASS_LABELS.get(classes[pred_idx]),
                           "prob": round(float(probs[pred_idx]), 4)},
            "ranked": [{"code": classes[i],
                        "label": CLASS_LABELS.get(classes[i], classes[i]),
                        "prob": round(float(probs[i]), 4)} for i in order],
            "input_image": to_data_uri(rep_img), "gradcam": overlay,
            "task": task["label"], "status": task["status"], "metrics": metrics,
            "elapsed": round(time.time() - t0, 2),
            "source_kind": "+".join(kinds),
            "n_slices_used": n_used,
            "subject_ids": sorted(_sids),
            "mixed_subjects": len(_sids) > 1,
            "mixed_subjects_note": (
                f"These {n_used} slices come from {len(_sids)} DIFFERENT people "
                f"({', '.join(sorted(_sids)[:4])}"
                f"{'...' if len(_sids) > 4 else ''}). The probabilities have been "
                "averaged together, which only makes sense for ONE person -- the "
                "reported accuracy is a per-person figure. Select a single subject's "
                "slices instead: they share a filename prefix."
                if len(_sids) > 1 else None),
            "n_slices_skipped": n_skipped,
            "n_files_received": n_in,
            "subject_level": multi,
            "aggregation": ("Mean of per-slice softmax probabilities — the same "
                            "subject-level soft vote used to compute the reported "
                            "metrics." if multi else
                            "Single slice: no aggregation. The reported metrics average "
                            "32 slices per subject, so this answer is noisier."),
            "representative_slice": rep + 1,
            "representative_note": (
                f"Grad-CAM is computed on slice {rep + 1} of {n_used} — the slice whose "
                "own probability sits closest to the averaged probability, so it is the "
                "single image that best represents the aggregate decision."
                if multi else
                "Grad-CAM is computed on the slice you uploaded."),
        }
        if n_skipped:
            result["skipped"] = skipped[:12]
            result["skipped_note"] = (
                f"{_plural(n_skipped, 'file')} could not be used and "
                f"{'was' if n_skipped == 1 else 'were'} skipped; the prediction is the "
                f"average over the {n_used} that passed validation.")
        # Per-slice spread is worth surfacing: it shows how much the individual slices
        # disagreed, which is the whole reason aggregation helps.
        if pos_idx is not None:
            col = stack[:, pos_idx]
            result["positive_prob"] = round(float(probs[pos_idx]), 4)
            result["positive_class"] = task["positive_class"]
            result["slice_probs"] = [round(float(v), 4) for v in col]
            result["slice_prob_range"] = [round(float(col.min()), 4),
                                          round(float(col.max()), 4)]
            if binary and thr is not None:
                result["threshold"] = thr
                result["slice_votes"] = {
                    task["positive_class"]: int((col >= thr).sum()),
                    classes[1 - pos_idx]: int((col < thr).sum()),
                }
                result["threshold_note"] = (
                    "The decision threshold was chosen on held-out validation data and is "
                    "applied to the averaged probability, not to individual slices. "
                    "Testing across scanner generations showed the ranking transfers to a "
                    "new scanner but this cut-point does not — so the probability is more "
                    "meaningful than the yes/no call.")
                # Rare but real: with a threshold below 0.5 the call can differ from the
                # largest probability. Say so rather than letting the bars contradict the
                # verdict silently.
                if pred_idx != int(order[0]):
                    result["threshold_flip"] = True
        yield ev(stage=6, pct=100, done_stage=True, label="Complete", result=result)

    except InvalidInput as e:
        yield ev(error=str(e))
    except (KeyError, FileNotFoundError):
        yield ev(error="Model unavailable on the server.")
    except Exception as e:  # noqa: BLE001
        yield ev(error=f"Unexpected error: {type(e).__name__}")


@app.post("/api/predict-stream")
async def predict_stream(image: List[UploadFile] = File(...),
                         task: str = Form("ad_vs_cn")):
    """Accepts one OR many files under the `image` field.

    `List[UploadFile]` also matches a single `image` part, so existing single-slice
    clients keep working with no change.
    """
    if get_task(task) is None:
        raise HTTPException(400, "Unknown or disabled task.")
    if not image:
        raise HTTPException(400, "Empty upload.")
    if len(image) > MAX_FILES:
        raise HTTPException(413, f"Too many files (maximum {MAX_FILES}).")

    uploads, total = [], 0
    for f in image:
        data = await f.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename or 'A file'} is too large "
                                     "(maximum 8 MB per file).")
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise HTTPException(413, "Upload too large in total (maximum 48 MB).")
        if data:
            uploads.append((data, f.filename or ""))
    if not uploads:
        raise HTTPException(400, "Empty upload.")

    return StreamingResponse(
        _analyse(uploads, task),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ---- static frontend (production) ------------------------------------------
# Vite builds to frontend/dist; in development the React dev server serves it
# instead and talks to this API over CORS.
_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_DIST, "assets")),
              name="assets")
    # Vite copies everything in frontend/public/ into dist/ root, which is where the
    # demo sample scans land. Without this mount they 404 in production while working
    # fine under the dev server, so it is worth an explicit mount rather than relying
    # on the SPA fallback.
    _SAMPLES = os.path.join(_DIST, "samples")
    if os.path.isdir(_SAMPLES):
        app.mount("/samples", StaticFiles(directory=_SAMPLES), name="samples")

    @app.get("/")
    def spa_root():
        return FileResponse(os.path.join(_DIST, "index.html"))
