# SRS & Design — MultiModel Alzheimer Detection (web demo)

Version 1.0 — 17 August 2026
Status: approved for build

---

## 1. Purpose and scope

A public, single-page web demonstration of this project's ADNI MRI classification work.
A visitor uploads one axial brain MRI slice and receives a prediction, a Grad-CAM
overlay showing which region drove that prediction, and the model's honestly-reported
validation metrics.

**Primary audience: technical recruiters and reviewers.** This shapes the design more
than anything else. A recruiter's judgement will rest less on the accuracy number than
on whether the work shows methodological care, so the page must surface *how the model
was validated*, not only what it predicts.

**This is not a medical device** and makes no clinical claim. See §7.

## 2. Users

Anonymous public visitors. No accounts, no authentication, no roles, no sessions.

## 3. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Visitor uploads one image without logging in | Must |
| FR-2 | Accepts PNG and JPEG | Must |
| FR-3 | Accepts DICOM (`.dcm`), reading pixel data via pydicom | Must |
| FR-4 | Rejects inputs that are not plausibly an axial brain slice, giving a reason | Must |
| FR-5 | Returns per-class probabilities for the active task | Must |
| FR-6 | Returns a Grad-CAM overlay of the attended region | Must |
| FR-7 | Displays model metrics (ROC AUC, accuracy, CI, test-set size) read from recorded result files, never hardcoded | Must |
| FR-8 | Displays a prominent non-diagnostic disclaimer | Must |
| FR-9 | Streams live pipeline progress while analysing, naming each real stage | Must |
| FR-10 | Explains that ADNI DICOMs are sagittal and the model needs axial | Must |
| FR-11 | Presents a "how this was validated" section covering the confound and cross-era testing | Must |
| FR-12 | Additional tasks can be enabled without changing routes or templates | Must |
| FR-13 | Disabled tasks are unreachable, including via crafted requests | Must |

### FR-9 — pipeline stages (must reflect real work, not a timer)

1. Reading file / decoding DICOM
2. Validating axial brain slice
3. Harmonising resolution (144px bottleneck, then 224px)
4. Running MobileNetV2 + mirrored test-time augmentation
5. Computing Grad-CAM
6. Complete

## 4. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | No uploaded image is written to disk or logged, ever |
| NFR-2 | End-to-end response under ~5 s on CPU-only hosting |
| NFR-3 | Upload capped at 8 MB |
| NFR-4 | Runs CPU-only; no GPU at deploy time |
| NFR-5 | Responsive down to 360px width |
| NFR-6 | Self-contained deployment (Dockerfile + requirements) |

## 5. Out of scope

User accounts; storing or retrieving past results; 3D volume upload; batch processing;
patient records; any clinical or diagnostic claim; the four-stage model (see §6).

## 6. The four-stage model is deliberately disabled

The registry contains a `four_stage` entry with `enabled: False`.

Four-way classification is **not** currently better than always guessing the most
common class to a statistically meaningful degree (best 45.2% against a 36.6%
baseline, confidence interval straddling it). Distinguishing early from late MCI is
defined in ADNI by a delayed-recall memory-test cutoff rather than by anatomy, so part
of that ceiling is in the labels.

Serving stage predictions now would look authoritative and would not be. Enabling it
later is a one-line change once cross-validation shows it clears baseline.

## 7. Safety and honesty requirements

- Prominent banner: research demonstration, not a diagnostic device.
- Metrics displayed are subject-level, from a subject-wise split, with confidence
  intervals and the majority baseline alongside — never a bare accuracy figure.
- The app must never load a `*_LEAKY` checkpoint. Those score ~96% only on people
  already in the training set and fall to roughly chance on a stranger's scan.
- The probability is presented as primary and the yes/no call as secondary, because
  cross-era testing showed the ranking transfers to a new scanner but the decision
  threshold does not.

## 8. Architecture

```
Browser (single page)
   |  multipart POST /predict-stream
   v
Flask app.py
   |-- tasks.py        registry: classes, checkpoint, metrics source, enabled flag
   |-- validation      cheap geometric checks for "is this an axial brain slice"
   |-- preprocess      DICOM/PNG -> grayscale -> 144px -> 224px  (matches training)
   |-- inference       torch, CPU, + mirrored TTA
   |-- gradcam.py      reused from scripts/, handles 2-class and 4-class heads
   v
Server-Sent Events: one event per completed stage, final event carries the result
```

Model weights load lazily on first request and are cached in-process.

## 9. Design language

Derived from the supplied purple dashboard reference.

| Role | Colour |
|---|---|
| Background | `#0B0518` → `#120A29` gradient |
| Panel | `#1A1035`, border `#2A1D4D` |
| Primary accent | `#8B5CF6` |
| Heading gradient | `#C084FC` → `#E879F9` |
| Positive | `#22C55E` |
| Caution | `#F59E0B` |
| High risk | `#EF4444` |
| Text / muted | `#F5F3FF` / `#A79FC4` |

Rounded cards, soft glow on stat tiles, gradient headline text, dashboard-style
stat row.

### Page structure

1. Hero — brand, tagline, one-line summary
2. Upload card — drag/drop, PNG/JPEG/DICOM, with the sagittal-vs-axial note
3. Live pipeline — stage list with progress bar (FR-9)
4. Result dashboard — stat tiles, probability bars, input vs Grad-CAM
5. "How this was validated" — confound, subject-wise split, cross-era results
6. Disclaimer footer

## 10. Acceptance criteria

- [ ] Uploading a known test slice returns a prediction, probabilities and a Grad-CAM
- [ ] Uploading a photo, a blank image or a wide banner is rejected with a clear reason
- [ ] A `.dcm` file is decoded and processed
- [ ] Requesting `task=four_stage` returns 400
- [ ] Displayed AUC matches `reports/mobilenetv2_ADvsCN_v3adcn_result.json`
- [ ] Progress stages appear in order during a real request
- [ ] No file is written to disk during a request
- [ ] Page is legible and usable at 360px width
