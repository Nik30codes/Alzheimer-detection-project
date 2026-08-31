# Detecting Alzheimer's Disease from Brain MRI

**A machine learning project on the ADNI dataset**
Nikunj Bhalla — August 2026

---

## 1. What the project does

It reads a brain MRI scan and predicts whether the person has Alzheimer's disease or is
cognitively normal.

The data is **real clinical MRI** from ADNI (Alzheimer's Disease Neuroimaging Initiative),
downloaded as raw DICOM files directly from the study archive — not a pre-packaged
teaching dataset. **853 people** in total.

A working web demo lets anyone upload a scan and see the prediction, the confidence, and
a heat-map of which brain region drove the decision.

---

## 2. The result

**Alzheimer's vs. cognitively normal: 74.1% accuracy, ROC AUC 0.784** (95% confidence
interval 0.743–0.826), measured by 5-fold cross-validation over all 501 AD/CN subjects.

Two things make this number trustworthy rather than just large:

- **The confidence interval excludes 0.5.** An AUC of 0.5 means random guessing. The
  entire interval sits well clear of it, so the result is not chance.
- **It was measured on every subject, not a favourable subset.** Cross-validation gives
  each of the 501 people a prediction from a model that never saw them during training.

For context, a 2026 foundation model (BrainDINO) pretrained on **6.6 million brain slices
from 20 datasets** reports AUC 0.850 [0.754, 0.947] on this same task. Our interval
overlaps theirs, from 501 subjects and no pretraining.

---

## 3. Why this was harder than it looks

The core difficulty in medical imaging ML is not building the model — it is proving the
model learned *medicine* rather than a shortcut.

A model can reach a high score by two completely different routes:

| Route | What it learns | Works on a new patient? |
|---|---|---|
| **Honest** | Brain shrinkage, enlarged ventricles, hippocampal atrophy | Yes |
| **Shortcut** | Which scanner took the image, how it was framed | **No** |

Both give a good test score. Only the first is worth anything. Published literature
suggests roughly **half of Alzheimer's deep-learning papers** contain a flaw of this kind.

This project found **four** such shortcuts by measuring for them, and fixed three.

---

## 4. The four problems found, and how

The method used throughout: **give a model only one piece of non-image information and
no picture at all — how well can it guess?** If it beats chance, that information is a
usable shortcut.

### Problem 1 — The same person in training and testing

Each person contributes 32 nearly identical slices. Split those randomly and the model
recognises the *person*, not the disease.

**Measured cost: +36.9 accuracy points of pure illusion.**

Fixed by splitting the data **per person, before any image is created** — so all 32 of
someone's slices stay on one side of the train/test boundary.

### Problem 2 — Scanner era was identical to diagnosis

In the original data every healthy and Alzheimer's subject came from ADNI's 2006–07
phase, and every mild-impairment subject from the 2010+ phase. So "which diagnosis" and
"which scanner" were *the same variable*.

**Measured: models identified the scanner era with 98–100% accuracy while performing at
chance on the actual disease.**

Fixed by expanding the dataset (439 → 853 people) so every class appears in both eras.
Scanner era now gives **+0.0%** over baseline — it carries no information about the label.

### Problem 3 — Slices landed on different anatomy in each person

Slices were being cut at a fixed *fraction* of image height. Because scans differ in how
much neck and empty space they include, the same slice number landed in a different place
in each person — and for some subjects the entire slice range **missed the hippocampus**,
the structure Alzheimer's damages first.

Fixed by anchoring slices in **millimetres below the top of the skull**, so the same
slice number reaches the same anatomy in everyone.

### Problem 4 — Every image stretched by a different amount

The code stacked slices without accounting for the physical gap between them, so each
person's scan was squashed by a factor that varied per person and tracked the scanning
protocol.

**Measured: given only that geometry and no pixels at all, a model scores 40.9% on the
four-stage task against a 36.7% baseline** — while the real model only manages 43.0%.

Fixed by resampling every scan to true 1 mm-per-pixel with a fixed physical field of
view. Geometry now gives **+0.0%**.

### Still open — which hospital did the scan

Site identity gives +8.0% over baseline. Reported rather than hidden. The literature has
named tools for this (ComBat, DeepComBat) that have not yet been tried.

---

## 5. Three independent checks that the result is real

Any single check could be fooled. These three use different methods and would fail in
different ways.

**Test on a completely different scanner generation.** Train on 2006-era machines, test
on 2010s machines — different hardware, different vendors, no shared patients. The signal
survived in 5 of 6 runs (AUC 0.68–0.79). A model reading scanner artifacts would collapse
here.

**Look at where the model looks.** Grad-CAM highlights the pixels driving each decision.
Early models put **77%** of their attention outside the brain — worse than a random
heat-map, which scores 58%. The current model is at **34%**, concentrated on ventricles
and the temporal lobe.

**Check which mistakes it makes.** Early models confused Alzheimer's with *healthy* more
often than with late mild impairment — backwards, clinically, and a signature of protocol
reading. That pattern has since flipped to the correct direction.

---

## 6. Honesty about what did not work

Recorded deliberately, with evidence, so they are not attempted again.

| Attempted | Result |
|---|---|
| Pretraining the four-stage model | Scored **below** random initialisation |
| Learning which slices matter (attention) | +0.005 AUC — interval includes zero |
| Using fewer, better-placed slices | −0.046 AUC — significantly worse |
| Keeping full image resolution | Two architectures worse, one better — noise |
| Warping brains to a standard template | Rejected: would delete the volume differences that *are* the signal |

The attention experiment produced a particularly clean negative: an *oracle* that cheats
by using the true labels gains only **+0.015 AUC**, because the 32 slices are so similar
they carry roughly **1.3 independent measurements, not 32**. That is a measured ceiling,
not a failed attempt.

---

## 7. The most important lesson: the headline was wrong, three times

Early in the project, results were measured on a **single held-out test set of 75
people**. That number was 82.7% accuracy, AUC 0.906.

Re-measuring the *identical model and code* with 5-fold cross-validation over all 501
people gave **70.9%, AUC 0.784**. The two confidence intervals **do not overlap**.

The single split had simply drawn a favourable 75 people.

This happened **three separate times** in the project — twice on experiments, once on the
headline itself. Each time, cross-validation caught it. The standing rule now:

> **No number is reported as a headline until it has been cross-validated.**

The website and all documents read their figures directly from cross-validated result
files, so they cannot silently drift back to optimistic values.

---

## 8. Four-stage classification: why it is not claimed

The project also attempts CN → EMCI → LMCI → AD (four stages). It reaches 43.0% against a
36.7% baseline — but this is **not** presented as a result, for two measured reasons.

**Most of the margin is artifact.** Acquisition geometry alone supplies 4.2 of the 6.3
points above baseline.

**The labels themselves are unreliable.** ADNI separates early from late mild impairment
by a *memory-test score*, not by anatomy. ADNI's own documentation states that at
follow-up visits, "LMCI" simply means any MCI — the distinction is a screening-time
stratification, not a maintained diagnosis. About 35% of MCI participants score in the
normal range on that test.

Merging the two MCI stages recovers 16 accuracy points instantly. Part of this ceiling is
in the data, not the model — so the four-stage model is **deliberately withheld** from the
public demo rather than shipped with a confident-looking output.

---

## 9. Technical implementation

**Pipeline:** raw DICOM → 3D volume reconstruction → axial reslicing → millimetre-anchored
slice band (48–92 mm below skull vertex, 32 slices) → denoising → resolution
harmonisation → 224×224 PNG.

**Models:** MobileNetV2, EfficientNet-B0, and a custom CNN, all trained from scratch.
ImageNet pretraining was tested and found to *hurt* — the domain gap between natural
photographs and grayscale MRI is too large.

**Prediction:** each of a person's 32 slices is scored independently and the probabilities
averaged into one per-person prediction. Averaging is worth ~8 accuracy points over
judging a single slice.

**Stack:** PyTorch, OpenCV, pydicom, scikit-learn. Web demo in React + Vite + Tailwind with
a FastAPI backend.

**Reproducibility:** every experiment writes a result JSON; documents and the website read
those files rather than hardcoded numbers.

---

## 10. What remains

- **Deploy the skull-masked model.** It measured AUC 0.814 under cross-validation and
  improved in *both* directions of the cross-scanner test — the best-supported
  configuration in the project, and not yet in production.
- **Address the site effect** using established harmonisation methods.
- **Longitudinal scans.** How fast a brain shrinks between visits is what actually
  separates early from late impairment. This is the strongest untried lever, and would
  require a new data download.

---

*Research project only. This is not a medical device and cannot be used for diagnosis.*
