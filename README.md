<div align="center">

# Alzheimer's Detection from Brain MRI

**A cross-validated deep learning pipeline on real ADNI clinical MRI — with four measured data leaks found and fixed, not assumed away.**

[Live demo](#) · [How it was validated](docs/PROJECT_EXPLANATION.md) · [Methodology docs](docs/)

</div>

---

## What this is

A model that reads an axial brain MRI slice and predicts Alzheimer's disease vs. cognitively normal, trained on **853 real clinical scans** downloaded directly from the [ADNI](https://adni.loni.usc.edu/) study archive as raw DICOM — not a pre-packaged Kaggle dataset. A web demo lets you upload a scan (or try a held-out sample) and see the prediction, a Grad-CAM heat-map of the region that drove it, and the model's honestly-reported, cross-validated metrics.

**Result: 74.1% accuracy, ROC AUC 0.784 (95% CI 0.743–0.826)**, measured by 5-fold cross-validation over all 501 AD/CN subjects — not a single favourable split. For context, a 2026 foundation model pretrained on 6.6M brain slices from 20 datasets reports AUC 0.850 [0.754, 0.947] on the same task; this project's interval overlaps it, from 501 subjects and no pretraining.

This is a research demonstration, not a medical device, and makes no diagnostic claim.

## Why this is more than another Kaggle classifier

The hard part of medical-imaging ML isn't training a model that scores well — it's proving the model learned *medicine* rather than a shortcut. Roughly half of published Alzheimer's deep-learning papers contain a data leak that inflates their results. This project measured for exactly that, four separate times:

| # | Confound found | Measured cost | Fix |
|---|---|---|---|
| 1 | Same subject's slices split across train/test | **+36.9 accuracy points of pure leakage** | Split per person before any image is created |
| 2 | Diagnosis and scanner era were the same variable | Models identified scanner era at 98–100% while at chance on the real disease | Rebuilt dataset so every class spans both scanner eras |
| 3 | Slices anchored to a fixed *fraction* of image height | Some subjects' extracted slices missed the hippocampus entirely | Anchored slices in millimetres below the skull vertex |
| 4 | Acquisition geometry stacked without true inter-slice spacing | Metadata-only classifier scored 40.9% vs. 36.7% baseline on the 4-way task, with zero pixels | Resampled to true mm-per-pixel geometry |

Full writeup: [`docs/PROJECT_EXPLANATION.md`](docs/PROJECT_EXPLANATION.md).

The project's standing rule, after three single-split headlines were withdrawn under cross-validation: **no number is reported until it has been cross-validated.** The web demo and every doc read their figures from recorded result JSON files (`reports/`), never a hardcoded number.

## Repo layout

```
src/            training/eval library code (models, datasets, data prep)
scripts/        experiment drivers, QC, cross-validation, confound audits
notebooks/      the pipeline walked through end-to-end (data → training → leakage proof)
docs/           methodology writeups (confound audits, ablations, feasibility reports)
reports/        every experiment's result JSON/figures — the evidence behind every number
demo/           a from-scratch demo training run (curves, confusion matrices)
webapp/         the live demo — FastAPI backend + React/Vite/Tailwind frontend
```

`data/` (raw DICOM + every processed image) is **not** in this repo — ADNI's Data Use Agreement prohibits redistributing the imaging data. See [ADNI access](https://adni.loni.usc.edu/data-samples/access-data/) if you want to retrain this on your own approved ADNI download. None of the other experimental checkpoints in `models/` are included either (~1GB, and none of them are what the demo serves) — **except** the one checkpoint the live demo actually uses (`models/checkpoints/mobilenetv2_ADvsCN.pt`, 9MB), which *is* tracked, so the demo below runs out of the box with no dataset or training required.

## Running the web demo locally

```bash
# backend
cd webapp
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# frontend, in a second terminal
cd webapp/frontend
npm install
npm run dev
```

## Stack

PyTorch · OpenCV · pydicom · scikit-learn for the model; FastAPI for the API; React 19 + Vite + Tailwind + Three.js for the web demo.

## Validation checks beyond cross-validation

- **Cross-scanner generalisation**: trained on one scanner generation, tested on a completely different one (different hardware, vendors, no shared subjects) — the signal survived in 5 of 6 runs (AUC 0.68–0.79).
- **Grad-CAM attention audit**: early models put 77% of their attention outside the brain (worse than a random heat-map); the current model concentrates on ventricles and the temporal lobe.
- **Negative results kept, not hidden**: pretraining hurt the model, learned slice-attention didn't beat a plain average, and narrowing the slice band made things worse — all measured and documented rather than quietly dropped. See [`docs/`](docs/).

## Paper

This project is also described in a research paper submitted to an academic conference and currently under peer review — not yet published, so it isn't included in this repo. Ask if you'd like more detail.

## Disclaimer

Research project only. This is not a medical device and cannot be used for diagnosis.
