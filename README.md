# MultiModel Alzheimer Detection

A research pipeline and website for structural MRI classification using the local ADNI cohort. The main task distinguishes the AD and cognitively normal research groups. The website is a research demonstration, not a diagnostic device.

## Recorded results

The five-fold participant-level evaluation contains 501 people (216 AD, 285 CN). Stored fold classifications give **355/501 correct: 70.86% accuracy**, with a 95% Wilson interval of 66.73–74.67%. AD-positive ROC AUC is **0.7845**, with an analytic interval of 0.7429–0.8261. These are internal development results, not external clinical validation.

The earlier 74.1% figure used a threshold selected on pooled out-of-fold scores. It is not the stored-fold classification accuracy. The serving checkpoint and threshold come from a separate development run.

A paired masking experiment improved AUC from 0.7643 to 0.8140; that model is not deployed. The completed 3D experiment achieved 64.3% accuracy and AUC 0.721. Four-stage classification remains exploratory and affected by acquisition geometry. Six cross-era runs range from AUC 0.552 to 0.786; scanner and site effects remain limitations.

## Website

The original amber design supports a patient batch or one-person upload, prepared MRI examples, per-person progress and results, CSV export, and representative MRI/Grad-CAM views. Optional clinical CSV values remain supplied context in the browser; they do not affect the MRI classifier. Clinical/fusion benchmark performance has not been measured. Recorded examples and aggregate audit results are available in expandable sections.

## Run locally

Backend, from webapp/: `pip install -r requirements.txt`, then `uvicorn backend.main:app --port 8000`.

Frontend, from webapp/frontend/: `npm ci`, then `npm run dev`. Use `npm run build` for production; FastAPI can also serve the built frontend. For separate Render services, set VITE_API_BASE to the backend URL at build time and include the frontend URL in backend CORS_ORIGINS.

Checks: `npm test` and `npm run lint` in the frontend; `python -B -m unittest discover -s webapp/tests -p test_inference_regressions.py -v` from the repository root.

## Project data

Raw/processed research datasets, participant manifests, local practice batches and report/presentation outputs stay outside the public repository. The included checkpoint supports the existing demo; training requires separately approved ADNI data. Public benchmark files contain aggregate results and de-identified example labels rather than participant records.

Stack: PyTorch, FastAPI, React, Vite, Tailwind and Three.js. Grad-CAM shows model sensitivity and does not identify diseased tissue or establish a correct diagnosis.
