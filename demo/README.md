# Demo: live model training + performance report

Self-contained script for showing the models actually working -- trains three
architectures from scratch on the real AD-vs-CN task and prints/saves full performance
metrics for each.

## Run it

```
conda activate ml
python demo/run_demo.py
```

Nothing else to configure. It reads data from the existing project data folder, so run
it from within this project as-is.

## What it does

For each of **Custom CNN**, **MobileNetV2**, and **EfficientNet-B0** (all trained from
scratch -- see the note in `run_demo.py` on why, not fine-tuned):

1. Trains on the project's real AD-vs-CN dataset (501 ADNI subjects, subject-wise
   train/val/test split so no person's scans appear in more than one split), printing
   every epoch's train/val loss and accuracy live.
2. Evaluates on the held-out test subjects: classification report, confusion matrix,
   accuracy with a 95% confidence interval, macro F1, ROC AUC with a 95% confidence
   interval, and a handful of example predictions.
3. Saves a confusion matrix PNG and a training-curves PNG per model to `demo/results/`.

At the end it prints a comparison table across all three models, **plus the project's
real 5-fold cross-validated headline number for the same task**, read live from
`reports/mobilenetv2_ADvsCN_cv_result.json` -- so the demo's own (faster, single-split)
numbers are never mistaken for the statistically established result.

## Expected runtime

Roughly 15-45 minutes total for all three models on a single consumer GPU (early
stopping usually ends a model well before the 40-epoch budget). To run faster for a
quick check, lower `EPOCHS` near the top of `run_demo.py`.

## Output

Everything lands in `demo/results/`:
- `{arch}_confusion_matrix.png`, `{arch}_training_curves.png` per model
- `{arch}_demo.pt` -- the trained weights from this run
- `summary.json` -- every number from the run in one file
