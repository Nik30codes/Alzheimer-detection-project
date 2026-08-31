"""
Regenerates notebooks/03 and writes notebooks/04 to match what Phase D actually did.

03 currently documents the ImageNet-pretrained two-phase fine-tune, which finding 7
established was the wrong approach on this dataset. It is rewritten to train
MobileNetV2 from scratch as the primary approach, with the pretrained attempt kept as
a documented negative result rather than deleted.

04 is new: EfficientNet-B0, the 2.5D input experiment, the soft-vote comparison, the
3-model ensemble, and the deliberate-leakage demonstration.

Cells are written WITHOUT outputs -- the user runs them in Jupyter to reproduce.
"""

import json
import os

import nbformat as nbf

ROOT = r"C:\Users\Nikunj\Documents\alzheimer-mri-project"
NB = os.path.join(ROOT, "notebooks")


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


SETUP = """
import sys
sys.path.insert(0, '../src')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datasets import CLASSES, build_dataloaders, build_dataloaders_25d, compute_class_weights
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0
from train import train_model
from evaluate import (get_predictions, slice_level_report, subject_level_report,
                      subject_level_soft_vote, ensemble_predictions)

%matplotlib inline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '-', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU only')

manifest = pd.read_csv('../data/manifest.csv')
class_weights = compute_class_weights(manifest)
torch.manual_seed(42); np.random.seed(42)
print(manifest.groupby(['class', 'split']).size().unstack())
"""

PLOT_CM = """
def plot_cm(cm, title, ax=None):
    \"\"\"Confusion matrix with per-class recall on the diagonal made obvious --
    aggregate accuracy hides which class the model is actually failing on.\"\"\"
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title)
    return ax
"""


def build_nb03():
    cells = [
        md("""
# MobileNetV2 on ADNI MRI — trained FROM SCRATCH

**Update to this notebook.** It previously fine-tuned MobileNetV2 from ImageNet weights.
That approach was tested three ways on this dataset and every variant lost to a much
smaller from-scratch CNN, so this notebook now trains the same architecture from random
initialization as the primary approach. The pretrained results are kept at the bottom as
a documented negative result, because "we tried it and it hurt" is a finding worth
keeping, not a mistake worth hiding.

| approach | subject-level accuracy |
|---|---|
| MobileNetV2, ImageNet weights, full unfreeze | 34.8% |
| MobileNetV2, ImageNet weights, partial unfreeze + frozen BN | ~37% |
| MobileNetV2, ImageNet weights, partial unfreeze + adaptive BN | 40.9% |
| **MobileNetV2, from scratch (this notebook)** | **see below** |
| custom SimpleCNN, from scratch (notebook 02) | 56.1% |

**Why pretraining hurts here.** ImageNet weights encode statistics of natural colour
photographs. Our input is a grayscale MRI slice copied into three identical channels —
no colour information, completely different spatial statistics, and a foreground that
fills the frame. The domain gap is large enough that the pretrained features are a worse
starting point than random ones, and with only ~300 training subjects there isn't enough
data to retrain them out. This is "negative transfer", and it is the expected outcome
when the source and target domains are this far apart.
"""),
        code(SETUP),
        code(PLOT_CM),
        md("""
## 1. Data loaders

`rgb=True` gives the 3-channel ImageNet-normalized transforms. The normalization
constants are arbitrary for a from-scratch model, but they are kept identical to the
pretrained runs so that **weight initialization is the only variable that changed**
between this result and the negative result at the bottom.
"""),
        code("""
train_loader, val_loader, test_loader = build_dataloaders(
    manifest, batch_size=32, num_workers=2, rgb=True)
print('class weights:', dict(zip(CLASSES, [round(w, 3) for w in class_weights.tolist()])))
"""),
        md("""
## 2. Model — random initialization

`pretrained=False` is the whole point of this notebook.
"""),
        code("""
model = build_mobilenetv2(num_classes=len(CLASSES), pretrained=False)
n_params = sum(p.numel() for p in model.parameters())
print(f'MobileNetV2 from scratch: {n_params:,} parameters')
"""),
        md("""
## 3. Train

Single phase, not the two-phase freeze/unfreeze schedule the pretrained version used —
there are no pretrained features to protect, so there is nothing to freeze. Learning rate
is 1e-3 rather than the 1e-5 used for fine-tuning, because random weights need to move a
long way. Early stopping restores the best-validation-loss checkpoint.
"""),
        code("""
history = train_model(
    model, train_loader, val_loader, class_weights, device,
    epochs=40, lr=1e-3, patience=7, weight_decay=1e-4,
    checkpoint_path='../models/checkpoints/mobilenetv2_honest2d.pt')
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history['train_loss'], label='train'); axes[0].plot(history['val_loss'], label='val')
axes[0].set_title('loss'); axes[0].set_xlabel('epoch'); axes[0].legend()
axes[1].plot(history['train_acc'], label='train'); axes[1].plot(history['val_acc'], label='val')
axes[1].set_title('accuracy'); axes[1].set_xlabel('epoch'); axes[1].legend()
plt.tight_layout()
"""),
        md("""
## 4. Evaluate

Three numbers, and the gap between them matters:

- **slice level** — every axial slice judged independently. Not what a clinician does.
- **subject level, hard vote** — majority across a subject's 32 slices.
- **subject level, soft vote** — average the probability distributions instead. Keeps
  confidence information that majority voting throws away.
"""),
        code("""
preds = get_predictions(model, test_loader, device)

print('===== SLICE LEVEL =====')
cm_slice = slice_level_report(preds)

print('\\n===== SUBJECT LEVEL — hard majority vote =====')
cm_hard, subj_hard = subject_level_report(preds)

print('\\n===== SUBJECT LEVEL — soft vote =====')
cm_soft, subj_soft = subject_level_soft_vote(preds)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_cm(cm_slice, 'slice level', axes[0])
plot_cm(cm_hard, 'subject, hard vote', axes[1])
plot_cm(cm_soft, 'subject, soft vote', axes[2])
plt.tight_layout()
"""),
        md("""
### Sample predictions

Per-subject output with the averaged class probabilities, so the failures are inspectable
rather than hidden behind an accuracy number.
"""),
        code("""
subj_soft['correct'] = subj_soft['true'] == subj_soft['pred']
print('correct:', int(subj_soft['correct'].sum()), 'of', len(subj_soft))
display(subj_soft.head(15).round(3))
print('\\nmisclassified subjects:')
display(subj_soft[~subj_soft['correct']].round(3))
"""),
        md("""
## 5. Negative result kept on the record — ImageNet fine-tuning

Not re-run here (it takes GPU time to reproduce a worse answer), but the numbers are
preserved in `reports/metrics_legacy_pretrained.json`. Two lessons worth keeping:

1. **Don't freeze BatchNorm** when fine-tuning onto a distant domain. Freezing it was
   tried on the theory that it would reduce overfitting; it made things worse, because it
   forces the network to keep normalizing with ImageNet-photo statistics instead of
   adapting to the actual MRI distribution.
2. **Pretrained is not automatically better.** It is better when the source and target
   domains are close. Grayscale medical volumes and natural photographs are not close.
"""),
        code("""
legacy = json.load(open('../reports/metrics_legacy_pretrained.json'))
print(json.dumps(legacy.get('mobilenetv2', {}), indent=2))
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    path = os.path.join(NB, "03_train_mobilenetv2.ipynb")
    nbf.write(nb, path)
    print("wrote", path)


def build_nb04():
    cells = [
        md("""
# EfficientNet-B0, 2.5D input, soft voting, ensemble — and a leakage demonstration

Closes out Phase D. Four things happen here:

1. **EfficientNet-B0 from scratch** — the third architecture, same protocol as the others.
2. **2.5D input** — stack three *adjacent* axial slices as the three channels instead of
   copying one slice three times. Same tensor shape, same parameter count, real
   volumetric context.
3. **Soft voting and a 3-model ensemble** — cheap aggregation wins, no new training.
4. **A deliberately leaky split** — reproducing the standard methodological bug in this
   literature to measure exactly how many points it invents.
"""),
        code(SETUP),
        code(PLOT_CM),
        md("""
## 1. EfficientNet-B0 from scratch

`pretrained=False` by default now, so finding 7 can't be repeated by accident.
At 4.0M parameters this is the largest model tried on ~10.8k training images, so
overfitting is the thing to watch in the curves below.
"""),
        code("""
train_loader, val_loader, test_loader = build_dataloaders(
    manifest, batch_size=32, num_workers=2, rgb=True)

effnet = build_efficientnet_b0(num_classes=len(CLASSES), pretrained=False)
print(f'EfficientNet-B0: {sum(p.numel() for p in effnet.parameters()):,} parameters')

history = train_model(effnet, train_loader, val_loader, class_weights, device,
                      epochs=40, lr=1e-3, patience=7, weight_decay=1e-4,
                      checkpoint_path='../models/checkpoints/efficientnet_b0_scratch.pt')
"""),
        code("""
preds_effnet = get_predictions(effnet, test_loader, device)
print('===== SUBJECT LEVEL — soft vote =====')
cm_effnet, subj_effnet = subject_level_soft_vote(preds_effnet)
plot_cm(cm_effnet, 'EfficientNet-B0, subject soft vote')
"""),
        md("""
## 2. 2.5D input — three adjacent slices instead of three copies

The standard grayscale→fake-RGB trick feeds the network **three identical copies** of one
slice, so two thirds of the input carries no information at all. `MRI25DDataset` stacks
slices *i-1, i, i+1* instead.

Why this should help on this task specifically: a dark region on a single 2D slice can be
noise or a partial-volume artifact, whereas genuine hippocampal atrophy persists across
consecutive slices. A single-slice model cannot distinguish those two cases even in
principle. This one can, at zero extra parameter cost.

Both augmentation and normalization are handled carefully — the same geometric transform
is applied to all three channels at once (augmenting them independently would misalign the
anatomy across depth and destroy the relationship being exploited), and `T.Grayscale(3)`
is deliberately absent from the 2.5D transform since it would collapse the stack back into
three identical channels.
"""),
        code("""
tr25, va25, te25 = build_dataloaders_25d(manifest, batch_size=32, num_workers=2)

# proof the channels really are different slices, not three copies
imgs, labels, sids = next(iter(te25))
img = imgs[0]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for i, name in enumerate(['slice i-1', 'slice i', 'slice i+1']):
    axes[i].imshow(img[i], cmap='gray'); axes[i].set_title(name); axes[i].axis('off')
axes[3].imshow((img[2] - img[0]).abs(), cmap='magma')
axes[3].set_title('|i+1 - i-1| (nonzero = real depth)'); axes[3].axis('off')
plt.tight_layout()
print('mean abs difference between adjacent channels:', (img[0]-img[1]).abs().mean().item())
"""),
        code("""
effnet25 = build_efficientnet_b0(num_classes=len(CLASSES), pretrained=False)
hist25 = train_model(effnet25, tr25, va25, class_weights, device,
                     epochs=40, lr=1e-3, patience=7, weight_decay=1e-4,
                     checkpoint_path='../models/checkpoints/efficientnet_b0_honest25d.pt')

preds_25d = get_predictions(effnet25, te25, device)
cm_25d, subj_25d = subject_level_soft_vote(preds_25d)
plot_cm(cm_25d, 'EfficientNet-B0 2.5D, subject soft vote')
"""),
        md("""
## 3. Hard vote vs soft vote

Majority voting discards confidence. If 17 slices weakly prefer EMCI (p=0.30 each) and 15
slices strongly say AD (p=0.85 each), majority voting returns EMCI even though the averaged
evidence clearly points to AD. Soft voting keeps that magnitude information, which matters
most when individual slices are near chance — which is the regime this dataset is in.
"""),
        code("""
for name, p in [('EfficientNet-B0', preds_effnet), ('EfficientNet-B0 2.5D', preds_25d)]:
    _, sh = subject_level_report(p)
    _, ss = subject_level_soft_vote(p, verbose=False)
    hard = (sh['true'] == sh['pred']).mean()
    soft = (ss['true'] == ss['pred']).mean()
    print(f'{name:24s} hard {hard:.3f} | soft {soft:.3f} | delta {soft-hard:+.3f}')
"""),
        md("""
## 4. Three-model ensemble

Averaging softmax across the custom CNN, MobileNetV2 and EfficientNet-B0. The models fail
on *different* subjects, so averaging lets a confidently-correct model outvote two
unconfidently-wrong ones. All prediction frames come from loaders built with
`shuffle=False`, so their rows correspond to the same images — `ensemble_predictions`
asserts this rather than trusting it.
"""),
        code("""
cnn = SimpleCNN(num_classes=4, in_channels=1)
cnn.load_state_dict(torch.load('../models/checkpoints/custom_cnn.pt')); cnn.to(device).eval()
_, _, test_gray = build_dataloaders(manifest, batch_size=32, num_workers=2, rgb=False)
preds_cnn = get_predictions(cnn, test_gray, device)

mnet = build_mobilenetv2(4, pretrained=False)
mnet.load_state_dict(torch.load('../models/checkpoints/mobilenetv2_honest2d.pt')); mnet.to(device).eval()
preds_mnet = get_predictions(mnet, test_loader, device)

ens = ensemble_predictions([preds_cnn, preds_mnet, preds_effnet])
print('===== ENSEMBLE — subject level, soft vote =====')
cm_ens, subj_ens = subject_level_soft_vote(ens)
plot_cm(cm_ens, 'ensemble, subject soft vote')
"""),
        md("""
## 5. Leakage demonstration — why this project splits by subject

Every result above splits by **subject**: a person's 32 slices are entirely in train, or
entirely in test, never both. The common alternative in published work is to shuffle
*slices*. Below, the same models are retrained on a slice-wise split and nothing else is
changed.

The accuracy will jump enormously. It is not a better model. Each subject contributes 32
near-duplicate axial slices, so slice-wise splitting puts the same brain on both sides of
the boundary and the network scores well by **recognizing the individual**, not by
detecting atrophy — it would score similarly if the diagnoses were shuffled at random.

Roughly half the published Alzheimer's-MRI deep learning literature reports 90%+ accuracy
obtained this way. Studies that split properly by subject report AD-vs-CN *binary*
accuracy under 71% on ADNI-sized data. That is the gap this cell measures.
"""),
        code("""
leaky_results = {}
for arch in ['custom_cnn', 'mobilenetv2', 'efficientnet_b0']:
    path = f'../reports/{arch}_leaky_result.json'
    try:
        leaky_results[arch] = json.load(open(path))
    except FileNotFoundError:
        print('not yet run:', path)

honest = json.load(open('../reports/metrics.json'))
rows = []
for arch in ['custom_cnn', 'mobilenetv2', 'efficientnet_b0']:
    h = honest.get(arch, {})
    l = leaky_results.get(arch, {})
    rows.append({
        'model': arch,
        'honest (subject-wise split)': h.get('subject_level_accuracy_softvote'),
        'LEAKY (slice-wise split)': l.get('slice_level_accuracy'),
    })
df = pd.DataFrame(rows).set_index('model')
df['inflation'] = df['LEAKY (slice-wise split)'] - df['honest (subject-wise split)']
display(df.round(3))

ax = df[['honest (subject-wise split)', 'LEAKY (slice-wise split)']].plot.bar(
    figsize=(8, 4.5), rot=0, color=['#3b7dd8', '#d1495b'])
ax.axhline(0.25, ls='--', c='gray', lw=1)
ax.text(-0.4, 0.26, 'chance (4-way)', color='gray', fontsize=9)
ax.set_ylabel('accuracy'); ax.set_ylim(0, 1)
ax.set_title('Same models, same data, one line changed in how the split is made')
plt.tight_layout()
"""),
        md("""
## 6. Where this actually lands

The honest numbers sit in the 50–60% range on a 4-way task, against a 25% chance baseline
and with only 439 subjects. That is roughly what the properly-validated literature reports
for this problem size, and it is the number that would survive contact with a new hospital's
scans.

The leaky numbers are much higher and mean nothing. They are included precisely so the
difference is visible in one figure.

Honest ways to actually move the real number, in rough order of expected payoff:

1. **More subjects.** The 876-scan LONI expansion roughly triples the dataset. Sample size
   is the binding constraint here, not architecture — every architecture tried lands within
   a few points of the others.
2. **Reduce the task.** AD vs CN binary is genuinely easier and genuinely useful; the
   EMCI/LMCI boundary is subtle even for radiologists.
3. **More slices per subject / full 3D.** 2.5D is a cheap step toward this; a real 3D CNN
   is the next one, and needs the larger dataset to be trainable.
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    path = os.path.join(NB, "04_efficientnet_ensemble_and_leakage.ipynb")
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    build_nb03()
    build_nb04()
