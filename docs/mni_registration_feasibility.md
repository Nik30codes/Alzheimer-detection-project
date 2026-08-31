# MNI registration + N4 bias correction: feasibility report

**Date:** 2026-08-17
**Scope:** research and design only. No project file was modified, nothing was installed, no
training or GPU work was started. All package checks were read-only import probes and
`pip index versions` queries.
**Question asked:** should the pipeline add (a) registration to MNI152 template space and
(b) N4 bias field correction, both of which the literature treats as mandatory for AD-MRI?

---

## 0. Verdict up front

**Recommendation: do NOT add nonlinear (SyN) registration. Do add N4 and a rigid-body
reorientation. Treat 12-DOF affine as an experimental arm, not an upgrade. And do a cheaper
fix first that neither of the two proposed steps was aimed at.**

Ranked by (expected benefit) / (cost x risk):

| # | Change | Cost | Risk to signal | Expected effect | Do it? |
|---|---|---|---|---|---|
| 0 | **Fix 3D voxel geometry** (isotropic resample, fixed mm/pixel, head-centred crop) — no template, no new dependency | ~0 (numpy/cv2) | low | removes a **measured** residual confound worth **+4.2 pts of 4-way accuracy** (§5.4) | **yes, first** |
| 1 | **N4 bias correction** (per 3D volume, before slicing) | ~1 h wall clock for 853 | very low | small/null on accuracy; plausible cross-site generalisation gain | **yes** |
| 2 | **Rigid (6-DOF) alignment to MNI** | ~1 h | low (no scaling applied) | fixes head tilt + true slice correspondence; the only part of "registration" this pipeline is genuinely missing | **yes** |
| 3 | **Affine (12-DOF) to MNI** | ~2.5–3 h | **medium** — normalises head size (§5.3) | unknown sign | **A/B arm only** |
| 4 | **Nonlinear SyN to MNI** | **21–45 h** wall clock | **high** — deforms away regional volume, i.e. the atrophy signal (§5.3) | no measured benefit over linear in the literature | **no** |

The headline reason the literature's "mandatory" framing does not transfer cleanly: the papers
that quantify registration's benefit compare it against **no anatomical anchoring at all**. This
project already replaced that with the millimetre-below-vertex band (decision 11), which is worth
most of the same thing and which already moved AD-vs-CN AUC from 0.35/0.51 to 0.67. What remains
unclaimed is tilt correction, isotropic geometry and exact slice correspondence — worth
distinctly less than the 6–7 points the registration papers report.

---

## 1. What is already installed

Env probed: `C:\Users\Nikunj\miniconda3\envs\ml\python.exe` — **Python 3.11.15, win-amd64**.

| package | status |
|---|---|
| `antspyx` / `ants` | **MISSING** |
| `SimpleITK` | **MISSING** |
| `nibabel` | **MISSING** |
| `dipy` | **MISSING** |
| `nilearn` | **MISSING** |
| `deepbet` | **MISSING** |
| `HD-BET` (`HD_BET`, `hd_bet`) | **MISSING** |
| `nipype`, `antspynet`, `templateflow`, `brainextractor` | **MISSING** |
| `scikit-image` | **MISSING** (worth knowing — the existing code is pure `cv2`) |

Present and relevant:

| package | version |
|---|---|
| numpy | **2.4.4** |
| scipy | 1.17.1 |
| opencv-python | 5.0.0.93 |
| pydicom | 3.0.2 |
| torch | 2.6.0+cu124 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| matplotlib | 3.11.0 |

**There is no neuroimaging stack in this environment at all.** Every option below requires a new
install.

### 1.1 The install has a hard conflict — read this before `pip install antspyx`

`antspyx` 0.6.3 (latest, 24 Feb 2026) declares **`numpy<2.4.0`**. This env has **numpy 2.4.4**.
Installing antspyx into `ml` will **downgrade numpy underneath PyTorch 2.6+cu124, pandas 3.0.3 and
OpenCV 5.0**, while a training job is running.

> Do the preprocessing work in a **separate conda env** (e.g. `preproc`), not in `ml`. The
> interface between them is PNG files on disk, so they never need to share an interpreter.
> `conda create -n preproc python=3.11 numpy=2.3 && pip install antspyx nibabel nilearn`.

Windows wheel availability (checked against the PyPI JSON API, nothing installed):

| package | latest | win_amd64 wheel for cp311? | notes |
|---|---|---|---|
| `antspyx` | 0.6.3 (2026-02-24) | **yes** — cp310/311/312/313 | needs MSVC redistributable; pins `numpy<2.4.0` |
| `SimpleITK` | 2.5.6 | **yes** — cp37…cp312 | no numpy pin declared |
| `itk-elastix` | 0.25.4 | yes (pip resolved it for this interpreter) | only needed if you want elastix's B-spline |

`pip index versions antspyx` resolving to 0.6.3 *on this interpreter* is itself evidence that a
compatible win-amd64 cp311 wheel exists — pip's finder applies platform tags. No admin rights are
needed for any of these; they are all pure wheels into a user-owned conda env.

---

## 2. Which library: ANTsPy vs SimpleITK

### (a) N4 bias correction — either works; ANTsPy is one line

ANTsPy exposes the identical ITK filter with far less ceremony
([source](https://raw.githubusercontent.com/ANTsX/ANTsPy/master/ants/ops/bias_correction.py)):

```python
ants.n4_bias_field_correction(image, mask=None, rescale_intensities=False,
                              shrink_factor=4,
                              convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
                              spline_param=None, return_bias_field=False,
                              verbose=False, weight_mask=None)
```

SimpleITK requires you to assemble it: `sitk.OtsuThreshold` for the mask,
`sitk.Shrink` for speed, `N4BiasFieldCorrectionImageFilter` +
`SetMaximumNumberOfIterations`, then `GetLogBiasFieldAsImage` and apply at full resolution
([docs](https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html)). Note the
docs' warning that the mask "must occupy the same physical space" as the input — a class of bug
ANTsPy hides from you. **Same algorithm, same speed; ANTsPy has the better API.**

### (b) Registration to MNI152 — ANTsPy, decisively

- **ANTsPy** gives the whole thing in one call, with named transform recipes
  ([source](https://raw.githubusercontent.com/ANTsX/ANTsPy/master/ants/registration/registration.py)):
  ```python
  ants.registration(fixed, moving, type_of_transform="Affine", initial_transform=None,
                    mask=None, moving_mask=None, aff_metric="mattes", aff_sampling=32,
                    aff_iterations=(2100, 1200, 1200, 10), aff_shrink_factors=(6, 4, 2, 1),
                    aff_smoothing_sigmas=(3, 2, 1, 0), reg_iterations=(40, 20, 0),
                    syn_metric="mattes", verbose=False, ...)
  # -> {"warpedmovout", "warpedfixout", "fwdtransforms", "invtransforms"}
  ```
  Valid `type_of_transform` includes `Translation, Rigid, QuickRigid, DenseRigid, Similarity,
  Affine, AffineFast, TRSAA, Elastic, ElasticSyN, SyN, SyNRA, SyNOnly, SyNCC, SyNabp, SyNAggro,
  antsRegistrationSyN[x], antsRegistrationSyNQuick[x]` and repro variants. The multi-resolution
  schedule is already tuned; you get ANTs' published defaults for free.
- **SimpleITK** has `ImageRegistrationMethod` (Mattes MI, gradient descent, multi-resolution,
  `SetInitialTransform` + `CenteredTransformInitializer`) — perfectly capable, but you write and
  tune the optimiser schedule yourself: ~60–100 lines and a real risk of a silently mediocre
  registration. Nonlinear in SimpleITK means hand-rolling a BSpline transform or adding
  `itk-elastix`.
- **API maturity:** ANTs' affine+SyN schedules are the most-cited T1-to-MNI recipe in the field;
  SimpleITK is a general toolkit with no opinion about brains.
- **Speed:** both are ITK underneath, so per-iteration cost is comparable. ANTs' defaults do more
  iterations than a naive SimpleITK script, so ANTs is often *slower* per subject and better.

**Use ANTsPy for both steps.** Use **SimpleITK only for DICOM reading** (see §6.1) — its
`ImageSeriesReader` + `GDCMSeriesFileNames` returns a correctly spaced and oriented 3D image in
two calls, which is exactly what the current `_build_volume()` does not do.

Wildcard worth knowing about: **`deepmriprep`** ([repo](https://github.com/wwu-mmll/deepmriprep))
does brain extraction (deepbet), affine registration (torchreg), segmentation and a *learned*
nonlinear warp in **~10 s on GPU / ~100 s on CPU per image**, i.e. 20–100x faster than ANTs SyN.
It is a legitimate escape hatch if nonlinear ever becomes necessary, at the price of a much less
validated method and a weights download. Not recommended for a first pass.

---

## 3. Where to get an MNI152 template

**`nilearn` bundles one inside the wheel — no download, fully offline.**
`nilearn/datasets/struct.py` defines
([source](https://raw.githubusercontent.com/nilearn/nilearn/main/nilearn/datasets/struct.py)):

```python
MNI152_FILE_PATH   = PACKAGE_DIRECTORY / "data" / "mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz"
GM_MNI152_FILE_PATH = PACKAGE_DIRECTORY / "data" / "mni_icbm152_gm_tal_nlin_sym_09a_converted.nii.gz"
WM_MNI152_FILE_PATH = PACKAGE_DIRECTORY / "data" / "mni_icbm152_wm_tal_nlin_sym_09a_converted.nii.gz"
```

Offline load, exact code:

```python
from nilearn.datasets import load_mni152_template
from nilearn.datasets.struct import MNI152_FILE_PATH   # a real path inside site-packages
tpl = load_mni152_template(resolution=1)               # nibabel image, 1 mm
import ants
tpl_ants = ants.image_read(str(MNI152_FILE_PATH))      # hand the same file to ANTs
```

**Critical caveat: nilearn's bundled MNI152 is *skull-stripped*** (its own docs describe it as "the
MNI152 skullstripped T1 template"). This pipeline deliberately keeps the skull (CLAUDE.md: masking
moved Grad-CAM attention but never touched the confound, and `v2mask` was indistinguishable from
`v2`). Registering a **whole head** to a **brain-only** template with Mattes MI will bias the
scaling — the optimiser tries to match scalp/skull intensity to background. Three ways out, in
order of preference:

1. **Whole-head ICBM152** — `nilearn.datasets.fetch_icbm152_2009()` returns keys
   `t1, t2, t2_relax, pd, gm, wm, csf, eye_mask, face_mask, mask`, downloaded once to
   `~/nilearn_data/`. The presence of `eye_mask` and `face_mask` confirms `t1` is whole-head. This
   is the right fixed image. Needs internet once.
2. **ANTsPy's bundled MNI** — `ants.get_ants_data("mni")` is **not** bundled: it downloads from
   figshare into `~/.antspy/` on first call
   ([source](https://raw.githubusercontent.com/ANTsX/ANTsPy/master/ants/utils/get_ants_data.py)),
   valid keys `r16, r27, r30, r62, r64, r85, ch2, mni, surf, pcasl`. Fine, but one-time internet
   too, and it is a coarse test asset rather than a curated template.
3. **Offline fallback** — use nilearn's bundled brain-only template and pass
   `moving_mask=<brain mask of the subject>` to `ants.registration`. The project already has a
   validated head/brain mask routine in `scripts/brain_mask.py` (threshold low → fill → erode past
   the skull; **do not use Otsu**, per CLAUDE.md, it deletes the cortical ribbon).

`SimpleITK` ships no templates. `templateflow` (the tidiest source of `MNI152NLin2009cAsym`) is not
installed and requires downloads.

---

## 4. Realistic runtime for 853 subjects

**Hardware reality check.** The AMD Ryzen 5 7640HS is **6 physical Zen 4 cores / 12 threads**, in a
35–54 W laptop envelope. The existing `reextract_v3.py` pattern (10 worker processes, each pinned
to one thread) is right, but expect the effective speedup on FP-heavy registration to be
**~6–7x, not 10x** — SMT gives little on saturated FPU work, and CLAUDE.md already records a run
that took 400 minutes instead of 60 purely from the "Balanced" power plan.

**Data size.** Every one of the 853 series was measured (header scan, this session):
`SpacingBetweenSlices = 1.2 mm` in **all 853**; in-plane spacing 0.938–1.354 mm; 160–184 sagittal
slices; matrices 192² to 256². So volumes are ~6–12 M voxels, i.e. squarely the size the ANTs
literature timings refer to.

| step | per subject, 1 thread | 853 subjects, 10 pinned workers |
|---|---|---|
| SimpleITK DICOM series read | 1–3 s | negligible |
| **N4** (`shrink_factor=4`, 4x50 iters) | 20–60 s | **~1.0–1.5 h** |
| **Rigid / QuickRigid** to MNI | 10–40 s | **~0.6–1.2 h** |
| **Affine** (ANTs default 4-level) | 30–120 s | **~1.5–3 h** |
| **SyN / SyNQuick** at 1 mm | 10–30 min ([ANTs users report 10–20 min for `antsRegistrationSyNQuick`](https://sourceforge.net/p/advants/discussion/840261/thread/44eb98df/), and a profiling study calls single-threaded ANTs registration runtimes "prohibitively large", [arXiv:2405.17650](https://arxiv.org/html/2405.17650v1)) | **21–45 h** |
| SyN at 2 mm (8x fewer voxels) | 1.5–5 min | 4–8 h |
| slice extraction + NLM + PNG write | ~9 s (measured, existing pipeline) | 9–15 min (measured: 8.6 min) |

**Answers to the question as asked:**

- **N4 + affine, all 853, is an evening job: ~3–4.5 h wall clock.** Comfortably overnight; probably
  done before you go to bed.
- **N4 + rigid: ~2–2.5 h.**
- **Full nonlinear SyN at native resolution is NOT an overnight job.** 21–45 h wall clock on this
  laptop, single-run, and any interruption (lid close, sleep, thermal throttle — all three are
  documented failure modes in CLAUDE.md) costs hours. It only becomes overnight-feasible by
  downsampling to 2 mm, which discards the fine-scale detail the nonlinear step exists to exploit.
- Mandatory implementation detail: **each worker must pin ITK to one thread**, exactly analogous to
  the existing `cv2.setNumThreads(1)` fix — `os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]="1"`
  **before** `import ants`, inside the worker. ANTs/ITK is internally multithreaded and will
  otherwise oversubscribe 6 cores 10-fold and run slower than serial.
- Disk: 853 registered volumes as compressed float32 NIfTI ≈ 8–15 MB each ≈ **7–13 GB**, plus a new
  PNG tree (~= `processed_v3/`). 160 GB free — fine. Caching the warped volumes is worth it: it lets
  you re-slice at different bands without re-registering.
- **The registration is not the expensive part of the experiment.** Answering "did it help" needs
  5-fold CV per configuration (§6.4), which is 5 trainings, and CLAUDE.md's own timings put that
  well above 3 h. Budget the A/B, not the preprocessing.

---

## 5. Expected benefit, and the honest case against

### 5.1 Evidence FOR

- **The one clean ablation, and it is the most decision-relevant paper found:** Klingenberg, Stark,
  Eitel & Ritter, *MRI Image Registration Considerably Improves CNN-Based Disease Classification*,
  MLCN 2021 — a 3D CNN on ADNI trained on the same data preprocessed three ways (no registration,
  linear, nonlinear). **Both linear and nonlinear registration raised balanced accuracy by ~6–7
  points over no registration, and there was no significant difference between linear and
  nonlinear.**
  <https://link.springer.com/chapter/10.1007/978-3-030-87586-2_5>
- **Wen et al. 2020** (*Medical Image Analysis* 63:101694, the reproducible-evaluation benchmark this
  project already leans on for its leakage numbers) builds both of its pipelines on **N4 + MNI
  registration**: "Minimal" = N4ITK + affine to MNI via ANTs; "Extensive" = bias correction +
  nonlinear (SPM12 Unified Segmentation) + skull-stripping. Note what this means: **N4 and
  registration are in *both* arms.** Nobody in the benchmark literature tests removing them, so the
  field's "mandatory" label reflects convention, not a measured effect size.
  <https://arxiv.org/abs/1904.07773>
- **One paper does report nonlinear > affine:** *Nonlinear registration as an effective preprocessing
  technique for deep learning based classification of disease*, EMBC 2021 — DARTEL nonlinear beat
  affine for AD classification, attributed to reduced spatial variance acting as a regulariser on
  small datasets. It reports the direction, not a large effect size, and it is contradicted by
  Klingenberg's controlled three-arm comparison.
  <https://pubmed.ncbi.nlm.nih.gov/34891933/> · <https://ieeexplore.ieee.org/document/9631044/>
- **Slice correspondence.** For a *2D-slice* pipeline the mechanism is more direct than for 3D:
  registration guarantees slice *i* is the same anatomy in every subject. That is precisely what
  decision 11 was about, and it was worth AUC 0.35→0.67. The vertex anchor is a 1-DOF approximation
  of a 6-DOF problem: it corrects translation along the S-I axis and **nothing else** — no pitch,
  roll or yaw. A 10° pitch (chin-up positioning, common and unmeasured here) tilts the axial cut
  plane through the whole band.

### 5.2 Evidence AGAINST

- **Tinauer et al., *Skull-stripping induces shortcut learning in MRI-based Alzheimer's disease
  classification*** (arXiv:2501.15831, latest v. Oct 2025), 990 matched ADNI T1w scans, 3D CNN,
  LRP relevance maps, McNemar tests with Bonferroni-Holm. Verbatim: "classification accuracy,
  sensitivity, and specificity remained stable across preprocessing conditions. Models trained on
  binarized images preserved performance, indicating minimal reliance on gray-white matter texture.
  Instead, volumetric features — particularly **brain contours introduced through skull-stripping**
  — were consistently used by the models." Conclusion: "a shortcut learning phenomenon, where
  preprocessing artifacts act as potentially unintended cues."
  **Transfer to this question:** the models were reading *shape outlines created by preprocessing*.
  Registration writes a new outline into every image (the template's), and does so via a transform
  whose parameters are estimated from protocol-dependent image content. Whether that outline becomes
  a cue or removes one is an empirical question, not a safe assumption. It also confirms that
  binarised images — no texture at all — classify as well, i.e. **the accuracy in this task class
  lives in shape and volume, exactly the quantities registration modifies.**
  <https://arxiv.org/abs/2501.15831>
- **Glocker, Robinson, Castro, Dou & Konukoglu**, *Machine Learning with Multi-Site Imaging Data*
  (arXiv:1910.04597): 592 age/sex-matched scans from two studies — "even after careful pre-processing
  with state-of-the-art neuroimaging pipelines a classifier can easily distinguish between the
  origin of the data with very high accuracy … current approaches to harmonize data are unable to
  remove scanner-specific bias." **Registration + N4 will not fix this project's open site confound
  (+8.0% on AD vs CN, decision 13).** Do not sell the change on that basis.
  <https://arxiv.org/abs/1910.04597>
- The harmonisation literature makes the reverse argument explicitly: preprocessing steps such as
  skull-stripping and spatial normalisation "may be sensitive to site effects", so "it is desirable
  to minimize the preprocessing needed"
  (<https://www.sciencedirect.com/science/article/pii/S1053811925003647>). Resampling itself changes
  intensities via interpolation, and this project's scans arrive at **7 different in-plane
  resolutions**, so a single resampling step is a *differently-sized* interpolation for different
  cohorts — a new era-correlated smoothing difference of exactly the kind decision 5 was written to
  prevent.
- **This project's own record is a graveyard of preprocessing variants.** `v2mask`, `v2crop`, and
  the removal of the 144px bottleneck (decision 21) all produced two-of-three-architectures
  agreement that reversed on the third. CLAUDE.md's own rule applies to this proposal too: *two out
  of three architectures agreeing at n=93 means nothing.*

### 5.3 The size question: is registration another `v2crop`?

This is the sharpest risk, and the answer is **"partly, and it depends on which DOF you use."** The
distinction matters, so here is the mechanism rather than an intuition:

- **`v2crop` normalised the *brain's own bounding box*.** Atrophy is the shrinkage of that exact
  object. Dividing an image by the quantity you are trying to measure destroys the measurement, and
  CLAUDE.md records precisely that outcome (cross-cohort errors rose 0%→7.1%, i.e. the cue moved,
  while the 4-way number and every AUC stayed inside noise).
- **A 12-DOF affine to MNI normalises the *skull / intracranial cavity*, not the brain.** The scale
  factors are driven by whole-head geometry. That determinant is a well-known quantity: it is the
  **Atlas Scaling Factor**, used in AD morphometry (Buckner et al. 2004, the OASIS ASF) *as* the
  head-size correction, because ICV is a nuisance variable that is fixed in adulthood and does not
  shrink with disease. Atrophy inside a fixed skull shows up as *relative* ventricular and sulcal
  enlargement, and **that survives affine normalisation.**
  <https://www.sciencedirect.com/science/article/abs/pii/S1053811904003271>
  So affine-to-MNI is **not** the same operation as `v2crop`, and the negative `v2crop` result does
  not automatically transfer. It is the theoretically *correct* normalisation for this task.
  Residual risk, stated plainly: if the MI optimiser is partly driven by brain content (it is —
  brain is most of the image's information), the fitted scale absorbs some atrophy, and this
  project has measured that head size carries both the scanner cue and part of the disease signal.
  That is why affine belongs in an A/B arm, not in the default pipeline.
- **Nonlinear (SyN) is the dangerous one, and for a documented reason.** A nonlinear warp to a
  template makes every subject's hippocampus, ventricles and cortical ribbon match the template's
  *shape* — the regional volume differences move out of the image and into the deformation field's
  Jacobian. This is not speculation; it is why VBM requires **Jacobian modulation**: multiplying by
  the Jacobian determinant "corrects for the volume changes that occurred during spatial
  normalization" and is what makes the analysis reflect "local gray matter volume as estimated in
  native space." An unmodulated nonlinear warp discards it.
  <https://www.sciencedirect.com/topics/medicine-and-dentistry/spatial-normalization>
  A 2D-slice CNN pipeline has no way to consume a Jacobian field. So SyN would hand the model
  intensity/texture only — and Tinauer et al. just showed texture is *not* what these models use.
  **SyN is 20–40x the compute to remove the feature the models actually rely on.** Combined with
  Klingenberg's finding that nonlinear buys nothing over linear: reject it.
- **Rigid (6-DOF) has none of this exposure.** No scaling, no shear, no warp — it only rotates and
  translates the head into a canonical orientation. It cannot remove a size cue and it cannot
  remove a size signal. It buys tilt correction and true slice correspondence, which is the part of
  "registration" this pipeline is actually missing. **This is the recommended default.**

### 5.4 New measurement made for this report: there is a *geometry* confound left in `v3go2`

While estimating volume sizes I audited the DICOM acquisition geometry of all 853 subjects
(header-only, no pixels). The reconstructed axial images are anisotropic in a way that varies by
cohort **and by class**, because `_build_volume()` stacks slices with `np.stack` and never applies
the 1.2 mm inter-slice spacing, and then `cv2.resize(..., (224,224))` squashes a non-square physical
field into a square:

| | ADNI1 (n=235) | ADNI-GO/2 (n=618) |
|---|---|---|
| L-R field of view (mm) | 194.6 ± 6.1 | 209.4 ± 3.1 |
| mm per output pixel, L-R | 0.869 ± 0.027 | 0.935 ± 0.014 |
| mm per output pixel, A-P | 1.078 ± 0.020 | 1.108 ± 0.043 |

So **every image is ~20–25% anisotropic** (structures stretched L-R relative to A-P), and the exact
factor differs per subject. Physical aspect ratio (L-R mm / A-P mm) by class:

| era | class | aspect | n |
|---|---|---|---|
| ADNI1 | AD | 0.807 ± 0.026 | 108 |
| ADNI1 | CN | 0.804 ± 0.024 | 127 |
| GO/2 | AD | 0.835 ± 0.046 | 108 |
| GO/2 | CN | 0.832 ± 0.041 | 158 |
| GO/2 | **EMCI** | **0.854 ± 0.038** | 227 |
| GO/2 | **LMCI** | **0.856 ± 0.037** | 125 |

Trained on **acquisition metadata only — no pixels** (RandomForest, subject-level stratified 5-fold):

| task | geometry-only accuracy | majority baseline |
|---|---|---|
| **`v3go2` 4-way (618 subjects)** | **40.9% ± 3.0** | 36.7% |
| `v3go2` {AD,CN} vs {EMCI,LMCI} | **71.5%** | 57.0% |
| aspect ratio alone, {EMCI,LMCI} vs {AD,CN} | AUC **0.643** | 0.5 |
| **AD vs CN, 501 subjects, both eras** | **52.3%** (below baseline) | 56.9% |

Two conclusions:

1. **The AD-vs-CN headline is clean on geometry** (52.3% < 56.9% baseline) — the 0.906 AUC result is
   not explained by this. Good news, and consistent with decision 13.
2. **The primary 4-way task is not.** Acquisition geometry alone reaches 40.9% against a 36.7%
   baseline. The measured 4-way result is 43.0% [39.2, 47.0] (decision 25). **A metadata-only
   classifier gets two-thirds of the way from the baseline to the model's score**, and unlike era or
   native resolution (both audited at +0.0%), this cue is *rendered into the pixels* as a
   per-subject anisotropic stretch. The reason it tracks class is structural: EMCI/LMCI come from
   the original 439-subject collection while most GO/2 AD/CN subjects came from the
   AlzheimerAdditional download, with different series families and fields of view.

**This is the cheapest and best-supported reason to touch preprocessing at all, and it needs no
template and no new dependency** — resample each volume to isotropic voxels at a *fixed mm-per-pixel
scale* with a head-centroid-centred crop, instead of resizing a variable physical field to a fixed
pixel count. Note what that does and does not normalise: it removes FOV/matrix/anisotropy (protocol)
while **preserving head size in millimetres** (anatomy). That is the separation CLAUDE.md says no
preprocessing variant has ever achieved — because every previous attempt (`v2crop`) normalised the
brain instead of the sampling grid.

### 5.5 Can the A/B even resolve the effect?

From decision 25: 5-fold CV over 618 subjects gives a **±3.9 point** interval; a single 93-subject
split gives **±10**. So:

- Klingenberg's +6–7 points would be detectable **by CV only**.
- The residual increment actually available here is smaller than theirs, because the vertex anchor
  already does part of the job. A realistic prior is **+0 to +4 points on the 4-way task**, i.e.
  **at or below the resolution of the best test this project can run.**
- A single-split A/B **cannot** answer this question and must not be used. Any run of
  `train_any.py <arch> v4...` against `v3go2` on the 93-subject split will produce a number that is
  noise, and this project has twice recorded believing such a number (the "masking hurts" episode,
  decision 21's bottleneck reversal).

**Honest expected outcome: the most likely result of a correctly-powered A/B is "no measurable
difference."** The value in doing it is then (a) removing the measured geometry cue in §5.4, which
is a validity fix rather than an accuracy fix, and (b) a methods section that matches what
reviewers expect. Both are real; neither is an accuracy gain. If the goal is a better number, the
evidence points elsewhere — CLAUDE.md decision 19's list (longitudinal MRI, hippocampal ROI
patches, slice attention) and more LMCI data.

---

## 6. Implementation plan

### 6.1 New file (design only — NOT created): `scripts/register_mni.py`

```python
"""N4 + template-space reslicing for the v4 dataset. Writes data/processed_v4_{mode}/.

Never overwrites processed_v3/ or any manifest_v3*.csv. Splits are inherited verbatim from
subject_manifest_v3.csv so the v3 vs v4 comparison moves zero subjects.

Run inside the SEPARATE `preproc` env: antspyx pins numpy<2.4.0 and would downgrade numpy
under PyTorch 2.6 in `ml`.
"""

# ---- module constants -------------------------------------------------------------------
MODES        = ("geom", "rigid", "affine", "syn")  # 'geom' = resample only, no template
TEMPLATE_MM  = 1.0
MNI_Z_RANGE  = (30.0, -14.0)   # superior -> inferior, millimetres in MNI z.
                               # The current band is 48-92mm below the vertex; the MNI vertex
                               # sits near z=+78, so 48-92mm below it maps to z=+30 .. -14,
                               # which spans the body of the lateral ventricles down to the
                               # hippocampal/medial-temporal level. Same anatomy, absolute frame.
N_SLICES     = 32              # == data_prep.N_SLICES_PER_SUBJECT
OUT_SIZE     = 224             # == data_prep.OUT_SIZE
BOTTLENECK   = 144             # == data_prep.BOTTLENECK_SIZE; keep, decision 21 found no
                               # evidence either way and this is not the experiment to change it in

# ---- I/O --------------------------------------------------------------------------------
def read_dicom_volume(dicom_dir: str) -> "SimpleITK.Image":
    """Read a DICOM series into a correctly SPACED and ORIENTED 3D image.

    Uses sitk.ImageSeriesReader + GDCMSeriesFileNames, which sorts by ImagePositionPatient
    rather than InstanceNumber and sets spacing/direction/origin from the headers. This is the
    fix for the geometry gap in data_prep._build_volume(): all 853 series carry
    SpacingBetweenSlices=1.2mm and in-plane spacing 0.938-1.354mm, and np.stack() discards both.
    """

def to_ants(img: "SimpleITK.Image") -> "ants.ANTsImage":
    """Hand a SimpleITK image to ANTs preserving spacing/origin/direction (no re-read)."""

# ---- preprocessing steps ----------------------------------------------------------------
def n4_correct(vol: "ants.ANTsImage", shrink_factor: int = 4,
               iters=(50, 50, 50, 50), mask: "ants.ANTsImage | None" = None
               ) -> "ants.ANTsImage":
    """N4 bias field correction on the 3D volume. Must run BEFORE registration (MI is
    sensitive to smooth intensity gradients) and BEFORE any per-slice percentile scaling."""

def head_mask(vol: "ants.ANTsImage") -> "ants.ANTsImage":
    """Coarse head/brain mask: threshold low -> binary fill -> erode past the skull.
    Ports the validated logic in scripts/brain_mask.py. NOT Otsu (CLAUDE.md: Otsu deletes the
    cortical ribbon, ~21% retention). Used as `moving_mask` when the template is brain-only,
    and as the N4 weight mask."""

def load_template(whole_head: bool = True, resolution_mm: float = TEMPLATE_MM
                  ) -> "ants.ANTsImage":
    """whole_head=True  -> nilearn.datasets.fetch_icbm152_2009()['t1']  (one-time download)
       whole_head=False -> nilearn.datasets.struct.MNI152_FILE_PATH     (bundled, offline,
                           SKULL-STRIPPED -> caller must pass moving_mask)"""

def register_to_template(vol: "ants.ANTsImage", template: "ants.ANTsImage",
                         mode: str = "rigid", moving_mask=None,
                         cache_path: "pathlib.Path | None" = None) -> dict:
    """mode -> ants.registration type_of_transform:
         'rigid'  -> 'Rigid'    (6 DOF; no scaling, cannot normalise head size)
         'affine' -> 'Affine'   (12 DOF; DOES normalise ICV -- experimental arm, see report 5.3)
         'syn'    -> 'SyNRA'    (NOT RECOMMENDED: 21-45h for 853, and removes regional volume)
       Returns {'warped', 'fwdtransforms', 'metric', 'scale_det'} where
         metric    = final Mattes MI against the template (QC),
         scale_det = determinant of the linear part = Atlas-Scaling-Factor-like head-size
                     estimate. LOG IT PER SUBJECT: if scale_det differs systematically by class
                     you have just moved the atrophy signal into a discarded scalar."""

def resample_isotropic(vol: "ants.ANTsImage", mm_per_px: float = 1.0,
                       fov_mm: float = 224.0) -> "ants.ANTsImage":
    """mode='geom': no template at all. Resample to isotropic mm_per_px and crop/pad to a
    fixed fov_mm cube centred on the head centroid. Removes FOV/matrix/anisotropy (protocol,
    the cue measured in report 5.4) while PRESERVING head size in mm (anatomy)."""

# ---- slicing ----------------------------------------------------------------------------
def extract_template_slices(vol_in_space: "ants.ANTsImage",
                            z_range=MNI_Z_RANGE, n_slices: int = N_SLICES,
                            out_size: int = OUT_SIZE, bottleneck: int | None = BOTTLENECK,
                            denoise: bool = True) -> list["np.ndarray"]:
    """Axial slices at FIXED PHYSICAL z coordinates (no vertex detection needed once the head
    is in template space). Then, unchanged from data_prep.extract_slices():
    percentile-clip -> uint8 -> bottleneck resize -> OUT_SIZE -> fastNlMeansDenoising(h=8).
    Keeping the tail identical is what makes v3-vs-v4 a one-variable experiment."""

# ---- driver -----------------------------------------------------------------------------
def process_one(task: tuple) -> tuple:
    """Worker. FIRST LINE, before `import ants`:
         os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
       plus cv2.setNumThreads(1). Same oversubscription trap as reextract_v3.py, different
       library. Resumable: skip if all 32 PNGs exist. Returns (subject_id, n, metric, err) and
       never raises -- one bad scan must not kill the pool."""

def main(mode: str = "rigid", n_workers: int = 10, cache_volumes: bool = True) -> None:
    """ProcessPoolExecutor over subject_manifest_v3.csv, mirroring reextract_v3.py exactly:
    per-subject markers, tqdm, error list, then build manifest_v4_{mode}.csv /
    _go2 / _adcn from what is ACTUALLY on disk. Writes reports/v4_{mode}_qc.csv with
    (subject_id, class, era, metric, scale_det, n_slices) for the QC gate in 6.3."""
```

### 6.2 Order of operations relative to the existing `extract_slices()`

Current (`src/data_prep.py`):

```
_build_volume (np.stack, spacing ignored)
  -> _row_spacing_mm + _find_vertex_row      -> band = vertex + 48..92mm
  -> volume[:, r, :]                          (anisotropic, 170x256-ish)
  -> _normalize_to_uint8  (per-slice 0.5/99.5 percentile)
  -> resize 144 -> resize 224                 (squashes non-square physical field)
  -> fastNlMeansDenoising(h=8)
```

Proposed v4 — three steps replaced, the whole tail kept byte-identical:

```
read_dicom_volume            (SimpleITK; REAL spacing + direction)      <- replaces _build_volume
  -> n4_correct              (3D, before anything intensity-based)      <- NEW
  -> register_to_template    (rigid | affine)  OR  resample_isotropic   <- replaces _find_vertex_row
                                                                            + _row_spacing_mm
  -> slice at fixed MNI z (or fixed mm offsets)                         <- replaces the band arithmetic
  -> _normalize_to_uint8        (UNCHANGED)
  -> bottleneck 144 -> 224      (UNCHANGED; now an isotropic->square resize, not a squash)
  -> fastNlMeansDenoising(h=8)  (UNCHANGED)
```

Two ordering rules that matter:

1. **N4 before registration, and before per-slice normalisation.** N4 estimates a smooth
   multiplicative field over the whole volume; running it after 2D percentile scaling per slice is
   meaningless, and running it after registration wastes the interpolation.
2. **Everything after slicing stays exactly as it is.** Changing NLM or the bottleneck at the same
   time would recreate the 2.5D mistake recorded in CLAUDE.md (two variables changed, conclusion
   unusable).

### 6.3 Mandatory QC gate before training on v4

Registration fails **silently**; a flipped or neck-anchored affine yields a plausible-looking image
of the wrong anatomy. Before any training run:

- `reports/v4_{mode}_qc.csv` — final MI per subject. Flag the worst 5% and eyeball them.
- **`scale_det` by class and era.** If the affine scale factor separates AD from CN, affine
  registration has moved the disease signal into a scalar you are throwing away. This is the
  single most informative number in the whole experiment and it is free.
- Re-run `scripts/qc_slice_anatomy.py` on `processed_v4_*` — it is what caught decision 11.
- Re-run the geometry audit from §5.4 on the v4 metadata: geometry-only 4-way accuracy should fall
  from 40.9% toward the 36.7% baseline. **If it does not, the change did not do the one thing it is
  best justified by.**
- Sanity-check 20 random subjects as a montage before processing all 853 (CLAUDE.md: the user wants
  a cheap reversible check before big runs).

### 6.4 A/B protocol that cannot overwrite `data/processed_v3/`

Isolation:

- New output trees: `data/processed_v4_geom/`, `processed_v4_rigid/`, `processed_v4_affine/`.
- New manifests: `manifest_v4_{mode}.csv`, `_go2`, `_adcn` — built by the same code path as
  `reextract_v3.py`, from files actually on disk.
- **Splits inherited verbatim** from `subject_manifest_v3.csv` by joining on `subject_id`; assert
  0 subjects moved (`reextract_v3_hires.py` already establishes this pattern and printed "0 split
  mismatches").
- `scripts/register_mni.py` writes nothing outside `data/processed_v4_*`, `data/manifest_v4_*` and
  `reports/v4_*`.
- Small additive changes needed in existing scripts (append-only, no behaviour change to v3 keys):
  - `train_any.py`: add `"v4geo","v4rigid","v4affine"` (all-853) and `"v4go2*"` keys to its
    `MANIFESTS` dict; epoch/patience entries alongside them.
  - `train_binary_adni1.py`: add `"v4adcn*"` keys — **and fix the checkpoint-name bug first**
    (CLAUDE.md: it writes `{arch}_ADvsCN.pt` with no manifest key and already destroyed the v2
    binary weights). Adding v4 arms without that fix will destroy the v3 AD-vs-CN checkpoint that
    decision 22's in-domain initialisation depends on.
  - `cross_validate.py`: it hardcodes `manifest_v3.csv` / `_go2` / `_adcn` at lines 122/154/198.
    Add an optional 5th CLI arg (or `MANIFEST_TAG` env var) that substitutes the tag; default
    unchanged.

Experiment schedule (in priority order, stopping early if a gate fails):

| # | run | why |
|---|---|---|
| 1 | QC gate §6.3 on `v4geom` | cheapest arm, and the one with a measured target |
| 2 | `cross_validate.py custom_cnn random 5` on `v4geom` vs the recorded `v3go2` 43.0% [39.2, 47.0] | the **only** 4-way test with a tight enough interval (±3.9 pts) |
| 3 | `train_binary_adni1.py <arch> v4geomadcn`, all three archs, AUC + CI | headline metric must not regress |
| 4 | repeat 2–3 for `v4rigid` | adds tilt correction on top |
| 5 | `v4affine` **as an ablation**, reported next to `scale_det`-by-class | tests the size question directly |
| — | `v4syn` | **not scheduled** (§5.3) |
| 6 | `train_cross_era.py` on the winner | registration's best theoretical case is cross-scanner generalisation (0.68–0.79 today vs 0.906 within-cohort). If it helps anywhere, it helps here. |

Decision rule, fixed in advance: **accept the v4 pipeline only if (a) the geometry-only audit drops
toward baseline AND (b) neither the 4-way CV interval nor the AD-vs-CN AUC regresses.** An accuracy
*gain* would be a bonus, not the criterion — with ±3.9 points of resolution and a realistic +0–4
point effect, demanding a gain would mean picking a preprocessing pipeline out of noise, which is
the mistake CLAUDE.md records three separate times.

No-leakage note: registering each subject to a **fixed external template** is a per-subject
deterministic operation and introduces no leakage. **Do not** build a study-specific template
(`ants.build_template`) — that fits to test subjects' anatomy, the same class of contamination the
per-fold pretraining in `cross_validate.py` exists to avoid.

---

## 7. Summary answers

1. **Installed:** none of it. No `antspyx`, `SimpleITK`, `nibabel`, `dipy`, `nilearn`, `deepbet` or
   `HD-BET`; also no `scikit-image`. Present: numpy 2.4.4, scipy 1.17.1, opencv 5.0.0.93,
   pydicom 3.0.2, torch 2.6.0+cu124, sklearn 1.9.0, pandas 3.0.3.
2. **Library:** ANTsPy 0.6.3 for both N4 and registration (native cp311 win_amd64 wheels exist);
   SimpleITK 2.5.6 for DICOM series reading. **Install into a new `preproc` env** — antspyx pins
   `numpy<2.4.0` and would downgrade numpy under the running PyTorch stack.
3. **Template:** `nilearn` bundles `mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz` inside the
   wheel (offline, exact path in §3) but it is **skull-stripped**; for a whole-head fixed image use
   `nilearn.datasets.fetch_icbm152_2009()['t1']` (one-time download). `ants.get_ants_data('mni')`
   also downloads. SimpleITK ships nothing.
4. **Runtime, 853 subjects, 6C/12T Ryzen 5 7640HS, 10 pinned workers:** N4 ~1–1.5 h; rigid
   ~0.6–1.2 h; affine ~1.5–3 h; **SyN 21–45 h — not an overnight job.** Nonlinear only fits
   overnight if downsampled to 2 mm (4–8 h), which defeats its purpose. **Affine-only, yes;
   nonlinear, no.** Must set `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1` per worker.
5. **Benefit/risk:** the one controlled ablation (Klingenberg 2021) reports +6–7 points for
   registration vs *none*, with **linear = nonlinear** — but its baseline is unanchored volumes,
   whereas this pipeline already has the vertex anchor. Against it: Tinauer 2025 shows these models
   read preprocessing-induced *shape*, not texture; Glocker 2019 shows scanner identity survives
   state-of-the-art preprocessing; the VBM modulation requirement shows unmodulated nonlinear
   warping deletes exactly the regional volume differences that are the AD signal. Affine is
   **not** the same operation as `v2crop` (it normalises the skull/ICV, which is a nuisance
   variable, not the brain, which is the signal) — but it does normalise size, and this project has
   measured that size carries both cue and signal, so it belongs in an A/B arm. Rigid carries
   neither exposure. **A correctly powered A/B will most likely show no measurable accuracy change**
   (available effect ~+0–4 points against a ±3.9-point CV resolution).
6. **Plan:** §6. Full signatures for `scripts/register_mni.py`, the exact three-step swap inside
   `extract_slices()` with the whole normalise/bottleneck/denoise tail unchanged, a mandatory
   registration-failure QC gate including `scale_det`-by-class, and an isolated `processed_v4_*` /
   `manifest_v4_*` A/B that inherits the v3 splits and is judged on 5-fold CV, never on the
   93-subject split.

**Most important single finding of this report is not about registration.** Acquisition geometry
alone — no pixels — classifies the primary 4-way `v3go2` task at **40.9% vs a 36.7% baseline**, and
separates {AD,CN} from {EMCI,LMCI} at **71.5% vs 57.0%**, because the 1.2 mm inter-slice spacing is
discarded and a variable physical field of view is squashed into a fixed 224x224 grid. Fixing that
costs nothing, needs no template and no new dependency, and normalises the sampling grid while
leaving head size intact. Do that before deciding whether MNI registration is worth 3 hours and a
new environment.

---

## Sources

- Klingenberg, Stark, Eitel, Ritter. *MRI Image Registration Considerably Improves CNN-Based Disease Classification.* MLCN 2021. <https://link.springer.com/chapter/10.1007/978-3-030-87586-2_5>
- Tinauer, Sackl, Stollberger, Schmidt, Ropele, Langkammer. *Skull-stripping induces shortcut learning in MRI-based Alzheimer's disease classification.* arXiv:2501.15831. <https://arxiv.org/abs/2501.15831>
- Wen, Thibeau-Sutre, Samper-González, Routier, Bottani, Durrleman, Burgos, Colliot. *Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation.* Medical Image Analysis 63:101694, 2020. <https://arxiv.org/abs/1904.07773> · <https://pubmed.ncbi.nlm.nih.gov/32417716/>
- *Nonlinear registration as an effective preprocessing technique for deep learning based classification of disease.* EMBC 2021. <https://pubmed.ncbi.nlm.nih.gov/34891933/> · <https://ieeexplore.ieee.org/document/9631044/>
- Glocker, Robinson, Castro, Dou, Konukoglu. *Machine Learning with Multi-Site Imaging Data: An Empirical Study on the Impact of Scanner Effects.* arXiv:1910.04597. <https://arxiv.org/abs/1910.04597>
- Buckner et al. *A unified approach for morphometric and functional data analysis … automated atlas-based head size normalization* (Atlas Scaling Factor). NeuroImage 2004. <https://www.sciencedirect.com/science/article/abs/pii/S1053811904003271>
- Spatial normalization / Jacobian modulation overview. <https://www.sciencedirect.com/topics/medicine-and-dentistry/spatial-normalization>
- PhyCHarm (harmonisation; "desirable to minimize the preprocessing needed"). <https://www.sciencedirect.com/science/article/pii/S1053811925003647>
- *An Analysis of Performance Bottlenecks in MRI Pre-Processing* (ANTs registration runtime profiling). <https://arxiv.org/html/2405.17650v1>
- ANTs users list, `antsRegistrationSyN` runtime reports. <https://sourceforge.net/p/advants/discussion/840261/thread/44eb98df/>
- ANTsPy installation notes and Windows wheels. <https://github.com/ANTsX/ANTsPy/wiki/Installing-ANTsPy> · <https://pypi.org/project/antspyx/>
- ANTsPy source: `registration.py`, `bias_correction.py`, `get_ants_data.py`. <https://github.com/ANTsX/ANTsPy>
- nilearn source `nilearn/datasets/struct.py` (bundled MNI152 paths) and template docs. <https://nilearn.github.io/dev/modules/generated/nilearn.datasets.load_mni152_template.html> · <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_icbm152_2009.html>
- SimpleITK N4 bias field correction docs. <https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html>
- deepbet / deepmriprep (fast learned alternative). <https://arxiv.org/pdf/2308.07003> · <https://github.com/wwu-mmll/deepmriprep>
