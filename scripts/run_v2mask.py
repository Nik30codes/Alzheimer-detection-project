"""
Tests the "mask but do not crop" hypothesis on the v2 slices.

The tension this resolves (or fails to):
  * v2 + crop lowered Grad-CAM attention outside the brain from 75% to 45% and produced
    the first non-zero cross-cohort error rate -- the scanner confound finally weakened.
  * But AD-vs-CN AUC fell from 0.667 to 0.576 under the same crop.

The likely reason is that brain_crop rescales every brain to fill the frame, which
normalises away absolute brain size -- and global atrophy is a real Alzheimer's marker.
So the crop removes a confound and a genuine signal at the same time.

v2mask zeroes skull, scalp and background but leaves framing alone, so brain volume
survives. If the hypothesis is right, AD-vs-CN AUC should recover toward 0.67 while
Grad-CAM attention stays better than the 75% of unmasked v2.
"""

import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
S = os.path.join(PROJ, "scripts")
R = os.path.join(PROJ, "reports")
LOGS = os.path.join(R, "run_logs")
os.makedirs(LOGS, exist_ok=True)

JOBS = [
    ("v2mask_data",        [PY, "-u", f"{S}/brain_mask.py", "runmask_v2"],
     f"{PROJ}/data/manifest_v2_masked.csv"),
    # 4-way + Grad-CAM: did masking alone weaken the cohort shortcut?
    ("mobilenetv2_v2mask", [PY, "-u", f"{S}/train_any.py", "mobilenetv2", "v2mask"],
     f"{R}/mobilenetv2_v2mask_result.json"),
    ("gradcam_v2mask",     [PY, "-u", f"{S}/gradcam.py", "mobilenetv2_v2mask", "3"],
     f"{R}/gradcam_mobilenetv2_v2mask.json"),
    # the decisive comparison: AD vs CN, all three architectures
    ("ADvsCN_cnn_v2mask",  [PY, "-u", f"{S}/train_binary_adni1.py", "custom_cnn", "v2mask"],
     f"{R}/custom_cnn_ADvsCN_v2mask_result.json"),
    ("ADvsCN_mnet_v2mask", [PY, "-u", f"{S}/train_binary_adni1.py", "mobilenetv2", "v2mask"],
     f"{R}/mobilenetv2_ADvsCN_v2mask_result.json"),
    ("ADvsCN_eff_v2mask",  [PY, "-u", f"{S}/train_binary_adni1.py", "efficientnet_b0", "v2mask"],
     f"{R}/efficientnet_b0_ADvsCN_v2mask_result.json"),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"v2mask driver pid {os.getpid()}, {len(JOBS)} jobs")
    for label, argv, marker in JOBS:
        if marker and os.path.exists(marker):
            log(f"SKIP  {label}")
            continue
        log(f"START {label}")
        t0 = time.time()
        lp = os.path.join(LOGS, f"{label}.log")
        with open(lp, "w", encoding="utf-8") as fh:
            code = subprocess.call(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJ)
        mins = (time.time() - t0) / 60
        if code == 0:
            log(f"OK    {label} in {mins:.1f} min")
        else:
            log(f"FAIL  {label} exit={code} after {mins:.1f} min")
            try:
                with open(lp, encoding="utf-8") as fh:
                    for line in fh.read().strip().splitlines()[-12:]:
                        log(f"      | {line}")
            except OSError:
                pass
    log("V2MASK RUN COMPLETE")


if __name__ == "__main__":
    main()
