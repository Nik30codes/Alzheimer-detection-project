"""
Re-runs the experiments that matter on the v2 (millimetre-anchored) slices.

Everything before this ran on slices whose axial level drifted between subjects and
systematically between cohorts, sometimes missing the hippocampus entirely. Those
numbers describe a broken dataset. This driver rebuilds the picture on data where slice
index actually corresponds to a consistent anatomical level.

Order is chosen so the most decisive results land first:
  1. v2 brain-crop dataset (CPU) so the cropped variants can run later
  2. MobileNetV2 4-way on v2 -- comparable to the 60.6% v1 number
  3. Grad-CAM on it -- is attention finally inside the brain?
  4. AD vs CN on v2, both architectures -- the clinically meaningful metric, and the
     one most likely to have been sabotaged by a missing hippocampus
  5. the same two things on v2 + brain crop
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
    ("v2_braincrop_data",  [PY, "-u", f"{S}/brain_mask.py", "runcrop_v2"],
     f"{PROJ}/data/manifest_v2_braincrop.csv"),

    ("mobilenetv2_v2",     [PY, "-u", f"{S}/train_any.py", "mobilenetv2", "v2"],
     f"{R}/mobilenetv2_v2_result.json"),
    ("gradcam_v2",         [PY, "-u", f"{S}/gradcam.py", "mobilenetv2_v2", "3"],
     f"{R}/gradcam_mobilenetv2_v2.json"),

    ("ADvsCN_mnet_v2",     [PY, "-u", f"{S}/train_binary_adni1.py", "mobilenetv2", "v2"],
     f"{R}/mobilenetv2_ADvsCN_v2_result.json"),
    ("ADvsCN_cnn_v2",      [PY, "-u", f"{S}/train_binary_adni1.py", "custom_cnn", "v2"],
     f"{R}/custom_cnn_ADvsCN_v2_result.json"),

    ("custom_cnn_v2",      [PY, "-u", f"{S}/train_any.py", "custom_cnn", "v2"],
     f"{R}/custom_cnn_v2_result.json"),
    ("efficientnet_b0_v2", [PY, "-u", f"{S}/train_any.py", "efficientnet_b0", "v2"],
     f"{R}/efficientnet_b0_v2_result.json"),

    ("mobilenetv2_v2crop", [PY, "-u", f"{S}/train_any.py", "mobilenetv2", "v2crop"],
     f"{R}/mobilenetv2_v2crop_result.json"),
    ("gradcam_v2crop",     [PY, "-u", f"{S}/gradcam.py", "mobilenetv2_v2crop", "3"],
     f"{R}/gradcam_mobilenetv2_v2crop.json"),
    ("ADvsCN_mnet_v2crop", [PY, "-u", f"{S}/train_binary_adni1.py", "mobilenetv2", "v2crop"],
     f"{R}/mobilenetv2_ADvsCN_v2crop_result.json"),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"v2 driver pid {os.getpid()}, {len(JOBS)} jobs")
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
    log("V2 RUN COMPLETE")


if __name__ == "__main__":
    main()
