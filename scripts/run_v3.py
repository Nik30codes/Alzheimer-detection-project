"""
Re-runs the AD-vs-CN experiments with validation-chosen decision thresholds.

Why: on v2 slices both architectures reached ROC AUC ~0.67 -- real ranking ability -- but
predicted CN for essentially every subject at the default 0.5 threshold, scoring exactly
the majority baseline. The probabilities were informative but compressed below 0.5. The
threshold is now chosen on the VALIDATION split (maximising Youden's J) and applied to
test, which turns that ranking ability into usable predictions without touching test data.

Also adds Hanley-McNeil confidence intervals on AUC, because at 26 test subjects a bare
point estimate overstates what the data can support.

Runs after run_v2.py finishes; the GPU fits one job at a time.
"""

import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
S = os.path.join(PROJ, "scripts")
LOGS = os.path.join(PROJ, "reports", "run_logs")
PREV_LOG = os.path.join(LOGS, "v2driver.log")

# No done-markers: these deliberately overwrite the earlier threshold-0.5 results.
JOBS = [
    ("ADvsCN_mnet_v2_thr",   [PY, "-u", f"{S}/train_binary_adni1.py", "mobilenetv2", "v2"]),
    ("ADvsCN_cnn_v2_thr",    [PY, "-u", f"{S}/train_binary_adni1.py", "custom_cnn", "v2"]),
    ("ADvsCN_effnet_v2_thr", [PY, "-u", f"{S}/train_binary_adni1.py", "efficientnet_b0", "v2"]),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"v3 driver pid {os.getpid()}")
    waited = 0
    while waited < 4 * 3600:
        try:
            with open(PREV_LOG, encoding="utf-8") as fh:
                if "V2 RUN COMPLETE" in fh.read():
                    break
        except OSError:
            pass
        time.sleep(60)
        waited += 60
    log(f"starting after {waited // 60} min wait")

    for label, argv in JOBS:
        log(f"START {label}")
        t0 = time.time()
        lp = os.path.join(LOGS, f"{label}.log")
        with open(lp, "w", encoding="utf-8") as fh:
            code = subprocess.call(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJ)
        mins = (time.time() - t0) / 60
        log(f"{'OK   ' if code == 0 else 'FAIL '} {label} in {mins:.1f} min")
        if code != 0:
            try:
                with open(lp, encoding="utf-8") as fh:
                    for line in fh.read().strip().splitlines()[-12:]:
                        log(f"      | {line}")
            except OSError:
                pass
    log("V3 RUN COMPLETE")


if __name__ == "__main__":
    main()
