"""
Second driver: de-confounding experiments + the leakage proof, then re-scoring.

Runs after run_remaining.py finishes (the GPU only fits one job at a time). Same
restart-safe design: every job records a result file, and a job whose result already
exists is skipped, so this can be relaunched freely after an interruption.

The headline question these jobs answer is NOT "did 4-way accuracy go up". It is
"did the model stop cheating": era accuracy should fall from ~100% toward 50% once the
skull and scalp are gone, because that is where much of the scanner-protocol cue lives.
4-way accuracy is expected to FALL at the same time, since roughly a third of it was the
free cohort split. A lower, honest number is the goal here.
"""

import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SCRIPTS = os.path.join(PROJ, "scripts")
R = os.path.join(PROJ, "reports")
LOGS = os.path.join(R, "run_logs")
os.makedirs(LOGS, exist_ok=True)

PREV_DRIVER_LOG = os.path.join(LOGS, "driver.log")
PREV_DONE_MARKER = "ALL REMAINING JOBS COMPLETE"

# Ordered by expected value. braincrop (mask + tight crop + rescale) comes first because
# Grad-CAM showed the unmasked model keying on the empty LEFT/RIGHT IMAGE MARGINS -- a
# field-of-view artifact of ADNI1 being 192x192 and ADNI-GO/2 being 256x256. Masking alone
# zeroes that background but leaves the head-size difference; cropping normalises both.
# The masked-only run is kept afterwards as an ablation, to show which half did the work.
JOBS = [
    ("mobilenetv2_braincrop",   [PY, "-u", f"{SCRIPTS}/train_any.py", "mobilenetv2", "braincrop2d"],
     f"{R}/mobilenetv2_braincrop2d_result.json"),
    ("gradcam_braincrop",       [PY, "-u", f"{SCRIPTS}/gradcam.py", "mobilenetv2_braincrop2d", "3"],
     f"{R}/gradcam_mobilenetv2_braincrop2d.json"),
    # one model, two test sets: leaked subjects vs genuinely new subjects
    ("leakage_proof",           [PY, "-u", f"{SCRIPTS}/leakage_proof.py", "mobilenetv2"],
     f"{R}/mobilenetv2_leakage_proof.json"),
    # ablation: masking without the crop, to attribute the effect
    ("mobilenetv2_masked",      [PY, "-u", f"{SCRIPTS}/train_any.py", "mobilenetv2", "masked2d"],
     f"{R}/mobilenetv2_masked2d_result.json"),
    ("custom_cnn_braincrop",    [PY, "-u", f"{SCRIPTS}/train_any.py", "custom_cnn", "braincrop2d"],
     f"{R}/custom_cnn_braincrop2d_result.json"),
    ("efficientnet_b0_braincrop", [PY, "-u", f"{SCRIPTS}/train_any.py", "efficientnet_b0", "braincrop2d"],
     f"{R}/efficientnet_b0_braincrop2d_result.json"),
    # refresh the consolidated table and the era decomposition
    ("final_eval",              [PY, "-u", f"{SCRIPTS}/final_eval.py"], None),
    ("within_era",              [PY, "-u", f"{SCRIPTS}/within_era_analysis.py"], None),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prev_driver_done():
    try:
        with open(PREV_DRIVER_LOG, encoding="utf-8") as fh:
            return PREV_DONE_MARKER in fh.read()
    except OSError:
        return False


def main():
    log(f"phase2 driver pid {os.getpid()}")
    waited = 0
    while not prev_driver_done() and waited < 6 * 3600:
        time.sleep(60)
        waited += 60
        if waited % 1800 == 0:
            log(f"still waiting for run_remaining.py ({waited // 60} min)")
    log(f"proceeding after {waited // 60} min (previous driver finished: {prev_driver_done()})")

    for label, argv, marker in JOBS:
        if marker and os.path.exists(marker):
            log(f"SKIP  {label}")
            continue
        log(f"START {label}")
        t0 = time.time()
        logpath = os.path.join(LOGS, f"{label}.log")
        with open(logpath, "w", encoding="utf-8") as fh:
            code = subprocess.call(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJ)
        mins = (time.time() - t0) / 60
        if code == 0:
            log(f"OK    {label} in {mins:.1f} min")
        else:
            log(f"FAIL  {label} exit={code} after {mins:.1f} min")
            try:
                with open(logpath, encoding="utf-8") as fh:
                    for line in fh.read().strip().splitlines()[-15:]:
                        log(f"      | {line}")
            except OSError:
                pass

    log("PHASE2 COMPLETE")


if __name__ == "__main__":
    main()
