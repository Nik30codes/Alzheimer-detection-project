"""
Single-process driver for everything left in Phase D.

Written as one long-lived Python process invoking each job as a subprocess, rather than
a chain of background shell scripts: the shell chains proved fragile (all three were
terminated mid-queue, orphaning the job that happened to be running). One process with
one sequential loop has no parent/child chain to lose, and each job is isolated so a
crash in one does not take down the rest.

The GPU is the bottleneck, so everything runs strictly sequentially.
"""

import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SCRIPTS = os.path.join(PROJ, "scripts")
LOGS = os.path.join(PROJ, "reports", "run_logs")
os.makedirs(LOGS, exist_ok=True)

R = os.path.join(PROJ, "reports")

# (label, argv, result-file-that-proves-it-is-done). Order matters: the training jobs all
# precede the final scoring, so metrics.json is written from complete numbers.
# `done_marker` makes the driver restart-safe -- background processes here have been
# reaped mid-queue more than once, and re-running a 20-minute job that already succeeded
# is pure waste. Pass None to always run.
JOBS = [
    # leaky runs. The first two originally used a 15-epoch budget: custom_cnn collapsed to
    # 35% (early stopping restored an epoch-1 checkpoint) and mobilenetv2 was still
    # improving when it hit the cap at 80.2%. Their old results are archived as
    # BAD_shortbudget_*. efficientnet_b0's run hung when its worker processes were killed.
    ("custom_cnn_leaky",   [PY, "-u", f"{SCRIPTS}/train_any.py", "custom_cnn", "leaky"],
     f"{R}/custom_cnn_leaky_result.json"),
    ("mobilenetv2_leaky",  [PY, "-u", f"{SCRIPTS}/train_any.py", "mobilenetv2", "leaky"],
     f"{R}/mobilenetv2_leaky_result.json"),
    ("efficientnet_b0_leaky", [PY, "-u", f"{SCRIPTS}/train_any.py", "efficientnet_b0", "leaky"],
     f"{R}/efficientnet_b0_leaky_result.json"),
    # 2.5D: three adjacent slices as channels instead of three copies of one
    ("custom_cnn_25d",      [PY, "-u", f"{SCRIPTS}/train_any.py", "custom_cnn", "honest25d"],
     f"{R}/custom_cnn_honest25d_result.json"),
    ("mobilenetv2_25d",     [PY, "-u", f"{SCRIPTS}/train_any.py", "mobilenetv2", "honest25d"],
     f"{R}/mobilenetv2_honest25d_result.json"),
    ("efficientnet_b0_25d", [PY, "-u", f"{SCRIPTS}/train_any.py", "efficientnet_b0", "honest25d"],
     f"{R}/efficientnet_b0_honest25d_result.json"),
    # the only comparison free of the cohort confound (decision 10)
    ("ADvsCN_custom_cnn",   [PY, "-u", f"{SCRIPTS}/train_binary_adni1.py", "custom_cnn"],
     f"{R}/custom_cnn_ADvsCN_result.json"),
    ("ADvsCN_mobilenetv2",  [PY, "-u", f"{SCRIPTS}/train_binary_adni1.py", "mobilenetv2"],
     f"{R}/mobilenetv2_ADvsCN_result.json"),
    # consolidated scoring + ensemble, then the era decomposition -- always re-run, they
    # are cheap and must reflect whatever finished above
    ("final_eval",          [PY, "-u", f"{SCRIPTS}/final_eval.py"], None),
    ("within_era",          [PY, "-u", f"{SCRIPTS}/within_era_analysis.py"], None),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"driver pid {os.getpid()} starting, {len(JOBS)} jobs")

    for label, argv, done_marker in JOBS:
        if done_marker and os.path.exists(done_marker):
            log(f"SKIP  {label} (already done: {os.path.basename(done_marker)})")
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
            log(f"FAIL  {label} exit={code} after {mins:.1f} min -- see {logpath}")
            try:
                with open(logpath, encoding="utf-8") as fh:
                    tail = fh.read().strip().splitlines()[-15:]
                for line in tail:
                    log(f"      | {line}")
            except OSError:
                pass

    log("ALL REMAINING JOBS COMPLETE")


if __name__ == "__main__":
    main()
