"""Sequential job driver for the post-expansion (v3) experiments.

One driver process running jobs one at a time, restart-safe via per-job result
markers. Deliberately NOT a chain of background shells: nested background bash
chains were reaped mid-queue on this machine while their child training processes
survived as orphans, one of which hung for an hour holding GPU memory (see the
infrastructure notes in CLAUDE.md).

Job order puts the primary task first, so the headline number exists even if the
queue is interrupted later:

  1-3  4-way within ADNI-GO/2 (v3go2). THE PRIMARY TASK -- 618 subjects, all four
       classes from one scanner era, so cohort carries no label information.
  4-6  AD vs CN across both eras (v3adcn). 501 subjects, era-balanced; the metric
       CLAUDE.md designates as the honest headline.
  7    4-way over all 853 subjects (v3). Kept for the size comparison only: EMCI and
       LMCI are still ADNI-GO/2-only here, so era partially predicts the label and
       this number stays contaminated. Runs last because it is the least meaningful.

Usage: python scripts/run_v3_expansion.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOGS = ROOT / "reports" / "run_logs"
ARCHS = ["custom_cnn", "mobilenetv2", "efficientnet_b0"]

# (script, args, marker file that means "already done")
JOBS = []
for arch in ARCHS:
    JOBS.append(("train_any.py", [arch, "v3go2"],
                 ROOT / "reports" / f"{arch}_v3go2_result.json"))
for arch in ARCHS:
    JOBS.append(("train_binary_adni1.py", [arch, "v3adcn"],
                 ROOT / "reports" / f"{arch}_ADvsCN_v3adcn_result.json"))
JOBS.append(("train_any.py", ["mobilenetv2", "v3"],
             ROOT / "reports" / "mobilenetv2_v3_result.json"))


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for i, (script, args, marker) in enumerate(JOBS, 1):
        tag = f"{script.replace('.py','')}_{'_'.join(args)}"
        if marker.exists():
            print(f"[{i}/{len(JOBS)}] SKIP {tag} (marker exists: {marker.name})",
                  flush=True)
            continue

        log_path = LOGS / f"v3_{tag}.log"
        print(f"[{i}/{len(JOBS)}] RUN  {tag}  -> {log_path.name}", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            # -u: without it Python block-buffers stdout when it is a file, so a job's
            # progress is invisible until it exits -- which makes a long queue
            # impossible to monitor and a hung job indistinguishable from a slow one.
            proc = subprocess.run(
                [PY, "-u", str(ROOT / "scripts" / script), *args],
                stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
            )
        mins = (time.time() - t0) / 60
        status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"[{i}/{len(JOBS)}] {status} {tag} in {mins:.1f} min", flush=True)
        if proc.returncode != 0:
            print(f"    last lines of {log_path}:", flush=True)
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            for line in tail:
                print("    | " + line, flush=True)

    print(f"\nQUEUE DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
