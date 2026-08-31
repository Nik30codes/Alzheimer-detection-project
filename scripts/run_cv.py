"""Driver for the cross-validation study.

Answers one question properly: are the pretraining gains real, or noise from a single
93-subject test set? Each job gives all 618 subjects an out-of-fold prediction, so the
confidence interval is computed on 618 rather than 93.

Cheap baselines run first so a comparison exists early even if the queue is cut short.
The pretrained configurations are far more expensive because pretraining is repeated
inside every fold -- reusing one pretrained encoder across folds would leak, since its
training subjects rotate into later test folds.

Usage: python scripts/run_cv.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOGS = ROOT / "reports" / "run_logs"

# (arch, config, k, ssl_epochs, rough cost)
JOBS = [
    ("custom_cnn",  "random", 5, 0,  "~75 min"),
    ("mobilenetv2", "random", 5, 0,  "~75 min"),
    ("mobilenetv2", "adcn",   5, 0,  "~2.5 h"),
    ("custom_cnn",  "ssl",    5, 10, "~4 h"),
]


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for i, (arch, config, k, ssl_ep, cost) in enumerate(JOBS, 1):
        tag = f"cv_{arch}_{config}_k{k}"
        marker = ROOT / "reports" / f"{tag}_result.json"
        if marker.exists():
            print(f"[{i}/{len(JOBS)}] SKIP {tag} (marker exists)", flush=True)
            continue
        log_path = LOGS / f"{tag}.log"
        print(f"[{i}/{len(JOBS)}] RUN  {tag} ({cost}) -> {log_path.name}", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                [PY, "-u", str(ROOT / "scripts" / "cross_validate.py"),
                 arch, config, str(k), str(ssl_ep)],
                stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        mins = (time.time() - t0) / 60
        status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"[{i}/{len(JOBS)}] {status} {tag} in {mins:.1f} min", flush=True)
        if proc.returncode != 0:
            for line in log_path.read_text(encoding="utf-8",
                                           errors="replace").splitlines()[-20:]:
                print("    | " + line, flush=True)

    print(f"\nCV QUEUE DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
