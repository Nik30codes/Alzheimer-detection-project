"""Cross-era external validation, re-run with macro-F1 checkpoint selection.

The val_loss pass demonstrated the problem again: MobileNetV2 go2->adni1 restored
epoch 2 (val_loss 0.650) while epoch 6 had val macro F1 0.718 against epoch 2's
0.587 -- a clearly better model was trained and discarded. Validation loss on this
data is too spiky to select on.

Results are tagged "_f1" so the val_loss pass stays on disk for comparison.

Usage: python scripts/run_cross_era_f1.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOGS = ROOT / "reports" / "run_logs"
ARCHS = ["mobilenetv2", "custom_cnn", "efficientnet_b0"]
DIRECTIONS = ["adni1->go2", "go2->adni1"]

JOBS = []
for arch in ARCHS:
    for d in DIRECTIONS:
        tag = f"{arch}_crossera_{d.replace('->', '_to_')}_f1"
        JOBS.append((arch, d, ROOT / "reports" / f"{tag}_result.json"))


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for i, (arch, direction, marker) in enumerate(JOBS, 1):
        label = f"{arch} {direction}"
        if marker.exists():
            print(f"[{i}/{len(JOBS)}] SKIP {label} (marker exists)", flush=True)
            continue
        log_path = LOGS / f"crossera_f1_{arch}_{direction.replace('->', '_to_')}.log"
        print(f"[{i}/{len(JOBS)}] RUN  {label} -> {log_path.name}", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                [PY, "-u", str(ROOT / "scripts" / "train_cross_era.py"),
                 arch, direction, "val_macro_f1"],
                stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
            )
        mins = (time.time() - t0) / 60
        status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"[{i}/{len(JOBS)}] {status} {label} in {mins:.1f} min", flush=True)
        if proc.returncode != 0:
            for line in log_path.read_text(encoding="utf-8",
                                           errors="replace").splitlines()[-20:]:
                print("    | " + line, flush=True)

    print(f"\nCROSS-ERA F1 QUEUE DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
