"""Re-run the primary four-way task selecting checkpoints on validation macro F1.

Why: the first pass selected on validation loss, which is spiky on this data (fp16
overflow in the AMP path). All three architectures early-stopped with their best
epoch at 2-5, and EfficientNet-B0 restored an epoch-2 checkpoint that never predicted
LMCI once -- 0% recall on an entire disease stage -- while still scoring near the
majority baseline by guessing the largest class.

Macro F1 weights the four stages equally, so a checkpoint that has stopped predicting
one of them is scored as the failure it is. That matches the actual goal here:
detecting all four stages, not maximising overall accuracy by playing the base rates.

Results are written under a "_f1" tag, so the val_loss runs stay on disk for
comparison rather than being overwritten.

Usage: python scripts/run_v3_f1.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOGS = ROOT / "reports" / "run_logs"
ARCHS = ["custom_cnn", "mobilenetv2", "efficientnet_b0"]

JOBS = [(a, ROOT / "reports" / f"{a}_v3go2_f1_result.json") for a in ARCHS]


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for i, (arch, marker) in enumerate(JOBS, 1):
        if marker.exists():
            print(f"[{i}/{len(JOBS)}] SKIP {arch} (marker exists)", flush=True)
            continue
        log_path = LOGS / f"v3f1_{arch}.log"
        print(f"[{i}/{len(JOBS)}] RUN  {arch} v3go2 val_macro_f1 -> {log_path.name}",
              flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                [PY, "-u", str(ROOT / "scripts" / "train_any.py"),
                 arch, "v3go2", "val_macro_f1"],
                stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
            )
        mins = (time.time() - t0) / 60
        status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"[{i}/{len(JOBS)}] {status} {arch} in {mins:.1f} min", flush=True)
        if proc.returncode != 0:
            for line in log_path.read_text(encoding="utf-8",
                                           errors="replace").splitlines()[-15:]:
                print("    | " + line, flush=True)

    print(f"\nF1-SELECTION QUEUE DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
