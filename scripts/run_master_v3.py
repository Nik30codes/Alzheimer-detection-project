"""Master driver: runs every remaining v3 experiment in order, unattended.

Restart-safe throughout -- each step has a result-file marker, so re-running skips
whatever already finished rather than repeating it.

Order is chosen so the cheapest, best-motivated experiments land first and the
expensive self-supervised stage runs last:

  STAGE 1  Four-way improvement attempts (scripts/run_v3_improve.py)
           - hi-res: drop the 144px bottleneck, which harmonizes ADNI1 192x192
             against GO/2 256x256 but is pure detail loss on this GO/2-only task
           - init:   start from the AD-vs-CN checkpoint (ROC AUC 0.906 on the same
             images) instead of random weights

  STAGE 2  Self-supervised pretraining (scripts/pretrain_autoencoder.py)
           Masked autoencoder over ~19,100 unlabelled train-split slices. Labels are
           the scarce resource here, not images, so this is the lever that targets
           the actual binding constraint.

  STAGE 3  Four-way runs initialised from the self-supervised encoder.
           Only custom_cnn: the encoder is architecturally SimpleCNN.features, so it
           does not fit MobileNetV2 or EfficientNet-B0.

  STAGE 4  Regenerate the Word report so the document matches what is on disk.

Usage: python scripts/run_master_v3.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOGS = ROOT / "reports" / "run_logs"
SCRIPTS = ROOT / "scripts"


def run(cmd, log_name, label, marker=None):
    """Run one subprocess, logging to reports/run_logs/. Returns True if it ran OK."""
    if marker is not None and Path(marker).exists():
        print(f"  SKIP {label} (marker exists: {Path(marker).name})", flush=True)
        return True
    log_path = LOGS / log_name
    print(f"  RUN  {label} -> {log_name}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    mins = (time.time() - t0) / 60
    ok = proc.returncode == 0
    print(f"  {'OK  ' if ok else f'FAIL rc={proc.returncode}'} {label} "
          f"in {mins:.1f} min", flush=True)
    if not ok:
        for line in log_path.read_text(encoding="utf-8",
                                       errors="replace").splitlines()[-20:]:
            print("      | " + line, flush=True)
    return ok


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("\n=== STAGE 1: four-way improvement attempts ===", flush=True)
    run([PY, "-u", str(SCRIPTS / "run_v3_improve.py")],
        "master_stage1_improve.log", "improvement queue (6 jobs)")

    print("\n=== STAGE 2: self-supervised pretraining ===", flush=True)
    ssl_ok = run([PY, "-u", str(SCRIPTS / "pretrain_autoencoder.py"), "25"],
                 "master_stage2_ssl.log", "masked autoencoder, 25 epochs",
                 marker=ROOT / "reports" / "ssl_pretrain_result.json")

    print("\n=== STAGE 3: four-way from the self-supervised encoder ===", flush=True)
    if not (ROOT / "models" / "checkpoints" / "ssl_encoder.pt").exists():
        print("  SKIP - ssl_encoder.pt was not produced by stage 2", flush=True)
    else:
        for mode in ("v3go2", "v3go2hi"):
            tag = f"custom_cnn_{mode}_f1_init-ssl_encoder"
            run([PY, "-u", str(SCRIPTS / "train_any.py"),
                 "custom_cnn", mode, "val_macro_f1", "ssl_encoder.pt"],
                f"master_stage3_{tag}.log", f"custom_cnn {mode} <- ssl_encoder",
                marker=ROOT / "reports" / f"{tag}_result.json")

    print("\n=== STAGE 4: regenerate the report ===", flush=True)
    run([PY, "-u", str(SCRIPTS / "make_report_docx.py")],
        "master_stage4_report.log", "Word report")

    print(f"\nMASTER QUEUE DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
