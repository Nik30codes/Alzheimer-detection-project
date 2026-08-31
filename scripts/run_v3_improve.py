"""Attempts to raise four-way accuracy on the era-matched ADNI-GO/2 task.

Baseline to beat: ~38-42% subject accuracy against a 36.6% majority baseline, with
every confidence interval straddling that baseline. Diagnosis of where the loss sits
(scratchpad analysis): merging EMCI and LMCI into a single MCI class recovers +15.9
accuracy points, and the EMCI-vs-LMCI sub-decision is right only 58.7% of the time
when the model already knows the subject is MCI. That boundary is defined in ADNI by
a delayed-recall memory-test cutoff rather than by imaging, so part of this ceiling is
in the labels, not the model.

Two levers are tested here, both chosen because they target something specific rather
than being generic knob-turning:

  HI-RES (v3go2hi)  Drop the 144px resolution-harmonization bottleneck. It equalizes
                    ADNI1 192x192 against GO/2 256x256, but this task is GO/2-only
                    where every scan is natively 256 rows -- so it low-passes to about
                    1.74mm/pixel and discards detail at the scale that separates
                    stages, in exchange for correcting a confound that is not present.

  INIT (init_from)  Initialise from the AD-vs-CN checkpoint instead of random. That
                    model reaches ROC AUC 0.906 on the same images, so its features
                    demonstrably encode real atrophy. Decision 7 showed ImageNet
                    weights hurt here, but that is a statement about the source domain
                    (natural photos), not about transfer in general -- this source is
                    the same modality, anatomy and preprocessing.

All runs select checkpoints on validation macro F1, since val_loss selection has
repeatedly restored near-untrained models on this data.

Usage: python scripts/run_v3_improve.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOGS = ROOT / "reports" / "run_logs"

# (arch, mode, init_from, marker-tag)
JOBS = [
    ("custom_cnn",      "v3go2hi", None,                      "custom_cnn_v3go2hi_f1"),
    ("mobilenetv2",     "v3go2hi", None,                      "mobilenetv2_v3go2hi_f1"),
    ("efficientnet_b0", "v3go2hi", None,                      "efficientnet_b0_v3go2hi_f1"),
    ("mobilenetv2",     "v3go2",   "mobilenetv2_ADvsCN.pt",
     "mobilenetv2_v3go2_f1_init-mobilenetv2_ADvsCN"),
    ("custom_cnn",      "v3go2",   "custom_cnn_ADvsCN.pt",
     "custom_cnn_v3go2_f1_init-custom_cnn_ADvsCN"),
    # both levers together
    ("mobilenetv2",     "v3go2hi", "mobilenetv2_ADvsCN.pt",
     "mobilenetv2_v3go2hi_f1_init-mobilenetv2_ADvsCN"),
]


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for i, (arch, mode, init_from, tag) in enumerate(JOBS, 1):
        marker = ROOT / "reports" / f"{tag}_result.json"
        label = f"{arch} {mode}" + (f" init={init_from}" if init_from else "")
        if marker.exists():
            print(f"[{i}/{len(JOBS)}] SKIP {label} (marker exists)", flush=True)
            continue
        log_path = LOGS / f"improve_{tag}.log"
        print(f"[{i}/{len(JOBS)}] RUN  {label} -> {log_path.name}", flush=True)
        cmd = [PY, "-u", str(ROOT / "scripts" / "train_any.py"),
               arch, mode, "val_macro_f1"]
        if init_from:
            cmd.append(init_from)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=str(ROOT))
        mins = (time.time() - t0) / 60
        status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"[{i}/{len(JOBS)}] {status} {label} in {mins:.1f} min", flush=True)
        if proc.returncode != 0:
            for line in log_path.read_text(encoding="utf-8",
                                           errors="replace").splitlines()[-20:]:
                print("    | " + line, flush=True)

    print(f"\nIMPROVE QUEUE DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
