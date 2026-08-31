"""Re-test skull/scalp masking on the v3 dataset, properly this time.

Masking was last evaluated on v2, where AD vs CN had 26 test subjects and every
confidence interval spanned roughly [0.25, 0.89] -- the result was uninformative, and
CLAUDE.md records it as "not distinguishable from noise". v3 has 501 subjects and a
75-subject test set, so the comparison is now actually resolvable.

The two things being measured are DIFFERENT questions and are expected to disagree:
  * WHERE the model looks   -- Grad-CAM attention outside the brain. Masking previously
                               moved this 75% -> 44%. Note the reference point is 58.5%,
                               not 0%: that is what a random heatmap scores, because only
                               ~42% of the frame is brain.
  * WHETHER it is more accurate -- AD vs CN AUC. Masking previously moved this the WRONG
                               way (0.673 -> 0.473 on one architecture) while attention
                               improved. Attention quality is an interpretability metric,
                               not an accuracy driver.

Stage 1 builds the masked dataset (CPU). Stage 2 trains two architectures on it (GPU).
Restart-safe via per-job markers.
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_EXE = sys.executable
LOGS = ROOT / "reports" / "run_logs"; LOGS.mkdir(parents=True, exist_ok=True)


def sh(cmd, log_name, label, marker=None):
    if marker and Path(marker).exists():
        print(f"  SKIP {label}", flush=True); return True
    print(f"  RUN  {label} -> {log_name}", flush=True)
    t0 = time.time()
    with open(LOGS / log_name, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"  {'OK  ' if p.returncode==0 else f'FAIL rc={p.returncode}'} {label} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    if p.returncode:
        for line in (LOGS/log_name).read_text(encoding='utf-8', errors='replace').splitlines()[-15:]:
            print("      | " + line, flush=True)
    return p.returncode == 0


t0 = time.time()
print("=== STAGE 1: build the masked v3 dataset (CPU) ===", flush=True)
ok = sh([PY_EXE, "-u", str(ROOT/"scripts"/"brain_mask.py"), "runmask_v3"],
        "mask_v3_build.log", "mask v3",
        marker=ROOT/"data"/"manifest_v3_masked.csv")

if ok:
    print("\n=== STAGE 2: train AD vs CN on masked slices (GPU) ===", flush=True)
    for arch in ("mobilenetv2", "custom_cnn"):
        sh([PY_EXE, "-u", str(ROOT/"scripts"/"train_binary_adni1.py"), arch, "v3mask"],
           f"mask_v3_{arch}.log", f"{arch} v3mask",
           marker=ROOT/"reports"/f"{arch}_ADvsCN_v3mask_result.json")

print(f"\nMASK TEST DONE in {(time.time()-t0)/60:.1f} min", flush=True)
