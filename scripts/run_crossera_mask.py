"""Cross-era test of the masked dataset: is decision 32's +0.05 AUC real or a shortcut?

Masking measured +0.050 AUC under paired 5-fold CV (decision 32) -- a genuine, significant
gain. But its Grad-CAM attention got 14 points WORSE (33.4% -> 47.5% outside the brain),
which is exactly what "Skull-stripping induces shortcut learning" (arxiv 2501.15831)
predicts: zeroing the background leaves a hard silhouette the model can read instead of
anatomy.

The discriminating test: train on ONE scanner generation, test on the whole of the other.
  * A real anatomical gain should transfer -- masked should hold up roughly as well as
    plain did (decision 17: AUC 0.68-0.79 across eras).
  * A silhouette artifact should collapse, because head framing and skull geometry differ
    between the two ADNI protocols.

Both directions, both datasets, so the comparison is like-for-like.
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "run_logs"; LOGS.mkdir(parents=True, exist_ok=True)
JOBS = [("adni1->go2", "manifest_v3_masked.csv", "masked"),
        ("go2->adni1", "manifest_v3_masked.csv", "masked")]

t0 = time.time()
for i, (d, mf, label) in enumerate(JOBS, 1):
    tag = f"mobilenetv2_crossera_{d.replace('->', '_to_')}_masked"
    marker = ROOT / "reports" / f"{tag}_result.json"
    if marker.exists():
        print(f"[{i}/{len(JOBS)}] SKIP {label} {d}", flush=True); continue
    log = LOGS / f"{tag}.log"
    print(f"[{i}/{len(JOBS)}] RUN  {label} {d} -> {log.name}", flush=True)
    s = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run([sys.executable, "-u", str(ROOT/"scripts"/"train_cross_era.py"),
                            "mobilenetv2", d, "val_loss", mf],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"[{i}/{len(JOBS)}] {'OK' if p.returncode==0 else f'FAIL rc={p.returncode}'} "
          f"in {(time.time()-s)/60:.1f} min", flush=True)
print(f"\nCROSSERA MASK DONE in {(time.time()-t0)/60:.1f} min", flush=True)
