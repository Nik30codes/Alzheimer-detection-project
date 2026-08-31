"""Train AD vs CN on the v4 (isotropic-geometry) dataset and compare against v3.

This is the experiment decision 31 says is still owed. v4 removes the acquisition
geometry confound (decision 27) while preserving head size in millimetres. AD vs CN was
already CLEAN of that confound -- geometry scored BELOW its baseline there -- so the
prediction is that v4 leaves this task roughly unchanged. A large FALL would mean the
isotropic resampling destroyed real signal (the v2crop failure mode of decision 21);
a large RISE would be surprising and would need explaining.

Two architectures, because a single one at n=75 cannot distinguish a real change from
noise (decision 21's lesson).

Usage: python scripts/run_v4_adcn.py
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "run_logs"; LOGS.mkdir(parents=True, exist_ok=True)

JOBS = [("mobilenetv2", "v4adcn"), ("custom_cnn", "v4adcn")]

t0 = time.time()
for i, (arch, key) in enumerate(JOBS, 1):
    marker = ROOT / "reports" / f"{arch}_ADvsCN_{key}_result.json"
    if marker.exists():
        print(f"[{i}/{len(JOBS)}] SKIP {arch} {key}", flush=True); continue
    log = LOGS / f"v4adcn_{arch}.log"
    print(f"[{i}/{len(JOBS)}] RUN  {arch} {key} -> {log.name}", flush=True)
    s = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run([sys.executable, "-u", str(ROOT/"scripts"/"train_binary_adni1.py"),
                            arch, key], stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"[{i}/{len(JOBS)}] {'OK' if p.returncode==0 else f'FAIL rc={p.returncode}'} "
          f"{arch} in {(time.time()-s)/60:.1f} min", flush=True)
print(f"\nV4 ADCN QUEUE DONE in {(time.time()-t0)/60:.1f} min", flush=True)
