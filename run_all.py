"""
run_all.py — Execute all pipeline steps in order.
"""
import subprocess, sys

PYTHON = sys.executable
steps = [
    ("Step 1: Crop",       "step1_crop.py"),
    ("Step 2: Slope",      "step2_slope.py"),
    ("Step 3: Route",      "step3_route.py"),
]

for label, script in steps:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([PYTHON, script], check=False)
    if result.returncode != 0:
        print(f"\nERROR in {script} — stopping.")
        sys.exit(result.returncode)

print("\n" + "="*60)
print("  All steps complete!")
print("  Serve the app: cd web && python -m http.server 8000")
print("  Open: http://localhost:8000")
print("="*60)
