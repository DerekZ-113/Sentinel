#!/usr/bin/env python
"""
Sentinel Pipeline Runner

Runs the complete ML pipeline:
1. Prepare data (feature engineering)
2. Train XGBoost classifier + evaluate

Usage (from anywhere — steps run as modules with the repo root on the path,
and artifacts always land in ml/):
    python -m ml.run_pipeline [--max-events 200000]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(name, module, extra_args=()):
    """Run a pipeline step as a module and report timing"""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}\n")

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", module, *extra_args],
        cwd=REPO_ROOT,
        env=env,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ {name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n✅ {name} completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the Sentinel ML pipeline")
    parser.add_argument("--max-events", type=int, default=200_000,
                        help="Cap training data at N whole events (0 = keep all)")
    args = parser.parse_args()

    print("="*60)
    print("SENTINEL ML PIPELINE (XGBoost)")
    print("="*60)

    total_start = time.time()
    timings = {}

    # Step 1: Prepare data
    timings['prepare'] = run_step(
        "Data Preparation", "ml.prepare_data",
        ["--max-events", str(args.max_events)])

    # Step 2: Train classifier + evaluate
    timings['train'] = run_step("XGBoost Training & Evaluation", "ml.train_classifier")

    # Summary
    total = time.time() - total_start

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nTimings:")
    print(f"  Data prep:      {timings['prepare']:>6.1f}s")
    print(f"  Train + Eval:   {timings['train']:>6.1f}s")
    print(f"  ─────────────────────")
    print(f"  Total:          {total:>6.1f}s ({total/60:.1f} min)")
    print()

if __name__ == "__main__":
    main()
