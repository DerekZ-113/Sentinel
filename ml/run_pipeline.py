#!/usr/bin/env python
"""
Sentinel Pipeline Runner

Runs the complete ML pipeline:
1. Prepare data (feature engineering)
2. Train VAE
3. Evaluate results

Usage: python run_pipeline.py
"""

import subprocess
import sys
import time

def run_step(name, script):
    """Run a pipeline step and report timing"""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}\n")
    
    start = time.time()
    result = subprocess.run([sys.executable, script], cwd='.')
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n❌ {name} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ {name} completed in {elapsed:.1f}s")
    return elapsed

def main():
    print("="*60)
    print("SENTINEL ML PIPELINE")
    print("="*60)
    
    total_start = time.time()
    timings = {}
    
    # Step 1: Prepare data
    timings['prepare'] = run_step("Data Preparation", "prepare_data.py")
    
    # Step 2: Train VAE
    timings['train'] = run_step("VAE Training", "train_vae.py")
    
    # Step 3: Evaluate
    timings['evaluate'] = run_step("Evaluation", "vae_alerter.py")
    
    # Summary
    total = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nTimings:")
    print(f"  Data prep:   {timings['prepare']:>6.1f}s")
    print(f"  Training:    {timings['train']:>6.1f}s")
    print(f"  Evaluation:  {timings['evaluate']:>6.1f}s")
    print(f"  ─────────────────────")
    print(f"  Total:       {total:>6.1f}s ({total/60:.1f} min)")
    print()

if __name__ == "__main__":
    main()
