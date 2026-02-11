"""
Run All Steps
=============
Runs the complete pipeline in sequence. Useful for testing everything at once.

Usage:
    python run_all_steps.py
"""

import subprocess
import sys

steps = [
    ("Step 1: Log Raw Data", "7_data_lineage/step1_log_raw_data.py"),
    ("Step 2: Preprocessing", "7_data_lineage/step2_preprocessing.py"),
    ("Step 3: Training", "7_data_lineage/step3_training.py"),
    ("Step 4: Evaluation", "7_data_lineage/step4_evaluation.py"),
]

print("=" * 60)
print("LINEAGE TUTORIAL - Running all steps")
print("=" * 60)

for name, script in steps:
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print("=" * 60 + "\n")
    
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"\n❌ {name} failed with code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {name} completed")

print("\n" + "=" * 60)
print("🎉 ALL STEPS COMPLETED!")
print("=" * 60)
print("\nNext steps:")
print("  1. Go to wandb.ai and open your 'lineage-tutorial' project")
print("  2. Click on 'Artifacts' in the left sidebar")
print("  3. Select any artifact (e.g., 'classifier')")
print("  4. Click the 'Lineage' tab to see the full dependency graph")
print()
