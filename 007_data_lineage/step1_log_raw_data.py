"""
Step 1: Log Raw Data
====================
Run this first. It creates dummy data and logs it as the starting artifact.

Usage:
    python step1_log_raw_data.py
"""

import wandb
import os

# Create dummy data structure
# In your real project, you'd skip this and point at actual data
os.makedirs("./data/raw/train", exist_ok=True)
os.makedirs("./data/raw/test", exist_ok=True)

print("Creating dummy data files...")
for i in range(10):
    with open(f"./data/raw/train/img_{i}.txt", "w") as f:
        f.write(f"training image {i}")

for i in range(3):
    with open(f"./data/raw/test/img_{i}.txt", "w") as f:
        f.write(f"test image {i}")

print(f"Created {10} training files and {3} test files")

# Log to W&B
with wandb.init(project="lineage-tutorial", job_type="data-collection") as run:
    raw_data = wandb.Artifact(
        name="raw-images",
        type="dataset",
        description="Raw collected images before preprocessing"
    )
    raw_data.add_dir("./data/raw/")
    run.log_artifact(raw_data)
    
    print(f"\n✓ Logged raw data artifact")
    print(f"  View at: {run.get_url()}")
