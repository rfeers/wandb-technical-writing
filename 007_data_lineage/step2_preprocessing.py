"""
Step 2: Preprocessing
=====================
Run after step1. Downloads raw data artifact, "processes" it, logs processed artifact.

Usage:
    python step2_preprocessing.py
"""

import wandb
import os
import shutil

with wandb.init(project="lineage-tutorial", job_type="preprocessing") as run:
    
    # Declare input - this creates the lineage link
    print("Downloading raw data artifact...")
    raw_artifact = run.use_artifact("raw-images:latest")
    raw_dir = raw_artifact.download()
    print(f"  Downloaded to: {raw_dir}")
    
    # Simulate preprocessing
    # In real life: resize, normalize, augment, etc.
    os.makedirs("./data/processed/train", exist_ok=True)
    os.makedirs("./data/processed/test", exist_ok=True)
    
    print("Processing files...")
    train_files = os.listdir(f"{raw_dir}/train")
    test_files = os.listdir(f"{raw_dir}/test")
    
    for f in train_files:
        shutil.copy(f"{raw_dir}/train/{f}", f"./data/processed/train/{f}")
    for f in test_files:
        shutil.copy(f"{raw_dir}/test/{f}", f"./data/processed/test/{f}")
    
    print(f"  Processed {len(train_files)} train + {len(test_files)} test files")
    
    # Log output artifact
    processed_data = wandb.Artifact(
        name="processed-images",
        type="dataset",
        description="Preprocessed and normalized images"
    )
    processed_data.add_dir("./data/processed/")
    run.log_artifact(processed_data)
    
    print(f"\n✓ Logged processed data artifact")
    print(f"  View at: {run.get_url()}")
