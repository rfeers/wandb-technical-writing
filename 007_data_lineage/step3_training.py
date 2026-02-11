"""
Step 3: Training
================
Run after step2. Downloads processed data, "trains" a model, logs model artifact.

Usage:
    python step3_training.py
"""

import wandb
import json
import os

with wandb.init(project="lineage-tutorial", job_type="training") as run:
    
    # Log hyperparameters - these get linked to the model artifact
    config = {
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "model": "resnet18"
    }
    run.config.update(config)
    print(f"Training config: {config}")
    
    # Declare input
    print("\nDownloading processed data artifact...")
    data_artifact = run.use_artifact("processed-images:latest")
    data_dir = data_artifact.download()
    print(f"  Downloaded to: {data_dir}")
    
    # Simulate training
    # In real life: load data, build model, train loop, etc.
    print("\nTraining model...")
    print("  (this is where your actual training code would go)")
    
    # Simulate some metrics
    for epoch in range(1, 6):
        loss = 1.0 / epoch
        acc = 0.5 + (epoch * 0.08)
        run.log({"epoch": epoch, "loss": loss, "accuracy": acc})
        print(f"  Epoch {epoch}: loss={loss:.3f}, acc={acc:.2f}")
    
    # Save and log model
    os.makedirs("./checkpoints", exist_ok=True)
    model_path = "./checkpoints/model.json"
    with open(model_path, "w") as f:
        json.dump({
            "weights": "placeholder_weights_data",
            "config": config,
            "final_accuracy": 0.92
        }, f, indent=2)
    
    model_artifact = wandb.Artifact(
        name="classifier",
        type="model",
        description="Trained image classifier"
    )
    model_artifact.add_file(model_path)
    model_artifact.metadata["accuracy"] = 0.92
    model_artifact.metadata["epochs_trained"] = 50
    run.log_artifact(model_artifact)
    
    print(f"\n✓ Logged model artifact")
    print(f"  View at: {run.get_url()}")
