"""
Step 4: Evaluation
==================
Run after step3. Downloads model + test data, evaluates, logs results.

Usage:
    python step4_evaluation.py
"""

import wandb
import json
import os

with wandb.init(project="lineage-tutorial", job_type="evaluation") as run:
    
    # Declare inputs - both model and data
    print("Downloading model artifact...")
    model_artifact = run.use_artifact("classifier:latest")
    model_path = model_artifact.download()
    print(f"  Model downloaded to: {model_path}")
    
    print("\nDownloading test data artifact...")
    data_artifact = run.use_artifact("processed-images:latest")
    data_dir = data_artifact.download()
    print(f"  Data downloaded to: {data_dir}")
    
    # Simulate evaluation
    # In real life: load model, run inference on test set, compute metrics
    print("\nRunning evaluation...")
    print("  (this is where your actual evaluation code would go)")
    
    # Fake some results
    test_accuracy = 0.89
    test_loss = 0.34
    test_samples = len(os.listdir(f"{data_dir}/test"))
    
    # Log metrics to run summary
    run.summary["test_accuracy"] = test_accuracy
    run.summary["test_loss"] = test_loss
    run.summary["test_samples"] = test_samples
    
    print(f"  Test accuracy: {test_accuracy}")
    print(f"  Test loss: {test_loss}")
    print(f"  Test samples: {test_samples}")
    
    # Log evaluation results as artifact
    os.makedirs("./results", exist_ok=True)
    results_path = "./results/eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "accuracy": test_accuracy,
            "loss": test_loss,
            "samples": test_samples,
            "model_version": model_artifact.version,
            "data_version": data_artifact.version
        }, f, indent=2)
    
    eval_artifact = wandb.Artifact(
        name="eval-results",
        type="evaluation",
        description="Test set evaluation results"
    )
    eval_artifact.add_file(results_path)
    run.log_artifact(eval_artifact)
    
    print(f"\n✓ Logged evaluation artifact")
    print(f"  View at: {run.get_url()}")
    print(f"\n🎉 Pipeline complete! Check the Artifacts tab and click 'Lineage' to see the full graph.")
