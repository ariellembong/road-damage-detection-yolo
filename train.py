"""
train.py - YOLO Training Pipeline for Road Damage Detection
Supports local execution and Kaggle cloud execution with W&B logging.
"""

import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO Model for Road Damage Detection")

    # Experiment Configuration
    parser.add_argument("--model", type=str, default="yolo12s.pt", help="Base model weights or path to checkpoint")
    parser.add_argument("--data", type=str, default="/kaggle/input/road-damage-dataset/data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--cfg", type=str, default="configs/hyp.yaml", help="Path to hyperparameter configuration file")
    
    # Run Metadata
    parser.add_argument("--name", type=str, default="exp_yolo12s", help="Unique run name on W&B and local runs")
    parser.add_argument("--project", type=str, default="road-damage-detection", help="W&B project name")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g. '0' or 'cpu')")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    
    # Optional CLI overrides (if specified, overrides configs/hyp.yaml)
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    
    return parser.parse_args()


def setup_wandb(project_name: str):
    """Authenticate and configure Weights & Biases."""
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_key)
    except Exception:
        if "WANDB_API_KEY" in os.environ:
            wandb.login(key=os.environ["WANDB_API_KEY"])
        else:
            print("ℹ️ W&B API Key not found in secrets. Running offline or with cached login.")

    os.environ["WANDB_PROJECT"] = project_name


def archive_results(source_dir: Path, zip_dest: str):
    """Zips training artifacts for 1-click download in Kaggle."""
    source_dir = Path(source_dir)
    if source_dir.exists():
        zip_path = shutil.make_archive(zip_dest, "zip", source_dir)
        print(f"📦 Successfully created downloadable archive: {zip_path}")
        return zip_path
    else:
        print(f"⚠️ Directory {source_dir} not found for archiving.")
        return None


def main():
    args = parse_args()

    # 1. Setup Tracking via Weights & Biases
    setup_wandb(args.project)

    print("==================================================")
    print(f"🚀 Starting Run: {args.name}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Config: {args.cfg}")
    print("==================================================")

    # 2. Initialize Model
    model = YOLO(args.model)

    # 3. Build Training Arguments
    train_kwargs = {
        "data": args.data,
        "cfg": args.cfg,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "save": True,
        "plots": True,
        "amp": True,
    }

    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.batch is not None:
        train_kwargs["batch"] = args.batch

    # 4. Train Model
    results = model.train(**train_kwargs)

    # 5. Final Validation on Test/Val Split
    print("\n--- 📊 Running Final Validation Evaluation ---")
    val_metrics = model.val()
    print(f"Validation mAP50:    {val_metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {val_metrics.box.map:.4f}")

    # 6. Archive results for 1-click download on Kaggle
    output_dir = Path(args.project) / args.name
    archive_dest = f"/kaggle/working/{args.name}_results" if Path("/kaggle/working").exists() else f"./results/{args.name}_results"
    archive_results(output_dir, archive_dest)

    wandb.finish()
    print("🎉 Run finished successfully!")


if __name__ == "__main__":
    main()