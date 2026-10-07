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


def prepare_dataset_yaml(data_path: str) -> str:
    """
    Validates and auto-corrects dataset YAML configuration.
    Fixes the common issue where 'path' in data.yaml was hardcoded to an absolute
    path during preprocessing (e.g. /kaggle/working/road-damage-detection-yolo/data),
    but the dataset is later mounted under /kaggle/input/...
    """
    yaml_file = Path(data_path).resolve()
    if not yaml_file.exists():
        return data_path

    try:
        import yaml
        with open(yaml_file, "r") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            return data_path

        val_subpath = cfg.get("val", "val/images")
        if isinstance(val_subpath, list):
            val_subpath = val_subpath[0] if val_subpath else "val/images"
        val_subpath = str(val_subpath)

        configured_path = Path(cfg.get("path", ""))

        # Check if validation images exist at the configured path
        path_valid = (configured_path / val_subpath).exists() if str(configured_path) else False

        if not path_valid:
            # Check if images actually exist relative to the YAML file's location
            actual_dir = yaml_file.parent
            if (actual_dir / val_subpath).exists():
                print(f"⚠️ Notice: 'path: {configured_path}' defined in data.yaml does not exist on disk.")
                print(f"🔄 Auto-redirecting dataset root path to: {actual_dir}")
                cfg["path"] = str(actual_dir)

                # Write fixed YAML to writable working directory
                fixed_yaml = Path("/kaggle/working/data_auto_fixed.yaml") if Path("/kaggle/working").exists() else Path("./data_auto_fixed.yaml")
                with open(fixed_yaml, "w") as f_out:
                    yaml.dump(cfg, f_out, default_flow_style=False)
                
                print(f"✅ Created auto-corrected data configuration at: {fixed_yaml}")
                return str(fixed_yaml.resolve())
    except Exception as e:
        print(f"⚠️ Warning during data.yaml verification: {e}")

    return data_path


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

    # Auto-adjust data.yaml if path was hardcoded in another environment
    data_path = prepare_dataset_yaml(args.data)

    # 1. Setup Tracking via Weights & Biases
    setup_wandb(args.project)

    print("==================================================")
    print(f"🚀 Starting Run: {args.name}")
    print(f"Model: {args.model}")
    print(f"Data: {data_path}")
    print(f"Config: {args.cfg}")
    print("==================================================")

    # 2. Initialize Model
    model = YOLO(args.model)

    # 3. Build Training Arguments
    train_kwargs = {
        "data": data_path,
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
    output_dir = Path(getattr(model.trainer, "save_dir", Path("runs/detect") / args.project / args.name))
    if not output_dir.exists():
        output_dir = Path(args.project) / args.name
    archive_dest = f"/kaggle/working/{args.name}_results" if Path("/kaggle/working").exists() else f"./results/{args.name}_results"
    archive_results(output_dir, archive_dest)

    wandb.finish()
    print("🎉 Run finished successfully!")


if __name__ == "__main__":
    main()