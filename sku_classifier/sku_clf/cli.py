import argparse
import sys
import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_args_with_config(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: Implement deep merge or override logic if needed
    # For POC, we'll focus on the most important overrides
    if hasattr(args, 'lr') and args.lr:
        config['training']['lr'] = args.lr
    if hasattr(args, 'epochs') and args.epochs:
        config['training']['epochs'] = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if hasattr(args, 'device') and args.device:
        config['training']['device'] = args.device
    return config

def main():
    parser = argparse.ArgumentParser(description="SKU Classifier CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the classifier")
    train_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    train_parser.add_argument("--data", type=str, help="Path to dataset (zip or directory)")
    train_parser.add_argument("--lr", type=float, help="Override learning rate")
    train_parser.add_argument("--epochs", type=int, help="Override number of epochs")
    train_parser.add_argument("--batch_size", type=int, help="Override batch size")
    train_parser.add_argument("--device", type=str, help="Override device (cpu, cuda, mps)")
    train_parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the classifier")
    eval_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--data", type=str, help="Path to evaluation dataset")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference on a single image")
    infer_parser.add_argument("--image", type=str, help="Path to image file")
    infer_parser.add_argument("--boxes", type=str, help="Path to YOLO boxes file")
    infer_parser.add_argument("--crops", type=str, help="Path to pre-cropped image")
    infer_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    infer_parser.add_argument("--save_vis", type=str, help="Path to save visualization")

    # Infer-batch command
    infer_batch_parser = subparsers.add_parser("infer-batch", help="Run inference on a batch of images")
    infer_batch_parser.add_argument("--images_dir", type=str, help="Directory containing images")
    infer_batch_parser.add_argument("--boxes_dir", type=str, help="Directory containing YOLO boxes")
    infer_batch_parser.add_argument("--crops_dir", type=str, help="Directory containing pre-cropped images")
    infer_batch_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    infer_batch_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    infer_batch_parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")

    # Init-config command
    init_config_parser = subparsers.add_parser("init-config", help="Interactively generate class configuration from dataset")
    init_config_parser.add_argument("--data", type=str, required=True, help="Path to dataset (zip or directory)")
    init_config_parser.add_argument("--config", type=str, default="config.yaml", help="Output config path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "init-config":
        from sku_clf.utils.config_gen import interactive_config_gen
        interactive_config_gen(args.data, args.config)
        sys.exit(0)

    config = load_config(args.config)
    config = merge_args_with_config(args, config)
    
    from sku_clf.utils.validators import validate_config
    try:
        validate_config(config)
    except Exception as e:
        print(f"ConfigError: {e}")
        sys.exit(1)

    if args.command == "train":
        from sku_clf.train import train
        train(config, args.data)
    elif args.command == "eval":
        from sku_clf.evaluate import evaluate
        evaluate(config, args.checkpoint, args.data)
    elif args.command == "infer":
        from sku_clf.infer import infer
        infer(config, args.checkpoint, image=args.image, boxes=args.boxes, crops=args.crops, save_vis=args.save_vis)
    elif args.command == "infer-batch":
        from sku_clf.infer_batch import infer_batch
        infer_batch(config, args.checkpoint, images_dir=args.images_dir, boxes_dir=args.boxes_dir, crops_dir=args.crops_dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
