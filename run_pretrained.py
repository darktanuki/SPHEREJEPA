"""Train a YAML pretraining run and evaluate frozen features with k-NN and a linear probe."""

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml

from dataloaders.galaxy10 import Galaxy10DataModule
from dataloaders.imagenet100 import ImageNet100DataModule
from dataloaders.mnist import MNISTDataModule
from evaluation import evaluate_classification
from lightningmodules.pretrained import PretrainedModule
from run_utils import (
    common_data_options, evaluation_device, load_run_config, make_parameters, make_trainer,
)


def make_data(config):
    name = config["name_data"]
    options = common_data_options(config)
    if name == "mnist":
        return MNISTDataModule(root=config["root"], image_size=config["image_size"], **options)
    if name == "galaxy10":
        return Galaxy10DataModule(root=config["root"], image_size=config["image_size"], **options)
    if name == "imagenet100":
        return ImageNet100DataModule(
            root=config["root"], source=config.get("source", "hf"),
            image_size=config["image_size"], global_views=config["global_views"], **options,
        )
    raise ValueError(f"No pretraining DataModule for {name!r}.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file in configs/ or an absolute path")
    parser.add_argument("--root", help="Override the dataset root")
    parser.add_argument("--output", help="Override the output directory")
    parser.add_argument("--epochs", type=int, help="Override pretraining epochs; use 1 for a full trial epoch")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "mps"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--linear-epochs", type=int, help="Frozen linear-probe epochs (default: 90)")
    parser.add_argument("--knn-k", type=int, help="Number of k-NN neighbors (default: 20)")
    parser.add_argument("--smoke", action="store_true", help="One epoch, two train batches, small feature evaluation")
    parser.add_argument("--checkpoint", type=Path, help="Resume training or load weights for --eval-only")
    parser.add_argument("--eval-only", action="store_true", help="Run frozen-feature evaluation from --checkpoint")
    parser.add_argument("--skip-eval", action="store_true", help="Skip k-NN and offline linear probing")
    args = parser.parse_args(argv)
    config, path = load_run_config(args.config, "pretrained")
    if args.eval_only and (args.checkpoint is None or args.skip_eval):
        parser.error("--eval-only requires --checkpoint and cannot be combined with --skip-eval")
    for argument, key in ((args.root, "root"), (args.output, "output"),
                          (args.epochs, "epochs"), (args.batch_size, "batch_size"),
                          (args.num_workers, "num_workers"), (args.accelerator, "accelerator"),
                          (args.seed, "seed"), (args.linear_epochs, "linear_probe_epochs"),
                          (args.knn_k, "knn_k")):
        if argument is not None:
            config[key] = argument
    if config["epochs"] < 1 or config["batch_size"] < 2 or config["num_workers"] < 0:
        parser.error("epochs must be >= 1, batch size >= 2, and workers >= 0")
    if config.get("linear_probe_epochs", 90) < 1 or config.get("knn_k", 20) < 1:
        parser.error("linear-probe epochs and k-NN neighbors must be positive")
    if args.smoke:
        config["epochs"] = 1
        config["batch_size"] = min(config["batch_size"], 8)
        config["num_workers"] = 0
    pl.seed_everything(config["seed"], workers=True)
    run_name = f"{path.stem}_seed{config['seed']}"
    run_dir = Path(config["output"]) / (f"smoke_{run_name}" if args.smoke else run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    data = make_data(config)
    model = PretrainedModule(make_parameters(config, "pretrained"))

    if args.eval_only:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(evaluation_device(config.get("accelerator", "auto"))).eval()
    else:
        trainer = make_trainer(config, run_dir, smoke=args.smoke)
        trainer.fit(model, datamodule=data, ckpt_path=str(args.checkpoint) if args.checkpoint else None)
        print(f"Checkpoint: {run_dir / 'checkpoints' / 'last.ckpt'}")

    if not args.skip_eval:
        metrics = evaluate_classification(model, data, config, smoke=args.smoke)
        (run_dir / "evaluation.yaml").write_text(yaml.safe_dump(metrics, sort_keys=False))
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
