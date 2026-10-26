"""Train or evaluate a YAML texture retrieval run with paired-view recall and mAP."""

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml

from dataloaders.textures import TexturesDataModule
from lightningmodules.retrieval import RetrievalModule
from run_utils import common_data_options, load_run_config, make_parameters, make_trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file in configs/ or an absolute path")
    parser.add_argument("--root", help="Override the texture dataset root")
    parser.add_argument("--output", help="Override the output directory")
    parser.add_argument("--epochs", type=int, help="Override training epochs; use 1 for a full trial epoch")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "mps"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true", help="One epoch, two train batches, one val/test batch")
    parser.add_argument("--checkpoint", type=Path, help="Resume training or load weights for --eval-only")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate --checkpoint on texture test split")
    args = parser.parse_args(argv)
    config, path = load_run_config(args.config, "retrieval")
    if args.eval_only and args.checkpoint is None:
        parser.error("--eval-only requires --checkpoint")
    for argument, key in ((args.root, "root"), (args.output, "output"),
                          (args.epochs, "epochs"), (args.batch_size, "batch_size"),
                          (args.num_workers, "num_workers"), (args.accelerator, "accelerator"),
                          (args.seed, "seed")):
        if argument is not None:
            config[key] = argument
    if config["epochs"] < 1 or config["batch_size"] < 2 or config["num_workers"] < 0:
        parser.error("epochs must be >= 1, batch size >= 2, and workers >= 0")
    if args.smoke:
        config["epochs"] = 1
        config["batch_size"] = 2
        config["num_workers"] = 0
    pl.seed_everything(config["seed"], workers=True)
    run_name = f"{path.stem}_seed{config['seed']}"
    run_dir = Path(config["output"]) / (f"smoke_{run_name}" if args.smoke else run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    data = TexturesDataModule(
        root=config["root"], families=config["families"],
        label_mode=config["label_mode"], image_size=config["image_size"],
        **common_data_options(config),
    )
    model = RetrievalModule(make_parameters(config, "retrieval"),
                            pretrained_backbone=args.checkpoint is None)
    trainer = make_trainer(config, run_dir, smoke=args.smoke)

    if args.eval_only:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        trainer.fit(model, datamodule=data, ckpt_path=str(args.checkpoint) if args.checkpoint else None)
        print(f"Checkpoint: {run_dir / 'checkpoints' / 'last.ckpt'}")
    metrics = trainer.test(model, datamodule=data)
    (run_dir / "evaluation.yaml").write_text(yaml.safe_dump(metrics[0], sort_keys=False))


if __name__ == "__main__":
    main()
