"""Configuration and Trainer helpers for the two command-line entry points."""

from dataclasses import fields
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from parameters import PretrainedParameters, RetrievalParameters


PROJECT_ROOT = Path(__file__).resolve().parent


def load_run_config(path, task):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    if config.get("status") == "reference_only":
        raise ValueError(f"{path.name} is reference-only; its model/data pipeline is not implemented.")
    if config.get("task") != task:
        raise ValueError(f"{path.name} has task={config.get('task')!r}; expected {task!r}.")
    for key in ("root", "output"):
        if key in config:
            value = Path(config[key]).expanduser()
            config[key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    return config, path


def make_parameters(config, task):
    cls = PretrainedParameters if task == "pretrained" else RetrievalParameters
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in config.items() if key in names})


def common_data_options(config):
    return {key: config[key] for key in ("batch_size", "num_workers", "num_views", "seed")}


def evaluation_device(accelerator):
    if accelerator in ("auto", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if accelerator in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if accelerator in ("gpu", "mps"):
        raise ValueError(f"Requested accelerator {accelerator!r} is unavailable.")
    return torch.device("cpu")


def make_trainer(config, run_dir, smoke=False):
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", save_last=True, save_top_k=0,
    )
    return pl.Trainer(
        default_root_dir=run_dir,
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", 1),
        max_epochs=1 if smoke else config["epochs"],
        limit_train_batches=2 if smoke else 1.0,
        limit_val_batches=1 if smoke else 1.0,
        limit_test_batches=1 if smoke else 1.0,
        num_sanity_val_steps=0,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=True,
    )
