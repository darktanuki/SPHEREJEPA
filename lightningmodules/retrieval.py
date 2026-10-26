"""Two-view SSL training and retrieval evaluation, adapted from Retrieval."""

import csv
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP

from losses.lejepa import LEJEPA
from losses.spherejepa import SPHEREJEPA
from parameters import RetrievalParameters

import timm 


class RetrievalModule(pl.LightningModule):
    """Match two augmented views, with paired retrieval metrics per batch."""

    def __init__(self, params: RetrievalParameters, pretrained_backbone=True):
        super().__init__()
        self.params = params
        self.save_hyperparameters(params.__dict__)


        self.loss_name = str(getattr(self.params, 'loss_name', 'spherejepa')).lower()
        self.backbone_name = str(getattr(self.params, 'backbone_name', 'convnext_s')).lower()
        self.normalize_output = self.loss_name == "spherejepa"

        self.in_channels = int(getattr(self.params, 'in_channels', 1))


        timm_name = {
            "convnext_s": "convnext_small.dinov3_lvd1689m",
            "convnext_b": "convnext_base.dinov3_lvd1689m",
            "convnext_l": "convnext_large.dinov3_lvd1689m",
        }.get(self.backbone_name)
        if timm_name is None:
            raise ValueError(f"Unsupported retrieval backbone: {self.backbone_name}")
        self.f = timm.create_model(timm_name, pretrained=pretrained_backbone, in_chans=self.in_channels, drop_path_rate=0.0, num_classes=0, global_pool="avg")
        self.g = MLP(in_channels=self.f.num_features, hidden_channels=[2048, 2048, self.params.projection_dim], activation_layer=nn.GELU, norm_layer=nn.BatchNorm1d, dropout=0.0) #type: ignore


        if self.loss_name == "spherejepa":
            self.loss = SPHEREJEPA(params=self.params)
        elif self.loss_name == "lejepa":
            self.loss = LEJEPA(params=self.params)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

    def forward(self, xs):
        embs = [self.f(x) for x in xs]
        zs = [self.g(emb) for emb in embs]
        if self.normalize_output:
            zs = [F.normalize(z, dim=-1) for z in zs]
        return zs, embs

    def info_loss(self, batch, batch_idx, mode="train"):
        xs, _ = batch
        zs, embs = self.forward(xs)
        loss, loss_metrics = self.loss(zs, mode=mode)

        if mode in ("val", "test"):
            emb1, emb2 = embs
            z1, z2 = zs
            # Each query's positive is the matching view at the same index.
            sim_emb = -(emb1.unsqueeze(0) - emb2.unsqueeze(1)).square().mean(-1)
            sim_z = -(z1.unsqueeze(0) - z2.unsqueeze(1)).square().mean(-1)
            metrics = {
                **self.compute_eval_stats(sim_emb, mode=mode, suffix="/emb"),
                **self.compute_eval_stats(sim_z, mode=mode, suffix="/z"),
            }
        else:
            metrics = loss_metrics
        return loss, metrics

    def training_step(self, batch, batch_idx, mode="train"):
        loss, metrics = self.info_loss(batch, batch_idx, mode=mode)
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, mode="val"):
        loss, metrics = self.info_loss(batch, batch_idx, mode=mode)
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx, mode="test"):
        loss, metrics = self.info_loss(batch, batch_idx, mode=mode)
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW( list(self.f.parameters()) + list(self.g.parameters()), lr=self.params.lr, weight_decay=self.params.weight_decay)
        warmup_steps = max(1, self.trainer.num_training_batches)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps) #type: ignore
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def compute_eval_stats(self, sim, mode="val", suffix="", sim_to_compare=None):
        """Compute batch Recall@1/3/5 and single-positive mAP from similarities."""
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
            raise ValueError("Expected a square (batch, batch) similarity matrix.")
        positive = sim.diagonal() if sim_to_compare is None else sim_to_compare
        ranks = (sim > positive.unsqueeze(-1)).sum(dim=1).float()
        return {f"{mode}/R@{k}{suffix}": (ranks < k).float().mean() for k in (1, 3, 5)} | {
            f"{mode}/mAP{suffix}": (1.0 / (ranks + 1.0)).mean()
        }
