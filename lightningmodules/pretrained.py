"""SSL pretraining with an online linear probe on detached encoder features."""

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP

from losses.lejepa import LEJEPA
from losses.spherejepa import SPHEREJEPA
from parameters import PretrainedParameters
from utils import ConvNet

import timm 
import torchvision 


class PretrainedModule(pl.LightningModule):
    """Use the same f/g/loss flow as RetrievalModule, with a fixed probe LR."""

    PROBE_LR = 0.1

    def __init__(self, params: PretrainedParameters):
        super().__init__()
        self.params = params
        self.save_hyperparameters(params.__dict__)


        self.name_data = str(getattr(self.params, 'name_data', 'imagenet100')).lower()
        self.loss_name = str(getattr(self.params, 'loss_name', 'spherejepa')).lower()
        self.backbone_name = str(getattr(self.params, 'backbone_name', 'resnet18')).lower()
        self.use_online_probe        = bool(getattr(self.params, "online_probe", False))
        self.normalize_output = self.loss_name == "spherejepa"

        self.proj_out_dim = int(getattr(self.params, 'projection_dim', getattr(self.params, 'proj_out_dim', 256)))
        self.lr = float(getattr(self.params, 'lr', 5e-2))
        self.probe_lr = self.PROBE_LR
        self.wd = float(getattr(self.params, 'weight_decay', getattr(self.params, 'wd', 5e-4)))

        num_classes = {"imagenet100": 100, "galaxy10": 10, "mnist": 10}.get(self.name_data, 10)


        if self.name_data == 'imagenet100':                
            if self.backbone_name == 'resnet18':
                resnet = torchvision.models.resnet18(weights=None)  # no pretraining
                feature_dim = list(resnet.children())[-1].in_features
                self.f = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten(start_dim=1))
            elif self.backbone_name == 'tinyvit': 
                self.f = timm.create_model("vit_tiny_patch16_224", pretrained=False, in_chans=3, drop_path_rate=0.0, dynamic_img_size=True, num_classes=0, global_pool="token")
                feature_dim = self.f.num_features
            else:
                raise ValueError(f"Unsupported ImageNet-100 backbone: {self.backbone_name}")
            
        elif self.name_data == 'galaxy10':
            resnet = torchvision.models.resnet50(weights=None)  # no pretraining
            feature_dim = list(resnet.children())[-1].in_features
            self.f = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten(start_dim=1))
        elif self.name_data == 'mnist':
            self.f = ConvNet(c_in=1, image_width=32)
            feature_dim = self.f.in_dim
        else:
            raise ValueError(f"Unsupported dataset: {self.name_data}")
            
        in_dim = feature_dim ; proj_hidden_dim = 2048; num_layers=3; out_dim=self.proj_out_dim
        hidden_channels = [proj_hidden_dim] * (num_layers - 1) + [out_dim]
        self.g = MLP(in_channels=in_dim, hidden_channels=hidden_channels, activation_layer=nn.GELU, norm_layer=nn.BatchNorm1d, dropout=0.0) # type: ignore



        if self.loss_name == "spherejepa":
            self.loss = SPHEREJEPA(params=self.params)
        elif self.loss_name == "lejepa":
            self.loss = LEJEPA(params=self.params)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")
        self.linear_prob = (nn.Linear(feature_dim, num_classes) if self.use_online_probe else None) #type:ignore 

    def forward(self, xs):
        embs = [self.f(x) for x in xs]
        zs = [self.g(emb) for emb in embs]
        if self.normalize_output:
            zs = [F.normalize(z, dim=-1) for z in zs]
        return zs, embs

    def info_loss(self, batch, batch_idx, mode="train"):
        xs, labels = batch
        zs, embs = self.forward(xs)
        ssl_loss, ssl_metrics = self.loss(zs, mode=mode)
        if self.linear_prob is None:
            return ssl_loss, ssl_metrics

        probe_logits = torch.cat([self.linear_prob(emb.detach()) for emb in embs])
        probe_labels = labels.long().repeat(len(embs))
        probe_loss = F.cross_entropy(probe_logits, probe_labels)
        probe_accuracy = (probe_logits.argmax(dim=1) == probe_labels).float().mean()
        metrics = {**ssl_metrics, f"{mode}/Lprob": probe_loss.detach(), f"{mode}/Acc": probe_accuracy}
        return ssl_loss + probe_loss, metrics

    def training_step(self, batch, batch_idx, mode="train"):
        loss, metrics = self.info_loss(batch, batch_idx, mode=mode)
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
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
        parameter_groups = [{ "params": list(self.f.parameters()) + list(self.g.parameters()), "lr": self.lr,}]

        if self.linear_prob is not None:
            parameter_groups.append({"params": list(self.linear_prob.parameters()), "lr": self.probe_lr})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=self.wd)

        warmup_steps = max(1, self.trainer.num_training_batches)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps) #type: ignore

        return { "optimizer": optimizer, "lr_scheduler": { "scheduler": scheduler, "interval": "step", "frequency": 1, }, }
