# Experiment configurations

These YAML files map the experiments in [SPHERE-JEPA, Section 6 and Appendices C–E](https://arxiv.org/pdf/2605.26900) to this project and to the launch defaults in `/Users/darktanuki/Retrieval`. Each file describes one run. The paper reports four independent pretraining runs for ImageNet-100/ResNet-18, Galaxy10/ResNet-50, and ImageNet-100/ViT-Tiny; `seed: 0` is one starting run, not the full reported mean and standard deviation. The paper does not state the four seed values.

| Files | Paper experiment | Main settings |
| --- | --- | --- |
| `imagenet100_resnet18_{spherejepa,lejepa}.yaml` | Table 4 | ResNet-18, 200 epochs, two views, batch 256, LR 0.05, WD 0.0005 |
| `galaxy10_resnet50_{spherejepa,lejepa}.yaml` | Table 5 | ResNet-50, same ResNet schedule |
| `imagenet100_tinyvit_{spherejepa,lejepa}.yaml` | Table 7 | ViT-Tiny, six views (two global), 200 epochs, batch 256 |
| `texture_{disk,cloud,flake,wood}_{spherejepa,lejepa}.yaml` | Tables 2 and 8–11 | DINOv3 ConvNeXt Small, two views, batch 100, 150 epochs |

The ResNet and ViT-Tiny settings also follow `Retrieval/run_pretrain.py` and its ImageNet-100/Galaxy10 launch scripts. Texture training settings follow `Retrieval/run_train.py`. The paper specifies a 256-dimensional projection head, regularization weight 0.05, and 1024 random projections for the main comparisons. `knots: 151` for SPHERE-JEPA and `knots: 51` for LeJEPA follow the original loss constructors in `Retrieval`; the paper does not state these quadrature counts.

The YAML keys `name_data`, `backbone_name`, `loss_name`, `in_channels`, `projection_dim`, `online_probe`, `lr`, `weight_decay`, `lambda_reg`, `global_views`, `num_projections`, `knots`, and `seed` correspond to the classes in `parameters.py`. The remaining keys configure a DataModule or a Lightning Trainer. Run supported files with `python -m run_pretrained --config configs/<file>.yaml` or `python -m run_retrieval --config configs/<file>.yaml`. Both commands accept `--smoke` for a one-epoch, two-training-batch pipeline check and `--epochs 1` for a complete first epoch. Configs marked `reference_only`, if added, are rejected before model or dataset loading. To construct parameters directly in Python:

```python
from dataclasses import fields
from pathlib import Path
import yaml
from parameters import PretrainedParameters

config = yaml.safe_load(Path("configs/imagenet100_resnet18_spherejepa.yaml").read_text())
names = {field.name for field in fields(PretrainedParameters)}
params = PretrainedParameters(**{key: value for key, value in config.items() if key in names})
```

The texture configurations point to the existing local dataset root, `/Users/darktanuki/Dataset`; change `root` on another machine. The paper uses 10,000 training, 500 validation, and 10,000 test images **per family**. The local Disk and Cloud folders have those counts, but the local Flake and Wood folders currently contain only 100 images per split. Their configurations are ready, but results from those folders are not directly comparable to the paper.

The current modules use a one-epoch linear warmup followed by a constant learning rate, while the paper describes cosine annealing. The current spherical regularizer also fixes `t_max=5`, whereas `Retrieval/src/spherejepa.py` uses `t_max=8`; YAML cannot change that today. Galaxy10 validation splitting and training `drop_last` also differ between this project and `Retrieval`. The online probe in pretraining is for monitoring; `run_pretrained.py` additionally evaluates a separate frozen-backbone linear probe and weighted k-NN on deterministic features. These differences must be resolved before claiming numerical reproduction of a table.

The ImageNet-1K experiments are omitted because this project has no ImageNet-1K loader, ViT-B/14 backbone, EMA teacher, or corresponding trainer. The lambda sweep can be run by copying an ImageNet-100/ResNet-18 YAML and changing `lambda_reg`. The paper's transfer datasets, ADE20K segmentation evaluation, and BYOL/VICReg texture baselines likewise need code beyond these YAML files. `mnist.yaml` and `textures.yaml` are local examples rather than experiments reported in the paper.
