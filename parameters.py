"""Default parameters for the two Lightning modules."""

from dataclasses import dataclass


@dataclass
class PretrainedParameters:
    """Pretraining defaults for MNIST; set dataset-specific values explicitly."""

    name_data: str = "mnist"
    backbone_name: str = "small_cnn"
    loss_name: str = "spherejepa"
    in_channels: int = 1
    projection_dim: int = 64
    online_probe: bool = True
    lr: float = 1e-3
    weight_decay: float = 5e-4
    lambda_reg: float = 0.05
    global_views: int = 2
    num_projections: int = 128
    knots: int = 51
    seed: int = 42


@dataclass
class RetrievalParameters:
    """Retrieval defaults for grayscale texture images and DINOv3 ConvNeXt."""

    name_data: str = "texture_wood"
    backbone_name: str = "convnext_s"
    loss_name: str = "spherejepa"
    in_channels: int = 1
    projection_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 5e-4
    lambda_reg: float = 0.05
    global_views: int = 2
    num_projections: int = 128
    knots: int = 51
    seed: int = 42
