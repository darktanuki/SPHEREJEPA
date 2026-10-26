#%%
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torch import nn
import kornia.augmentation as K
from dataloaders.utils import (
    ImageDataModule, ViewDataset, MultiViewTransform,
    normalize_function, unnormalize_function,
)

class RandomAffineWithPhotometricNoise(nn.Module):
    """Center-pad, affine, brightness/contrast, resize, clamp, normalize."""

    def __init__(
        self,
        img_size=(32, 32),
        padded_size=42,
        degrees=(-20, 20),
        translate=(0.2, 0.2),
        scale=(1.4, 1.4),
        shear=(-10, 10),
        brightness_max=0.4,
        contrast_max=0.4,
        p_affine=1.0,
        p_photometric=1.0,
        normalize_mean=0.5,
        normalize_std=0.5,
    ):
        super().__init__()

        # ---- Geometric augmentation ----
        self.affine = K.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            p=p_affine,
            keepdim=True,
            padding_mode="zeros",
        )

        # ---- Photometric augmentations ----
        self.brightness = K.RandomBrightness(
            brightness=(1 - brightness_max, 1 + brightness_max),
            p=p_photometric,
        )

        self.contrast = K.RandomContrast(
            contrast=(1 - contrast_max, 1 + contrast_max),
            p=p_photometric,
        )



        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.normalize = normalize_function(
            mean=self.normalize_mean,
            std=self.normalize_std,
        )

        # inverse transform (for visualization/debug)
        self.unnormalize = unnormalize_function(
            mean=self.normalize_mean,
            std=self.normalize_std,
        )
        self.img_size = img_size
        self.padded_size = padded_size
        # ---- Final resize (ensure fixed output resolution) ----
        # Kornia expects size as (H, W)
        self.final_resize = K.Resize(size=self.img_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the full augmentation pipeline.
        """
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        device = x.device
        x = x.cpu()

        # ---- Embed into black square before augmentation ----
        B, C, H, W = x.shape
        padded_size = self.padded_size
        padded_batch = torch.zeros((B, C, padded_size, padded_size), dtype=x.dtype)

        # center placement
        start_h = (padded_size - H) // 2
        start_w = (padded_size - W) // 2
        padded_batch[:, :, start_h:start_h + H, start_w:start_w + W] = x

        x = padded_batch

        # ---- Geometric ----
        x = self.affine(x)

        # ---- Photometric ----
        x = self.brightness(x)
        x = self.contrast(x)


        # ---- Ensure fixed output resolution ----
        # (B, 1, H, W) -> (B, 1, img_size[0], img_size[1])
        x = self.final_resize(x)

        # Clamp before normalization
        x = torch.clamp(x, 0.0, 1.0)

        # ---- Normalize to [-1, 1] using configured mean/std ----
        x = self.normalize(x)

        # Back to original device
        x = x.to(device)

        if single:
            x = x.squeeze(0)

        return x


class DeterministicPadResizeNormalize(nn.Module):
    """
    Deterministic preprocessing that mirrors the non-random structural steps of
    RandomAffineWithPhotometricNoise:
    - embed the image into a black square of size `padded_size`
    - resize to `img_size`
    - clamp to [0, 1]
    - normalize with the configured mean/std

    Expected input: Tensor of shape (B, C, H, W) or (C, H, W)
    """

    def __init__(
        self,
        img_size=(28, 28),
        padded_size=42,
        normalize_mean=0.5,
        normalize_std=0.5,
    ):
        super().__init__()
        self.img_size = img_size
        self.padded_size = padded_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.normalize = normalize_function(mean=normalize_mean, std=normalize_std)
        self.unnormalize = unnormalize_function(mean=normalize_mean, std=normalize_std)
        self.final_resize = K.Resize(size=self.img_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        device = x.device
        x = x.cpu()

        B, C, H, W = x.shape
        padded_size = self.padded_size
        padded_batch = torch.zeros((B, C, padded_size, padded_size), dtype=x.dtype)

        start_h = (padded_size - H) // 2
        start_w = (padded_size - W) // 2
        padded_batch[:, :, start_h:start_h + H, start_w:start_w + W] = x
        x = padded_batch

        x = self.final_resize(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.normalize(x)
        x = x.to(device)

        if single:
            x = x.squeeze(0)

        return x



class MNISTDataModule(ImageDataModule):
    num_classes = 10
    in_channels = 1

    def __init__(self, root="data", image_size=32, val_size=5000,
                 padded_size=42, eval_mode=False, **kwargs):
        super().__init__(**kwargs)
        self.root = str(Path(root))
        if image_size < 1 or padded_size < 28:
            raise ValueError("image_size must be positive and padded_size must fit a 28x28 digit.")
        self.image_size = image_size
        self.padded_size = padded_size
        self.eval_mode = eval_mode
        self.val_size = val_size

    def prepare_data(self):
        MNIST(self.root, train=True, download=True)
        MNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        raw = MNIST(self.root, train=True)
        if not 0 < self.val_size < len(raw):
            raise ValueError("val_size must leave nonempty train and validation splits.")
        train, val = random_split(raw, [len(raw) - self.val_size, self.val_size],
                                 generator=torch.Generator().manual_seed(self.seed))
        training = T.Compose([
            T.ToTensor(),
            RandomAffineWithPhotometricNoise(
                img_size=(self.image_size, self.image_size), padded_size=self.padded_size,
            ),
        ])
        evaluation = T.Compose([
            T.ToTensor(),
            DeterministicPadResizeNormalize(
                img_size=(self.image_size, self.image_size), padded_size=self.padded_size,
            ),
        ])
        train_tf = MultiViewTransform(evaluation if self.eval_mode else training, self.num_views)
        eval_tf = MultiViewTransform(evaluation, self.num_views)
        self.train_dataset = ViewDataset(train, train_tf)
        self.val_dataset = ViewDataset(val, eval_tf)
        self.test_dataset = ViewDataset(MNIST(self.root, train=False), eval_tf)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloaders.utils import plot_dataset_batch

    # Edit these defaults directly before running this cell.
    root = "data"
    image_size = 32
    num_images = 6
    split = "train"

    data = MNISTDataModule(
        root=root, image_size=image_size, batch_size=num_images, num_workers=0,
    )
    plot_dataset_batch(data, "MNIST", split=split)
    plt.show()

# %%
