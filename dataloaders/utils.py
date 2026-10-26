"""Shared dataset adapters, augmentations, and Lightning data utilities."""


import lightning.pytorch as pl
import torch
from torch import nn
import kornia
import kornia.augmentation as K
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class ViewDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), int(label)


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=0, seed=42, num_views=2,
                 retrieval_eval=False):
        super().__init__()
        if batch_size < 1 or num_workers < 0 or num_views < 2:
            raise ValueError("Invalid batch size, worker count or number of views.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.num_views = num_views
        self.retrieval_eval = retrieval_eval

    def _loader(self, dataset, training=False):
        if dataset is None:
            raise ValueError("This dataset has no independent test split.")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=training,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          drop_last=False)

    def train_dataloader(self):
        return self._loader(self.train_dataset, training=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)


class MultiViewTransform:
    """Wrap a single-view transform, a Lightly pair, or a list of DINO crops."""

    def __init__(self, transform, num_views=2):
        if num_views < 2:
            raise ValueError("Self-supervised learning requires at least two views.")
        self.transform = transform
        self.num_views = num_views
        if isinstance(transform, (list, tuple)) and len(transform) != num_views:
            raise ValueError("The crop transform count must match num_views.")

    def __call__(self, image):
        if isinstance(self.transform, (list, tuple)):
            return [transform(image) for transform in self.transform]
        first = self.transform(image)
        if isinstance(first, (list, tuple)):
            if len(first) != self.num_views:
                raise ValueError("The transform returned a different number of views.")
            return list(first)
        return [first] + [self.transform(image) for _ in range(self.num_views - 1)]


class HFImages(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample["image"].convert("RGB"), sample["label"]


def load_hf(dataset_id, cache_dir):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError('Install Hugging Face support: uv sync --extra hf') from exc
    return load_dataset(dataset_id, cache_dir=str(cache_dir))


def _preview_view_batches(views):
    """Return BCHW batches, including nested pairs and BVCHW crop batches."""
    if isinstance(views, (list, tuple)):
        return [batch for group in views for batch in _preview_view_batches(group)]
    if not isinstance(views, torch.Tensor):
        raise TypeError("Preview views must be tensors or lists of tensors.")
    if views.ndim == 5:
        # Keep every crop; a singleton crop dimension becomes one BCHW batch.
        return list(views.unbind(dim=1))
    if views.ndim == 4:
        return [views]
    raise ValueError(f"Expected BCHW or BVCHW views, received shape {tuple(views.shape)}.")


def plot_dataset_batch(datamodule, title, split="train", mean=0.5, std=0.5):
    """Plot each augmented view in its own row, undoing normalization.

    Dataset entry points call plt.show() after this helper builds the figure.
    Matplotlib is imported only when a preview is requested.
    """
    import matplotlib.pyplot as plt

    datamodule.prepare_data()
    datamodule.setup()
    loaders = {"train": datamodule.train_dataloader,
               "val": datamodule.val_dataloader,
               "test": datamodule.test_dataloader}
    if split not in loaders:
        raise ValueError("split must be 'train', 'val', or 'test'.")
    views, labels = next(iter(loaders[split]()))
    views = _preview_view_batches(views)
    count = len(labels)
    for batch in views:
        if batch.shape[0] != count or batch.shape[1] not in (1, 3):
            raise ValueError(
                f"Expected ({count}, 1 or 3, H, W), received {tuple(batch.shape)}."
            )
    figure, axes = plt.subplots(len(views), count, squeeze=False,
                                figsize=(2.5 * count, 2.5 * len(views)))
    classes = getattr(datamodule, "classes", None)
    for row, batch in enumerate(views):
        for column in range(count):
            image = batch[column].detach().cpu()
            image = image * image.new_tensor(std).reshape(-1, 1, 1)
            image = (image + image.new_tensor(mean).reshape(-1, 1, 1)).clamp(0, 1)
            axis = axes[row, column]
            if image.shape[0] == 1:
                axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(image.permute(1, 2, 0).numpy())
            label = int(labels[column])
            label_name = classes[label] if classes else str(label)
            axis.set_title(f"View {row + 1} | {label_name}")
            axis.axis("off")
    figure.suptitle(f"{title} | {split}")
    figure.tight_layout()
    return figure


# Texture augmentation implementation preserved from Retrieval.
def unnormalize_function(mean, std):
    # If mean and std are scalars, wrap them in a list.
    if isinstance(mean, (int, float)):
        mean = [mean]
    if isinstance(std, (int, float)):
        std = [std]
    # Compute unnormalize parameters.
    return Normalize(mean=[-m/s for m, s in zip(mean, std)],
                     std=[1/s for s in std])


def normalize_function(mean, std):
    # If mean and std are scalars, wrap them in a list.
    if isinstance(mean, (int, float)):
        mean = [mean]
    if isinstance(std, (int, float)):
        std = [std]
    return Normalize(mean=mean, std=std)


class RandomAffineWithPhotometricNoise(nn.Module):
    """
    Apply RandomAffine followed by photometric and noise augmentations:
    - Brightness
    - Contrast
    - Uniform noise
    - Random erasing

    Expected input: Tensor of shape (B, C, H, W) or (C, H, W)
    All operations are done on CPU (Kornia-safe).
    """

    def __init__(
        self,
        img_size=(224, 224),
        degrees=(0, 360),
        translate=(0.2, 0.2),
        scale=(1.4, 1.4),
        shear=(-20, 20),
        brightness_max=0.4,
        contrast_max=0.4,
        uniform_noise_scale=0.05,
        erasing_scale=(0.02, 0.1),
        erasing_ratio=(0.5, 2.0),
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

        # ---- Noise ----
        self.uniform_noise_scale = uniform_noise_scale

        # ---- Erasing ----
        self.erasing = K.RandomErasing(
            scale=erasing_scale,
            ratio=erasing_ratio,
            value=0.0,
            p=1.0,
            same_on_batch=False,
        )

        # ---- Grayscale + Normalization ----
        # NOTE: Kornia RandomGrayscale keeps 3 channels.
        # We explicitly convert to 1-channel grayscale.
        self.force_grayscale = True

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
        # ---- Final resize (ensure fixed output resolution) ----
        # Kornia expects size as (H, W)
        self.final_resize = K.Resize(size=self.img_size)

    def add_uniform_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add uniform noise in [-scale, +scale].
        """
        if self.uniform_noise_scale <= 0:
            return x
        noise = torch.empty_like(x).uniform_(
            -self.uniform_noise_scale,
            self.uniform_noise_scale
        )
        return x + noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the full augmentation pipeline.
        """
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True


        # ---- Grayscale (force 1 channel) ----
        if self.force_grayscale:
            # If already grayscale, keep as-is. Otherwise convert RGB -> gray.
            if x.shape[1] == 3:
                x = kornia.color.rgb_to_grayscale(x)
            elif x.shape[1] != 1:
                raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # IMPORTANT: Kornia RandomAffine must run on CPU
        device = x.device
        x = x.cpu()

        # ---- Geometric ----
        x = self.affine(x)

        # ---- Photometric ----
        x = self.brightness(x)
        x = self.contrast(x)

        # ---- Noise + Occlusion ----
        x = self.add_uniform_noise(x)
        x = self.erasing(x)


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
