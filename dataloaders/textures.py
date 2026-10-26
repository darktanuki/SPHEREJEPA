# %% Imports
from pathlib import Path

"""Load Retrieval's texture families or its disk/flake class-folder dataset."""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from dataloaders.utils import ImageDataModule, ViewDataset, MultiViewTransform, RandomAffineWithPhotometricNoise

TEXTURE_FAMILIES = ("texture_disk", "texture_cloud", "texture_wood", "texture_flake")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class TextureImages(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("L"), label


class TexturesDataModule(ImageDataModule):
    """Use family labels or explicit class folders; never fabricate labels.

    family: root/<family>/<split>/all_images/*.png, label = family index.
    folders: root/<dataset>/<split>/<class>/*.png, label = class index.
    """

    in_channels = 1

    def __init__(self, root, families=TEXTURE_FAMILIES, image_size=224,
                 label_mode="family", **kwargs):
        super().__init__(**kwargs)
        if label_mode not in {"family", "folders"}:
            raise ValueError("label_mode must be 'family' or 'folders'.")
        self.root = Path(root)
        self.families = tuple(families)
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("Provide at least one unique texture family.")
        if label_mode == "folders" and len(self.families) != 1:
            raise ValueError("Folder labels require exactly one dataset name.")
        self.image_size = image_size
        self.label_mode = label_mode
        self.classes = list(self.families) if label_mode == "family" else []

    @property
    def num_classes(self):
        if not self.classes:
            self._discover_classes()
        return len(self.classes)

    def _discover_classes(self):
        path = self.root / self.families[0] / "train"
        if not path.is_dir():
            raise FileNotFoundError(f"Missing texture directory: {path}")
        self.classes = sorted(p.name for p in path.iterdir() if p.is_dir())
        if not self.classes or "all_images" in self.classes:
            raise ValueError("Folder mode requires real class directories, e.g. disk/ and flake/.")

    def _samples(self, split):
        directories = []
        if self.label_mode == "family":
            directories = [(self.root / family / split / "all_images", label)
                           for label, family in enumerate(self.families)]
        else:
            base = self.root / self.families[0] / split
            if not base.is_dir():
                raise FileNotFoundError(f"Missing texture directory: {base}")
            names = sorted(p.name for p in base.iterdir() if p.is_dir())
            if names != self.classes:
                raise ValueError(f"Texture classes in {split} differ from train: {names}")
            directories = [(base / name, label) for label, name in enumerate(self.classes)]
        samples = []
        for directory, label in directories:
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Missing texture directory: {directory.resolve()}. "
                    "Set root to your existing dataset directory, or create images "
                    "with dataloaders/generate_dataset.py."
                )
            files = sorted(p for p in directory.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if not files:
                raise ValueError(f"No images in {directory}")
            samples.extend((path, label) for path in files)
        return samples

    def setup(self, stage=None):
        if self.label_mode == "folders":
            self._discover_classes()
        # Preserve Retrieval's preprocessing order, including normalization
        # before Kornia and again inside the augmentation module.
        augmentation = T.Compose([
            T.ToTensor(), T.Normalize([0.5], [0.5]),
            RandomAffineWithPhotometricNoise(img_size=(self.image_size, self.image_size)),
        ])
        views = MultiViewTransform(augmentation, self.num_views)
        # Retrieval applies the same stochastic pipeline to all texture splits.
        self.train_dataset = ViewDataset(TextureImages(self._samples("train")), views)
        self.val_dataset = ViewDataset(TextureImages(self._samples("val")), views)
        self.test_dataset = ViewDataset(TextureImages(self._samples("test")), views)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloaders.utils import plot_dataset_batch

    # Edit these defaults directly before running this cell.
    root = Path.home() / "Dataset"
    families = TEXTURE_FAMILIES
    label_mode = "family"
    image_size = 224
    num_images = 6
    split = "train"

    data = TexturesDataModule(
        root=root, families=families, label_mode=label_mode,
        image_size=image_size, batch_size=num_images, num_workers=0,
    )
    plot_dataset_batch(data, "Textures", split=split)
    plt.show()
