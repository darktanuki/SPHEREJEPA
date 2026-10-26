# %% 
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from lightly.transforms import BYOLTransform
from lightly.transforms.byol_transform import BYOLView1Transform, BYOLView2Transform
from lightly.transforms.dino_transform import DINOViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from dataloaders.utils import HFImages, load_hf, ImageDataModule, ViewDataset, MultiViewTransform

class ImageNet100DataModule(ImageDataModule):
    num_classes = 100
    in_channels = 3
    dataset_id = "clane9/imagenet-100"

    def __init__(self, root="data/imagenet100", source="hf", image_size=224, global_views=2, **kwargs):
        super().__init__(**kwargs)
        if source not in {"hf", "folder"}:
            raise ValueError("source must be 'hf' or 'folder'.")
        if not 1 <= global_views <= self.num_views:
            raise ValueError("global_views must be between 1 and num_views.")
        self.global_views = global_views
        self.root = Path(root)
        self.source = source
        self.image_size = image_size

    def prepare_data(self):
        if self.source == "hf":
            load_hf(self.dataset_id, self.root)

    def setup(self, stage=None):
        if self.source == "hf":
            ds = load_hf(self.dataset_id, self.root)
            train, val = HFImages(ds["train"]), HFImages(ds["validation"])
            test = HFImages(ds["test"]) if "test" in ds else None
        else:
            train, val = ImageFolder(self.root / "train"), ImageFolder(self.root / "val")
            test = ImageFolder(self.root / "test") if (self.root / "test").is_dir() else None
            if len(train.classes) != 100:
                raise ValueError("ImageNet100 must contain exactly 100 training classes.")
            for split in (val, test):
                if split is not None and split.class_to_idx != train.class_to_idx:
                    raise ValueError("Class mappings differ between ImageNet100 splits.")
        train_tf = self.training_transform()
        evaluation = T.Compose([
            T.Resize(self.image_size), T.CenterCrop(self.image_size), T.ToTensor(),
            T.Normalize(IMAGENET_NORMALIZE["mean"], IMAGENET_NORMALIZE["std"]),
        ])
        eval_tf = train_tf if self.retrieval_eval else MultiViewTransform(evaluation, self.num_views)
        self.train_dataset = ViewDataset(train, train_tf)
        self.val_dataset = ViewDataset(val, eval_tf)
        self.test_dataset = ViewDataset(test, eval_tf) if test is not None else None

    def training_transform(self):
        """Match Retrieval: BYOL for two views, DINO crops for more views."""
        if self.num_views == 2:
            byol = BYOLTransform(
                view_1_transform=BYOLView1Transform(input_size=self.image_size),
                view_2_transform=BYOLView2Transform(input_size=self.image_size),
            )
            return MultiViewTransform(byol, self.num_views)
        crops = [
            DINOViewTransform(
                crop_size=self.image_size, crop_scale=(0.4, 1.0),
                gaussian_blur=1.0 if index == 0 else 0.1,
                solarization_prob=0.2 if index == 1 else 0.0,
                normalize=IMAGENET_NORMALIZE,
            )
            for index in range(self.global_views)
        ]
        crops.extend(
            DINOViewTransform(
                crop_size=96, crop_scale=(0.05, 0.4), gaussian_blur=0.5,
                solarization_prob=0.0, normalize=IMAGENET_NORMALIZE,
            )
            for _ in range(self.num_views - self.global_views)
        )
        return MultiViewTransform(crops, self.num_views)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloaders.utils import plot_dataset_batch

    # Edit these defaults directly before running this cell.
    root = "data/imagenet100"
    source = "hf"
    image_size = 224
    num_images = 2
    num_views = 2
    global_views = 2
    split = "train"

    data = ImageNet100DataModule(
        root=root, source=source, image_size=image_size, batch_size=num_images,
        num_workers=0, num_views=num_views, global_views=global_views,
    )
    plot_dataset_batch(
        data, "ImageNet100", split=split,
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    plt.show()

# %%
