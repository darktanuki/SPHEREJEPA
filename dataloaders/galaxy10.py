# %% 

import torch
from torch.utils.data import random_split
from torchvision import transforms as T
from lightly.transforms import BYOLTransform
from lightly.transforms.byol_transform import BYOLView1Transform, BYOLView2Transform
from dataloaders.utils import HFImages, load_hf, ImageDataModule, ViewDataset, MultiViewTransform



class Galaxy10DataModule(ImageDataModule):
    num_classes = 10
    in_channels = 3
    dataset_id = "matthieulel/galaxy10_decals"

    def __init__(self, root="data/galaxy10", image_size=224, val_fraction=0.1, **kwargs):
        super().__init__(**kwargs)
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1).")
        if self.num_views != 2:
            raise ValueError("Galaxy10 uses exactly two BYOL views, as in Retrieval.")
        self.root = root
        self.image_size = image_size
        self.val_fraction = val_fraction

    def prepare_data(self):
        load_hf(self.dataset_id, self.root)

    def setup(self, stage=None):
        ds = load_hf(self.dataset_id, self.root)
        train = HFImages(ds["train"])
        if "validation" in ds:
            val = HFImages(ds["validation"])
        else:
            count = max(1, int(len(train) * self.val_fraction))
            if count >= len(train):
                raise ValueError("Galaxy10 training split is too small.")
            train, val = random_split(train, [len(train) - count, count],
                                     generator=torch.Generator().manual_seed(self.seed))
        normalize = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        byol = BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=self.image_size, normalize=normalize),
            view_2_transform=BYOLView2Transform(input_size=self.image_size, normalize=normalize),
        )
        train_tf = MultiViewTransform(byol, self.num_views)
        evaluation = T.Compose([
            T.Resize(self.image_size), T.CenterCrop(self.image_size), T.ToTensor(),
            T.Normalize(normalize["mean"], normalize["std"]),
        ])
        eval_tf = train_tf if self.retrieval_eval else MultiViewTransform(evaluation, self.num_views)
        self.train_dataset = ViewDataset(train, train_tf)
        self.val_dataset = ViewDataset(val, eval_tf)
        self.test_dataset = ViewDataset(HFImages(ds["test"]), eval_tf) if "test" in ds else None


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloaders.utils import plot_dataset_batch

    # Edit these defaults directly before running this cell.
    root = "data/galaxy10"
    image_size = 224
    num_images = 2
    split = "train"

    data = Galaxy10DataModule(
        root=root, image_size=image_size, batch_size=num_images, num_workers=0,
    )
    plot_dataset_batch(data, "Galaxy10", split=split)
    plt.show()

# %%
