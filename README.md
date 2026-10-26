# SPHEREJEPA

A structured PyTorch/Lightning project adapted from the original `Retrieval`
codebase. Learn representations from augmented views of an image, then evaluate
those representations with **online linear probing** or **paired-view retrieval**.

SPHEREJEPA normalizes projections onto the unit sphere and combines cosine
invariance with SUSReg, targeting a uniform spherical distribution. LEJEPA uses
squared-error invariance and SIGReg with a Gaussian target. Both use:

```text
loss = (1 - lambda_reg) * invariance + lambda_reg * regularization
```

The center is the mean of the first `global_views` views and remains attached
to the computation graph.

## Project layout

```text
SPHEREJEPA/
├── losses/                          # Each objective is self-contained
│   ├── lejepa.py                    # LEJEPA and SIGReg
│   └── spherejepa.py                # SPHEREJEPA and SUSReg
├── dataloaders/                     # All dataset loaders
│   ├── __init__.py                  # Python package
│   ├── mnist.py
│   ├── galaxy10.py
│   ├── imagenet100.py
│   ├── textures.py
│   ├── generate_dataset.py          # Original texture generators from Retrieval
│   └── utils.py                     # Shared dataset adapters and augmentations
├── lightningmodules/
│   ├── pretrained.py               # Pretraining and online linear probe
│   └── retrieval.py                # Training and paired-view retrieval
├── configs/                        # Paper experiment YAMLs and local examples
│   ├── README.md                   # Experiment map and fidelity notes
│   ├── imagenet100_resnet18_spherejepa.yaml
│   ├── galaxy10_resnet50_spherejepa.yaml
│   ├── texture_disk_spherejepa.yaml
│   ├── mnist.yaml
│   └── ...                         # Other losses and texture families
├── run_pretrained.py               # Pretraining plus frozen-feature evaluation
├── run_retrieval.py                # Texture training and retrieval evaluation
├── run_utils.py                    # Shared YAML and Trainer setup
├── evaluation.py                   # Frozen linear probe and weighted k-NN
├── parameters.py                   # PretrainedParameters and RetrievalParameters
└── pyproject.toml
```

## Reproducible environment with uv

Python is pinned to **3.12.13** in `.python-version`. `pyproject.toml` declares
runtime dependencies and the `dev` group includes `ipykernel` for VS Code
interactive cells. `uv.lock` records exact package versions and artifact hashes.

From the project root:

```bash
uv sync --extra hf
```

This creates `.venv/`, installs the project in editable mode, and installs
Lightning, PyTorch, Matplotlib, the Jupyter kernel, and Hugging Face support.
The `hf` extra installs the optional `datasets` package used by Galaxy10 and
ImageNet100. For MNIST and local textures alone, use `uv sync` and omit the
extra from run commands.
`timm` is part of the locked environment for TinyViT and DINOv3 ConvNeXt.
The ConvNeXt weights are downloaded when that encoder is created.

Run commands through `uv run` so they use this environment. Add `--extra hf`
when a command needs a Hugging Face dataset. `uv` uses `uv.lock` automatically
and updates it when project dependencies change; `--locked` is only needed if
you want such a change to stop the command instead.
For example:

```bash
uv run python -c "import sys; print(sys.executable)"
```

In VS Code, select **Python: Select Interpreter**, then `.venv/bin/python`.
For `# %%` cells, also select that environment in the **Interactive Window's
kernel picker** and restart the kernel. Changing the terminal's environment
does not switch an already running notebook kernel. `.vscode/settings.json`
sets the default interpreter for this workspace.

Optionally register a named kernel inside the project environment:

```bash
uv run python -m ipykernel install --prefix .venv \
  --name spherejepa --display-name "SPHEREJEPA (uv)"
```

Keep `pyproject.toml`, `uv.lock`, and `.python-version` under version control;
`.venv/` is local and ignored. Recreate the environment with the same sync
command. To deliberately change dependencies, use `uv add <package>` or
`uv lock --upgrade-package <package>`, then review the resulting lockfile.
Avoid `%pip install` in cells: it would modify the environment outside the lock.
The lockfile fixes the software environment; device choice and nondeterministic
training operations can still affect numerical results.

## YAML configurations and training

All YAML files live in `configs/`; see [configs/README.md](configs/README.md)
for the paper experiment map, parameter values, and reproducibility limits.
Run a supported config from the project root with `uv`:

```bash
uv run python -m run_pretrained --config configs/mnist.yaml --smoke
uv run --extra hf python -m run_pretrained --config configs/imagenet100_resnet18_spherejepa.yaml --epochs 1 --linear-epochs 1
uv run python -m run_retrieval --config configs/texture_wood_spherejepa.yaml --smoke
```

`--smoke` trains for one epoch with two training batches, one validation batch,
and at most two feature-extraction batches per split. `--epochs 1` runs a **complete** training epoch;
for pretraining, `--linear-epochs 1` also shortens the separate linear probe.
Remove these overrides for the YAML's full run. For a checkpoint already saved,
use `--eval-only --checkpoint /path/to/last.ckpt` with the same config. The
checkpoint is saved under `outputs/<config-name>_seed<seed>/checkpoints/last.ckpt`;
`--smoke` uses a separate `smoke_` directory. Resolved YAML and evaluation
metrics are saved alongside it. `run_pretrained` reports frozen-backbone
weighted k-NN and linear-probe Top-1 on validation and, when present, test.
`run_retrieval` reports paired-view Recall@1/3/5 and mAP on test. The online
probe in `PretrainedModule` remains separate from the frozen evaluation probe.

`configs/mnist.yaml` is a local pretraining example with an online linear
probe; `configs/textures.yaml` is a local multi-family retrieval example. They
are not paper experiments. ImageNet-1K experiments cannot run yet.

The texture layout from `Retrieval` is supported:

```text
textures/
├── texture_disk/{train,val,test}/all_images/*.png
├── texture_cloud/{train,val,test}/all_images/*.png
├── texture_wood/{train,val,test}/all_images/*.png
└── texture_flake/{train,val,test}/all_images/*.png
```

Family labels follow the order of `families` and remain consistent across
splits. For explicitly labeled class folders, the mixed dataset layout
`texture_disk_flake/{train,val,test}/{disk,flake}/*.png` is also supported
by setting `label_mode: folders` and `families: [texture_disk_flake]`.

## Lightning modules

- The two modules use `PretrainedParameters` and `RetrievalParameters` from
  `parameters.py`, with an encoder `f`,
  projector `g`, objective `loss`, and `info_loss`. Their `forward(views)`
  methods return lists of projections and embeddings. Each parameter class
  has task-specific defaults, which can be overridden in its constructor.
- For ImageNet100, pretraining uses an untrained ResNet18 or
  `vit_tiny_patch16_224`. Galaxy10 uses an untrained ResNet50. MNIST uses
  the compact CNN. All use a three-layer `torchvision.ops.MLP`
  with 2048 hidden units, GELU, BatchNorm1d, and no dropout.
- Retrieval uses a pretrained DINOv3 ConvNeXt (`convnext_s`, `convnext_b`, or
  `convnext_l`) through `timm` and the same MLP layout as pretraining.
- `PretrainedModule` trains with AdamW and an online linear probe on detached
  encoder features. **The probe learning rate stays at `0.1`**, while the
  encoder/projector learning rate warms up for one epoch. Probe loss and
  accuracy are averaged across views.
- `RetrievalModule` compares the two views in each validation or test batch
  using negative mean squared distance. Query `i` has one positive at index
  `i`; class labels do not define retrieval positives. Recall@1/3/5 and mAP
  are logged separately for encoder features (`/emb`) and projections (`/z`).
  With one positive, mAP equals mean reciprocal rank. Ties use the strict
  comparison from the original `Retrieval` class. Set the retrieval loader to
  exactly two views.
- The retrieval optimizer uses AdamW, one epoch of linear learning-rate
  warmup, then a constant rate.
- MNIST, Galaxy10, and ImageNet100 use deterministic probe evaluation views.
  With `retrieval_eval=True`, their evaluation views are augmented independently.
  Textures always use independent augmented views on every split, matching
  Retrieval. A fixed seed reproduces a complete run.
- Retrieval metrics are computed from each batch's similarity matrix, so
  their candidate set is determined by the evaluation batch size.
- Logs and checkpoints are saved under `outputs/`. Final evaluation uses the
  checkpoint with the best validation probe accuracy or projection Recall@1.
  `--fast-dev-run` processes one batch per phase for a quick smoke check.

## Dataset loaders

`dataloaders/` contains the four dataset modules, the original texture
generation script, and `utils.py`. Shared
DataModule logic, view transforms, and Hugging Face adapters live in
`dataloaders/utils.py`.

Every batch follows `(views, labels)`: `views` is a list of `(B, C, H, W)` tensors
and `labels` is an integer tensor of shape `(B,)`. DataModules expose
`num_classes`, `in_channels`, `train_dataloader()`, `val_dataloader()`, and
`test_dataloader()`.

```python
from parameters import PretrainedParameters
from dataloaders.galaxy10 import Galaxy10DataModule
from dataloaders.imagenet100 import ImageNet100DataModule
from lightningmodules.pretrained import PretrainedModule

data = Galaxy10DataModule(root="data/galaxy10", batch_size=64)
# Hugging Face alternative:
# data = ImageNet100DataModule(root="data/imagenet100")
# Local alternative:
# data = ImageNet100DataModule(root="/data/imagenet100", source="folder")
params = PretrainedParameters(name_data="galaxy10", backbone_name="resnet50",
                              in_channels=3, projection_dim=256)
model = PretrainedModule(params)
```

Galaxy10 uses `matthieulel/galaxy10_decals`, matching the source project. If no
validation split exists, one is taken from training; the test split remains
independent. ImageNet100 uses `clane9/imagenet-100` or local
`{train,val,test}/<class>/*.jpg` folders with exactly 100 classes. When no
independent test split exists, `test_dataloader()` raises an explicit error.

## Augmentations from Retrieval

The Galaxy10, ImageNet100, and texture pipelines reproduce
`Retrieval/dataloader/image_dataloader.py` using Lightly and Kornia. These
packages are included in `pyproject.toml` and pinned by `uv.lock`.

| Dataset | Training augmentations | Default evaluation |
| --- | --- | --- |
| Galaxy10 | Lightly BYOLView1Transform and BYOLView2Transform; normalization mean/std 0.5 | Resize, center crop, tensor conversion, normalization |
| ImageNet100, 2 views | Lightly BYOLTransform with ImageNet normalization | Resize, center crop, tensor conversion, ImageNet normalization |
| ImageNet100, more than 2 views | DINO crops with the original crop scales, blur and solarization probabilities | Same deterministic transform repeated for each view |
| Textures | Original Kornia affine, brightness, contrast, uniform noise, erasing and resize pipeline | Same stochastic pipeline as training |
| MNIST | Supplied Kornia pipeline: black padding, affine, brightness/contrast, resize and normalization | Black padding, resize and normalization |

For ImageNet100, the first `global_views` crops have size 224 by default, scale
`(0.4, 1.0)`, blur probabilities 1.0 for the first crop and 0.1 for the others,
and solarization probability 0.2 only for the second crop. Remaining crops
have size 96, scale `(0.05, 0.4)`, blur probability 0.5, and no solarization.
Configure views through the DataModule, or edit `num_views` and
`global_views` in the ImageNet100 preview cell:

```python
data = ImageNet100DataModule(num_views=6, global_views=2)
# Use PretrainedParameters(global_views=2) for the matching SSL objective.
```

```bash
uv run --extra hf python -m dataloaders.imagenet100
```

Texture defaults match the reference: rotation `(0, 360)`, translation
`(0.2, 0.2)`, scale `(1.4, 1.4)`, shear `(-20, 20)`, brightness/contrast
strength 0.4, uniform noise amplitude 0.05, erasing scale `(0.02, 0.1)` and
ratio `(0.5, 2.0)`, followed by a 224 by 224 resize. The normalization before
Kornia and the normalization after augmentation are both preserved exactly
from the source. Labels still identify texture families or explicit classes.
`image_size` remains configurable; the texture examples default to 224.

MNIST uses the separately supplied MNIST example. It converts the 28 by 28
image to a tensor, centers it on a black 42 by 42 canvas, applies Kornia affine
augmentation (rotation `(-20, 20)`, translation `(0.2, 0.2)`, fixed scale 1.4,
shear `(-10, 10)`), brightness and contrast strength 0.4, resizes to 32 by 32,
clamps to `[0, 1]`, and normalizes with mean/std 0.5. No noise or erasing is
applied to MNIST. Validation and test use the supplied deterministic padding,
resize, clamp, and normalization. `MNISTDataModule(eval_mode=True)` also uses
this deterministic transform for training. Galaxy10 requires exactly two views.
The independent validation/test partitions of this project are preserved;
only augmentation behavior is matched to the reference.

## Create texture datasets

The original generator from `Retrieval/dataloader/generate_dataset.py` is
included as `dataloaders/generate_dataset.py`. Its disk, cloud, wood, flake,
and mixed disk/flake algorithms are preserved. The unused OpenCV import and
CLI parser are omitted; the last cloud function in the source is retained
because it is the one Python actually uses.

The existing local datasets are in `/Users/darktanuki/Dataset`. The texture
preview defaults to `Path.home() / "Dataset"`, so it can use them directly.
For training with those images, set `root` in the texture YAML to that path,
or pass `--root /Users/darktanuki/Dataset` to the training example.

To create a new copy, open the generator and edit the fixed parameters in its
last `# %%` cell, then run that cell or:

```bash
uv run python -m dataloaders.generate_dataset
```

Defaults reproduce the source's sizes: 10,000 training, 500 validation, and
10,000 test images per family, generated at 800 by 800 and saved at 256 by 256.
The four families are written beneath `data/textures/` with the loader's
`<family>/<split>/all_images/*.png` layout. No files are generated on import.
The entry point refuses to replace an existing family directory; choose an
unused output root when recreating data. The source seed and split-specific
seed offsets are preserved. For the mixed dataset, set
`families = ("texture_disk_flake",)` and load it with `label_mode="folders"`.

## Visualize a dataset

Each dataset file has an `if __name__ == "__main__":` entry point that plots a
batch with Matplotlib and calls `plt.show()`. Preview parameters are plain
variables in the visualization cell: there is no argument parser in the
loaders. Edit `root`, `image_size`, `num_images`, and `split` directly.

From the project root:

```bash
uv run python -m dataloaders.mnist
uv run --extra hf python -m dataloaders.galaxy10
uv run --extra hf python -m dataloaders.imagenet100
uv run python -m dataloaders.textures
```

Set `split = "val"` to preview validation. ImageNet100 also exposes `source`,
`num_views`, and `global_views`; set `source = "folder"` and edit `root` for
local data, or `num_views = 6` to preview DINO crops. For textures, edit `root`,
`families`, and `label_mode` to select your local datasets.

Each row shows one view of the same samples, with labels above the images.
Normalization is reversed for display, including ImageNet's channel-specific
normalization. An interactive Matplotlib backend is required to open the window.

### VS Code interactive cells

Open the dataset file and run its `# %%` cells in order using **Run Cell** or
**Shift+Enter**. Edit the defaults in the visualization cell, then run it to
call `plt.show()`.

Select the `.venv/bin/python` kernel created by uv. The project is installed
in editable mode, so `dataloaders` imports normally without `sys.path` changes.
Direct script execution from the project root also works:

```bash
uv run --extra hf python dataloaders/galaxy10.py
```

MNIST and Hugging Face datasets are downloaded if they are not cached. Texture
images must already exist locally. Setting `split = "test"` requires an
independent test split. Training runs use the `run_pretrained` and
`run_retrieval` commands above.

## Implementation choices

The objectives are adapted from `Retrieval/src/spherejepa.py` and
`Retrieval/src/lejepa.py`. Both loss classes receive one of the parameter classes as their
`params` argument and return stage-prefixed `L`, `Linv`, and `Lreg` metrics.
SUSReg receives the configured projection dimension by keyword. SIGReg uses
`dt = t_max / (knots - 1)` rather than a hardcoded integration interval.
The spherical target uses a Bessel function up to dimension 256 and a Gaussian
approximation above that, as in the source implementation. Regularization is
scaled by batch size. The examples use 128 projection directions and 51
integration knots; both settings are configurable.

The default pretraining backbone is the compact CNN trained from scratch.
Set `name_data` and `backbone_name` in `PretrainedParameters` to choose the
dataset-specific pretraining encoder. `RetrievalParameters()` selects the
pretrained DINOv3 ConvNeXt Small by default.
The BatchNorm projector requires batches of at least two during training.
