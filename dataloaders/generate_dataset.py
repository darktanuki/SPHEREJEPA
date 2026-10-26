# %% 
from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
from PIL import Image


@dataclass
class Parameters:
    root: Path = Path("data/textures")
    dataset_name: str = "texture_disk"
    img_size: tuple[int, int] = (800, 800)
    img_size_to_save: tuple[int, int] = (256, 256)
    nb_train: int = 10_000
    nb_val: int = 500
    nb_test: int = 10_000
    seed: int = 42

    @property
    def path_dataset(self):
        return str(Path(self.root) / self.dataset_name)


def laplacian_kernel(size):
    x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1), np.arange(-size // 2 + 1, size // 2 + 1))
    kernel = np.exp(-np.sqrt(x ** 2 + y ** 2))
    return kernel / kernel.sum()

def laplacian_in_fourier(image, kernel_size):
    laplacian_kernel_ = laplacian_kernel(image.shape[0])
    h, w = image.shape
    padded_kernel = np.zeros_like(image, dtype=np.float32)
    kh, kw = laplacian_kernel_.shape
    padded_kernel[:kh, :kw] = laplacian_kernel_
    padded_kernel = fftshift(padded_kernel)
    image_fft = fft2(image)
    kernel_fft = fft2(padded_kernel)
    result_fft = image_fft * kernel_fft
    result = np.real(ifft2(result_fft))
    return result

def generate_texture_with_levy_noise(img_size, percentage_to_clip, min_max_texture, alpha=0.5, beta=0):
    """
    Generate a texture with Lévy noise, apply a Laplacian blur in the Fourier domain,
    and clip the resulting values to a specified range.

    Args:
        img_size (tuple): Size of the texture (height, width).
        percentage_to_clip (float): Fraction of extreme values to clip (e.g., 0.01 for 1%).
        min_max_texture (tuple): Range of the output texture (min, max).
        alpha (float): Stability parameter for Lévy noise (0 < alpha ≤ 2).
        beta (float): Skewness parameter for Lévy noise (-1 ≤ beta ≤ 1).

    Returns:
        np.ndarray: Generated texture.
    """
    C = np.zeros(img_size)
    noise = levy_stable.rvs(alpha, beta, size=C.shape)
    C_noisy = C + noise
    C_blurred = laplacian_in_fourier(C_noisy, C_noisy.shape[0])
    lower_bound = np.percentile(C_blurred, percentage_to_clip * 100)
    upper_bound = np.percentile(C_blurred, (1 - percentage_to_clip) * 100)
    C_clip = np.clip(C_blurred, lower_bound, upper_bound)
    C_scale = (C_clip - lower_bound) / (upper_bound - lower_bound) * (min_max_texture[1] - min_max_texture[0]) + min_max_texture[0]
    return C_scale

def cnoise(w, h, σ, s=42):
    from numpy.random import default_rng
    x = default_rng(s).standard_normal((h, w))
    from numpy.fft import fft2, ifft2, fftfreq
    from numpy import meshgrid
    p, q = meshgrid(fftfreq(w), fftfreq(h))
    r2 = p ** 2 + q ** 2
    r2[0, 0] = 1
    X = fft2(x)
    F = r2 ** (-σ / 2)
    F[0, 0] = 0
    Y = F * X
    y = ifft2(Y).real
    return y

def generate_cloud_texture(seed: int, shape: tuple[int, int]):
    """Generate the original smooth cloud texture from multiple Gaussian-filtered noise layers."""
    rng = np.random.default_rng(seed)
    h, w = shape
    base_scale = float(min(h, w))
    sigmas = (base_scale / 6.0, base_scale / 12.0, base_scale / 24.0, base_scale / 48.0)
    weights = (0.52, 0.27, 0.14, 0.07)
    cloud = np.zeros((h, w), dtype=np.float32)
    for sigma, weight in zip(sigmas, weights):
        layer = rng.standard_normal((h, w)).astype(np.float32)
        layer = gaussian_filter(layer, sigma=sigma, mode='reflect')
        cloud += weight * layer
    envelope = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma=base_scale / 3.5, mode='reflect')
    envelope = _normalize_with_percentiles(envelope, lower=2.0, upper=98.0)
    cloud *= 0.75 + 0.5 * envelope
    cloud = _normalize_with_percentiles(cloud, lower=1.0, upper=99.0)
    cloud = np.power(cloud, 0.85).astype(np.float32)
    cloud = gaussian_filter(cloud, sigma=base_scale / 96.0, mode='reflect')
    cloud = _normalize_with_percentiles(cloud, lower=0.5, upper=99.5)
    return cloud.astype(np.float32)

def convolve_with_full_kernel(image, kernel_full):
    """Convolve an image with a full-size kernel through FFT multiplication."""
    img_fft = fft2(image)
    ker_fft = fft2(kernel_full)
    result = np.real(ifft2(img_fft * ker_fft))
    return result

def flake_kernel_full(shape, decay=8, branch=5):
    """Build and normalize the original positive radial flake kernel."""
    H, W = shape
    cy, cx = (H // 2, W // 2)
    y, x = np.indices((H, W))
    xv = x - cx
    yv = y - cy
    z = xv.astype(np.complex128) + 1j * yv.astype(np.complex128)
    zb = z ** branch
    real_zb = np.real(zb)
    r = np.sqrt(xv ** 2 + yv ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = real_zb / r ** decay
    k[cy, cx] = 0.0
    k = np.maximum(0.0, k)
    s = k.sum()
    if s > 0:
        k /= s
    return k.astype(np.float32)

def convolve_in_fourier(image, kernel):
    """Convolve equal-sized image and centered kernel arrays in Fourier space."""
    img_fft = fft2(image)
    kernel_shifted = fftshift(kernel)
    ker_fft = fft2(kernel_shifted)
    res = np.real(ifft2(img_fft * ker_fft))
    return res

def generate_texture_with_flake_noise(img_size, percentage_to_clip, min_max_texture, alpha=0.5, beta=0.0):
    """Generate Levy noise, convolve with the flake kernel, clip percentiles, and rescale."""
    H, W = img_size
    noise = levy_stable.rvs(alpha, beta, size=(H, W)).astype(np.float32)
    kernel_full = flake_kernel_full((H, W))
    convolved = convolve_with_full_kernel(noise, kernel_full)
    lo = np.percentile(convolved, percentage_to_clip * 100)
    hi = np.percentile(convolved, (1 - percentage_to_clip) * 100)
    clipped = np.clip(convolved, lo, hi)
    mn, mx = min_max_texture
    scaled = (clipped - lo) / (hi - lo)
    scaled = scaled * (mx - mn) + mn
    return scaled.astype(np.float32)

def next_ds_size(n):
    """
    Given a desired output size n (any integer), compute the smallest size of the form (2^k + 1)
    that is ≥ (n + 1). We'll generate a Diamond–Square grid of that size, then center‐crop to n.
    """
    if isinstance(n, tuple):
        n = max(n)
    return 2 ** int(np.ceil(np.log2(n))) + 1

def diamond_square(size, roughness, seed=None):
    """
    Generate a heightmap of shape (size x size) using the Diamond–Square algorithm.
    `size` must be of the form (2^k + 1). `roughness` in (0,1).
    Returns a float32 array normalized to [0,1].
    """
    if seed is not None:
        np.random.seed(seed)
    grid = np.zeros((size, size), dtype=np.float32)
    grid[0, 0] = np.random.rand()
    grid[0, size - 1] = np.random.rand()
    grid[size - 1, 0] = np.random.rand()
    grid[size - 1, size - 1] = np.random.rand()
    step = size - 1
    scale = 1.0
    while step > 1:
        half = step // 2
        for x in range(0, size - 1, step):
            for y in range(0, size - 1, step):
                avg = (grid[x, y] + grid[x + step, y] + grid[x, y + step] + grid[x + step, y + step]) * 0.25
                grid[x + half, y + half] = avg + (np.random.rand() * 2 - 1) * scale
        for x in range(0, size, half):
            for y in range((x + half) % step, size, step):
                total = 0.0
                count = 0
                if x - half >= 0:
                    total += grid[x - half, y]
                    count += 1
                if x + half < size:
                    total += grid[x + half, y]
                    count += 1
                if y - half >= 0:
                    total += grid[x, y - half]
                    count += 1
                if y + half < size:
                    total += grid[x, y + half]
                    count += 1
                avg = total / count
                grid[x, y] = avg + (np.random.rand() * 2 - 1) * scale
        scale *= roughness
        step = half
    grid -= grid.min()
    grid /= grid.max() - grid.min()
    return grid

def compute_intensity_grayscale(heightmap, scale_x):
    """
    Given a heightmap (H×W), compute the “Wood” intensity map in grayscale:
      i(x,y) = | sin( (X + 2*h(x,y)) * π ) |,
    where X = (x_norm * scale_x) and x_norm = x / (W-1).
    This uses vectorized NumPy operations only (no Python loops).
    """
    H, W = heightmap.shape
    x_norm = np.linspace(0.0, 1.0, W, dtype=np.float32)
    X = x_norm * scale_x
    warped = X[np.newaxis, :] + 2.0 * heightmap
    intensity = np.abs(np.sin(warped * np.pi))
    intensity -= intensity.min()
    intensity /= intensity.max() - intensity.min()
    return intensity.astype(np.float32)

def generate_wood_grayscale(img_size, roughness=0.75, seed=0, scale_x=15.0):
    """
    1) Determine the smallest Diamond–Square grid size ≥ (img_size+1).
    2) Generate that heightmap.
    3) Center-crop to (img_size × img_size).
    4) Compute the grayscale “wood” intensity without any Python loops.

    Args:
        img_size (int): desired square output resolution (any positive integer).
        roughness (float): roughness parameter for Diamond–Square in (0,1).
        seed (int): random seed for reproducibility.
        scale_x (float): controls horizontal band spacing.

    Returns:
        np.ndarray of shape (img_size, img_size), dtype float32, values in [0,1].
    """
    ds_size = next_ds_size(img_size)
    heightmap_full = diamond_square(ds_size, roughness, seed)
    H, W = img_size
    start_h = (ds_size - H) // 2
    start_w = (ds_size - W) // 2
    heightmap = heightmap_full[start_h:start_h + H, start_w:start_w + W]
    wood_gray = compute_intensity_grayscale(heightmap, scale_x)
    return wood_gray

def save_image(image, path, filename, img_size=None, use_bilinear=False):
    """Save the image as a grayscale .png file with optional resizing."""
    os.makedirs(path, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    elif not isinstance(image, Image.Image):
        raise TypeError('The input image must be a numpy array or a PIL image.')
    if img_size is not None and image.size != img_size:
        resample = Image.BILINEAR if use_bilinear else Image.LANCZOS
        image = image.resize(img_size, resample)
    full_path = os.path.join(path, filename)
    image.save(full_path, format='PNG')

def plot_image(img, title='Image'):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()

def _normalize_with_percentiles(image, lower=1.0, upper=99.0):
    lo = np.percentile(image, lower)
    hi = np.percentile(image, upper)
    image = np.clip(image, lo, hi)
    denom = hi - lo
    if denom <= 1e-08:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - lo) / denom).astype(np.float32)

def generate_dataset(params):
    if params.dataset_name not in ('texture_disk', 'texture_cloud', 'texture_wood', 'texture_flake', 'texture_disk_flake'):
        raise ValueError(f'Unknown texture dataset: {params.dataset_name}')
    if min(params.nb_train, params.nb_val, params.nb_test) < 1:
        raise ValueError('All split sizes must be positive.')
    if params.img_size[0] != params.img_size[1]:
        raise ValueError('The original disk generator requires square source images.')
    np.random.seed(params.seed)
    os.makedirs(params.path_dataset, exist_ok=True)
    os.makedirs(os.path.join(params.path_dataset, 'train'), exist_ok=True)
    os.makedirs(os.path.join(params.path_dataset, 'val'), exist_ok=True)
    os.makedirs(os.path.join(params.path_dataset, 'test'), exist_ok=True)
    dataset_sizes = {'train': params.nb_train, 'val': params.nb_val, 'test': params.nb_test}
    max_dataset_size = max(dataset_sizes.values())
    for j, (split, size) in enumerate(dataset_sizes.items()):
        split_path = os.path.join(params.path_dataset, split)
        for i in range(size):
            filename = f'texture_{i:04d}.png'
            texture_seed = j * max_dataset_size + i
            if params.dataset_name == 'texture_disk':
                texture = generate_texture_with_levy_noise(params.img_size, 0.01, (0.0, 1.0), alpha=0.5, beta=0.0)
                save_image(texture, os.path.join(split_path, 'all_images'), filename, img_size=params.img_size_to_save)
            elif params.dataset_name == 'texture_cloud':
                texture = generate_cloud_texture(seed=texture_seed, shape=params.img_size_to_save)
                save_image(texture, os.path.join(split_path, 'all_images'), filename, img_size=params.img_size_to_save, use_bilinear=True)
            elif params.dataset_name == 'texture_wood':
                texture = generate_wood_grayscale(params.img_size, roughness=0.75, seed=texture_seed, scale_x=15.0)
                save_image(texture, os.path.join(split_path, 'all_images'), filename, img_size=params.img_size_to_save)
            elif params.dataset_name == 'texture_flake':
                texture = generate_texture_with_flake_noise(params.img_size, 0.01, (0.0, 1.0), alpha=0.5, beta=0.0)
                save_image(texture, os.path.join(split_path, 'all_images'), filename, img_size=params.img_size_to_save)
            elif params.dataset_name == 'texture_disk_flake':
                if i % 2 == 0:
                    texture = generate_texture_with_levy_noise(params.img_size, 0.01, (0.0, 1.0), alpha=0.5, beta=0.0)
                    save_image(texture, os.path.join(split_path, 'disk'), filename, img_size=params.img_size_to_save)
                else:
                    texture = generate_texture_with_flake_noise(params.img_size, 0.01, (0.0, 1.0), alpha=0.5, beta=0.0)
                    save_image(texture, os.path.join(split_path, 'flake'), filename, img_size=params.img_size_to_save)
            else:
                raise ValueError(f'Unknown name : {params.dataset_name}')
            if (i + 1) % 20 == 0 or i + 1 == size:
                percentage = (i + 1) / size * 100
                print(f'[{split}] Progress: {i + 1}/{size} ({percentage:.2f}%)')


if __name__ == "__main__":
    # Edit these defaults directly. Existing datasets are not regenerated automatically.
    root = Path("data/textures")
    families = ("texture_disk", "texture_cloud", "texture_wood", "texture_flake")
    img_size = (800, 800)
    img_size_to_save = (256, 256)
    nb_train = 10_000
    nb_val = 500
    nb_test = 10_000
    seed = 42

    for family in families:
        params = Parameters(
            root=root, dataset_name=family, img_size=img_size,
            img_size_to_save=img_size_to_save, nb_train=nb_train,
            nb_val=nb_val, nb_test=nb_test, seed=seed,
        )
        if Path(params.path_dataset).exists():
            raise FileExistsError(f"Output already exists: {params.path_dataset}. Choose an unused root.")
        print(f"Generating {family} in {params.path_dataset}")
        generate_dataset(params)

# %%
