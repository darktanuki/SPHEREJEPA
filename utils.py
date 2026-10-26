"""Small image encoder used for MNIST pretraining."""

from torch import Tensor, nn


class ConvNet(nn.Module):
    """LeNet-like encoder returning 128 features from square grayscale images."""

    in_dim = 128

    def __init__(self, c_in: int = 1, image_width: int = 32, dropout_rate: float = 0.0):
        super().__init__()
        if image_width < 16:
            raise ValueError("image_width must be at least 16.")

        width = (image_width - 4) // 2  # 5x5 convolution, then 2x2 pooling
        width = (width - 4) // 2
        self.network = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=5),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(64 * width * width, self.in_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
