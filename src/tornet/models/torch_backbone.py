"""The authors' TorNet CNN as plain PyTorch modules, with weight transfer from the Keras model.

`TorNetEncoder` is the backbone: four blocks of CoordConv layers (a 3x3 convolution over the
features concatenated with two coordinate channels), each block followed by 2x2 max pooling of
both features and coordinates and a dropout. It returns the feature pyramid in the convention
Segmentation Models PyTorch uses, one tensor per stride: the pre-pool output of each block at
strides 1, 2, 4, 8 and the bottleneck at stride 16, so it drops into an SMP U-Net decoder.
`TorNetDetector` adds the authors' max-pool head and reproduces the published model's logit,
which `load_keras_weights` verifies.

Inputs are channels-first: the 12 normalised, background-filled radar channels plus the two
range-folded channels (14), and the 2 coordinate channels, exactly the tensors the Keras model
builds internally before its first convolution (tornet.data.forecast_torch.normalize_fields).
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # the checkpoint is read through Keras

import numpy as np
import torch
from torch import nn

BLOCK_CONVS = (2, 2, 3, 3)
IN_CHANNELS = 14  # 6 products x 2 sweeps + 2 range-folded channels
COORD_CHANNELS = 2


class CoordConv(nn.Module):
    """3x3 convolution over [features, coordinates], ReLU; returns features only."""

    def __init__(self, in_channels: int, out_channels: int, coord_channels: int = COORD_CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + coord_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(torch.cat([x, coords], dim=1)))


class TorNetEncoder(nn.Module):
    """The backbone. `out_channels` lists the channels of each returned feature, strides
    1, 2, 4, 8, 16 (SMP convention: features[0] is at the input resolution)."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        start_filters: int = 48,
        block_convs: tuple[int, ...] = BLOCK_CONVS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = in_channels
        self.out_channels: list[int] = []
        for b, n_convs in enumerate(block_convs):
            filters = start_filters * 2**b
            layers = nn.ModuleList()
            for _ in range(n_convs):
                layers.append(CoordConv(channels, filters))
                channels = filters
            self.blocks.append(layers)
            self.out_channels.append(filters)
        self.out_channels.append(channels)  # the bottleneck after the last pool
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # Keras padding="same"
        self.dropout = nn.Dropout(dropout)
        self.depth = len(block_convs)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> list[torch.Tensor]:
        features = []
        c = coords
        for layers in self.blocks:
            for layer in layers:
                x = layer(x, c)
            features.append(x)
            x = self.dropout(self.pool(x))
            c = self.pool(c)
        features.append(x)
        return features

    def conv_layers(self) -> list[nn.Conv2d]:
        return [layer.conv for layers in self.blocks for layer in layers]


class TorNetHead(nn.Module):
    """The authors' max-pool head: two 1x1 convolutions, a 1x1 heat map (one map per output),
    max over locations. The map before the max is the paper's spatial likelihood map."""

    def __init__(
        self, in_channels: int, hidden: tuple[int, ...] = (512, 256), out_channels: int = 1
    ):
        super().__init__()
        layers, channels = [], in_channels
        for units in hidden:
            layers += [nn.Conv2d(channels, units, 1), nn.ReLU()]
            channels = units
        self.hidden = nn.Sequential(*layers)
        self.heatmap = nn.Conv2d(channels, out_channels, 1)

    def heat(self, features: torch.Tensor) -> torch.Tensor:
        """Per-location logits, (batch, out_channels, H, W)."""
        return self.heatmap(self.hidden(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.heat(features).amax(dim=(2, 3))  # (batch, out_channels)

    def conv_layers(self) -> list[nn.Conv2d]:
        return [m for m in self.hidden if isinstance(m, nn.Conv2d)] + [self.heatmap]


class TorNetDetector(nn.Module):
    """Backbone plus head: the published tornado detector, as PyTorch."""

    def __init__(self, start_filters: int = 48, in_channels: int = IN_CHANNELS):
        super().__init__()
        self.encoder = TorNetEncoder(in_channels=in_channels, start_filters=start_filters)
        self.head = TorNetHead(self.encoder.out_channels[-1])

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, coords)[-1])

    def heat(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.head.heat(self.encoder(x, coords)[-1])


def _copy_conv(torch_conv: nn.Conv2d, kernel: np.ndarray, bias: np.ndarray) -> None:
    """Keras (kh, kw, in, out) -> torch (out, in, kh, kw)."""
    weight = torch.as_tensor(np.ascontiguousarray(np.transpose(kernel, (3, 2, 0, 1))))
    if tuple(weight.shape) != tuple(torch_conv.weight.shape):
        raise ValueError(f"kernel shape {tuple(weight.shape)} != {tuple(torch_conv.weight.shape)}")
    with torch.no_grad():
        torch_conv.weight.copy_(weight)
        torch_conv.bias.copy_(torch.as_tensor(np.ascontiguousarray(bias)))


def load_keras_weights(module: TorNetEncoder | TorNetDetector, keras_model) -> int:
    """Copy the Keras model's CoordConv2D layers (and, for a detector, its head's Conv2D
    layers) into the torch module in order. Returns the number of layers copied."""
    from tornet.models.layers import CoordConv2D

    coord_layers = [layer for layer in keras_model.layers if isinstance(layer, CoordConv2D)]
    encoder = module.encoder if isinstance(module, TorNetDetector) else module
    targets = encoder.conv_layers()
    if len(coord_layers) != len(targets):
        raise ValueError(
            f"{len(coord_layers)} CoordConv2D layers in Keras, {len(targets)} in torch"
        )
    for src, dst in zip(coord_layers, targets, strict=True):
        _copy_conv(dst, *src.get_weights())
    copied = len(targets)
    if isinstance(module, TorNetDetector):
        import keras

        convs = [
            layer
            for layer in keras_model.layers
            if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, CoordConv2D)
        ]
        heads = module.head.conv_layers()
        if len(convs) != len(heads):
            raise ValueError(f"{len(convs)} head Conv2D layers in Keras, {len(heads)} in torch")
        for src, dst in zip(convs, heads, strict=True):
            _copy_conv(dst, *src.get_weights())
        copied += len(heads)
    return copied


def pretrained_detector(
    path: str | None = None, device: str | torch.device = "cpu"
) -> TorNetDetector:
    """The published detector as a frozen torch module ("pretrained" or a .keras path)."""
    import keras

    from tornet.scripts.train_keras import resolve_init_from

    keras_model = keras.saving.load_model(resolve_init_from(path or "pretrained"), compile=False)
    first = next(layer for layer in keras_model.layers if layer.__class__.__name__ == "CoordConv2D")
    start_filters = int(first.get_weights()[0].shape[-1])
    detector = TorNetDetector(start_filters=start_filters)
    load_keras_weights(detector, keras_model)
    detector.eval().requires_grad_(False)
    return detector.to(device)
