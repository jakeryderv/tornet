"""Radar forecaster: a U-Net whose encoder is the pretrained TorNet backbone.

Inputs are the detector's dict of radar variables with a time axis of `n_in` scans (plus the
range-folded mask and coordinates). The backbone runs on each input scan with shared weights,
its pre-pool feature maps at every scale are concatenated across scans and used as skip
connections, and a decoder upsamples back to the chip resolution to emit `horizons` future scans
in the detector's normalised, NaN-filled representation: (batch, horizons, azimuth, range, 12).

With a frozen `detector`, the model also outputs the detector's tornado logit for every forecast
scan (batch, horizons), computed by denormalising the forecast and handing it to the detector
with the last input scan's range-folded mask and the chip's coordinates. Training on that output
against the horizon's label is the "detector as a loss" variant: the forecaster is pushed to
produce scans the detector reads correctly.
"""

import keras
import numpy as np
from keras import ops

from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.models.cnn_baseline import normalize
from tornet.models.layers import CoordConv2D, FillNaNs
from tornet.models.temporal import FoldTime, FramesToChannels, RepeatFrames, UnfoldTime

# The authors' backbone: convolutions per block; filters are start_filters * 2**block.
BLOCK_CONVS = (2, 2, 3, 3)
BLOCK_DROPOUT = (0.1, 0.1, 0.1, 0.1)


@keras.saving.register_keras_serializable()
class Denormalize(keras.layers.Layer):
    """Normalised channels of one variable back to physical units (no NaNs)."""

    def __init__(self, variable: str, **kwargs):
        super().__init__(**kwargs)
        self.variable = variable
        lo, hi = CHANNEL_MIN_MAX[variable]
        self.offset, self.scale = (lo + hi) / 2, (hi - lo) / 2

    def call(self, x):
        return x * self.scale + self.offset

    def get_config(self):
        return {**super().get_config(), "variable": self.variable}


@keras.saving.register_keras_serializable()
class TakeFrame(keras.layers.Layer):
    """(batch, time, ...) -> (batch, ...) at a fixed index."""

    def __init__(self, index: int, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + tuple(input_shape[2:])

    def get_config(self):
        return {**super().get_config(), "index": self.index}


@keras.saving.register_keras_serializable()
class ChannelSlice(keras.layers.Layer):
    """(..., C) -> (..., size) channels starting at `start`."""

    def __init__(self, start: int, size: int, **kwargs):
        super().__init__(**kwargs)
        self.start, self.size = start, size

    def call(self, x):
        return x[..., self.start : self.start + self.size]

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.size,)

    def get_config(self):
        return {**super().get_config(), "start": self.start, "size": self.size}


@keras.saving.register_keras_serializable()
class SplitHorizons(keras.layers.Layer):
    """(batch, H, W, horizons * C) -> (batch, horizons, H, W, C)."""

    def __init__(self, horizons: int, **kwargs):
        super().__init__(**kwargs)
        self.horizons = horizons

    def call(self, x):
        s = ops.shape(x)
        x = ops.reshape(x, (s[0], s[1], s[2], self.horizons, -1))
        return ops.transpose(x, (0, 3, 1, 2, 4))

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (b, self.horizons, h, w, None if c is None else c // self.horizons)

    def get_config(self):
        return {**super().get_config(), "horizons": self.horizons}


def pooled_sizes(shape: tuple[int, int], levels: int = 4) -> list[tuple[int, int]]:
    """Spatial sizes after each of the backbone's `levels` max-pools (padding "same")."""
    sizes, h, w = [], shape[0], shape[1]
    for _ in range(levels):
        h, w = -(-h // 2), -(-w // 2)
        sizes.append((h, w))
    return sizes


def build_forecaster(
    shape: tuple[int, int, int] = (120, 240, 2),
    n_in: int = 2,
    horizons: int = 2,
    start_filters: int = 48,
    decoder_filters: tuple[int, ...] = (256, 128, 64, 32),
    l2_reg: float = 1e-5,
    background_flag: float = -3.0,
    detector: keras.Model | None = None,
    name: str = "forecaster",
) -> keras.Model:
    """The U-Net described in the module docstring. `shape` is one scan's (azimuth, range,
    tilt). `decoder_filters` are the widths at the 15x30, 30x60, 60x120 and 120x240 stages."""
    inputs = {}
    in_shape = (n_in,) + tuple(shape)
    for v in ALL_VARIABLES:
        inputs[v] = keras.Input(in_shape, name=v)
    inputs["range_folded_mask"] = keras.Input(in_shape[:-1] + (shape[2],), name="range_folded_mask")
    inputs["coordinates"] = keras.Input(tuple(shape[:2]) + (2,), name="coordinates")

    x = keras.layers.Concatenate(axis=-1)([normalize(inputs[v], v) for v in ALL_VARIABLES])
    x = FillNaNs(background_flag)(x)
    x = keras.layers.Concatenate(axis=-1)([x, inputs["range_folded_mask"]])
    x, c = FoldTime()(x), RepeatFrames(n_in)(inputs["coordinates"])

    # Encoder: the backbone's blocks, keeping each block's pre-pool map as a skip connection.
    skips = []
    for b, (n_convs, drop) in enumerate(zip(BLOCK_CONVS, BLOCK_DROPOUT, strict=True)):
        for j in range(n_convs):
            x, c = CoordConv2D(
                filters=start_filters * 2**b,
                kernel_size=3,
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                padding="same",
                activation="relu",
                name=f"enc_b{b}_c{j}",
            )([x, c])
        skips.append(x)
        x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
        c = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(c)
        if drop > 0:
            x = keras.layers.Dropout(drop)(x)

    def merge_frames(t):  # (batch * n_in, H, W, C) -> (batch, H, W, n_in * C)
        return FramesToChannels(n_in)(UnfoldTime(n_in)(t))

    x = merge_frames(x)
    skips = [merge_frames(s) for s in skips]

    # Decoder: upsample to each skip's resolution, concatenate, convolve.
    sizes = [tuple(shape[:2])] + pooled_sizes(tuple(shape[:2]))[
        :-1
    ]  # 120x240, 60x120, 30x60, 15x30
    for level, (skip, size, filters) in enumerate(
        zip(reversed(skips), reversed(sizes), decoder_filters, strict=True)
    ):
        x = keras.layers.Resizing(size[0], size[1], interpolation="bilinear")(x)
        x = keras.layers.Concatenate(axis=-1)([x, skip])
        for j in range(2):
            x = keras.layers.Conv2D(
                filters,
                3,
                padding="same",
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f"dec_l{level}_c{j}",
            )(x)
    channels = 2 * len(ALL_VARIABLES) if shape[2] == 2 else shape[2] * len(ALL_VARIABLES)
    out = keras.layers.Conv2D(horizons * channels, 1, name="fields_flat")(x)
    fields = SplitHorizons(horizons, name="fields")(out)
    outputs = {"fields": fields}

    if detector is not None:
        detector.trainable = False
        rfm = TakeFrame(n_in - 1)(inputs["range_folded_mask"])
        logits = []
        for h in range(horizons):
            f = TakeFrame(h)(fields)
            det_in = {"range_folded_mask": rfm, "coordinates": inputs["coordinates"]}
            for i, v in enumerate(ALL_VARIABLES):
                det_in[v] = Denormalize(v)(ChannelSlice(i * shape[2], shape[2])(f))
            logit = detector(det_in, training=False)
            if isinstance(logit, (list, tuple)):  # the published model lists its one output
                logit = logit[0]
            logits.append(logit)
        outputs["logits"] = keras.layers.Concatenate(axis=-1, name="logits")(logits)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def load_encoder_weights(forecaster: keras.Model, pretrained: keras.Model) -> int:
    """Copy the pretrained backbone's convolutions into the forecaster's encoder, in order.
    Returns the number of layers set."""
    src = [layer for layer in pretrained.layers if isinstance(layer, CoordConv2D)]
    dst = [layer for layer in forecaster.layers if isinstance(layer, CoordConv2D)]
    if len(src) != len(dst):
        raise ValueError(f"pretrained has {len(src)} CoordConv2D layers, forecaster {len(dst)}")
    for s, d in zip(src, dst, strict=True):
        ws, wd = s.get_weights(), d.get_weights()
        if [w.shape for w in ws] != [w.shape for w in wd]:
            raise ValueError(
                f"shape mismatch {s.name} -> {d.name}: {[w.shape for w in ws]} vs {[w.shape for w in wd]}"
            )
        d.set_weights(ws)
    return len(dst)


def set_encoder_trainable(forecaster: keras.Model, trainable: bool) -> None:
    for layer in forecaster.layers:
        if isinstance(layer, CoordConv2D):
            layer.trainable = trainable


def persistence_fields(last_fields: np.ndarray, horizons: int) -> np.ndarray:
    """The persistence forecast: the last input scan repeated for every horizon."""
    return np.repeat(last_fields[:, None], horizons, axis=1)
