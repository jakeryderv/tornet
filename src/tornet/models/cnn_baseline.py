"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import keras
import numpy as np

from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.models.heads import HEADS
from tornet.models.layers import CoordConv2D, FillNaNs
from tornet.models.temporal import TEMPORAL_HEADS, FoldTime, RepeatFrames, UnfoldTime

# Name of the backbone's last layer (the dropout after block 4). Heads attach to its output.
FEATURES_LAYER = "features"


def build_model(
    shape: tuple[int] = (120, 240, 2),
    c_shape: tuple[int] = (120, 240, 2),
    input_variables: list[str] = ALL_VARIABLES,
    start_filters: int = 64,
    l2_reg: float = 0.001,
    background_flag: float = -3.0,
    include_range_folded: bool = True,
    head="maxpool",
    head_kwargs=None,
    frames: int = 1,
):
    """The authors' CNN with a choice of head.

    `shape` is one frame's (azimuth, range, tilt). With `frames` > 1 the inputs are
    (frames, azimuth, range, tilt), the backbone runs on every frame with shared weights, and
    `head` must be one of `TEMPORAL_HEADS`, which combine the per-frame feature maps.
    """
    temporal = frames > 1
    if temporal and head not in TEMPORAL_HEADS:
        raise ValueError(f"frames={frames} needs a temporal head, one of {sorted(TEMPORAL_HEADS)}")
    if not temporal and head in TEMPORAL_HEADS:
        raise ValueError(f"head {head!r} needs frames > 1")
    # Create input layers for each input_variables
    inputs = {}
    in_shape = (frames,) + tuple(shape) if temporal else tuple(shape)
    for v in input_variables:
        inputs[v] = keras.Input(in_shape, name=v)
    n_sweeps = shape[2]

    # Normalize inputs and concate along channel dim
    normalized_inputs = keras.layers.Concatenate(axis=-1, name="Concatenate1")(
        [normalize(inputs[v], v) for v in input_variables]
    )

    # Replace nan pixel with background flag
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)

    # Add channel for range folded gates
    if include_range_folded:
        range_folded = keras.Input(in_shape[:-1] + (n_sweeps,), name="range_folded_mask")
        inputs["range_folded_mask"] = range_folded
        normalized_inputs = keras.layers.Concatenate(axis=-1, name="Concatenate2")(
            [normalized_inputs, range_folded]
        )

    # Input coordinate information
    cin = keras.Input(c_shape, name="coordinates")
    inputs["coordinates"] = cin

    x, c = normalized_inputs, cin
    if temporal:  # frames into the batch axis; the backbone is shared across frames
        x, c = FoldTime()(x), RepeatFrames(frames)(c)

    x, c = vgg_block(
        x, c, filters=start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1
    )  # (60,120)
    x, c = vgg_block(
        x, c, filters=2 * start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1
    )  # (30,60)
    x, c = vgg_block(
        x, c, filters=4 * start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1
    )  # (15,30)
    x, c = vgg_block(
        x,
        c,
        filters=8 * start_filters,
        ksize=3,
        l2_reg=l2_reg,
        n_convs=3,
        drop_rate=0.1,
        name=FEATURES_LAYER,
    )  # (8,15): the backbone's output, where heads attach
    # x,c = vgg_block(x,c, filters=8*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3)  # (3,7)

    head_kwargs = dict(head_kwargs or {})
    if temporal:
        x = UnfoldTime(frames)(x)
        head_kwargs.setdefault("l2_reg", l2_reg)
        output = TEMPORAL_HEADS[head](x, **head_kwargs)
    else:
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {sorted(HEADS)}")
        if head == "maxpool":
            head_kwargs.setdefault("l2_reg", l2_reg)
        output = HEADS[head](x, **head_kwargs)
    return keras.Model(inputs=inputs, outputs=output)


def vgg_block(x, c, filters=64, ksize=3, n_convs=2, l2_reg=1e-6, drop_rate=0.0, name=None):
    """`name`, if given, names the block's final dropout layer."""
    for _ in range(n_convs):
        x, c = CoordConv2D(
            filters=filters,
            kernel_size=ksize,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            padding="same",
            activation="relu",
        )([x, c])
    x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    c = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(c)
    if drop_rate > 0:
        x = keras.layers.Dropout(rate=drop_rate, name=name)(x)
    return x, c


def normalize(x, name: str):
    """
    Channel-wise normalization using known CHANNEL_MIN_MAX
    """
    min_max = np.array(CHANNEL_MIN_MAX[name])  # [2,]
    n_sweeps = x.shape[-1]

    # choose mean,var to get approximate [-1,1] scaling
    var = ((min_max[1] - min_max[0]) / 2) ** 2  # scalar
    var = np.array(
        n_sweeps
        * [
            var,
        ]
    )  # [n_sweeps,]

    offset = (min_max[0] + min_max[1]) / 2  # scalar
    offset = np.array(
        n_sweeps
        * [
            offset,
        ]
    )  # [n_sweeps,]

    return keras.layers.Normalization(mean=offset, variance=var, name=f"Normalize_{name}")(x)


def backbone_end(model: keras.Model) -> int:
    """Index in `model.layers` of the backbone's last layer.

    Models built here name it `FEATURES_LAYER`; upstream checkpoints predate that name, and for
    them (maxpool head only) it is the last Dropout layer.
    """
    names = [layer.name for layer in model.layers]
    if FEATURES_LAYER in names:
        return names.index(FEATURES_LAYER)
    dropouts = [
        i for i, layer in enumerate(model.layers) if isinstance(layer, keras.layers.Dropout)
    ]
    if not dropouts:
        raise ValueError("model has neither a 'features' layer nor a Dropout layer to cut at")
    return dropouts[-1]


def backbone_layers(model: keras.Model) -> list[keras.layers.Layer]:
    return model.layers[: backbone_end(model) + 1]


def load_backbone_weights(model: keras.Model, source: keras.Model | str) -> int:
    """Copy backbone weights from `source` (a model or a .keras path) into `model`, layer by layer.

    Only the backbone is copied; the head keeps its fresh initialisation. Returns the number of
    layers whose weights were set. Raises if the two backbones do not line up.
    """
    if isinstance(source, str):
        source = keras.saving.load_model(source, compile=False)
    src = [layer for layer in backbone_layers(source) if layer.weights]
    dst = [layer for layer in backbone_layers(model) if layer.weights]
    if len(src) != len(dst):
        raise ValueError(
            f"backbones differ: {len(src)} weighted layers in source, {len(dst)} in model"
        )
    for a, b in zip(src, dst, strict=True):
        wa, wb = a.get_weights(), b.get_weights()
        if [w.shape for w in wa] != [w.shape for w in wb]:
            raise ValueError(
                f"shape mismatch at {a.name} -> {b.name}: {[w.shape for w in wa]} vs {[w.shape for w in wb]}"
            )
        b.set_weights(wa)
    return len(dst)
