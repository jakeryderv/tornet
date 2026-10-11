"""Temporal variants: the backbone runs on each of the last `frames` radar scans with shared
weights, and a head combines the per-frame feature maps.

Inputs are (batch, time, azimuth, range, tilt); `fold_time` runs the shared backbone on
(batch * time, ...) and `UnfoldTime` restores (batch, time, H, W, C). Heads:

- `temporal_attention`: at each of the H*W locations, self-attention over the `frames` tokens
  (with a learned per-frame position embedding), then the last frame's token at each location,
  then the authors' maxpool head. Attention is the only thing that mixes frames.
- `temporal_concat`: the no-attention control. The frames are concatenated along the channel axis
  at each location and the authors' maxpool head follows. Its first 1x1 convolution grows from
  C to frames*C inputs, which at C=384, frames=4 adds about as many parameters as one attention
  layer (590k), so the two heads are parameter-matched at depth 1.

Both keep the baseline's per-location logit and global max, so whole-scan inference is unchanged.
"""

import keras
from keras import ops

from tornet.models.heads import TakeToken, maxpool_head


@keras.saving.register_keras_serializable()
class FoldTime(keras.layers.Layer):
    """(batch, time, H, W, C) -> (batch * time, H, W, C)."""

    def call(self, x):
        s = ops.shape(x)
        return ops.reshape(x, (-1, s[2], s[3], s[4]))

    def compute_output_shape(self, input_shape):
        b, t, h, w, c = input_shape
        return (None if b is None or t is None else b * t, h, w, c)


@keras.saving.register_keras_serializable()
class UnfoldTime(keras.layers.Layer):
    """(batch * time, H, W, C) -> (batch, time, H, W, C) for a known number of frames."""

    def __init__(self, frames: int, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames

    def call(self, x):
        s = ops.shape(x)
        return ops.reshape(x, (-1, self.frames, s[1], s[2], s[3]))

    def compute_output_shape(self, input_shape):
        bt, h, w, c = input_shape
        return (None if bt is None else bt // self.frames, self.frames, h, w, c)

    def get_config(self):
        return {**super().get_config(), "frames": self.frames}


@keras.saving.register_keras_serializable()
class RepeatFrames(keras.layers.Layer):
    """(batch, H, W, C) -> (batch * time, H, W, C): the same coordinates for every frame."""

    def __init__(self, frames: int, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames

    def call(self, x):
        return ops.repeat(x, self.frames, axis=0)

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (None if b is None else b * self.frames, h, w, c)

    def get_config(self):
        return {**super().get_config(), "frames": self.frames}


@keras.saving.register_keras_serializable()
class LocationsToSequences(keras.layers.Layer):
    """(batch, time, H, W, C) -> (batch * H * W, time, C): one time sequence per location."""

    def call(self, x):
        s = ops.shape(x)
        x = ops.transpose(x, (0, 2, 3, 1, 4))
        return ops.reshape(x, (-1, s[1], s[4]))

    def compute_output_shape(self, input_shape):
        b, t, h, w, c = input_shape
        n = None if None in (b, h, w) else b * h * w
        return (n, t, c)


@keras.saving.register_keras_serializable()
class SequencesToLocations(keras.layers.Layer):
    """Inverse of LocationsToSequences after pooling time: given (batch * H * W, C) and the
    (batch, time, H, W, C) tensor it came from, returns (batch, H, W, C)."""

    def call(self, inputs):
        pooled, ref = inputs
        s = ops.shape(ref)
        return ops.reshape(pooled, (-1, s[2], s[3], ops.shape(pooled)[-1]))

    def compute_output_shape(self, input_shape):
        pooled_shape, ref_shape = input_shape
        return (ref_shape[0], ref_shape[2], ref_shape[3], pooled_shape[-1])


@keras.saving.register_keras_serializable()
class FramesToChannels(keras.layers.Layer):
    """(batch, time, H, W, C) -> (batch, H, W, time * C)."""

    def __init__(self, frames: int, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames

    def call(self, x):
        s = ops.shape(x)
        x = ops.transpose(x, (0, 2, 3, 1, 4))
        return ops.reshape(x, (-1, s[2], s[3], self.frames * s[4]))

    def compute_output_shape(self, input_shape):
        b, t, h, w, c = input_shape
        return (b, h, w, None if c is None else self.frames * c)

    def get_config(self):
        return {**super().get_config(), "frames": self.frames}


@keras.saving.register_keras_serializable()
class FramePosition(keras.layers.Layer):
    """Adds a learned embedding per time step to (n, time, C)."""

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(1, input_shape[1], input_shape[2]),
            initializer="zeros",
            trainable=True,
            name="frame_position",
        )

    def call(self, x):
        return x + self.embedding


def temporal_attention_head(
    x,
    depth: int = 1,
    num_heads: int = 6,
    dropout: float = 0.1,
    l2_reg: float = 1e-5,
    hidden=(512, 256),
    outputs=("logit",),
):
    """Per-location self-attention over frames, then the authors' maxpool head on the last frame's
    attended token at each location."""
    channels = x.shape[-1]
    tokens = FramePosition()(LocationsToSequences()(x))
    for _ in range(depth):
        y = keras.layers.LayerNormalization(epsilon=1e-6)(tokens)
        y = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=channels // num_heads, dropout=dropout
        )(y, y)
        y = keras.layers.Dropout(dropout)(y)
        tokens = keras.layers.Add()([tokens, y])
    tokens = keras.layers.LayerNormalization(epsilon=1e-6)(tokens)
    last = TakeToken(-1)(tokens)
    maps = SequencesToLocations()([last, x])
    return maxpool_head(maps, l2_reg=l2_reg, hidden=hidden, outputs=outputs)


def temporal_concat_head(x, l2_reg: float = 1e-5, hidden=(512, 256), outputs=("logit",)):
    """No-attention control: frames concatenated along channels, then the maxpool head."""
    frames = x.shape[1]
    return maxpool_head(FramesToChannels(frames)(x), l2_reg=l2_reg, hidden=hidden, outputs=outputs)


TEMPORAL_HEADS = {
    "temporal_attention": temporal_attention_head,
    "temporal_concat": temporal_concat_head,
}
