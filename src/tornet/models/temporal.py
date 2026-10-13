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

- `temporal_diff_concat`: `temporal_concat` handed the frame-to-frame differences as well.
- `spacetime_transformer`: self-attention over every cell of every frame (time * H * W tokens),
  the one head here that can relate a feature to a different location in another scan.

All keep a per-location (or per-token) logit and a global max, so whole-scan inference is
unchanged. Every head accepts `output_units` to widen a named extra output, e.g. one logit per
frame for the per-frame label target.
"""

import keras
from keras import ops

from tornet.models.heads import TakeToken, _encoder, _pool_tokens, maxpool_head


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
class LastFrame(keras.layers.Layer):
    """(batch, time, H, W, C) -> (batch, H, W, C): the current scan only, so a single-frame head
    can train on a multi-frame feature cache as the control."""

    def call(self, x):
        return x[:, -1]

    def compute_output_shape(self, input_shape):
        b, _t, h, w, c = input_shape
        return (b, h, w, c)


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


@keras.saving.register_keras_serializable()
class FrameEmbedding(keras.layers.Layer):
    """Adds a learned embedding per time step to (batch, time, H, W, C)."""

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(1, input_shape[1], 1, 1, input_shape[-1]),
            initializer="zeros",
            trainable=True,
            name="frame_embedding",
        )

    def call(self, x):
        return x + self.embedding


@keras.saving.register_keras_serializable()
class SpaceTimeTokens(keras.layers.Layer):
    """(batch, time, H, W, C) -> (batch, time * H * W, C): every cell of every frame a token."""

    def call(self, x):
        s = ops.shape(x)
        return ops.reshape(x, (s[0], -1, s[-1]))

    def compute_output_shape(self, input_shape):
        b, t, h, w, c = input_shape
        n = None if None in (t, h, w) else t * h * w
        return (b, n, c)


@keras.saving.register_keras_serializable()
class FrameDifferences(keras.layers.Layer):
    """(batch, time, H, W, C) -> (batch, time - 1, H, W, C): each frame minus the previous one."""

    def call(self, x):
        return x[:, 1:] - x[:, :-1]

    def compute_output_shape(self, input_shape):
        b, t, h, w, c = input_shape
        return (b, None if t is None else t - 1, h, w, c)


def temporal_attention_head(
    x,
    depth: int = 1,
    num_heads: int = 6,
    dropout: float = 0.1,
    l2_reg: float = 1e-5,
    hidden=(512, 256),
    outputs=("logit",),
    output_units: dict | None = None,
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
    return maxpool_head(
        maps, l2_reg=l2_reg, hidden=hidden, outputs=outputs, output_units=output_units
    )


def temporal_concat_head(
    x,
    l2_reg: float = 1e-5,
    hidden=(512, 256),
    outputs=("logit",),
    output_units: dict | None = None,
    differences: bool = False,
):
    """No-attention control: frames concatenated along channels, then the maxpool head.

    With `differences`, each frame-to-frame difference is concatenated too, so the head is handed
    the change between scans directly rather than having to learn a subtraction."""
    frames = x.shape[1]
    if differences:
        x = keras.layers.Concatenate(axis=1)([x, FrameDifferences()(x)])
        frames = 2 * frames - 1
    return maxpool_head(
        FramesToChannels(frames)(x),
        l2_reg=l2_reg,
        hidden=hidden,
        outputs=outputs,
        output_units=output_units,
    )


def temporal_diff_concat_head(x, **kwargs):
    """`temporal_concat` with frame differences."""
    return temporal_concat_head(x, differences=True, **kwargs)


def spacetime_transformer_head(
    x,
    depth: int = 2,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    pool: str = "max",
    outputs=("logit",),
    output_units: dict | None = None,
):
    """Self-attention over every cell of every frame at once (time * H * W tokens), so a feature
    can be related to any location in any other scan. Position: a convolutional encoding within
    each frame (as `transformer_head`) plus a learned embedding per frame. Pooling as in
    `transformer_head`: `max` keeps a logit per token and takes the strongest."""
    if pool == "cls":
        raise ValueError("spacetime_transformer pools with max or mean")
    frames, channels = x.shape[1], x.shape[-1]
    folded = FoldTime()(x)
    folded = keras.layers.Add()([folded, keras.layers.DepthwiseConv2D(3, padding="same")(folded)])
    x = FrameEmbedding()(UnfoldTime(frames)(folded))
    tokens = _encoder(SpaceTimeTokens()(x), depth, num_heads, channels, mlp_ratio, dropout)
    return _pool_tokens(tokens, pool, dropout, outputs, output_units=output_units)


TEMPORAL_HEADS = {
    "temporal_attention": temporal_attention_head,
    "temporal_concat": temporal_concat_head,
    "temporal_diff_concat": temporal_diff_concat_head,
    "spacetime_transformer": spacetime_transformer_head,
}


def temporal_backbone(backbone: keras.Model, frames: int) -> keras.Model:
    """Wrap a single-frame backbone (dict inputs, one feature-map output) so it takes
    (frames, azimuth, range, tilt) radar inputs plus one set of coordinates and returns
    (batch, frames, H, W, C): frames fold into the batch axis, exactly as `build_model(frames=N)`
    does, so caching per-frame features of a pretrained model uses the same shared weights."""
    inputs, folded = {}, {}
    for name, tensor in backbone.input.items():
        shape = tuple(tensor.shape[1:])
        if name == "coordinates":
            inputs[name] = keras.Input(shape, name=name)
            folded[name] = RepeatFrames(frames)(inputs[name])
        else:
            inputs[name] = keras.Input((frames,) + shape, name=name)
            folded[name] = FoldTime()(inputs[name])
    out = UnfoldTime(frames)(backbone(folded))
    return keras.Model(inputs=inputs, outputs=out, name=f"{backbone.name}_{frames}f")
