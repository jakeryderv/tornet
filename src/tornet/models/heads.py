"""Classification heads that map a backbone feature map (batch, H, W, C) to one logit.

Every head is a function `head(x, **kwargs) -> logit tensor` so it can be attached either to the
CNN backbone inside `build_model` or to cached backbone features via a plain `keras.Input`.

Heads:

- `maxpool`: the authors' head, unchanged. Two 1x1 convolutions with ReLU, a 1x1 "heatmap"
  convolution giving a logit per location, then a global max. The sigmoid is applied in the loss.
- `transformer`: tokens from the feature map, a convolutional position encoding (depthwise 3x3
  over the grid, added to the tokens, size-agnostic), `depth` pre-norm transformer encoder layers,
  then one of three poolings.
- `token_mlp`: the no-attention control. The same residual feed-forward blocks and pooling as the
  transformer head, no attention and no position encoding. Its `hidden` width is chosen so the
  parameter count matches a transformer head of the same depth (see `matched_hidden`).

Every head accepts `outputs=("logit",)`; naming more outputs, e.g. `("logit", "aux")`, adds one
extra scalar output per name from the same pooled representation (or per-token-then-max for the
`max` pooling), for auxiliary regression targets. The result is then a dict keyed by name.

Pooling (shared by `transformer` and `token_mlp`):

- `max`: a logit per token, then the maximum over tokens. Same semantics as the baseline: the
  strongest location decides, and the head stays fully convolutional for whole-scan inference.
- `mean`: average the tokens, then a dense layer.
- `cls`: a learned class token attends with the rest (transformer only), then a dense layer.
"""

import keras
from keras import ops


@keras.saving.register_keras_serializable()
class ClassToken(keras.layers.Layer):
    """Prepends a learned token to a (batch, tokens, dim) sequence."""

    def build(self, input_shape):
        self.cls = self.add_weight(
            shape=(1, 1, input_shape[-1]), initializer="zeros", trainable=True, name="cls"
        )

    def call(self, x):
        batch = ops.shape(x)[0]
        cls = ops.broadcast_to(self.cls, (batch, 1, ops.shape(x)[-1]))
        return ops.concatenate([cls, x], axis=1)

    def compute_output_shape(self, input_shape):
        tokens = None if input_shape[1] is None else input_shape[1] + 1
        return (input_shape[0], tokens, input_shape[2])


@keras.saving.register_keras_serializable()
class TakeToken(keras.layers.Layer):
    """Selects one token from a (batch, tokens, dim) sequence."""

    def __init__(self, index=0, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return {**super().get_config(), "index": self.index}


@keras.saving.register_keras_serializable()
class ToTokens(keras.layers.Layer):
    """(batch, H, W, C) -> (batch, H*W, C), with H and W allowed to be unknown."""

    def call(self, x):
        shape = ops.shape(x)
        return ops.reshape(x, (shape[0], -1, shape[-1]))

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        tokens = None if h is None or w is None else h * w
        return (b, tokens, c)


def _finish(x, names: tuple[str, ...]):
    """A single tensor when one output is requested, else a dict keyed by name."""
    return x[names[0]] if len(names) == 1 else x


def maxpool_head(
    x,
    l2_reg: float = 1e-5,
    hidden=(512, 256),
    outputs: tuple[str, ...] = ("logit",),
    output_units: dict | None = None,
):
    """The authors' head: per-location MLP via 1x1 convolutions, then the max over locations.

    With one output the heatmap layer is named "heatmap" as upstream. Extra outputs each get their
    own per-location 1x1 convolution and global max, sharing the hidden layers; `output_units`
    widens a named output (e.g. one logit per frame for a per-frame target).
    """
    output_units = output_units or {}
    for units in hidden:
        x = keras.layers.Conv2D(
            filters=units,
            kernel_size=1,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            activation="relu",
        )(x)
    out = {}
    for name in outputs:
        layer_name = "heatmap" if len(outputs) == 1 else f"{name}_heatmap"
        h = keras.layers.Conv2D(filters=output_units.get(name, 1), kernel_size=1, name=layer_name)(
            x
        )
        out[name] = keras.layers.GlobalMaxPooling2D(name=name)(h)
    return _finish(out, outputs)


def _ffn(x, hidden: int, dim: int, dropout: float, activation: str = "gelu"):
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(hidden, activation=activation)(y)
    y = keras.layers.Dropout(dropout)(y)
    y = keras.layers.Dense(dim)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])


def _encoder(tokens, depth: int, num_heads: int, channels: int, mlp_ratio: float, dropout: float):
    """`depth` pre-norm transformer blocks over (batch, tokens, channels)."""
    for _ in range(depth):
        y = keras.layers.LayerNormalization(epsilon=1e-6)(tokens)
        y = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=channels // num_heads, dropout=dropout
        )(y, y)
        y = keras.layers.Dropout(dropout)(y)
        tokens = keras.layers.Add()([tokens, y])
        tokens = _ffn(tokens, int(mlp_ratio * channels), channels, dropout)
    return keras.layers.LayerNormalization(epsilon=1e-6)(tokens)


def _pool_tokens(
    tokens,
    pool: str,
    dropout: float,
    outputs: tuple[str, ...] = ("logit",),
    output_units: dict | None = None,
):
    output_units = output_units or {}
    out = {}
    if pool == "max":
        for name in outputs:  # logit per token, then the maximum over tokens
            units = output_units.get(name, 1)
            per_token = keras.layers.Dense(units, name=f"{name}_per_token")(tokens)
            out[name] = keras.layers.GlobalMaxPooling1D(name=name)(per_token)
        return _finish(out, outputs)
    if pool == "mean":
        pooled = keras.layers.GlobalAveragePooling1D()(tokens)
        pooled = keras.layers.Dropout(dropout)(pooled)
    elif pool == "cls":
        pooled = TakeToken(0)(tokens)
        pooled = keras.layers.LayerNormalization(epsilon=1e-6)(pooled)
    else:
        raise ValueError(f"unknown pool {pool!r}; expected max, mean or cls")
    for name in outputs:
        out[name] = keras.layers.Dense(output_units.get(name, 1), name=name)(pooled)
    return _finish(out, outputs)


def transformer_head(
    x,
    depth: int = 2,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    pool: str = "max",
    dim: int | None = None,
    outputs: tuple[str, ...] = ("logit",),
    output_units: dict | None = None,
):
    """Transformer encoder over the feature-map tokens with a convolutional position encoding."""
    channels = x.shape[-1]
    if dim is not None and dim != channels:
        x = keras.layers.Conv2D(dim, 1)(x)
        channels = dim
    # Convolutional position encoding: depthwise 3x3 over the grid, added to the tokens. It carries
    # relative position without a fixed grid size, so the head still works on larger scans.
    x = keras.layers.Add()([x, keras.layers.DepthwiseConv2D(3, padding="same")(x)])
    tokens = ToTokens()(x)
    if pool == "cls":
        tokens = ClassToken()(tokens)
    tokens = _encoder(tokens, depth, num_heads, channels, mlp_ratio, dropout)
    return _pool_tokens(tokens, pool, dropout, outputs, output_units=output_units)


def matched_hidden(channels: int, mlp_ratio: float = 4.0) -> int:
    """FFN width for `token_mlp` that matches a transformer layer's parameter count.

    Per layer a transformer has ~4C^2 attention parameters plus 2*C*(mlp_ratio*C) in its FFN; the
    control has only 2*C*hidden. Equating gives hidden = (2 + mlp_ratio) * C.
    """
    return int((2 + mlp_ratio) * channels)


def token_mlp_head(
    x,
    depth: int = 2,
    hidden: int | None = None,
    dropout: float = 0.1,
    pool: str = "max",
    dim: int | None = None,
    outputs: tuple[str, ...] = ("logit",),
    output_units: dict | None = None,
):
    """No-attention control: the transformer head's feed-forward blocks and pooling, nothing else.

    No position encoding and no mixing between tokens, so any gap to `transformer_head` is
    attributable to attention. `hidden` defaults to the parameter-matched width.
    """
    channels = x.shape[-1]
    if dim is not None and dim != channels:
        x = keras.layers.Conv2D(dim, 1)(x)
        channels = dim
    if hidden is None:
        hidden = matched_hidden(channels)
    if pool == "cls":
        raise ValueError("token_mlp has no attention, so a class token cannot gather information")
    tokens = ToTokens()(x)
    for _ in range(depth):
        tokens = _ffn(tokens, hidden, channels, dropout)
    tokens = keras.layers.LayerNormalization(epsilon=1e-6)(tokens)
    return _pool_tokens(tokens, pool, dropout, outputs, output_units=output_units)


HEADS = {
    "maxpool": maxpool_head,
    "transformer": transformer_head,
    "token_mlp": token_mlp_head,
}


def build_head_model(
    feature_shape, head: str = "maxpool", outputs: tuple[str, ...] = ("logit",), **head_kwargs
) -> keras.Model:  # noqa: E302
    """A standalone model from a cached feature map to a logit (or a dict of named outputs when
    more than one is requested), for head-only training.

    A four-dimensional `feature_shape` (frames, H, W, C) is a multi-frame cache: temporal heads
    see every frame, and the single-frame heads here see only the last frame, which makes them the
    controls for the same cache."""
    inputs = keras.Input(feature_shape, name="features")
    x = inputs
    if len(feature_shape) == 4:
        # imported here because tornet.models.temporal builds on the heads in this module
        from tornet.models.temporal import TEMPORAL_HEADS, LastFrame

        if head in TEMPORAL_HEADS:
            out = TEMPORAL_HEADS[head](x, outputs=tuple(outputs), **head_kwargs)
            return keras.Model(inputs=inputs, outputs=out, name=head)
        x = LastFrame()(x)
    out = HEADS[head](x, outputs=tuple(outputs), **head_kwargs)
    return keras.Model(inputs=inputs, outputs=out, name=head)


# --------------------------------------------------------------------------------------------
# Metric-learning head: an L2-normalised embedding plus a logit from learned class prototypes.
# --------------------------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class L2Normalize(keras.layers.Layer):
    def call(self, x):
        return x / (ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True)) + 1e-8)


@keras.saving.register_keras_serializable()
class PrototypeLogit(keras.layers.Layer):
    """Logit = (cos(e, p_tornado) - cos(e, p_null)) / temperature with two learned prototypes.

    A scalar score derived from distances to class centres, so it can be trained with the usual
    binary loss, thresholded, and scored like any other logit.
    """

    def __init__(self, temperature: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def build(self, input_shape):
        self.prototypes = self.add_weight(
            shape=(2, input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
            name="prototypes",
        )

    def call(self, e):
        p = self.prototypes / (
            ops.sqrt(ops.sum(ops.square(self.prototypes), axis=-1, keepdims=True)) + 1e-8
        )
        cos = ops.matmul(e, ops.transpose(p))  # (batch, 2): [tornado, null]
        return (cos[:, 0:1] - cos[:, 1:2]) / self.temperature

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        return {**super().get_config(), "temperature": self.temperature}


def embedding_head(
    x,
    depth: int = 2,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    pool: str = "cls",
    dim: int | None = None,
    embed_dim: int = 128,
    temperature: float = 0.1,
    outputs: tuple[str, ...] = ("logit", "embedding"),
):
    """Transformer tokens -> pooled -> normalised embedding; the logit comes from prototypes.

    Always returns a dict with "logit" and "embedding" (the `outputs` argument is accepted for
    interface symmetry with the other heads but the two outputs are fixed).
    """
    channels = x.shape[-1]
    if dim is not None and dim != channels:
        x = keras.layers.Conv2D(dim, 1)(x)
        channels = dim
    x = keras.layers.Add()([x, keras.layers.DepthwiseConv2D(3, padding="same")(x)])
    tokens = ToTokens()(x)
    if pool == "cls":
        tokens = ClassToken()(tokens)
    for _ in range(depth):
        y = keras.layers.LayerNormalization(epsilon=1e-6)(tokens)
        y = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=channels // num_heads, dropout=dropout
        )(y, y)
        y = keras.layers.Dropout(dropout)(y)
        tokens = keras.layers.Add()([tokens, y])
        tokens = _ffn(tokens, int(mlp_ratio * channels), channels, dropout)
    tokens = keras.layers.LayerNormalization(epsilon=1e-6)(tokens)
    if pool == "cls":
        pooled = TakeToken(0)(tokens)
    elif pool == "mean":
        pooled = keras.layers.GlobalAveragePooling1D()(tokens)
    else:
        raise ValueError("embedding head supports pool='cls' or 'mean'")
    pooled = keras.layers.Dropout(dropout)(pooled)
    embedding = L2Normalize(name="embedding")(keras.layers.Dense(embed_dim)(pooled))
    logit = PrototypeLogit(temperature=temperature, name="logit")(embedding)
    return {"logit": logit, "embedding": embedding}


HEADS["embedding"] = embedding_head


@keras.saving.register_keras_serializable()
class SoftSupervisedContrastive(keras.losses.Loss):
    """Supervised contrastive loss over a batch of normalised embeddings with soft positives.

    `y_true` is one scalar target per sample (a label, or a continuous value such as the proximity
    target); the positive weight of pair (i, j) is exp(-|t_i - t_j| / sigma), so samples with
    equal targets are full positives and the weight decays smoothly with target distance. With
    binary labels and small sigma this reduces to standard supervised contrastive loss. Returns
    one loss value per anchor.
    """

    def __init__(self, temperature: float = 0.1, sigma: float = 0.25, name="soft_supcon", **kw):
        super().__init__(name=name, **kw)
        self.temperature = temperature
        self.sigma = sigma

    def call(self, y_true, y_pred):
        t = ops.reshape(ops.cast(y_true, y_pred.dtype), (-1, 1))
        n = ops.shape(y_pred)[0]
        sim = ops.matmul(y_pred, ops.transpose(y_pred)) / self.temperature
        not_self = 1.0 - ops.eye(n, dtype=y_pred.dtype)
        sim = sim - (1.0 - not_self) * 1e9  # exclude self from the softmax
        log_prob = sim - ops.logsumexp(sim, axis=1, keepdims=True)
        w = ops.exp(-ops.abs(t - ops.transpose(t)) / self.sigma) * not_self
        denom = ops.sum(w, axis=1)
        loss = -ops.sum(w * log_prob, axis=1) / ops.maximum(denom, 1e-8)
        return ops.where(denom > 0, loss, ops.zeros_like(loss))

    def get_config(self):
        return {**super().get_config(), "temperature": self.temperature, "sigma": self.sigma}
