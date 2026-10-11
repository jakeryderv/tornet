"""Optimizers for end-to-end training with per-layer learning rates."""

import keras


@keras.saving.register_keras_serializable()
class ScaledAdam(keras.optimizers.Adam):
    """Adam whose learning rate is multiplied per variable, for fine-tuning a pretrained backbone
    under a fresh head.

    `scales` maps a variable path (`variable.path`) to a multiplier; unlisted variables use 1.0.
    Scaling the learning rate, not the gradient, is what makes this work: Adam's update is
    invariant to the gradient's scale.
    """

    def __init__(self, learning_rate=1e-3, scales: dict[str, float] | None = None, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.scales = dict(scales or {})

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        groups: dict[float, tuple[list, list]] = {}
        for grad, var in zip(grads, trainable_variables, strict=True):
            gs, vs = groups.setdefault(float(self.scales.get(var.path, 1.0)), ([], []))
            gs.append(grad)
            vs.append(var)
        for scale, (gs, vs) in groups.items():
            lr = learning_rate if scale == 1.0 else learning_rate * scale
            super()._backend_update_step(gs, vs, lr)

    def get_config(self):
        return {**super().get_config(), "scales": self.scales}


def scales_for(layers, scale: float) -> dict[str, float]:
    """Learning-rate scales for every trainable variable of `layers`."""
    return {v.path: scale for layer in layers for v in layer.trainable_variables}
