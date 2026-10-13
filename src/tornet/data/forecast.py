"""Batches for radar forecasting: the oldest scans in, the newer scans as targets.

Each TorNet sample has up to four scans. `ForecastLoader` returns the first `n_in` as model
inputs (the same dict layout the detector takes, with a time axis) and the following `horizons`
scans as targets in the detector's own normalised, NaN-filled representation, so that a forecast
can be handed straight to the frozen detector and a forecast error is measured in the space the
detector sees. Targets also carry each horizon's tornado label (the file's per-frame label) and
the minutes from the last input scan to each horizon, from the real scan times.
"""

import numpy as np

from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.keras_loader import KerasDataLoader
from tornet.data.loader import read_file

BACKGROUND = -3.0  # the detector's FillNaNs value, in normalised units


def channel_scale(name: str) -> tuple[float, float]:
    """(offset, scale) such that normalised = (raw - offset) / scale, as the detector does."""
    lo, hi = CHANNEL_MIN_MAX[name]
    return (lo + hi) / 2, (hi - lo) / 2


def normalize_fields(frames: dict[str, np.ndarray], background: float = BACKGROUND) -> np.ndarray:
    """{variable: (..., azimuth, range, tilt)} -> (..., azimuth, range, 2 * len(ALL_VARIABLES))
    normalised and NaN-filled, channel order as the detector's concatenation."""
    out = []
    for v in ALL_VARIABLES:
        offset, scale = channel_scale(v)
        x = (frames[v].astype(np.float32) - offset) / scale
        out.append(np.where(np.isnan(x), background, x))
    return np.concatenate(out, axis=-1)


def denormalize_fields(fields: np.ndarray) -> dict[str, np.ndarray]:
    """Inverse of `normalize_fields` without re-introducing NaNs: the detector normalises these
    back to exactly the given values, so background gates stay at its fill value."""
    out, n = {}, fields.shape[-1] // len(ALL_VARIABLES)
    for i, v in enumerate(ALL_VARIABLES):
        offset, scale = channel_scale(v)
        out[v] = fields[..., i * n : (i + 1) * n] * scale + offset
    return out


class ForecastLoader(KerasDataLoader):
    """Batches of (inputs, targets) for the forecaster.

    inputs:  {variable: (B, n_in, A, R, T)}, range_folded_mask (B, n_in, A, R, T),
             coordinates (B, A, R, 2), plus the metadata keys of the sample, and
             last_fields (B, A, R, 12): the last input scan normalised (persistence)
    targets: fields (B, horizons, A, R, 12), logits (B, horizons) tornado labels,
             minutes (B, horizons) from the last input scan, time (B, horizons) epoch seconds
    """

    def __init__(
        self,
        *args,
        n_in: int = 2,
        horizons: int = 2,
        target_keys: list[str] | None = None,
        **kwargs,
    ):
        kwargs["n_frames"] = n_in + horizons
        super().__init__(*args, **kwargs)
        self.n_in, self.horizons, self.target_keys = n_in, horizons, target_keys

    @staticmethod
    def model_input_keys() -> list[str]:
        return ALL_VARIABLES + ["range_folded_mask", "coordinates"]

    def __getitem__(self, idx: int):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.file_list))
        elements = [
            read_file(f, variables=ALL_VARIABLES, n_frames=self.n_frames, tilt_last=self.tilt_last)
            for f in self.file_list[low:high]
        ]
        for el in elements:
            pp.add_coordinates(el, include_az=self.include_az, backend=np, tilt_last=self.tilt_last)
            el["coordinates"] = el["coordinates"][None, ...]
        n = self.n_in
        x, y = {}, {}
        for v in ALL_VARIABLES + ["range_folded_mask"]:
            x[v] = np.stack([el[v][:n] for el in elements])
        x["coordinates"] = np.concatenate([el["coordinates"] for el in elements])
        x["last_fields"] = np.stack(
            [normalize_fields({v: el[v][n - 1] for v in ALL_VARIABLES}) for el in elements]
        )
        for key in elements[0]:
            if key in ALL_VARIABLES or key in ("range_folded_mask", "coordinates", "label", "time"):
                continue
            x[key] = np.concatenate([el[key] for el in elements])
        x["time"] = np.array([el["time"][n - 1] for el in elements])  # last input scan
        y["fields"] = np.stack(
            [normalize_fields({v: el[v][n:] for v in ALL_VARIABLES}) for el in elements]
        )
        y["logits"] = np.stack([el["label"][n:] for el in elements]).astype(np.float32)
        y["time"] = np.stack([el["time"][n:] for el in elements])
        y["minutes"] = ((y["time"] - x["time"][:, None]) / 60).astype(np.float32)
        if self.select_keys:
            x = {k: v for k, v in x.items() if k in self.select_keys}
        if self.target_keys:
            y = {k: v for k, v in y.items() if k in self.target_keys}
        return x, y
