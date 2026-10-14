"""PyTorch dataset for radar forecasting: the oldest `n_in` scans in, the next `horizons` out.

Every tensor is channels-first and already in the detector's representation (normalised as the
detector normalises, empty gates filled with its background value; `normalize_fields`), so the
torch TorNet encoder consumes `x` directly and a forecast can be handed to the torch detector
without further processing.

Per sample:
    x            (n_in, 14, A, R)       inputs: 12 radar channels + 2 range-folded channels
    coords       (2, A, R)              CoordConv coordinates (range, 1/range)
    future       (horizons, 14, A, R)   the target scans: all 12 channels + their own
                                        range-folded channels (regression targets are a subset)
    labels       (horizons,)            tornado label of each target scan
    frame_labels (n_in + horizons,)     every frame's label, for scoping the evaluation
    minutes      (horizons,)            lead from the last input scan, from real timestamps
    weight       ()                     the authors' sample weight (category / EF)
    meta         dict of filename, category, ef_number, event_id, time (last input), times
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import query_catalog, read_file

BACKGROUND = -3.0  # the detector's FillNaNs value, in normalised units
CATEGORY_CODES = {"TOR": 0, "NUL": 1, "WRN": 2}
# channel index of each product's sweeps in the 12-channel layout
CHANNEL_INDEX = {v: [2 * i, 2 * i + 1] for i, v in enumerate(ALL_VARIABLES)}


def channel_indices(products: list[str]) -> list[int]:
    return [i for p in products for i in CHANNEL_INDEX[p]]


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
    """Inverse of `normalize_fields` without re-introducing NaNs, for viewing forecasts in
    physical units; background gates stay at the fill value's physical equivalent."""
    out, n = {}, fields.shape[-1] // len(ALL_VARIABLES)
    for i, v in enumerate(ALL_VARIABLES):
        offset, scale = channel_scale(v)
        out[v] = fields[..., i * n : (i + 1) * n] * scale + offset
    return out


def sample_weight(category: str, ef_number: int, w: dict) -> float:
    """The authors' rule (tornet.data.preprocess.compute_sample_weight), one sample."""
    if category == "WRN":
        return float(w.get("wW", 0.5))
    if ef_number < 0:
        return float(w.get("wN", 1.0))
    if ef_number == 0:
        return float(w.get("w0", 1.0))
    if ef_number == 1:
        return float(w.get("w1", 1.0))
    return float(w.get("w2", 2.0))


class ForecastDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        partition: str,
        years: list[int],
        n_in: int = 2,
        horizons: int = 2,
        weights: dict | None = None,
        random_state: int = 1234,
        max_files: int | None = None,
        catalog=None,
    ):
        self.data_root = data_root
        self.n_in, self.horizons = n_in, horizons
        self.weights = weights or {}
        self.files = query_catalog(data_root, partition, list(years), random_state, catalog=catalog)
        if max_files:
            self.files = self.files[:max_files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        n = self.n_in
        d = read_file(self.files[idx], variables=ALL_VARIABLES, n_frames=n + self.horizons)
        coords = pp.compute_coordinates(
            d, include_az=False, tilt_last=True, backend=np
        )  # (A, R, 2)
        fields = normalize_fields({v: d[v] for v in ALL_VARIABLES})  # (T, A, R, 12)
        rfm = d["range_folded_mask"]  # (T, A, R, 2)
        x = np.concatenate([fields[:n], rfm[:n]], axis=-1)  # (n_in, A, R, 14)
        labels = d["label"].astype(np.float32)
        times = d["time"].astype(np.int64)
        category = {0: "TOR", 1: "NUL", 2: "WRN"}[int(d["category"][0])]
        ef = int(d["ef_number"][0])
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float(),
            "coords": torch.from_numpy(np.ascontiguousarray(coords.transpose(2, 0, 1))).float(),
            "future": torch.from_numpy(
                np.ascontiguousarray(
                    np.concatenate([fields[n:], rfm[n:]], axis=-1).transpose(0, 3, 1, 2)
                )
            ).float(),
            "labels": torch.from_numpy(labels[n:]),
            "frame_labels": torch.from_numpy(labels),
            "minutes": torch.from_numpy(((times[n:] - times[n - 1]) / 60).astype(np.float32)),
            "weight": torch.tensor(sample_weight(category, ef, self.weights), dtype=torch.float32),
            "meta": {
                "filename": self.files[idx].replace(self.data_root, "").lstrip("/"),
                "category": category,
                "ef_number": ef,
                "event_id": int(d["event_id"][0]),
                "time": int(times[n - 1]),
                "times": times[n:].tolist(),
            },
        }


def collate(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "meta"}
    out["meta"] = [b["meta"] for b in batch]
    return out
