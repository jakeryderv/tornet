"""Multi-task radar forecasting and tornado prediction on TorNet, in PyTorch with Segmentation
Models PyTorch (SMP). One file, top to bottom: config, data, model, losses, training,
evaluation, CLI.

    observed scans (n_in x 14 channels, polar)        all channels in
            |
    encoder: the pretrained TorNet CNN ported to torch (or any SMP encoder, e.g. an ImageNet
             ResNet34, for the encoder-swap experiment), run per scan, features concatenated
             across scans at every scale
            |
    bottleneck: identity (CNN) or a small transformer over the 8x15 tokens
            |
    SMP U-Net decoder ---------------------------> regression head: VEL + DBZ at each horizon
    bottleneck features ------------------------> classification head: tornado logit per horizon

Losses: weighted L1 on the predicted products in the detector's normalised space (optionally
weighted spatially by the frozen detector's heat map on the real target scan), weighted
cross-entropy on the per-horizon labels, and optionally the frozen detector's cross-entropy on
the forecast chip (forecast products plus the other products and the range-folded channels
persisted from the last scan; the truth chip keeps its own range-folded channels, which do
change between scans and move the detector's logit).

Evaluation (test years), per horizon: the classification head's probability, and the frozen
TorNet detector on persistence / forecast / truth, written as tornet-score CSVs with the per-frame
label pattern so the pre-onset subset (no input frame tornadic, last target frame tornadic) can
be scored on its own; summary.json holds AUC/AUCPR per view for all and pre-onset cases, MAE
against truth for forecast and persistence, and paired bootstraps against persistence.

Usage: tornet-forecast <config.json>
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tornet.data.forecast_torch import ForecastDataset, channel_indices, collate
from tornet.models.torch_backbone import TorNetEncoder, load_keras_weights, pretrained_detector
from tornet.scripts.score import VIEWS, paired_difference, roc_auc, view_metrics
from tornet.utils.general import git_revision, make_exp_dir

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "train_years": [2013, 2015],
    "val_years": [2014],
    "test_years": [2014],
    "n_in": 2,
    "horizons": 2,
    "target_products": ["VEL", "DBZ"],  # predicted; the rest are persisted for the detector
    # model
    "encoder": "tornet",  # "tornet" (the pretrained TorNet CNN) or an SMP encoder name
    "encoder_weights": "pretrained",  # tornet: "pretrained" | .keras path | null; SMP: "imagenet" | null
    "encoder_trainable": True,
    "start_filters": 48,
    "bottleneck": "cnn",  # "cnn" (identity) or "transformer"
    "transformer": {"depth": 2, "heads": 8, "dropout": 0.1},
    "decoder_channels": [256, 128, 64, 32],
    "cls_dropout": 0.2,
    # losses
    "field_weight": 1.0,
    "product_weights": {"VEL": 1.0, "DBZ": 1.0},
    "heatmap_weight": 0.0,  # spatial weight = 1 + heatmap_weight * detector heat map on truth
    "cls_weight": 1.0,
    "detector_weight": 0.0,  # frozen detector's loss on the forecast chip
    "sample_weights": {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 2.0, "wW": 0.5},
    "detector": "pretrained",
    # training
    "epochs": 6,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "encoder_lr_scale": 0.1,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "amp": True,
    "workers": 8,
    "max_files": None,
    "seed": 1234,
    # evaluation
    "bootstrap": 1000,
    "exp_name": "forecast",
    "exp_dir": os.environ.get("EXP_DIR", "experiments"),
}

# --------------------------------------------------------------------------------------------
# 2. Data
# --------------------------------------------------------------------------------------------


def make_loader(cfg, partition, years, shuffle):
    ds = ForecastDataset(
        os.environ["TORNET_ROOT"],
        partition,
        years,
        n_in=cfg["n_in"],
        horizons=cfg["horizons"],
        weights=cfg["sample_weights"],
        random_state=cfg["seed"],
        max_files=cfg.get("max_files"),
    )
    workers = int(cfg["workers"])
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


# --------------------------------------------------------------------------------------------
# 3. Model
# --------------------------------------------------------------------------------------------


class TransformerBottleneck(nn.Module):
    """Self-attention over the bottleneck's spatial tokens with a learned position embedding."""

    def __init__(
        self, channels: int, grid: tuple[int, int], depth: int, heads: int, dropout: float
    ):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, grid[0] * grid[1], channels))
        layer = nn.TransformerEncoderLayer(
            channels,
            heads,
            dim_feedforward=4 * channels,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        pos = self.pos
        if pos.shape[1] != h * w:  # another chip size: interpolate the embedding grid
            side = int(math.sqrt(pos.shape[1]))
            pos = (
                nn.functional.interpolate(
                    pos.transpose(1, 2).reshape(1, c, side, -1), size=(h, w), mode="bilinear"
                )
                .flatten(2)
                .transpose(1, 2)
            )
        tokens = self.norm(self.encoder(tokens + pos))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ForecastNet(nn.Module):
    """Encoder per scan -> features concatenated across scans -> bottleneck -> SMP U-Net decoder
    -> regression head; bottleneck -> classification head."""

    def __init__(self, cfg, grid: tuple[int, int] = (8, 15)):
        super().__init__()
        from segmentation_models_pytorch.base import ClassificationHead, SegmentationHead
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
        from segmentation_models_pytorch.encoders import get_encoder

        self.n_in, self.horizons = int(cfg["n_in"]), int(cfg["horizons"])
        self.target_channels = channel_indices(cfg["target_products"])
        if cfg["encoder"] == "tornet":
            self.encoder = TorNetEncoder(start_filters=int(cfg["start_filters"]))
            self.tornet_encoder = True
            channels = self.encoder.out_channels
        else:
            self.encoder = get_encoder(
                cfg["encoder"], in_channels=14, depth=4, weights=cfg.get("encoder_weights") or None
            )
            self.tornet_encoder = False
            channels = list(self.encoder.out_channels)
        enc_channels = [c * self.n_in for c in channels]
        if cfg["bottleneck"] == "transformer":
            t = cfg["transformer"]
            self.bottleneck = TransformerBottleneck(
                enc_channels[-1], grid, t["depth"], t["heads"], t["dropout"]
            )
        else:
            self.bottleneck = nn.Identity()
        dec = list(cfg["decoder_channels"])
        self.decoder = UnetDecoder(enc_channels, dec, n_blocks=len(dec), use_norm="batchnorm")
        self.regression = SegmentationHead(
            dec[-1], self.horizons * len(self.target_channels), kernel_size=3
        )
        self.classification = ClassificationHead(
            enc_channels[-1], self.horizons, pooling="max", dropout=float(cfg["cls_dropout"])
        )

    def encode(self, x: torch.Tensor, coords: torch.Tensor) -> list[torch.Tensor]:
        b, n = x.shape[:2]
        flat = x.flatten(0, 1)  # (B * n_in, 14, A, R)
        if self.tornet_encoder:
            feats = self.encoder(flat, coords.repeat_interleave(n, dim=0))
        else:
            feats = self.encoder(flat)
        # (B * n_in, C, h, w) -> (B, n_in * C, h, w): scans side by side in the channel axis
        return [f.reshape(b, n * f.shape[1], *f.shape[2:]) for f in feats]

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encode(x, coords)
        feats[-1] = self.bottleneck(feats[-1])
        fields = self.regression(self.decoder(feats))
        b, _, a, r = fields.shape
        return {
            "fields": fields.reshape(b, self.horizons, len(self.target_channels), a, r),
            "logits": self.classification(feats[-1]),
        }


def build_model(cfg) -> ForecastNet:
    model = ForecastNet(cfg)
    if cfg["encoder"] == "tornet" and cfg.get("encoder_weights"):
        import keras

        from tornet.scripts.train_keras import resolve_init_from

        source = keras.saving.load_model(resolve_init_from(cfg["encoder_weights"]), compile=False)
        n = load_keras_weights(model.encoder, source)
        logging.info(f"encoder initialised from {cfg['encoder_weights']}: {n} layers")
    if not cfg.get("encoder_trainable", True):
        model.encoder.requires_grad_(False)
    return model


# --------------------------------------------------------------------------------------------
# 4. Losses and the frozen detector
# --------------------------------------------------------------------------------------------


def full_chip(
    forecast: torch.Tensor, last_x: torch.Tensor, target_channels: list[int]
) -> torch.Tensor:
    """Forecast products placed into the last observed scan (persisting the other products and
    the range-folded channels): (B, 14, A, R) for the detector."""
    chip = last_x.clone()
    chip[:, target_channels] = forecast
    return chip


class Losses:
    def __init__(self, cfg, model: ForecastNet, device):
        self.cfg = cfg
        self.target_channels = model.target_channels
        w = []
        for p in cfg["target_products"]:
            w += [float(cfg["product_weights"].get(p, 1.0))] * 2
        self.channel_w = torch.tensor(w, device=device).view(1, 1, -1, 1, 1)
        self.detector = None
        if float(cfg["heatmap_weight"]) > 0 or float(cfg["detector_weight"]) > 0:
            self.detector = pretrained_detector(cfg["detector"], device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def spatial_weight(self, batch, device) -> torch.Tensor | None:
        """1 + heatmap_weight * the detector's heat map (min-max normalised per scan) on the real
        target scan, upsampled to the chip: (B, horizons, 1, A, R)."""
        hw = float(self.cfg["heatmap_weight"])
        if hw <= 0:
            return None
        future, coords = batch["future"].to(device), batch["coords"].to(device)
        b, h, _, a, r = future.shape
        with torch.no_grad():
            heat = self.detector.heat(future.flatten(0, 1), coords.repeat_interleave(h, dim=0))
            heat = nn.functional.interpolate(heat, size=(a, r), mode="bilinear")
            lo = heat.amin(dim=(2, 3), keepdim=True)
            hi = heat.amax(dim=(2, 3), keepdim=True)
            heat = (heat - lo) / (hi - lo + 1e-6)
        return 1 + hw * heat.reshape(b, h, 1, a, r)

    def __call__(self, out: dict, batch: dict, device) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        target = batch["future"][:, :, self.target_channels].to(device)
        err = (out["fields"] - target).abs() * self.channel_w
        sw = self.spatial_weight(batch, device)
        if sw is not None:
            err = err * sw
        field = err.mean()
        weight = batch["weight"].to(device)[:, None]
        labels = batch["labels"].to(device)
        cls = (self.bce(out["logits"], labels) * weight).mean()
        total = float(cfg["field_weight"]) * field + float(cfg["cls_weight"]) * cls
        parts = {"field": float(field), "cls": float(cls)}
        dw = float(cfg["detector_weight"])
        if dw > 0:
            x, coords = batch["x"].to(device), batch["coords"].to(device)
            det_logits = []
            for h in range(out["fields"].shape[1]):
                chip = full_chip(out["fields"][:, h], x[:, -1], self.target_channels)
                det_logits.append(self.detector(chip, coords))
            det_logits = torch.cat(det_logits, dim=1)
            det = (self.bce(det_logits, labels) * weight).mean()
            total = total + dw * det
            parts["detector"] = float(det)
        return total, parts


# --------------------------------------------------------------------------------------------
# 5. Training
# --------------------------------------------------------------------------------------------


def run_epoch(model, loader, losses, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    sums, n = {}, 0
    logits_all, labels_all = [], []
    for batch in loader:
        x, coords = (
            batch["x"].to(device, non_blocking=True),
            batch["coords"].to(device, non_blocking=True),
        )
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device.type, enabled=amp and device.type == "cuda"),
        ):
            out = model(x, coords)
            out = {k: v.float() for k, v in out.items()}
            total, parts = losses(out, batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            if losses.cfg["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), float(losses.cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
        bs = x.shape[0]
        for k, v in {"loss": float(total), **parts}.items():
            sums[k] = sums.get(k, 0.0) + v * bs
        n += bs
        logits_all.append(out["logits"].detach().cpu().numpy())
        labels_all.append(batch["labels"].numpy())
    metrics = {k: v / max(n, 1) for k, v in sums.items()}
    logits, labels = np.concatenate(logits_all), np.concatenate(labels_all)
    for h in range(labels.shape[1]):
        metrics[f"AUC_h{h}"] = (
            roc_auc(labels[:, h].astype(bool), logits[:, h])
            if 0 < labels[:, h].sum() < len(labels)
            else float("nan")
        )
    return metrics


def train(cfg, model, device, expdir):
    train_loader = make_loader(cfg, "train", cfg["train_years"], shuffle=True)
    val_loader = (
        make_loader(cfg, "train", cfg["val_years"], shuffle=False) if cfg["val_years"] else None
    )
    losses = Losses(cfg, model, device)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    lr = float(cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in enc_params if p.requires_grad],
                "lr": lr * float(cfg["encoder_lr_scale"]),
            },
            {"params": rest, "lr": lr},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    epochs = int(cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(enabled=bool(cfg["amp"]) and device.type == "cuda")
    history, best, best_path = [], -np.inf, None
    os.makedirs(os.path.join(expdir, "checkpoints"), exist_ok=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, losses, device, optimizer, scaler, amp=bool(cfg["amp"]))
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}}
        if val_loader is not None:
            va = run_epoch(model, val_loader, losses, device, amp=bool(cfg["amp"]))
            row.update({f"val_{k}": v for k, v in va.items()})
            score = np.nanmean([v for k, v in va.items() if k.startswith("AUC_h")])
            if score > best:
                best, best_path = score, os.path.join(expdir, "checkpoints", "best.pt")
                torch.save(model.state_dict(), best_path)
        scheduler.step()
        row["seconds"] = time.time() - t0
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(expdir, "history.csv"), index=False)
        logging.info(
            " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        )
    last_path = os.path.join(expdir, "checkpoints", "last.pt")
    torch.save(model.state_dict(), last_path)
    return history, best_path or last_path, last_path


# --------------------------------------------------------------------------------------------
# 6. Evaluation
# --------------------------------------------------------------------------------------------


def pattern_flags(frame_labels: np.ndarray, n_in: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """pre_onset: no input frame tornadic and the last frame tornadic; ongoing: an input frame
    tornadic. Also the label pattern as a string, e.g. '0011'."""
    inputs = frame_labels[:, :n_in].max(axis=1) > 0
    last = frame_labels[:, -1] > 0
    pre_onset = (~inputs) & last
    patterns = ["".join(str(int(v)) for v in row) for row in frame_labels]
    return pre_onset, inputs, patterns


def views_for(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for key, (_desc, cats) in VIEWS.items():
        sub = df[df["category"].isin(cats)]
        if sub["label"].sum() == 0 or sub["label"].sum() == len(sub):
            continue
        out[key] = view_metrics(sub, n_boot, rng)
    return out


@torch.no_grad()
def evaluate(cfg, model, device, expdir, years):
    model.eval()
    loader = make_loader(cfg, "test", years, shuffle=False)
    detector = pretrained_detector(cfg["detector"], device)
    tc = model.target_channels
    horizons = int(cfg["horizons"])
    rows = {w: [[] for _ in range(horizons)] for w in ("head", "persistence", "forecast", "truth")}
    mae = {"forecast": [[] for _ in range(horizons)], "persistence": [[] for _ in range(horizons)]}
    for batch in loader:
        x, coords = batch["x"].to(device), batch["coords"].to(device)
        with torch.autocast(device.type, enabled=bool(cfg["amp"]) and device.type == "cuda"):
            out = model(x, coords)
        fields = out["fields"].float()
        head_logits = out["logits"].float().cpu().numpy()
        future = batch["future"].to(device)
        last = x[:, -1]
        pre_onset, ongoing, patterns = pattern_flags(
            batch["frame_labels"].numpy(), int(cfg["n_in"])
        )
        for h in range(horizons):
            chips = {
                "persistence": last,
                "forecast": full_chip(fields[:, h], last, tc),
                "truth": future[:, h],  # the real scan with its own range-folded channels
            }
            logits = {w: detector(c, coords).squeeze(1).cpu().numpy() for w, c in chips.items()}
            logits["head"] = head_logits[:, h]
            mae["forecast"][h].append(
                (fields[:, h] - future[:, h, tc]).abs().mean(dim=(1, 2, 3)).cpu().numpy()
            )
            mae["persistence"][h].append(
                (last[:, tc] - future[:, h, tc]).abs().mean(dim=(1, 2, 3)).cpu().numpy()
            )
            for j, meta in enumerate(batch["meta"]):
                base = {
                    "filename": meta["filename"],
                    "label": int(batch["labels"][j, h]),
                    "category": meta["category"],
                    "ef_number": meta["ef_number"],
                    "event_id": meta["event_id"],
                    "time": int(meta["times"][h]),
                    "minutes": float(batch["minutes"][j, h]),
                    "pattern": patterns[j],
                    "pre_onset": bool(pre_onset[j]),
                    "ongoing": bool(ongoing[j]),
                }
                for w in rows:
                    lg = float(logits[w][j])
                    rows[w][h].append({**base, "logit": lg, "prob": 1 / (1 + math.exp(-lg))})
    out_dir = os.path.join(expdir, "forecast_scores")
    os.makedirs(out_dir, exist_ok=True)
    n_boot, seed = int(cfg["bootstrap"]), int(cfg["seed"])
    summary = {
        "years": list(years),
        "n_in": int(cfg["n_in"]),
        "target_products": cfg["target_products"],
        "horizons": [],
    }
    for h in range(horizons):
        preds = {}
        for w in rows:
            df = pd.DataFrame(rows[w][h])
            df["year"] = pd.to_datetime(df["time"], unit="s").dt.year
            df.to_csv(os.path.join(out_dir, f"predictions_{w}_h{h}.csv"), index=False)
            preds[w] = df
        scoped = {w: df[df["pre_onset"] | (df["category"] != "TOR")] for w, df in preds.items()}
        entry = {
            "horizon": h,
            "lead_minutes_median": float(preds["head"]["minutes"].median()),
            "n": int(len(preds["head"])),
            "positives": int(preds["head"]["label"].sum()),
            "pre_onset_cases": int(preds["head"]["pre_onset"].sum()),
            "ongoing_cases": int(preds["head"]["ongoing"].sum()),
            "mae_forecast": float(np.concatenate(mae["forecast"][h]).mean()),
            "mae_persistence": float(np.concatenate(mae["persistence"][h]).mean()),
            "all": {w: views_for(df, n_boot, seed) for w, df in preds.items()},
            "pre_onset": {w: views_for(df, n_boot, seed) for w, df in scoped.items()},
            "pre_onset_paired_vs_persistence": {
                w: paired_difference(scoped["persistence"], scoped[w], n_boot, seed)
                for w in ("head", "forecast", "truth")
            },
        }
        summary["horizons"].append(entry)
        line = " ".join(
            f"{w}={entry['pre_onset'].get(w, {}).get('WC', {}).get('auc', float('nan')):.4f}"
            for w in ("persistence", "forecast", "truth", "head")
        )
        logging.info(
            f"h{h} lead {entry['lead_minutes_median']:.0f} min, pre-onset WC AUC: {line}; "
            f"MAE forecast {entry['mae_forecast']:.4f} persistence {entry['mae_persistence']:.4f}"
        )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------------------------
# 7. CLI
# --------------------------------------------------------------------------------------------


def main(cfg):
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device {device}; config {cfg}")
    expdir = make_exp_dir(exp_dir=cfg["exp_dir"], prefix=cfg["exp_name"])
    git_commit, git_dirty = git_revision()
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"{cfg['encoder']} encoder, {cfg['bottleneck']} bottleneck: {n_params:,} params, {n_train:,} trainable"
    )
    with open(os.path.join(expdir, "params.json"), "w") as f:
        json.dump({"config": cfg, "params": n_params, "trainable": n_train}, f, indent=2)
    history, best_path, last_path = train(cfg, model, device, expdir)
    model.load_state_dict(torch.load(best_path, map_location=device))
    summary = evaluate(cfg, model, device, expdir, cfg["test_years"]) if cfg["test_years"] else None
    results = {
        "exp_name": cfg["exp_name"],
        "encoder": cfg["encoder"],
        "bottleneck": cfg["bottleneck"],
        "params": n_params,
        "trainable": n_train,
        "epochs_run": len(history),
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "history_last": history[-1] if history else None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "expdir": expdir,
    }
    if summary:
        results["pre_onset_WC_AUC"] = {
            f"h{e['horizon']}": {
                w: e["pre_onset"].get(w, {}).get("WC", {}).get("auc")
                for w in ("persistence", "forecast", "truth", "head")
            }
            for e in summary["horizons"]
        }
        results["mae"] = {
            f"h{e['horizon']}": {"forecast": e["mae_forecast"], "persistence": e["mae_persistence"]}
            for e in summary["horizons"]
        }
    with open(os.path.join(expdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"results: {results}")
    return results


def cli():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            loaded = json.load(f)
        for k in ("transformer", "product_weights", "sample_weights"):
            if k in loaded:
                loaded[k] = {**cfg[k], **loaded[k]}
        cfg.update(loaded)
    main(cfg)


if __name__ == "__main__":
    cli()
