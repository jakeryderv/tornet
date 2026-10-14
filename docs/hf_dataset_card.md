---
license: cc-by-4.0
pretty_name: TorNet
tags:
  - weather
  - radar
  - tornado
  - nexrad
  - dual-pol
  - meteorology
  - severe-weather
  - earth-science
size_categories:
  - 100K<n<1M
task_categories:
  - image-classification
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train/*/*.parquet
      - split: test
        path: data/test/*/*.parquet
  - config_name: catalog
    data_files:
      - split: train
        path: catalog.parquet
  - config_name: "2013"
    data_files:
      - split: train
        path: data/train/2013/*.parquet
      - split: test
        path: data/test/2013/*.parquet
  - config_name: "2014"
    data_files:
      - split: train
        path: data/train/2014/*.parquet
      - split: test
        path: data/test/2014/*.parquet
  - config_name: "2015"
    data_files:
      - split: train
        path: data/train/2015/*.parquet
      - split: test
        path: data/test/2015/*.parquet
  - config_name: "2016"
    data_files:
      - split: train
        path: data/train/2016/*.parquet
      - split: test
        path: data/test/2016/*.parquet
  - config_name: "2017"
    data_files:
      - split: train
        path: data/train/2017/*.parquet
      - split: test
        path: data/test/2017/*.parquet
  - config_name: "2018"
    data_files:
      - split: train
        path: data/train/2018/*.parquet
      - split: test
        path: data/test/2018/*.parquet
  - config_name: "2019"
    data_files:
      - split: train
        path: data/train/2019/*.parquet
      - split: test
        path: data/test/2019/*.parquet
  - config_name: "2020"
    data_files:
      - split: train
        path: data/train/2020/*.parquet
      - split: test
        path: data/test/2020/*.parquet
  - config_name: "2021"
    data_files:
      - split: train
        path: data/train/2021/*.parquet
      - split: test
        path: data/test/2021/*.parquet
  - config_name: "2022"
    data_files:
      - split: train
        path: data/train/2022/*.parquet
      - split: test
        path: data/test/2022/*.parquet
---

# TorNet

TorNet is a benchmark dataset of storm-centered NEXRAD WSR-88D dual-polarisation radar samples
for tornado detection, released by MIT Lincoln Laboratory (Veillette et al., 2024). This is the
complete v1.1 dataset, all ten years and both partitions, repackaged as Parquet so it loads
with `datasets` by year and partition without downloading or extracting per-year archives.
Nothing is dropped or altered: every field of every netCDF sample is here, the radar values in
their physical units.

- **203,133 samples**, 2013 to 2022: 171,666 `train`, 31,467 `test` (the authors' partitions).
- **Four scans** per sample, about five minutes apart, the newest last.
- **Six products** per scan on a polar grid of 120 azimuths × 240 range gates × 2 tilts
  (0.5° and 0.9°): reflectivity `DBZ` (dBZ), radial velocity `VEL` (m/s), specific differential
  phase `KDP` (°/km), correlation coefficient `RHOHV`, differential reflectivity `ZDR` (dB),
  spectrum width `WIDTH` (m/s). Empty gates are NaN.
- **Three categories**: `TOR` (confirmed tornado, with EF rating), `WRN` (tornado-warned storm
  with no tornado), `NUL` (random non-tornadic storm).
- **Per-scan labels** (`frame_labels`): 1 if a tornado is present in that scan, so a sample's
  four labels read like `0011` for a tornado that appears in the third scan.

## Layout

```
data/train/<year>/part-NNNNN-of-MMMMM.parquet    256 samples per shard, zstd
data/test/<year>/part-NNNNN-of-MMMMM.parquet
catalog.parquet                                  one row per sample, every scalar field
```

## Loading

```python
from datasets import load_dataset

ds = load_dataset("jakeryderv/tornet", "2015")            # one year, train and test splits
ds = load_dataset("jakeryderv/tornet")                     # all ten years
ds = load_dataset("jakeryderv/tornet", data_files={"train": ["data/train/2013/*.parquet",
                                                             "data/train/2015/*.parquet"],
                                                   "test": "data/test/2014/*.parquet"})  # a fold
catalog = load_dataset("jakeryderv/tornet", "catalog", split="train").to_pandas()  # 1 s, no arrays

sample = ds["train"].with_format("numpy")[0]
sample["VEL"].shape      # (4, 120, 240, 2): scan, azimuth, range, tilt
sample["frame_labels"]   # array([0, 0, 1, 1])
```

Radar arrays are stored as float16 (the netCDF values are exactly representable; the round
trip is lossless) and come back as float32 under `with_format("numpy")`. The range-folded mask
is uint8. Selecting a year or partition downloads only those shards.

## Fields

| field | type | meaning |
|---|---|---|
| `DBZ`, `VEL`, `KDP`, `RHOHV`, `ZDR`, `WIDTH` | float16 (4, 120, 240, 2) | the products, physical units, NaN where empty |
| `range_folded_mask` | uint8 (4, 120, 240, 2) | 1 where `VEL` / `WIDTH` are range-folded |
| `filename` | string | the netCDF name in the original release, e.g. `train/2015/TOR_150425_...nc` |
| `partition`, `year` | string, int16 | the authors' split and the sample's year |
| `category` | string | `TOR`, `WRN` or `NUL` |
| `ef_number` | int8 | EF rating for `TOR`, −1 otherwise |
| `event_id`, `episode_id` | int64 | NWS storm event and episode ids |
| `site` | string | NEXRAD site, e.g. `KTLX` |
| `lat`, `lon` | float32 | storm location |
| `label` | int8 | the newest scan's label |
| `frame_labels` | int8[4] | tornado present in each scan |
| `time` | int64[4] | each scan's time, Unix seconds |
| `az_lower`, `az_upper` | float32 | azimuth limits of the chip, degrees |
| `rng_lower`, `rng_upper` | float32 | range limits of the chip, metres |
| `tornado_start_time`, `tornado_end_time` | int64 | the tornado's interval, Unix seconds (v1.1); 0 when none |

The CoordConv coordinates used by the published detector are derived from the azimuth and
range limits; they are not stored.

## Source and license

Original release: Veillette, M. S., et al., *TorNet: A benchmark dataset for tornado detection
and prediction using full-resolution polarimetric weather radar data*, Artificial Intelligence
for the Earth Systems, 2024, <https://doi.org/10.1175/AIES-D-24-0006.1>. Data on Zenodo,
<https://doi.org/10.5281/zenodo.12636522> (per-year records), code at
<https://github.com/mit-ll/tornet>, pretrained detector at
<https://huggingface.co/tornet-ml/tornado_detector_baseline_v1>.

The dataset is distributed under CC BY 4.0, as released. This repackaging changes the container
format only; cite the paper and the Zenodo record when you use it.

```bibtex
@article{veillette2024tornet,
  title   = {TorNet: A Benchmark Dataset for Tornado Detection and Prediction Using Full-Resolution Polarimetric Weather Radar Data},
  author  = {Veillette, Mark S. and Kurdzo, James M. and Stepanian, Phillip M. and McDonald, Joseph and Samsi, Siddharth and Cho, John Y. N.},
  journal = {Artificial Intelligence for the Earth Systems},
  year    = {2024},
  doi     = {10.1175/AIES-D-24-0006.1}
}
```
