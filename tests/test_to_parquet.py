"""netCDF -> Parquet shards round-trip on a few real files (needs TORNET_ROOT: the synthetic
fixture's chips are not 120 x 240)."""

import os

import numpy as np
import pandas as pd
import pytest

from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import read_file
from tornet.scripts.to_parquet import convert, merge_catalog


@pytest.fixture
def real_root():
    root = os.environ.get("TORNET_ROOT")
    if not root or not os.path.exists(os.path.join(root, "test", "2014")):
        pytest.skip("TORNET_ROOT with test/2014 needed")
    return root


def test_convert_round_trip(real_root, tmp_path):
    from datasets import load_dataset

    root = real_root
    cat = pd.read_csv(os.path.join(root, "catalog.csv"), parse_dates=["start_time", "end_time"])
    small = cat[(cat["type"] == "test") & (cat["start_time"].dt.year == 2014)].head(6)
    data = tmp_path / "data"
    (data / "test" / "2014").mkdir(parents=True)
    for f in small["filename"]:
        os.symlink(os.path.join(root, f), data / f)
    small.to_csv(data / "catalog.csv", index=False)
    out = tmp_path / "out"
    convert(str(data), str(out), ["test"], [2014], shard_size=4, workers=2)
    merge_catalog(str(out))
    shards = sorted(os.listdir(out / "data" / "test" / "2014"))
    assert shards == ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"]
    ds = load_dataset("parquet", data_files=str(out / "data/test/2014/*.parquet"), split="train")
    ds = ds.with_format("numpy")
    assert len(ds) == 6 and ds.features["VEL"].shape == (4, 120, 240, 2)
    row = ds[1]
    ref = read_file(os.path.join(root, row["filename"]), variables=ALL_VARIABLES, n_frames=4)
    for v in ALL_VARIABLES:
        assert np.array_equal(row[v].astype(np.float16), ref[v].astype(np.float16), equal_nan=True)
    assert np.array_equal(row["range_folded_mask"], ref["range_folded_mask"].astype(np.uint8))
    assert list(row["frame_labels"]) == ref["frame_labels"][0].tolist()
    assert list(row["time"]) == ref["time"].tolist()
    catalog = pd.read_parquet(out / "catalog.parquet")
    assert len(catalog) == 6 and set(catalog["shard"]) == {0, 1}
    assert catalog["filename"].tolist() == sorted(small["filename"])
    assert "lat" in catalog and "tornado_start_time" in catalog


def test_convert_rejects_unknown_year(real_root, tmp_path):
    convert(real_root, str(tmp_path), ["test"], [1999], shard_size=4, workers=1)
    assert not (tmp_path / "data").exists()
    with pytest.raises(FileNotFoundError):
        merge_catalog(str(tmp_path / "nothing"))
