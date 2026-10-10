# TorNet

Software to work with the TorNet dataset as described in the paper [*A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data*](https://journals.ametsoc.org/view/journals/aies/4/1/AIES-D-24-0006.1.xml)

## Updates (7/9/24)

* The TorNet dataset has been updated to version 1.1.  This update fixes a small number of incorrectly labeled frames in v1 of the dataset, and fixes the event and episode IDs of the warning categoies.   Version 1.1 also provides the tornado start and end times in the metadata.  We recommend re-downloading the newer version of the data using the links below. 

* The code and pretrained models are now compatible with `keras` 3.0.   Users can now select their deep learning backend from `tensorflow`, `torch`, or `jax`.  Backend-agnostic data loaders are also provided.  Read more about this library at [keras's website](https://keras.io/).  Users of `tf.keras` should use the `tf_keras` branch of this repo.  

* The pretrained CNN model is now available on [huggingface (tornet-ml/tornado_detector_baseline_v1)](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1).   Instructions for downloading and using the pre-trained model can be found in `models/README.md` and in the `VisualizeSamples.ipynb` notebook.


![Alt text](tornet_image.png?raw=true "sample")



## Downloading the Data

The TorNet dataset can be downloaded from the following location:

#### Zenodo

TorNet is split across 10 files, each containing 1 year of data. There is also a catalog CSV file that is used by some functions in this repository.    

* Tornet 2013 (3 GB) and catalog: [https://doi.org/10.5281/zenodo.12636522](https://doi.org/10.5281/zenodo.12636522)
* Tornet 2014 (15 GB): [https://doi.org/10.5281/zenodo.12637032](https://doi.org/10.5281/zenodo.12637032)
* Tornet 2015 (17 GB): [https://doi.org/10.5281/zenodo.12655151](https://doi.org/10.5281/zenodo.12655151)
* Tornet 2016 (16 GB): [https://doi.org/10.5281/zenodo.12655179](https://doi.org/10.5281/zenodo.12655179)
* Tornet 2017 (15 GB): [https://doi.org/10.5281/zenodo.12655183](https://doi.org/10.5281/zenodo.12655183)
* Tornet 2018 (12 GB): [https://doi.org/10.5281/zenodo.12655187](https://doi.org/10.5281/zenodo.12655187)
* Tornet 2019 (18 GB): [https://doi.org/10.5281/zenodo.12655716](https://doi.org/10.5281/zenodo.12655716)
* Tornet 2020 (17 GB): [https://doi.org/10.5281/zenodo.12655717](https://doi.org/10.5281/zenodo.12655717)
* Tornet 2021 (18 GB): [https://doi.org/10.5281/zenodo.12655718](https://doi.org/10.5281/zenodo.12655718)
* Tornet 2022 (19 GB): [https://doi.org/10.5281/zenodo.12655719](https://doi.org/10.5281/zenodo.12655719)

If downloading through your browser is slow, we recommend downloading these using `zenodo_get` (https://gitlab.com/dvolgyes/zenodo_get).

After downloading, there should be 11 files, `catalog.csv`, and 10 files named as `tornet_YYYY.tar.gz`.   Move and untar these into a target directory, which will be referenced using the `TORNET_ROOT` environment variable in the code.  After untarring the 10 files, this directory should contain `catalog.csv` along with sub-directories `train/` and `test/` filled with `.nc` files for each year in the dataset.


## Setup

This fork uses [uv](https://docs.astral.sh/uv/) and a standard `pyproject.toml`. Python 3.11 or newer is required.

```
uv sync --extra torch --group dev
```

Extras:

* `torch`: Keras 3 on the torch backend, plus the standalone Lightning port in `tornet.models.torch`. This is the only backend locked in `uv.lock`.
* `keras`: Keras 3 only, bring your own backend.
* `hf`: `huggingface_hub`, for downloading the pretrained weights.
* `notebooks`: Jupyter.

* `tensorflow`: Keras 3 on the tensorflow backend, the authors' original stack, plus the TFDS reformat path.

Torch is the default because PyPI tensorflow wheels carry no kernels for Blackwell GPUs such as the
RTX 50 series, so tensorflow falls back to CPU there. For a full-scale run on an H100-class GPU
that matches the paper's stack exactly:

```
uv sync --no-group torch --group tensorflow
export KERAS_BACKEND=tensorflow
```

The model is written against Keras ops, so the architecture is identical on either backend and the
pretrained weights load on both. The jax path remains in the tree but is not locked.

### Development checks

```
uv run ruff format      # format
uv run ruff check       # lint
uv run ty check         # type check (advisory)
uv run pytest           # tests
```

Tests that need the real dataset are skipped unless `TORNET_ROOT` is set. The rest run against small synthetic NetCDF files built in `tests/conftest.py`.

## Loading and visualizing TorNet

Start with `notebooks/DataLoaders.ipynb` to get an overview on loading and visualizing the dataset.

To run inference on TorNet samples using a pretrained model,  look at `notebooks/VisualizeSamples.ipynb`.

## Train CNN baseline model

### Multiple backend support with Keras 3
The model uses Keras 3 which supports multiple backends. The environment variable
KERAS_BACKEND can be used to choose the backend. 

The `tornet-train` and `tornet-test` commands default to `torch` when the variable is unset.

```
export KERAS_BACKEND=torch
# export KERAS_BACKEND=tensorflow
# export KERAS_BACKEND=jax
```

The following trains the CNN baseline model described in the paper.  If you run this out-of-the-box, it will run very slowly because it uses the basic dataloader.  Read the DataLoader notebook for tips on how to optimize the data loader.
```
# Set path to dataset
export TORNET_ROOT=/path/to/tornet     

# Run training
uv run tornet-train configs/tornado_baseline.json
```

### Local iteration on a subset

You do not need the full dataset to develop against the baseline. Every yearly archive contains both
`train/` and `test/` files, so any subset of years gives you a held-out test set for free. The
configs in `configs/` set up a two-fold cross validation over 2013 and 2014 (about 18 GB), mirroring
the paper's year-pair folds:

```
uv run tornet-train configs/fold_a.json   # train 2013, validate 2014
uv run tornet-train configs/fold_b.json   # train 2014, validate 2013
uv run tornet-train configs/final.json    # retrain on both, no validation
```

Each run writes a directory `experiments/<exp_name>_<timestamp>/` containing `results.json` (best
validation epoch, AUC, AUCPR, git commit), `history.csv` (every metric per epoch), the config used,
and `checkpoints/best.keras` plus `checkpoints/last.keras` (set `save_all_checkpoints` to keep every
epoch). `experiments/latest` always points at the most recent run.

Tabulate runs, averaging folds that share an experiment name:

```
uv run tornet-results --group
```

Then evaluate the final checkpoint on the subset's test files:

```
uv run tornet-test --model_path experiments/latest/checkpoints/last.keras --years 2013 2014 --output scores.json
```

Scaling to the full dataset is the same commands with all ten years listed in the config.

## Evaluate trained model
To evaluate this model on the test set, run

```
# Set path to dataset
export TORNET_ROOT=/path/to/tornet  

# Evaluate trained model on all test years (use --years to restrict, --output to save JSON)
uv run tornet-test
```

This will compute and print various metrics computed on the test set.  Note that this script will attempt to download pretrained weights from huggingface, so ensure there is internet connectivity.  Alternatively, manually download the pretrained yourself and provide with `--model_path`


### Disclosure
```
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
```
