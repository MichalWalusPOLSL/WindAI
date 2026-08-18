# Wind speed classification

Beaufort-class classification of 100 m wind speed from ERA5 reanalysis
(Open-Meteo Historical Weather API) for three Polish locations.

## Layout

```
config.yaml               the experiment grid: locations x models x variants
data/                     frozen input CSV, one file per location
src/dataset.py            loading, Beaufort target, pressure/PBL tendency
src/splits.py             LeaveOneYearOut, WalkForward
src/registry.py           preprocessing variants + model factories
src/run.py                config-driven experiment loop -> predictions
src/metrics.py            metrics computed from stored predictions
src/stats.py              McNemar, Wilcoxon, Friedman + Nemenyi
results/predictions/      one file per (location, model, variant, horizon, splitter)
results/tables/           tables for the thesis
tools/gpu_check.py        versions, driver and the device each model will use
```

## Data

One CSV per location, 2015-2023 hourly, 78 888 rows and 29 features after the
tendency lags. `data/` may be a directory junction to the dataset folder outside
the repository; the CSV themselves stay out of git.

The AI Studio exports have two quirks the loader absorbs, both checked in
`_read` and `_regular_hours`:

- **cp1252, not utf-8.** The degree sign in the temperature columns is a single
  0xb0 byte, so a plain `read_csv` raises `UnicodeDecodeError`.
- **Local-time labels with DST damage.** Every day carries 24 rows, switch days
  included, so the hourly sequence is contiguous while the labels skip 02:00 in
  March and repeat the following hour: 18 irregular steps over nine years. The
  rows themselves are distinct observations, so the loader rebuilds the index
  from the first label and keeps a single fixed offset. Anything beyond two
  irregular steps per year is a real gap and still raises.

## Setup

`Pipfile.lock` pins the exact versions the results were produced with, so the
same environment can be rebuilt on any machine with Python 3.13:

```
pip install --user pipenv
pipenv install --dev
pipenv run python tools/gpu_check.py
```

On Windows use the launcher if `python` is not on PATH: `py -m pip install
--user pipenv`, then `py -m pipenv install --dev`. Set `PIPENV_VENV_IN_PROJECT=1`
to keep the virtualenv in `.venv/` next to the sources.

For another interpreter version, edit `[requires]` in the `Pipfile` and re-lock
with `pipenv lock`. `requirements.txt` is kept in sync as an unpinned fallback
for environments without pipenv.

## Hardware

The optional boosters are the only models with a GPU path, and only xgboost has
one that works out of the box:

| model | device | why |
| --- | --- | --- |
| `xgb` | CUDA | the PyPI wheel ships CUDA kernels; 3.x is the first release with Blackwell (sm_120) support |
| `lgbm` | CPU | the PyPI wheel has no GPU tree learner, it needs a source build with `-DUSE_GPU=1` |
| everything else | CPU | scikit-learn has no CUDA backend |

`registry` probes both libraries with a throwaway fit and falls back to CPU when
the GPU path is missing, so the same config runs on a workstation and on a
laptop. `WIND_XGB_DEVICE` / `WIND_LGBM_DEVICE` pin the device explicitly.

GPU is not automatically faster here. On ~20k rows a CUDA `xgb` fit measured
8.0 s against 2.1 s on CPU: the dataset is far too small to amortise the kernel
launches, and CUDA only starts paying off if the grid grows by orders of
magnitude. `WIND_XGB_DEVICE=cpu` is the faster setting for the thesis runs.

## Usage

The grid in `config.yaml` is the whole interface: the runner takes the cartesian
product of `locations`, `models`, `preprocessing`, `horizons` and `splitters`,
so the default file is 3 x 4 x 1 x 1 x 1 = 12 runs.

```
pipenv run run                                    # the configured grid
pipenv run rerun                                  # the same, --overwrite
pipenv run gpucheck                               # versions, driver, devices
pipenv run python src/run.py --config other.yaml  # any other config
```

Inside `pipenv shell` the same commands work as plain `python src/run.py
--config config.yaml`. `run.py` resolves every path from its own location, so
the working directory does not matter. On Windows prefix with the launcher when
`python` is not on PATH: `py -m pipenv run run`.

Runs already present in `results/predictions/` are skipped unless `--overwrite`
is given, so the grid can be extended incrementally: add a model to the config,
run again, and only the missing combinations are trained.

Beyond the configured four, the registry offers `svm`, `nb`, `lda`, `xgb`,
`lgbm`, `voting` and `stacking`, the `scaled` / `binned_width` / `binned_freq`
variants, and the `walk_forward` splitter. `registry.available_models()` lists
them.

### Reading the results

Metrics are computed from the stored predictions, never during training, so this
step needs no retraining and can be repeated as the grid grows. `metrics` and
`stats` live in `src/`, which has to be importable:

```
pipenv run python -c "import sys; sys.path.insert(0, 'src'); import metrics; print(metrics.summarise(metrics.load_predictions('results/predictions')).to_string())"
```

For an interactive session, work from `src/` and point the loader one level up:

```python
import metrics, stats
p = metrics.load_predictions("../results/predictions")
metrics.summarise(p)                        # one row per run
metrics.per_class(p)                        # one row per class within a run
metrics.confusion(p)                        # pooled confusion matrix
metrics.frequency_recall_correlation(metrics.per_class(p))
stats.mcnemar_matrix(p, location="hel")     # pairwise, Holm-corrected
stats.friedman_nemenyi(stats.fold_scores(metrics.summarise(p, by=["model", "fold"])))
```
