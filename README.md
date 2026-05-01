# IMERG precipitation correction (Hunan)

Train **random forest (RF)** and **linear regression (LR)** models to bias-correct GPM IMERG daily precipitation against a gridded gauge analysis (**CN05.1**), with **ERA5** (u10, v10, tcwv) and **DEM** as extra predictors. The workflow interpolates all fields to a common 0.25° grid over Hunan Province (China), fits on summer months (JJA) for selected years, evaluates on held-out years, and exports **figures**, **CSV metrics**, and an optional **Excel** summary.

## Pipeline

1. **`train`** — Load NetCDF inputs, align grids/times, build train/test samples, fit RF & LR, save `joblib` models + `npz` predictions + NetCDF slices for plotting.
2. **`plot_all`** — Maps, scatter metrics, time series, rain-intensity RMSE, permutation importance, SHAP summary & dependence, bias maps (requires **geopandas** + `assets/hunan.geojson`).
3. **`make_all_tables`** — Writes `outputs/*.csv` and `outputs/model_summary.xlsx` (needs **openpyxl**).

Entry point:

```bash
python main.py
```

Run steps separately if needed (from the repo root, with `src` on `PYTHONPATH`):

```bash
python -c "import sys; sys.path.insert(0,'src'); from train_model import train; train()"
python -c "import sys; sys.path.insert(0,'src'); from plot_figures import plot_all; plot_all()"
python -c "import sys; sys.path.insert(0,'src'); from make_tables import make_all_tables; make_all_tables()"
```

## Requirements

- **Python** 3.9+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

See `requirements.txt` for pinned-style listing (numpy, pandas, xarray, scikit-learn, matplotlib, netCDF4, joblib, scipy, shap, geopandas, openpyxl).

## Input data

Set the data directory with the environment variable **`DATA_ROOT`** (default in `src/config.py` is `/mnt/data/imerg_correction_hunan`). Under `DATA_ROOT`, the pipeline expects:

| File               | Role                                                               |
| ------------------ | ------------------------------------------------------------------ |
| `imerg.nc`         | IMERG precipitation (time × lat × lon)                             |
| `cn051.nc`         | CN05.1 (or other) gridded daily precipitation for training targets |
| `era5.nc`          | ERA5: u10, v10, tcwv (and consistent `time`)                       |
| `dem_hunan_025.nc` | Static DEM on or near the target grid                              |

Variable names are resolved via candidate lists in `src/config.py` (`*_VAR_CANDIDATES`). Datasets are renamed to `lat` / `lon` / `time` where needed and interpolated to the target extent:

- Longitude **108.65–114.35°E**, latitude **24.50–30.30°N**, resolution **0.25°** (WGS84).

Training and test years, and summer months, are configured in `config.py` (`TRAIN_YEARS`, `TEST_YEARS`, `SUMMER_MONTHS`).

## Configuration

Edit **`src/config.py`** for:

- `DATA_ROOT` default or use `export DATA_ROOT=/path/to/data`
- Train/test years and `SUMMER_MONTHS`
- `MAX_SAMPLES_PER_TIME`, `N_ESTIMATORS`, `RANDOM_STATE`
- Optional overrides (not required): `FONTS_DIR`, `HUNAN_GEOJSON`, `BOUNDARY_SIMPLIFY_TOL`, `SHAP_MAX_SAMPLES` — see `plot_figures.py` / `getattr(config, ...)`.

## Assets

- **`assets/hunan.geojson`** — Province boundary for map clipping and masks.
- **`assets/fonts/`** (optional) — Drop `.ttf`/`.otf` here if you want custom sans-serif fonts for figures.

## Outputs

| Location       | Contents                                                                                                                                                                                                                                                             |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`models/`**  | `rf_model.joblib`, `lr_model.joblib`, `train_test_data.npz`, `dem_reference.nc`, `time_index.nc`, `obs_for_plot.nc`, `imerg_for_plot.nc`, `u10_for_plot.nc`, `v10_for_plot.nc`, `tcwv_for_plot.nc`                                                                   |
| **`outputs/`** | Metrics CSVs (`test_metrics.csv`, `rain_class_rmse_bias.csv`, `ablation_results.csv`, `per_year_metrics.csv`, `grid_alignment_report.csv`, `SHAP_Directionality_Table.csv`, …), high-res **PNG/TIFF** figures, and **`model_summary.xlsx`** if openpyxl is available |

## Figures (non-exhaustive)

- Scatter: raw IMERG vs CN05.1 and RF vs CN05.1
- Spatial: observed vs RF-corrected precipitation by year
- RMSE by rain-intensity class (IMERG / LR / RF)
- Domain-mean daily time series
- Permutation importance and SHAP (summary + tcwv dependence)
- IMERG vs RF bias maps
- Supplementary raw IMERG spatial means

## Troubleshooting

- **`geopandas` / boundary errors** — Install geopandas and ensure `assets/hunan.geojson` exists; plotting uses the same mask philosophy as training.
- **Empty time intersection** — Check that IMERG, CN05.1, and ERA5 share overlapping `time` after preprocessing.
- **Excel skipped** — Install `openpyxl`; CSV tables are still written.
- **Chinese labels in old runs** — Current code uses English table headers and sheet names; re-run `make_all_tables()` for English `model_summary.xlsx`.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

If you use this repository in published work, please cite the code itself using the Zenodo DOI and refer to the data sources as appropriate:

Yi Xu. (2026). *IMERG precipitation correction pipeline over Hunan Province (v1.0.0).* Zenodo. [https://doi.org/10.5281/zenodo.19937476](https://doi.org/10.5281/zenodo.19937476)

Additionally, remember to cite the IMERG, CN05.1, and ERA5 products you used.
