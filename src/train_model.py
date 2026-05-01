"""Train RF/LR correction models and write aligned NetCDF slices for plotting."""
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import config
import os

def ensure_file_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")

def find_var(ds, candidates, ds_name="dataset"):
    for v in candidates:
        if v in ds.data_vars:
            return ds[v]
    raise KeyError(
        f"{ds_name}: none of {candidates} in data_vars; available: {list(ds.data_vars)}"
    )

def to_dataarray(obj):
    if isinstance(obj, xr.Dataset):
        return obj[list(obj.data_vars)[0]]
    return obj

def standardize_latlon(da):
    for old, new in [("latitude","lat"),("longitude","lon"),("Latitude","lat"),("Longitude","lon")]:
        if old in da.coords:
            da = da.rename({old:new})
    return da.sortby("lat").sortby("lon")

def standardize_time(da):
    if "valid_time" in da.dims:
        da = da.rename({"valid_time":"time"})
    return da

def ensure_latlon_order(da):
    return da.transpose(...,"lat","lon") if "lat" in da.dims and "lon" in da.dims else da

def deduplicate_time(da, name="dataset"):
    if "time" not in da.dims:
        return da
    time_index = da.get_index("time")
    duplicated = time_index.duplicated(keep="first")
    if duplicated.any():
        keep_idx = np.where(~duplicated)[0]
        removed = int(duplicated.sum())
        print(f"{name}: removed {removed} duplicate time indices (kept first).")
        da = da.isel(time=keep_idx)
    return da

def write_with_clean_time_encoding(obj, path):
    encoding = {}
    if "time" in obj.coords:
        encoding["time"] = {
            "units": "days since 0001-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }
    obj.to_netcdf(path, encoding=encoding if encoding else None)

def _grid_summary(da, name):
    lat = da.lat.values
    lon = da.lon.values
    dlat = float(np.median(np.diff(lat))) if lat.size > 1 else np.nan
    dlon = float(np.median(np.diff(lon))) if lon.size > 1 else np.nan
    lat_min = float(lat.min())
    lat_max = float(lat.max())
    lon_min = float(lon.min())
    lon_max = float(lon.max())
    return {
        "name": name,
        "shape": f"lat={lat.size}, lon={lon.size}",
        "lat_count": int(lat.size),
        "lon_count": int(lon.size),
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_range": f"[{lat_min:.4f}, {lat_max:.4f}]",
        "lon_range": f"[{lon_min:.4f}, {lon_max:.4f}]",
        "dlat": dlat,
        "dlon": dlon,
    }

def _coord_metadata(da, name):
    lat_attrs = da.lat.attrs if "lat" in da.coords else {}
    lon_attrs = da.lon.attrs if "lon" in da.coords else {}
    return {
        "dataset": name,
        "lat_dim": ",".join(da.lat.dims) if "lat" in da.coords else "",
        "lon_dim": ",".join(da.lon.dims) if "lon" in da.coords else "",
        "lat_standard_name": str(lat_attrs.get("standard_name", "")),
        "lon_standard_name": str(lon_attrs.get("standard_name", "")),
        "lat_axis": str(lat_attrs.get("axis", "")),
        "lon_axis": str(lon_attrs.get("axis", "")),
        "lat_units": str(lat_attrs.get("units", "")),
        "lon_units": str(lon_attrs.get("units", "")),
    }

def print_grid_alignment_report(dem, imerg, obs, u10, v10, tcwv):
    print("\n=== Grid alignment ===")
    summaries = [
        _grid_summary(dem, "DEM"),
        _grid_summary(imerg, "IMERG"),
        _grid_summary(obs, "OBS"),
        _grid_summary(u10, "U10"),
        _grid_summary(v10, "V10"),
        _grid_summary(tcwv, "TCWV"),
    ]
    for s in summaries:
        print(
            f"{s['name']:>6} | {s['shape']} | lat {s['lat_range']} | lon {s['lon_range']} | "
            f"dlat={s['dlat']:.4f}, dlon={s['dlon']:.4f}"
        )
    metadata_rows = [
        _coord_metadata(dem, "DEM"),
        _coord_metadata(imerg, "IMERG"),
        _coord_metadata(obs, "OBS"),
        _coord_metadata(u10, "U10"),
        _coord_metadata(v10, "V10"),
        _coord_metadata(tcwv, "TCWV"),
    ]
    print("--- Coordinate metadata ---")
    for m in metadata_rows:
        print(
            f"{m['dataset']:>6} | lat_dim={m['lat_dim']} ({m['lat_standard_name']},{m['lat_axis']},{m['lat_units']}) "
            f"| lon_dim={m['lon_dim']} ({m['lon_standard_name']},{m['lon_axis']},{m['lon_units']})"
        )

    target_lat = dem.lat.values
    target_lon = dem.lon.values
    offset_rows = []
    for da, name in [(imerg, "IMERG"), (obs, "OBS"), (u10, "U10"), (v10, "V10"), (tcwv, "TCWV")]:
        lat_offset = float(np.max(np.abs(da.lat.values - target_lat)))
        lon_offset = float(np.max(np.abs(da.lon.values - target_lon)))
        print(f"{name:>6} vs DEM max |Δlat|={lat_offset:.6f}, |Δlon|={lon_offset:.6f}")
        offset_rows.append({
            "source": name,
            "target": "DEM",
            "max_abs_dlat": lat_offset,
            "max_abs_dlon": lon_offset,
        })
    print("===\n")

    summary_rows = []
    for s in summaries:
        summary_rows.append({
            "section": "grid_summary",
            "dataset": s["name"],
            "lat_count": s["lat_count"],
            "lon_count": s["lon_count"],
            "lat_min": s["lat_min"],
            "lat_max": s["lat_max"],
            "lon_min": s["lon_min"],
            "lon_max": s["lon_max"],
            "dlat": float(s["dlat"]),
            "dlon": float(s["dlon"]),
            "target": "",
            "max_abs_dlat": np.nan,
            "max_abs_dlon": np.nan,
        })
    for row in offset_rows:
        summary_rows.append({
            "section": "grid_offset_vs_dem",
            "dataset": row["source"],
            "lat_count": np.nan,
            "lon_count": np.nan,
            "lat_min": np.nan,
            "lat_max": np.nan,
            "lon_min": np.nan,
            "lon_max": np.nan,
            "dlat": np.nan,
            "dlon": np.nan,
            "target": row["target"],
            "max_abs_dlat": row["max_abs_dlat"],
            "max_abs_dlon": row["max_abs_dlon"],
        })
    for m in metadata_rows:
        summary_rows.append({
            "section": "coord_metadata",
            "dataset": m["dataset"],
            "lat_count": np.nan,
            "lon_count": np.nan,
            "lat_min": np.nan,
            "lat_max": np.nan,
            "lon_min": np.nan,
            "lon_max": np.nan,
            "dlat": np.nan,
            "dlon": np.nan,
            "target": "",
            "max_abs_dlat": np.nan,
            "max_abs_dlon": np.nan,
            "lat_dim": m["lat_dim"],
            "lon_dim": m["lon_dim"],
            "lat_standard_name": m["lat_standard_name"],
            "lon_standard_name": m["lon_standard_name"],
            "lat_axis": m["lat_axis"],
            "lon_axis": m["lon_axis"],
            "lat_units": m["lat_units"],
            "lon_units": m["lon_units"],
        })
    report_df = pd.DataFrame(summary_rows)
    report_path = os.path.join(config.OUTPUTS_DIR, "grid_alignment_report.csv")
    report_df.to_csv(report_path, index=False)
    print(report_path)

def build_samples(imerg_da, u_da, v_da, q_da, dem_da, obs_da, time_mask):
    X_list, y_list = [], []
    dem2d = dem_da.transpose("lat","lon").values
    for t_idx in np.where(time_mask)[0]:
        im = imerg_da.isel(time=t_idx).transpose("lat","lon").values
        u = u_da.isel(time=t_idx).transpose("lat","lon").values
        v = v_da.isel(time=t_idx).transpose("lat","lon").values
        q = q_da.isel(time=t_idx).transpose("lat","lon").values
        ob = obs_da.isel(time=t_idx).transpose("lat","lon").values
        mask = (np.isfinite(im) & np.isfinite(u) & np.isfinite(v) &
                np.isfinite(q) & np.isfinite(dem2d) & np.isfinite(ob))
        if mask.sum() == 0:
            continue
        X = np.stack([im[mask], u[mask], v[mask], q[mask], dem2d[mask]], axis=1)
        y = ob[mask]
        if len(y) > config.MAX_SAMPLES_PER_TIME:
            idx = np.random.choice(len(y), config.MAX_SAMPLES_PER_TIME, replace=False)
            X, y = X[idx], y[idx]
        X_list.append(X)
        y_list.append(y)
    if not X_list:
        raise ValueError(
            "Empty training sample: check time range, variable names, and valid-data mask."
        )
    return np.concatenate(X_list), np.concatenate(y_list)

def build_samples_with_years(imerg_da, u_da, v_da, q_da, dem_da, obs_da, time_mask, time_coord):
    X_list, y_list, year_list = [], [], []
    dem2d = dem_da.transpose("lat","lon").values
    for t_idx in np.where(time_mask)[0]:
        im = imerg_da.isel(time=t_idx).transpose("lat","lon").values
        u = u_da.isel(time=t_idx).transpose("lat","lon").values
        v = v_da.isel(time=t_idx).transpose("lat","lon").values
        q = q_da.isel(time=t_idx).transpose("lat","lon").values
        ob = obs_da.isel(time=t_idx).transpose("lat","lon").values
        mask = (np.isfinite(im) & np.isfinite(u) & np.isfinite(v) &
                np.isfinite(q) & np.isfinite(dem2d) & np.isfinite(ob))
        if mask.sum() == 0:
            continue
        X = np.stack([im[mask], u[mask], v[mask], q[mask], dem2d[mask]], axis=1)
        y = ob[mask]
        if len(y) > config.MAX_SAMPLES_PER_TIME:
            idx = np.random.choice(len(y), config.MAX_SAMPLES_PER_TIME, replace=False)
            X, y = X[idx], y[idx]
        year_val = time_coord.isel(time=t_idx).dt.year.values
        year_arr = np.full(len(y), year_val)
        X_list.append(X)
        y_list.append(y)
        year_list.append(year_arr)
    if not X_list:
        raise ValueError(
            "Empty test sample: check TEST_YEARS, SUMMER_MONTHS, and time alignment."
        )
    return np.concatenate(X_list), np.concatenate(y_list), np.concatenate(year_list)

def train():
    np.random.seed(config.RANDOM_STATE)

    ensure_file_exists(config.IMERG_DIR, "IMERG_FILE")
    ensure_file_exists(config.OBS_FILE, "OBS_FILE")
    ensure_file_exists(config.ERA5_FILE, "ERA5_FILE")
    ensure_file_exists(config.DEM_FILE, "DEM_FILE")

    imerg_ds = xr.open_dataset(config.IMERG_DIR)
    imerg = ensure_latlon_order(standardize_time(standardize_latlon(
        to_dataarray(find_var(imerg_ds, config.IMERG_VAR_CANDIDATES, "IMERG")))))

    obs_ds = xr.open_dataset(config.OBS_FILE)
    obs = ensure_latlon_order(standardize_time(standardize_latlon(
        to_dataarray(find_var(obs_ds, config.OBS_VAR_CANDIDATES, "OBS")))))

    era5_ds = xr.open_dataset(config.ERA5_FILE)
    def clean_era5(da):
        da = standardize_time(standardize_latlon(to_dataarray(da)))
        for dim in ["number","expver"]:
            if dim in da.dims:
                da = da.isel({dim:0}, drop=True)
        return ensure_latlon_order(da)
    u10 = clean_era5(find_var(era5_ds, config.U10_VAR_CANDIDATES, "U10"))
    v10 = clean_era5(find_var(era5_ds, config.V10_VAR_CANDIDATES, "V10"))
    tcwv = clean_era5(find_var(era5_ds, config.TCWV_VAR_CANDIDATES, "TCWV"))

    dem_ds = xr.open_dataset(config.DEM_FILE)
    dem = ensure_latlon_order(standardize_latlon(to_dataarray(
        find_var(dem_ds, config.DEM_VAR_CANDIDATES, "DEM"))))
    if "time" in dem.dims:
        dem = dem.isel(time=0, drop=True)
    dem = dem.transpose("lat","lon")

    target_lon = np.arange(
        config.TARGET_LON_MIN,
        config.TARGET_LON_MAX + config.TARGET_RES / 2.0,
        config.TARGET_RES,
    )
    target_lat = np.arange(
        config.TARGET_LAT_MIN,
        config.TARGET_LAT_MAX + config.TARGET_RES / 2.0,
        config.TARGET_RES,
    )

    imerg = imerg.interp(lat=target_lat, lon=target_lon, method="nearest")
    obs   = obs.interp(lat=target_lat, lon=target_lon, method="nearest")
    u10   = u10.interp(lat=target_lat, lon=target_lon, method="nearest")
    v10   = v10.interp(lat=target_lat, lon=target_lon, method="nearest")
    tcwv  = tcwv.interp(lat=target_lat, lon=target_lon, method="nearest")
    dem   = dem.interp(lat=target_lat, lon=target_lon, method="nearest")

    imerg = deduplicate_time(imerg, "IMERG")
    obs = deduplicate_time(obs, "OBS")
    u10 = deduplicate_time(u10, "U10")
    v10 = deduplicate_time(v10, "V10")
    tcwv = deduplicate_time(tcwv, "TCWV")

    common = imerg.time.values
    for da in [obs, u10, v10, tcwv]:
        common = np.intersect1d(common, da.time.values)
    if len(common) == 0:
        raise ValueError(
            "Empty time intersection for IMERG/OBS/ERA5: check `time` coverage in each file."
        )
    imerg = imerg.sel(time=common)
    obs   = obs.sel(time=common)
    u10   = u10.sel(time=common)
    v10   = v10.sel(time=common)
    tcwv  = tcwv.sel(time=common)

    print_grid_alignment_report(dem, imerg, obs, u10, v10, tcwv)

    train_mask = (imerg.time.dt.year.isin(config.TRAIN_YEARS) &
                  imerg.time.dt.month.isin(config.SUMMER_MONTHS))
    test_mask  = (imerg.time.dt.year.isin(config.TEST_YEARS) &
                  imerg.time.dt.month.isin(config.SUMMER_MONTHS))

    print(f"train_days={int(train_mask.sum())} test_days={int(test_mask.sum())}")

    X_train, y_train = build_samples(imerg, u10, v10, tcwv, dem, obs, train_mask.values)
    X_test,  y_test, years_test = build_samples_with_years(imerg, u10, v10, tcwv, dem, obs, test_mask.values, imerg.time)

    rf = RandomForestRegressor(n_estimators=config.N_ESTIMATORS,
                               random_state=config.RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    lr = LinearRegression().fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    lr_pred = lr.predict(X_test)
    raw_pred = X_test[:,0]

    joblib.dump(rf, config.RF_MODEL_PATH)
    joblib.dump(lr, config.LR_MODEL_PATH)
    np.savez(config.DATA_NPZ_PATH,
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             raw_pred=raw_pred, rf_pred=rf_pred, lr_pred=lr_pred,
             years_test=years_test)

    unique_years = np.unique(years_test)
    year_rows = []
    for yr in unique_years:
        mask_yr = years_test == yr
        yt = y_test[mask_yr]
        rawp = raw_pred[mask_yr]
        rfp = rf_pred[mask_yr]
        lrp = lr_pred[mask_yr]
        year_rows.append([
            int(yr), int(mask_yr.sum()),
            r2_score(yt, rawp), np.sqrt(mean_squared_error(yt, rawp)),
            mean_absolute_error(yt, rawp), np.mean(rawp - yt),
            r2_score(yt, lrp), np.sqrt(mean_squared_error(yt, lrp)),
            mean_absolute_error(yt, lrp), np.mean(lrp - yt),
            r2_score(yt, rfp), np.sqrt(mean_squared_error(yt, rfp)),
            mean_absolute_error(yt, rfp), np.mean(rfp - yt)
        ])
    year_df = pd.DataFrame(year_rows, columns=[
        "Year", "N",
        "IMERG_R2", "IMERG_RMSE", "IMERG_MAE", "IMERG_Bias",
        "LR_R2", "LR_RMSE", "LR_MAE", "LR_Bias",
        "RF_R2", "RF_RMSE", "RF_MAE", "RF_Bias"
    ])
    print(year_df.round(3))
    year_df.to_csv(os.path.join(config.OUTPUTS_DIR, "per_year_metrics.csv"), index=False)

    dem.to_netcdf(config.DEM_REF_NC)
    write_with_clean_time_encoding(imerg.to_dataset(name="imerg"), config.IMERG_NC)
    write_with_clean_time_encoding(obs.to_dataset(name="obs"), config.OBS_NC)
    write_with_clean_time_encoding(u10.to_dataset(name="u10"), config.U10_NC)
    write_with_clean_time_encoding(v10.to_dataset(name="v10"), config.V10_NC)
    write_with_clean_time_encoding(tcwv.to_dataset(name="tcwv"), config.TCWV_NC)
    time_index_ds = xr.Dataset(coords={"time": imerg.time})
    write_with_clean_time_encoding(time_index_ds, config.TIME_NC)

    print(config.MODELS_DIR)

if __name__ == "__main__":
    train()