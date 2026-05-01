"""CSV/Excel summary tables from saved predictions and grids."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib
import xarray as xr
import config
import os

def reduce_samples(X, y, mx=2000):
    if len(y) <= mx:
        return X, y
    idx = np.random.choice(len(y), mx, replace=False)
    return X[idx], y[idx]

def build_ablation_X(imerg_da, u_da, v_da, q_da, dem_da, obs_da, time_mask, config_vars):
    X_list, y_list = [], []
    dem2d = dem_da.transpose("lat","lon").values
    for t_idx in np.where(time_mask)[0]:
        feats = []
        ob = obs_da.isel(time=t_idx).transpose("lat","lon").values
        mask = np.isfinite(ob)
        if "IMERG" in config_vars:
            im = imerg_da.isel(time=t_idx).transpose("lat","lon").values
            feats.append(im); mask &= np.isfinite(im)
        if "u10" in config_vars:
            u = u_da.isel(time=t_idx).transpose("lat","lon").values
            feats.append(u); mask &= np.isfinite(u)
        if "v10" in config_vars:
            v = v_da.isel(time=t_idx).transpose("lat","lon").values
            feats.append(v); mask &= np.isfinite(v)
        if "tcwv" in config_vars:
            q = q_da.isel(time=t_idx).transpose("lat","lon").values
            feats.append(q); mask &= np.isfinite(q)
        if "DEM" in config_vars:
            feats.append(dem2d); mask &= np.isfinite(dem2d)
        if mask.sum() == 0:
            continue
        X = np.stack([f[mask] for f in feats], axis=1)
        y = ob[mask]
        X, y = reduce_samples(X, y)
        X_list.append(X); y_list.append(y)
    if not X_list:
        raise ValueError(
            f"Empty ablation sample for config_vars={config_vars}: check time mask and NaNs."
        )
    return np.concatenate(X_list), np.concatenate(y_list)

def collect_valid_obs_values(imerg_da, u_da, v_da, q_da, dem_da, obs_da, time_mask):
    y_all = []
    dem2d = dem_da.transpose("lat","lon").values
    dem_valid = np.isfinite(dem2d)
    for t_idx in np.where(time_mask)[0]:
        im = imerg_da.isel(time=t_idx).transpose("lat","lon").values
        u = u_da.isel(time=t_idx).transpose("lat","lon").values
        v = v_da.isel(time=t_idx).transpose("lat","lon").values
        q = q_da.isel(time=t_idx).transpose("lat","lon").values
        ob = obs_da.isel(time=t_idx).transpose("lat","lon").values
        mask = (
            np.isfinite(im) & np.isfinite(u) & np.isfinite(v) &
            np.isfinite(q) & dem_valid & np.isfinite(ob)
        )
        if mask.sum():
            y_all.append(ob[mask])
    if not y_all:
        return np.array([])
    return np.concatenate(y_all)

def make_all_tables():
    data = np.load(config.DATA_NPZ_PATH)
    X_test = data['X_test']
    y_test = data['y_test']
    raw_pred = data['raw_pred']
    rf_pred = data['rf_pred']
    lr_pred = data['lr_pred']

    def metrics(yt, yp):
        return [r2_score(yt,yp), np.sqrt(mean_squared_error(yt,yp)),
                mean_absolute_error(yt,yp), np.mean(yp-yt)]

    tbl1 = pd.DataFrame(
        [["IMERG"] + metrics(y_test, raw_pred),
         ["LR"] + metrics(y_test, lr_pred),
         ["RF"] + metrics(y_test, rf_pred)],
        columns=["Model","R2","RMSE","MAE","Bias"])
    test_metrics_csv = os.path.join(config.OUTPUTS_DIR, "test_metrics.csv")
    tbl1.to_csv(test_metrics_csv, index=False)

    classes = [(0.1,10,"Light"),(10,25,"Moderate"),(25,50,"Heavy"),(50,1000,"Torrential")]
    rows = []
    for low,high,name in classes:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum():
            rows.append([name, mask.sum(),
                         np.sqrt(mean_squared_error(y_test[mask], raw_pred[mask])),
                         np.sqrt(mean_squared_error(y_test[mask], lr_pred[mask])),
                         np.sqrt(mean_squared_error(y_test[mask], rf_pred[mask])),
                         np.mean(raw_pred[mask] - y_test[mask]),
                         np.mean(lr_pred[mask] - y_test[mask]),
                         np.mean(rf_pred[mask] - y_test[mask])])
        else:
            rows.append([name, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    rain_df = pd.DataFrame(rows, columns=["Class","N","IMERG_RMSE","LR_RMSE","RF_RMSE",
                                          "IMERG_Bias","LR_Bias","RF_Bias"])
    rain_class_csv = os.path.join(config.OUTPUTS_DIR, "rain_class_rmse_bias.csv")
    rain_df.to_csv(rain_class_csv, index=False)

    dem_da = xr.open_dataarray(config.DEM_REF_NC)
    obs_da = xr.open_dataarray(config.OBS_NC)
    imerg_da = xr.open_dataarray(config.IMERG_NC)
    u10_da = xr.open_dataarray(config.U10_NC)
    v10_da = xr.open_dataarray(config.V10_NC)
    tcwv_da = xr.open_dataarray(config.TCWV_NC)
    time_ds = xr.open_dataset(config.TIME_NC)
    train_mask = (time_ds.time.dt.year.isin(config.TRAIN_YEARS) &
                  time_ds.time.dt.month.isin(config.SUMMER_MONTHS))
    test_mask  = (time_ds.time.dt.year.isin(config.TEST_YEARS) &
                  time_ds.time.dt.month.isin(config.SUMMER_MONTHS))

    ablation_configs = [
        ("IMERG", ["IMERG"]),
        ("IMERG+DEM", ["IMERG","DEM"]),
        ("IMERG+ERA5", ["IMERG","u10","v10","tcwv"]),
        ("Full", ["IMERG","u10","v10","tcwv","DEM"])
    ]
    from sklearn.ensemble import RandomForestRegressor
    ablation_rows = []
    for name, cfg in ablation_configs:
        Xtr, ytr = build_ablation_X(imerg_da, u10_da, v10_da, tcwv_da, dem_da, obs_da, train_mask.values, cfg)
        Xte, yte = build_ablation_X(imerg_da, u10_da, v10_da, tcwv_da, dem_da, obs_da, test_mask.values, cfg)
        rf = RandomForestRegressor(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE, n_jobs=1)
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xte)
        ablation_rows.append([name] + metrics(yte, pred))
    ablation_df = pd.DataFrame(ablation_rows, columns=["Model","R2","RMSE","MAE","Bias"])
    ablation_csv = os.path.join(config.OUTPUTS_DIR, "ablation_results.csv")
    ablation_df.to_csv(ablation_csv, index=False)

    metrics_csv_df = pd.read_csv(test_metrics_csv)
    rain_csv_df = pd.read_csv(rain_class_csv)
    ablation_csv_df = pd.read_csv(ablation_csv)
    grid_report_csv = os.path.join(config.OUTPUTS_DIR, "grid_alignment_report.csv")
    grid_report_df = pd.read_csv(grid_report_csv) if os.path.exists(grid_report_csv) else None

    def pct_str(v):
        return f"{v*100:.1f}%"

    y_train_all = collect_valid_obs_values(imerg_da, u10_da, v10_da, tcwv_da, dem_da, obs_da, train_mask.values)
    y_test_all = collect_valid_obs_values(imerg_da, u10_da, v10_da, tcwv_da, dem_da, obs_da, test_mask.values)
    n_train = int(len(y_train_all))
    n_test = int(len(y_test_all))
    n_total = n_train + n_test
    train_days = 0
    test_days = 0
    if "time" in time_ds:
        train_days = int(train_mask.sum())
        test_days = int(test_mask.sum())
    train_years_txt = f"{min(config.TRAIN_YEARS)}–{max(config.TRAIN_YEARS)}"
    test_years_txt = f"{min(config.TEST_YEARS)}"
    if len(config.TEST_YEARS) > 1:
        test_years_txt = f"{min(config.TEST_YEARS)}–{max(config.TEST_YEARS)}"
    section1 = pd.DataFrame([
        ["Total valid grid-day samples (incl. zero rain)", n_total],
        [f"Train samples ({train_years_txt} summer, {train_days} days)", n_train],
        [f"Test samples ({test_years_txt} summer, {test_days} days)", n_test],
    ], columns=["Item", "Value"])

    rain_classes = [
        ("Light", 0.1, 10.0),
        ("Moderate", 10.0, 25.0),
        ("Heavy", 25.0, 50.0),
        ("Torrential", 50.0, np.inf),
    ]
    class_rows = []
    for cname, low, high in rain_classes:
        if np.isinf(high):
            m = y_test_all >= low
            thr = "≥ 50.0"
        else:
            m = (y_test_all >= low) & (y_test_all < high)
            thr = f"{low:.1f} – {high-0.1:.1f}"
        class_rows.append([cname, thr, int(m.sum())])
    zero_rain_n = int((y_test_all < 0.1).sum())
    section2 = pd.DataFrame(class_rows, columns=["Category", "Threshold (mm/d)", "Count (N)"])
    section2_note = pd.DataFrame([
        ["No-rain samples (<0.1 mm/d)", zero_rain_n]
    ], columns=["Item", "Value"])

    imerg_row = metrics_csv_df.loc[metrics_csv_df["Model"] == "IMERG"].iloc[0]
    section3 = pd.DataFrame([
        ["R²", imerg_row["R2"]],
        ["RMSE (mm/d)", imerg_row["RMSE"]],
        ["MAE (mm/d)", imerg_row["MAE"]],
        ["Bias (mm/d)", imerg_row["Bias"]],
    ], columns=["Metric", "Value"])

    lr_row = metrics_csv_df.loc[metrics_csv_df["Model"] == "LR"].iloc[0]
    rf_row = metrics_csv_df.loc[metrics_csv_df["Model"] == "RF"].iloc[0]
    section41 = pd.DataFrame([
        ["LR-Full", lr_row["R2"], lr_row["RMSE"], lr_row["MAE"], lr_row["Bias"]],
        ["RF-Full", rf_row["R2"], rf_row["RMSE"], rf_row["MAE"], rf_row["Bias"]],
    ], columns=["Model", "R²", "RMSE (mm/d)", "MAE (mm/d)", "Bias (mm/d)"])

    section42 = ablation_csv_df.copy()
    section42["Model"] = section42["Model"].replace({
        "IMERG": "RF-IMERG",
        "IMERG+DEM": "RF-IMERG+DEM",
        "IMERG+ERA5": "RF-IMERG+ERA5",
        "Full": "RF-Full (IMERG+ERA5+DEM)"
    })
    section42 = section42.rename(columns={
        "R2": "R²",
        "RMSE": "RMSE (mm/d)",
        "MAE": "MAE (mm/d)",
        "Bias": "Bias (mm/d)"
    })

    rf_full = joblib.load(config.RF_MODEL_PATH)
    perm = permutation_importance(
        rf_full, X_test, y_test, scoring="r2",
        n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=1
    )
    feat_names = ["IMERG", "u10", "v10", "tcwv", "DEM"]
    imp_df = pd.DataFrame({
        "Feature": feat_names,
        "Importance (ΔR²)": perm.importances_mean
    }).sort_values("Importance (ΔR²)", ascending=False).reset_index(drop=True)

    rows6 = []
    for _, r in rain_csv_df.iterrows():
        cname = r["Class"]
        rows6.append([cname, "Raw IMERG", r["IMERG_RMSE"], r["IMERG_Bias"]])
        rows6.append([cname, "LR-Full", r["LR_RMSE"], r["LR_Bias"]])
        rows6.append([cname, "RF-Full", r["RF_RMSE"], r["RF_Bias"]])
    section6 = pd.DataFrame(rows6, columns=["Intensity", "Model", "RMSE (mm/d)", "Bias (mm/d)"])

    section7 = pd.DataFrame([
        ["n_estimators", getattr(rf_full, "n_estimators", config.N_ESTIMATORS)],
        ["max_depth", getattr(rf_full, "max_depth", None)],
        ["random_state", getattr(rf_full, "random_state", config.RANDOM_STATE)],
        ["Other", "scikit-learn defaults except fields above"],
    ], columns=["Parameter", "Value"])

    rmse_improve = np.nan if np.isclose(imerg_row["RMSE"], 0.0) else (imerg_row["RMSE"] - rf_row["RMSE"]) / imerg_row["RMSE"]
    mae_improve = np.nan if np.isclose(imerg_row["MAE"], 0.0) else (imerg_row["MAE"] - rf_row["MAE"]) / imerg_row["MAE"]
    section8 = pd.DataFrame([
        ["RMSE reduction", "N/A" if np.isnan(rmse_improve) else pct_str(rmse_improve)],
        ["MAE reduction", "N/A" if np.isnan(mae_improve) else pct_str(mae_improve)],
        ["R²", f"{imerg_row['R2']:.3f} -> {rf_row['R2']:.3f}"],
    ], columns=["Metric", "RF-Full vs raw IMERG"])

    excel_path = os.path.join(config.OUTPUTS_DIR, "model_summary.xlsx")
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            if grid_report_df is not None:
                grid_report_df.to_excel(writer, sheet_name="0_Grid_alignment", index=False)
            section1.to_excel(writer, sheet_name="1_Sample_counts", index=False)
            section2.to_excel(writer, sheet_name="2_Intensity_counts", index=False)
            section2_note.to_excel(writer, sheet_name="2_Intensity_counts", index=False, startrow=len(section2) + 2)
            section3.to_excel(writer, sheet_name="3_Raw_IMERG_metrics", index=False)
            section41.to_excel(writer, sheet_name="4_Model_performance", index=False)
            section42.to_excel(writer, sheet_name="4_Ablation", index=False)
            imp_df.to_excel(writer, sheet_name="5_Feature_importance", index=False)
            section6.to_excel(writer, sheet_name="6_RMSE_Bias_by_intensity", index=False)
            section7.to_excel(writer, sheet_name="7_RF_hyperparameters", index=False)
            section8.to_excel(writer, sheet_name="8_Improvement_vs_IMERG", index=False)
        print(excel_path)
    except ModuleNotFoundError:
        print("openpyxl missing, skip xlsx")

    print(config.OUTPUTS_DIR)

if __name__ == "__main__":
    make_all_tables()