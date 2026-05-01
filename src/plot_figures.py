"""Publication-style figures: maps, diagnostics, SHAP (requires trained models in config paths)."""
import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import joblib
from joblib import parallel_backend
import config
import shap

try:
    import geopandas as gpd
except ImportError:
    gpd = None

COLORS = {
    "obs": "#1A1A1A",
    "imerg": "#5B8FB9",
    "lr": "#E39D3E",
    "rf": "#3E8F6A",
    "identity": "#6F6F6F",
    "grid": "#CFCFCF",
}

# Map display extent (Hunan, WGS84); slightly wider fill extent reduces edge NaNs in plots
DISPLAY_LON_MIN = 108.65
DISPLAY_LON_MAX = 114.35
DISPLAY_LAT_MIN = 24.50
DISPLAY_LAT_MAX = 30.30
FILL_LON_MIN = 108.45
FILL_LON_MAX = 114.55
FILL_LAT_MIN = 24.30
FILL_LAT_MAX = 30.50

def setup_custom_fonts():
    """Register fonts under assets/fonts (if present) for matplotlib."""
    custom_families = set()
    custom_files = set()
    fonts_dir = getattr(config, "FONTS_DIR", os.path.join(config.BASE_DIR, "assets", "fonts"))
    if os.path.isdir(fonts_dir):
        for fname in os.listdir(fonts_dir):
            if fname.lower().endswith((".ttf", ".otf", ".ttc")):
                fpath = os.path.join(fonts_dir, fname)
                try:
                    fm.fontManager.addfont(fpath)
                    custom_families.add(fm.FontProperties(fname=fpath).get_name())
                    custom_files.add(fname.lower())
                except Exception as exc:
                    print(f"Font skip: {fpath} ({exc})")
    return custom_families, custom_files

def apply_nature_style():
    """Apply rcParams tuned for line weights, type sizes, and grid styling."""
    custom_families, custom_files = setup_custom_fonts()
    preferred = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    installed = {f.name for f in fm.fontManager.ttflist}
    available = installed.union(custom_families)
    if any(n in custom_files for n in ["arial.ttf", "arialn.ttf", "ariali.ttf", "arialbd.ttf"]):
        font_family = "Arial"
    elif any(n in custom_files for n in ["helveti1.ttf"]):
        font_family = "Helvetica"
    else:
        font_family = next((f for f in preferred if f in available), "DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "axes.facecolor": "#F5F5F5",
        "axes.edgecolor": "#3A3A3A",
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": "#222222",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
        "font.size": 10,
        "font.family": font_family,
        "font.sans-serif": preferred + sorted(custom_families),
        "mathtext.fontset": "stix",
    })

def format_map_axis(ax):
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(DISPLAY_LON_MIN, DISPLAY_LON_MAX)
    ax.set_ylim(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX)
    ax.set_facecolor("#E8E8E8")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)

def add_panel_label(ax, label):
    ax.text(
        0.015, 0.985, label, transform=ax.transAxes, va="top", ha="left",
        fontsize=10, fontweight="bold", color="#222222",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85)
    )

def get_cell_edge_extent(da):
    """imshow extent from cell centers so domain edges are not half-cell clipped."""
    lon = da.lon.values
    lat = da.lat.values
    if lon.size > 1:
        dlon = float(np.median(np.diff(lon)))
    else:
        dlon = 0.25
    if lat.size > 1:
        dlat = float(np.median(np.diff(lat)))
    else:
        dlat = 0.25
    return [
        float(lon.min() - 0.5 * dlon),
        float(lon.max() + 0.5 * dlon),
        float(lat.min() - 0.5 * dlat),
        float(lat.max() + 0.5 * dlat),
    ]

def load_hunan_boundary():
    if gpd is None:
        raise ImportError("geopandas is required to load the Hunan boundary GeoJSON.")
    hunan_geojson = getattr(config, "HUNAN_GEOJSON", os.path.join(config.BASE_DIR, "assets", "hunan.geojson"))
    if not os.path.exists(hunan_geojson):
        raise FileNotFoundError(f"Hunan boundary GeoJSON not found: {hunan_geojson}")
    gdf = gpd.read_file(hunan_geojson)
    if gdf.empty:
        raise ValueError(f"Hunan boundary GeoJSON is empty: {hunan_geojson}")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf

def simplify_boundary_for_plot(boundary_gdf, tolerance=0.005):
    if boundary_gdf is None:
        return None
    simplified = boundary_gdf.copy()
    simplified["geometry"] = simplified.geometry.simplify(tolerance, preserve_topology=True)
    return simplified

def overlay_boundary(ax, boundary_gdf):
    if boundary_gdf is None:
        return
    boundary_gdf.boundary.plot(ax=ax, edgecolor="#5C5C5C", linewidth=0.33, alpha=0.72, zorder=5)
    outer = boundary_gdf.dissolve()
    outer.boundary.plot(ax=ax, edgecolor="white", linewidth=1.8, zorder=6)
    outer.boundary.plot(ax=ax, edgecolor="#222222", linewidth=1.05, zorder=7)

def build_clip_patch(ax, boundary_gdf):
    if boundary_gdf is None:
        return None
    outer = boundary_gdf.dissolve()
    geom = outer.geometry.iloc[0]
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    verts, codes = [], []
    for poly in polys:
        ext = np.asarray(poly.exterior.coords)
        verts.extend(ext.tolist())
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ext) - 2) + [Path.CLOSEPOLY])
    path = Path(np.asarray(verts), np.asarray(codes))
    return PathPatch(path, transform=ax.transData)

def build_boundary_mask(lat_vals, lon_vals, boundary_gdf):
    """True where grid cell center lies inside province polygons."""
    if boundary_gdf is None:
        return None
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    mask = np.zeros(pts.shape[0], dtype=bool)
    for geom in boundary_gdf.geometry:
        if geom is None:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            ext = np.asarray(poly.exterior.coords)
            inside = Path(ext).contains_points(pts)
            for hole in poly.interiors:
                hcoords = np.asarray(hole.coords)
                inside &= ~Path(hcoords).contains_points(pts)
            mask |= inside
    return mask.reshape(lat2d.shape)

def fill_display_gaps(da):
    """Nearest-neighbor fill of NaNs inside DISPLAY_* (visualization only)."""
    if da is None or "lat" not in da.dims or "lon" not in da.dims:
        return da
    out = da.copy()
    lon_mask = (out.lon >= DISPLAY_LON_MIN) & (out.lon <= DISPLAY_LON_MAX)
    lat_mask = (out.lat >= DISPLAY_LAT_MIN) & (out.lat <= DISPLAY_LAT_MAX)
    sub = out.sel(lat=out.lat[lat_mask], lon=out.lon[lon_mask])
    if sub.size == 0:
        return out
    sub_filled = sub.interpolate_na(dim="lon", method="nearest", fill_value="extrapolate")
    sub_filled = sub_filled.interpolate_na(dim="lat", method="nearest", fill_value="extrapolate")
    out.loc[dict(lat=sub_filled.lat, lon=sub_filled.lon)] = sub_filled
    return out

def load_data():
    data = np.load(config.DATA_NPZ_PATH)
    rf = joblib.load(config.RF_MODEL_PATH)
    lr = joblib.load(config.LR_MODEL_PATH)
    dem_da = xr.open_dataarray(config.DEM_REF_NC)
    obs_da = xr.open_dataarray(config.OBS_NC)
    imerg_da = xr.open_dataarray(config.IMERG_NC)
    u10_da = xr.open_dataarray(config.U10_NC)
    v10_da = xr.open_dataarray(config.V10_NC)
    tcwv_da = xr.open_dataarray(config.TCWV_NC)
    time_ds = xr.open_dataset(config.TIME_NC)
    return data, rf, lr, dem_da, obs_da, imerg_da, u10_da, v10_da, tcwv_da, time_ds

def save_fig(fig, name):
    png_path = os.path.join(config.OUTPUTS_DIR, f"{name}.png")
    tiff_path = os.path.join(config.OUTPUTS_DIR, f"{name}.tiff")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(tiff_path)

def add_spatial_corr_text(ax, cn05_data, rf_data):
    cn = np.asarray(cn05_data).flatten()
    rf = np.asarray(rf_data).flatten()
    valid = np.isfinite(cn) & np.isfinite(rf)
    if valid.sum() >= 2:
        r_spatial, _ = pearsonr(cn[valid], rf[valid])
        txt = f"RF vs CN05.1\nr = {r_spatial:.3f}"
    else:
        txt = "RF vs CN05.1\nr = NaN"
    ax.text(
        0.95, 0.05, txt, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

def plot_all():
    apply_nature_style()
    hunan_boundary = load_hunan_boundary()
    simplify_tol = float(getattr(config, "BOUNDARY_SIMPLIFY_TOL", 0.0))
    if simplify_tol > 0:
        hunan_boundary = simplify_boundary_for_plot(hunan_boundary, tolerance=simplify_tol)
    data, rf, lr, dem, obs_da, imerg_da, u10_da, v10_da, tcwv_da, time_ds = load_data()
    X_test = data['X_test']
    y_test = data['y_test']
    raw_pred = data['raw_pred']
    rf_pred = data['rf_pred']
    lr_pred = data['lr_pred']

    boundary_mask_2d = build_boundary_mask(dem.lat.values, dem.lon.values, hunan_boundary)
    if boundary_mask_2d is None:
        raise ValueError(
            "Could not build boundary mask from GeoJSON. "
            "Provide assets/hunan.geojson and a working geopandas install "
            "(DEM-only mask fallback is disabled for consistency with training)."
        )
    test_time = time_ds.sel(time=time_ds.time.dt.year.isin(config.TEST_YEARS) &
                                  time_ds.time.dt.month.isin(config.SUMMER_MONTHS)).time
    if test_time.size == 0:
        raise ValueError(
            "Empty test period: check TEST_YEARS, SUMMER_MONTHS, and models/time_index.nc."
        )
    obs_mean = obs_da.sel(time=test_time).mean("time")
    imerg_mean = imerg_da.sel(time=test_time).mean("time")

    maps = []
    for t in test_time.values:
        im = imerg_da.sel(time=t).transpose("lat","lon").values
        u = u10_da.sel(time=t).transpose("lat","lon").values
        v = v10_da.sel(time=t).transpose("lat","lon").values
        q = tcwv_da.sel(time=t).transpose("lat","lon").values
        d2 = dem.transpose("lat","lon").values
        mask = (np.isfinite(im) & np.isfinite(u) & np.isfinite(v) &
                np.isfinite(q) & np.isfinite(d2) & boundary_mask_2d)
        grid = np.full(im.shape, np.nan)
        if mask.sum():
            grid[mask] = rf.predict(np.stack([im[mask], u[mask], v[mask], q[mask], d2[mask]], axis=1))
        maps.append(grid)
    if len(maps) == 0:
        raise ValueError("Empty RF spatial stack: no valid grid cells in the test period.")
    rf_ts = xr.DataArray(
        np.stack(maps, axis=0),
        coords={"time": test_time.values, "lat": obs_mean.lat.values, "lon": obs_mean.lon.values},
        dims=("time", "lat", "lon"),
    )
    rf_mean = rf_ts.mean("time")
    obs_mean_plot = obs_mean
    imerg_mean_plot = imerg_mean
    rf_mean_plot = rf_mean
    map_years = [int(y) for y in sorted(config.TEST_YEARS)]
    year_fields = {}
    for yr in map_years:
        tmask = test_time.dt.year == yr
        if int(tmask.sum()) == 0:
            continue
        year_fields[yr] = {
            "obs": obs_da.sel(time=test_time.sel(time=tmask)).mean("time"),
            "imerg": imerg_da.sel(time=test_time.sel(time=tmask)).mean("time"),
            "rf": fill_display_gaps(rf_ts.sel(time=rf_ts.time.dt.year == yr).mean("time")),
        }

    def metrics(yt, yp):
        return (r2_score(yt,yp), np.sqrt(mean_squared_error(yt,yp)),
                mean_absolute_error(yt,yp), np.mean(yp-yt))
    im_met = metrics(y_test, raw_pred)
    rf_met = metrics(y_test, rf_pred)

    fig, axes = plt.subplots(1,2,figsize=(11,5), constrained_layout=True)
    axes[0].scatter(y_test, raw_pred, s=5, alpha=0.30, color=COLORS["imerg"], edgecolors="none")
    lim = [min(y_test.min(), raw_pred.min()), max(y_test.max(), raw_pred.max())]
    axes[0].plot(lim, lim, ls="--", lw=1.1, color=COLORS["identity"])
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("CN05.1"); axes[0].set_ylabel("IMERG"); axes[0].set_title("Raw IMERG")
    add_panel_label(axes[0], "(a)")
    txt = f"R²={im_met[0]:.3f}\nRMSE={im_met[1]:.2f}\nMAE={im_met[2]:.2f}\nBias={im_met[3]:.2f}\nn={len(y_test)}"
    axes[0].text(0.05,0.95,txt,transform=axes[0].transAxes,fontsize=9,va='top',
                 bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
    axes[1].scatter(y_test, rf_pred, s=5, alpha=0.30, color=COLORS["rf"], edgecolors="none")
    axes[1].plot(lim, lim, ls="--", lw=1.1, color=COLORS["identity"])
    axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel("CN05.1"); axes[1].set_ylabel("RF"); axes[1].set_title("Random Forest")
    add_panel_label(axes[1], "(b)")
    txt = f"R²={rf_met[0]:.3f}\nRMSE={rf_met[1]:.2f}\nMAE={rf_met[2]:.2f}\nBias={rf_met[3]:.2f}\nn={len(y_test)}"
    axes[1].text(0.05,0.95,txt,transform=axes[1].transAxes,fontsize=9,va='top',
                 bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
    for ax in axes:
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    save_fig(fig, "Scatter_IMERG_RF")

    if not year_fields:
        raise ValueError("No per-year spatial fields for plotting: check TEST_YEARS and data.")
    vmin = min(
        np.nanmin(year_fields[yr]["obs"].values) for yr in year_fields
    )
    vmax = max(
        np.nanmax(year_fields[yr]["rf"].values) for yr in year_fields
    )
    cmap = plt.cm.get_cmap("YlGnBu").copy(); cmap.set_bad(color="#E8E8E8",alpha=1.0)
    fig, axes = plt.subplots(len(year_fields), 2, figsize=(14, 5.8 * len(year_fields)), constrained_layout=True)
    if len(year_fields) == 1:
        axes = np.array([axes])
    panel_idx = 0
    for r, yr in enumerate(sorted(year_fields.keys())):
        obs_year = year_fields[yr]["obs"].values
        rf_year = year_fields[yr]["rf"].values
        for c, (key, tt) in enumerate([("obs", f"Observed CN05.1 ({yr})"), ("rf", f"RF-Corrected ({yr})")]):
            ax = axes[r, c]
            d = year_fields[yr][key]
            extent = get_cell_edge_extent(d)
            im = ax.imshow(d.values, origin="lower", cmap=cmap, extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
            clip_patch = build_clip_patch(ax, hunan_boundary)
            if clip_patch is not None:
                im.set_clip_path(clip_patch)
            ax.set_title(tt)
            add_panel_label(ax, f"({chr(97+panel_idx)})")
            panel_idx += 1
            format_map_axis(ax)
            overlay_boundary(ax, hunan_boundary)
            add_spatial_corr_text(ax, obs_year, rf_year)
    cb = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.028)
    cb.set_label("Precipitation (mm d$^{-1}$)")
    save_fig(fig, "Spatial_Observed_RF")

    classes = [(0.1,10,"Light"),(10,25,"Moderate"),(25,50,"Heavy"),(50,1000,"Torrential")]
    rows = []
    for low,high,name in classes:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum():
            rows.append([name, mask.sum(),
                         np.sqrt(mean_squared_error(y_test[mask], raw_pred[mask])),
                         np.sqrt(mean_squared_error(y_test[mask], lr_pred[mask])),
                         np.sqrt(mean_squared_error(y_test[mask], rf_pred[mask]))])
        else:
            rows.append([name,0,np.nan,np.nan,np.nan])
    rdf = pd.DataFrame(rows, columns=["Class","N","IMERG_RMSE","LR_RMSE","RF_RMSE"])
    fig, ax = plt.subplots(figsize=(6.8,4.8), constrained_layout=True)
    w = 0.25; x = np.arange(len(rdf))
    labels = [f"{r['Class']}\n(n={int(r['N'])})" for _,r in rdf.iterrows()]
    ax.bar(x-w, rdf["IMERG_RMSE"], w, label="IMERG", color=COLORS["imerg"])
    ax.bar(x, rdf["LR_RMSE"], w, label="LR", color=COLORS["lr"])
    ax.bar(x+w, rdf["RF_RMSE"], w, label="RF", color=COLORS["rf"])
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("RMSE (mm/d)")
    ax.legend(loc="upper left"); ax.set_title("RMSE by Rainfall Intensity")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "RainClass_RMSE")

    tt = test_time
    dobs, di = [], []
    for t in tt.values:
        ot = obs_da.sel(time=t).values; it = imerg_da.sel(time=t).values
        ot[~boundary_mask_2d]=np.nan; it[~boundary_mask_2d]=np.nan
        dobs.append(np.nanmean(ot)); di.append(np.nanmean(it))
    dobs=np.array(dobs); di=np.array(di)

    lr_maps=[]
    for t in tt.values:
        im = imerg_da.sel(time=t).transpose("lat","lon").values
        u = u10_da.sel(time=t).transpose("lat","lon").values
        v = v10_da.sel(time=t).transpose("lat","lon").values
        q = tcwv_da.sel(time=t).transpose("lat","lon").values
        d2 = dem.transpose("lat","lon").values
        mask = (np.isfinite(im)&np.isfinite(u)&np.isfinite(v)&np.isfinite(q)&np.isfinite(d2)&boundary_mask_2d)
        grid = np.full(im.shape, np.nan)
        if mask.sum():
            grid[mask] = lr.predict(np.stack([im[mask],u[mask],v[mask],q[mask],d2[mask]],axis=1))
        lr_maps.append(grid)
    dlr = np.array([np.nanmean(g) for g in lr_maps]); dlr = np.maximum(dlr,0)
    drf = np.array([np.nanmean(g) for g in maps])
    years = pd.to_datetime(tt.values).year
    year_panels = [2021, 2022]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True, constrained_layout=True)
    for i, (ax, yr) in enumerate(zip(axes, year_panels)):
        m = years == yr
        ax.plot(tt.values[m], dobs[m], label="CN05.1", lw=1.8, color=COLORS["obs"])
        ax.plot(tt.values[m], di[m], label="IMERG", lw=1.2, alpha=0.8, color=COLORS["imerg"])
        ax.plot(tt.values[m], drf[m], label="RF", lw=1.8, color=COLORS["rf"])
        ax.plot(tt.values[m], dlr[m], label="LR", lw=1.2, alpha=0.85, color=COLORS["lr"], ls="--")
        ax.set_ylabel("mm/d")
        ax.set_title(f"Daily Time Series ({yr} Summer)")
        add_panel_label(ax, f"({chr(97+i)})")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Date")
    save_fig(fig, "TimeSeries")

    feat_names = np.array(["IMERG", "u10", "v10", "tcwv", "DEM"])
    with parallel_backend("threading"):
        perm = permutation_importance(
            rf, X_test, y_test, scoring="r2",
            n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1
        )
    order = np.argsort(perm.importances_mean)[::-1]
    imp_mean = perm.importances_mean[order]
    imp_std = perm.importances_std[order]
    imp_names = feat_names[order]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    y_pos = np.arange(len(imp_names))
    bar_colors = [COLORS["rf"] if i == 0 else "#8FBF88" for i in range(len(imp_names))]
    ax.barh(
        y_pos, imp_mean, xerr=imp_std, color=bar_colors, edgecolor="none",
        error_kw={"ecolor": COLORS["identity"], "elinewidth": 0.9, "capsize": 2}
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_names)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (ΔR²)")
    ax.set_title("Feature Importance (Permutation, RF)")
    ax.grid(True, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, "(a)")
    save_fig(fig, "Permutation_Importance_RF")

    feat_names = ["IMERG", "u10", "v10", "tcwv", "DEM"]
    n_shap_cfg = int(getattr(config, "SHAP_MAX_SAMPLES", 300))
    n_shap = min(max(n_shap_cfg, 50), X_test.shape[0])
    shap_rng = np.random.default_rng(config.RANDOM_STATE)
    sample_idx = shap_rng.choice(X_test.shape[0], size=n_shap, replace=False)
    X_shap = X_test[sample_idx]
    X_shap_df = pd.DataFrame(X_shap, columns=feat_names)
    explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_shap_df, check_additivity=False, approximate=True)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    fig = plt.figure(figsize=(8.4, 5.6))
    shap.summary_plot(
        shap_values, X_shap_df, feature_names=feat_names, show=False, plot_size=None
    )
    ax = plt.gca()
    ax.set_title("SHAP Summary (RF)")
    ax.set_xlabel("SHAP value (impact on model output)")
    save_fig(fig, "SHAP_Summary_RF")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    shap_tcwv = shap_values[:, feat_names.index("tcwv")]
    tcwv_vals = X_shap_df["tcwv"].values
    ax1.scatter(tcwv_vals, shap_tcwv, s=6, alpha=0.55, c="#5B8FB9", edgecolors="none")
    ax1.axhline(y=0, color="#6F6F6F", lw=0.8, linestyle="--")
    ax1.set_xlabel("tcwv")
    ax1.set_ylabel("SHAP value for tcwv")
    ax1.set_title("SHAP Dependence: tcwv")
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    add_panel_label(ax1, "(a)")

    dem_vals = X_shap_df["DEM"].values
    sc = ax2.scatter(
        tcwv_vals, shap_tcwv, s=8, alpha=0.7, c=dem_vals,
        cmap=plt.cm.get_cmap("PRGn"), edgecolors="none"
    )
    ax2.axhline(y=0, color="#6F6F6F", lw=0.8, linestyle="--")
    ax2.set_xlabel("tcwv")
    ax2.set_ylabel("SHAP value for tcwv")
    ax2.set_title("SHAP Dependence: tcwv × DEM")
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    add_panel_label(ax2, "(b)")

    cb = fig.colorbar(sc, ax=ax2, pad=0.02)
    cb.set_label("DEM (m)")

    save_fig(fig, "SHAP_Dependence_tcwv_RF")

    direction_rows = []
    for idx, name in enumerate(feat_names):
        rho, p_value = spearmanr(X_shap_df[name].values, shap_values[:, idx], nan_policy="omit")
        direction_rows.append({
            "feature": name,
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            "direction": "positive" if np.isfinite(rho) and rho > 0 else ("negative" if np.isfinite(rho) and rho < 0 else "neutral"),
            "n_samples": int(np.sum(np.isfinite(X_shap_df[name].values) & np.isfinite(shap_values[:, idx]))),
        })
    direction_df = pd.DataFrame(direction_rows).sort_values(by="spearman_rho", ascending=False)
    direction_csv = os.path.join(config.OUTPUTS_DIR, "SHAP_Directionality_Table.csv")
    direction_df.to_csv(direction_csv, index=False)
    print(direction_csv)

    bias_fields = {}
    for yr in year_fields:
        bias_fields[yr] = {
            "ib": year_fields[yr]["imerg"] - year_fields[yr]["obs"],
            "rb": year_fields[yr]["rf"] - year_fields[yr]["obs"],
        }
    cmap = plt.cm.get_cmap("PRGn").copy(); cmap.set_bad(color="#E8E8E8",alpha=1.0)
    fig, axes = plt.subplots(len(bias_fields), 2, figsize=(14, 5.8 * len(bias_fields)), constrained_layout=True)
    if len(bias_fields) == 1:
        axes = np.array([axes])
    vmax_abs = max(
        max(np.nanmax(np.abs(bias_fields[yr]["ib"].values)), np.nanmax(np.abs(bias_fields[yr]["rb"].values)))
        for yr in bias_fields
    )
    panel_idx = 0
    for r, yr in enumerate(sorted(bias_fields.keys())):
        for c, (key, tt) in enumerate([("ib", f"IMERG Bias ({yr})"), ("rb", f"RF Bias ({yr})")]):
            ax = axes[r, c]
            d = bias_fields[yr][key]
            extent = get_cell_edge_extent(d)
            im = ax.imshow(d.values, origin="lower", cmap=cmap, extent=extent, aspect="auto", vmin=-vmax_abs, vmax=vmax_abs)
            clip_patch = build_clip_patch(ax, hunan_boundary)
            if clip_patch is not None:
                im.set_clip_path(clip_patch)
            ax.set_title(tt)
            add_panel_label(ax, f"({chr(97+panel_idx)})")
            panel_idx += 1
            format_map_axis(ax)
            overlay_boundary(ax, hunan_boundary)
    cb = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.028)
    cb.set_label("Bias (mm d$^{-1}$)")
    save_fig(fig, "Bias_Comparison")

    fig, axes = plt.subplots(1, len(year_fields), figsize=(7.5 * len(year_fields), 7), constrained_layout=True)
    if len(year_fields) == 1:
        axes = [axes]
    cmap = plt.cm.get_cmap("YlGnBu").copy(); cmap.set_bad(color="#E8E8E8",alpha=1.0)
    panel_idx = 0
    for ax, yr in zip(axes, sorted(year_fields.keys())):
        d = year_fields[yr]["imerg"]
        extent = get_cell_edge_extent(d)
        im = ax.imshow(d.values, origin="lower", cmap=cmap, extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
        clip_patch = build_clip_patch(ax, hunan_boundary)
        if clip_patch is not None:
            im.set_clip_path(clip_patch)
        ax.set_title(f"Raw IMERG Mean ({yr} Summer)")
        add_panel_label(ax, f"({chr(97+panel_idx)})")
        panel_idx += 1
        format_map_axis(ax)
        overlay_boundary(ax, hunan_boundary)
    cb = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.045)
    cb.set_label("Precipitation (mm d$^{-1}$)")
    save_fig(fig, "Supp_IMERG_Spatial")

if __name__ == "__main__":
    plot_all()
