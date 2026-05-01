import os

# Project root (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory containing input NetCDF files; override with environment variable DATA_ROOT
DATA_ROOT = os.getenv("DATA_ROOT", "/mnt/data/imerg_correction_hunan")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

IMERG_DIR = os.path.join(DATA_ROOT, "imerg.nc")
OBS_FILE  = os.path.join(DATA_ROOT, "cn051.nc")
ERA5_FILE = os.path.join(DATA_ROOT, "era5.nc")
DEM_FILE  = os.path.join(DATA_ROOT, "dem_hunan_025.nc")

# Training / reproducibility
RANDOM_STATE = 42
MAX_SAMPLES_PER_TIME = 2000
N_ESTIMATORS = 200

# Train / test years and summer season (months)
TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020]
TEST_YEARS  = [2021, 2022]
SUMMER_MONTHS = [6, 7, 8]

# Variable name candidates when opening heterogeneous NetCDF sources
IMERG_VAR_CANDIDATES = ["precipitationCal", "precipitation", "precip", "IMERG"]
OBS_VAR_CANDIDATES = ["pre", "precip", "precipitation", "CN05.1"]
U10_VAR_CANDIDATES = ["u10"]
V10_VAR_CANDIDATES = ["v10"]
TCWV_VAR_CANDIDATES = ["tcwv"]
DEM_VAR_CANDIDATES = ["dem", "elevation", "DEM", "height", "__xarray_dataarray_variable__"]

# Saved models and gridded intermediates for plotting
RF_MODEL_PATH  = os.path.join(MODELS_DIR, "rf_model.joblib")
LR_MODEL_PATH  = os.path.join(MODELS_DIR, "lr_model.joblib")
DATA_NPZ_PATH  = os.path.join(MODELS_DIR, "train_test_data.npz")
DEM_REF_NC     = os.path.join(MODELS_DIR, "dem_reference.nc")
TIME_NC        = os.path.join(MODELS_DIR, "time_index.nc")
OBS_NC         = os.path.join(MODELS_DIR, "obs_for_plot.nc")
IMERG_NC       = os.path.join(MODELS_DIR, "imerg_for_plot.nc")
U10_NC         = os.path.join(MODELS_DIR, "u10_for_plot.nc")
V10_NC         = os.path.join(MODELS_DIR, "v10_for_plot.nc")
TCWV_NC        = os.path.join(MODELS_DIR, "tcwv_for_plot.nc")

# Target grid over Hunan (EPSG:4326), ~0.25 deg
TARGET_LON_MIN = 108.65
TARGET_LON_MAX = 114.35
TARGET_LAT_MIN = 24.50
TARGET_LAT_MAX = 30.30
TARGET_RES = 0.25