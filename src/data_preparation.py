"""
data_preparation.py — Data loading, feature engineering, and sequence preparation.

Reproduces the exact pipeline described in Section 3.1-3.2 of:
    "Explainable Dual-Attention Encoder-Decoder Model for Natural Gas
     Consumption Forecasting Using Algerian Hourly Data"
    R. Ladlani et al., ICAIABA 2026.

Dataset note:
    The raw Algerian hourly natural gas consumption data (2014) was provided
    by Dr. Oussama Laib (Laib et al., ICIF 2018) under a restricted arrangement
    and cannot be redistributed. To obtain the data, please contact the original
    authors via the citation below.

    Meteorological variables (temperature, wind speed, humidity, wet bulb
    temperature) were sourced from NASA POWER and are freely accessible at:
    https://power.larc.nasa.gov/

    Citation:
        Laib, O. et al.: A Gaussian Process Regression for Natural Gas
        Consumption Prediction Based on Time Series Data.
        In: Proc. ICIF (2018). https://doi.org/10.23919/icif.2018.8455447
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


# ── Feature column definition ─────────────────────────────────────────────────
FEATURE_COLS = [
    "Normalized Consumption",     # target (lagged values used as input)
    "temperature",
    "Hour",
    "DayOfWeek",
    "DayOfMonth",
    "Month",
    "holidaybinary",
    "HourSin",
    "DayOfWeekSin",
    "DayOfMonthSin",
    "MonthSin",
    "HolidaySin",
    "Wet_Bulb_Temperature_2_Meters",
    "Wind_Speed_at_10_Meters",
    "temperture_2_metre",
    "Relative_Humidity_at_2_Meters",
]

TARGET_COL = "Normalized Consumption"
INPUT_STEPS = 24    # 24-hour lookback window
OUTPUT_STEPS = 12   # 12-hour forecast horizon


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset CSV file.

    Args:
        csv_path: Path to the CSV file (e.g. 'data/new_data_temp.csv').

    Returns:
        Raw DataFrame sorted by date.
    """
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df = df.dropna()
    print(f"Dataset loaded: {df.shape[0]} hourly records, {df.shape[1]} columns.")
    return df


def prepare_enhanced_data(
    df: pd.DataFrame,
    input_steps: int = INPUT_STEPS,
    output_steps: int = OUTPUT_STEPS,
    target_col: str = TARGET_COL,
    feature_cols: list = None,
):
    """
    Full data preparation pipeline: sliding window, chronological split,
    z-score normalization of features, Min-Max normalization of target,
    and decoder input construction with teacher forcing.

    All scaler parameters are fitted exclusively on the training set to
    prevent data leakage.

    Args:
        df:           Preprocessed DataFrame with Date index.
        input_steps:  Encoder lookback window (default: 24 hours).
        output_steps: Decoder prediction horizon (default: 12 hours).
        target_col:   Name of the target column.
        feature_cols: List of input feature columns. Defaults to FEATURE_COLS.

    Returns:
        Tuple of:
            X_train, decoder_input_train, y_train,
            X_val,   decoder_input_val,   y_val,
            X_test,  decoder_input_test,  y_test,
            scalers (dict with 'feature_scaler' and 'target_scaler'),
            feature_cols (list)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # ── 1. Build sliding-window sequences (unscaled) ──────────────────────────
    X_raw, y_raw = [], []
    for i in range(len(df) - input_steps - output_steps + 1):
        seq_x = df[feature_cols].iloc[i : i + input_steps].values
        seq_y = df[target_col].iloc[i + input_steps : i + input_steps + output_steps].values
        X_raw.append(seq_x)
        y_raw.append(seq_y)

    X_raw = np.array(X_raw)   # (N, input_steps, n_features)
    y_raw = np.array(y_raw)   # (N, output_steps)

    # ── 2. Chronological split — no shuffling ────────────────────────────────
    # Train 70% | Val 15% | Test 15%
    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_raw, y_raw, test_size=0.30, shuffle=False
    )
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp_raw, y_temp_raw, test_size=0.50, shuffle=False
    )

    print(
        f"Split — Train: {len(X_train_raw):,} | "
        f"Val: {len(X_val_raw):,} | Test: {len(X_test_raw):,} sequences"
    )

    # ── 3. Fit scalers on training data only ──────────────────────────────────
    n_features = len(feature_cols)

    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    # Fit on training set (features = all cols except target at index 0)
    feature_scaler.fit(X_train_raw[:, :, 1:].reshape(-1, n_features - 1))
    target_scaler.fit(y_train_raw.reshape(-1, 1))

    # ── 4. Apply scaling ──────────────────────────────────────────────────────
    def scale_X(X):
        X_scaled = np.zeros_like(X, dtype=np.float32)
        # Target column (index 0) scaled with MinMaxScaler
        X_scaled[:, :, 0] = target_scaler.transform(
            X[:, :, 0].reshape(-1, 1)
        ).reshape(X.shape[0], X.shape[1])
        # Remaining features scaled with StandardScaler
        X_scaled[:, :, 1:] = feature_scaler.transform(
            X[:, :, 1:].reshape(-1, n_features - 1)
        ).reshape(X.shape[0], X.shape[1], n_features - 1)
        return X_scaled

    def scale_y(y):
        return target_scaler.transform(y.reshape(-1, 1)).reshape(y.shape).astype(np.float32)

    X_train = scale_X(X_train_raw)
    X_val   = scale_X(X_val_raw)
    X_test  = scale_X(X_test_raw)

    y_train = scale_y(y_train_raw)
    y_val   = scale_y(y_val_raw)
    y_test  = scale_y(y_test_raw)

    # ── 5. Build decoder inputs with teacher forcing ──────────────────────────
    def create_decoder_input(X_batch, y_batch):
        """
        Decoder input: starts from the last encoder step, then uses
        ground-truth target values (teacher forcing) for subsequent steps.
        """
        dec = np.zeros((len(X_batch), output_steps, n_features), dtype=np.float32)
        for i in range(len(X_batch)):
            dec[i, 0, :] = X_batch[i, -1, :]
            for j in range(1, output_steps):
                dec[i, j, :] = dec[i, j - 1, :]
                dec[i, j, 0] = y_batch[i, j - 1]   # teacher forcing on target
        return dec

    decoder_input_train = create_decoder_input(X_train, y_train)
    decoder_input_val   = create_decoder_input(X_val,   y_val)
    decoder_input_test  = create_decoder_input(X_test,  y_test)

    scalers = {
        "feature_scaler": feature_scaler,
        "target_scaler":  target_scaler,
    }

    return (
        X_train, decoder_input_train, y_train,
        X_val,   decoder_input_val,   y_val,
        X_test,  decoder_input_test,  y_test,
        scalers,
        feature_cols,
    )
