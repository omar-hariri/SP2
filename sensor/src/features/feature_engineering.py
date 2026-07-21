# src/features/feature_engineering.py
# =================================

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
# from scipy.stats import skew, kurtosis as scipy_kurtosis

# ─────────────────────────────────────────
# Feature Definitions
# ─────────────────────────────────────────

FEATURE_VECTOR = [
    # raw feats
    "accel_x", "accel_y", "accel_z",
    "roll", "pitch", "yaw",
    "speed",
    "course", "course_variation",
    # diff feats
    "accel_x_diff","accel_y_diff","accel_z_diff",
    "speed_diff",
    "acc_mag",
    "jerk_mag"
]

STAT_FEATURES = [
    # raw feats
    "accel_x", "accel_y", "accel_z",
    "roll", "pitch", "yaw",
    "speed",
    "course", "course_variation",
    # diff feats
    "accel_x_diff","accel_y_diff","accel_z_diff",
    "speed_diff",
    "acc_mag",
    "jerk_mag"
]

# ─────────────────────────────────────────
# Computing Features
# ─────────────────────────────────────────
# me: goes for stat and vector

def compute_features(df):
    """
     Computes the new columns.
    """
    df = df.copy()

    df["accel_x_diff"] = df["accel_x"].diff().fillna(0)
    df["accel_y_diff"] = df["accel_y"].diff().fillna(0)
    df["accel_z_diff"] = df["accel_z"].diff().fillna(0)

    df["speed_diff"] = df["speed"].diff().fillna(0)
    
    # acc_mag = sqrt(accel_x² + accel_y² + accel_z²)
    df["acc_mag"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)

    df["jerk_mag"] = np.sqrt(df["accel_x_diff"]**2 + df["accel_y_diff"]**2 + df["accel_z_diff"]**2)

    return df

# ─────────────────────────────────────────
# Normalization (scaler)
# ─────────────────────────────────────────

def fit_scaler(train_dfs):
    combined = pd.concat(train_dfs, ignore_index=True)

    scaler = StandardScaler()
    scaler.fit(combined[FEATURE_VECTOR])

    print(f"  Scaler fitted on {len(combined)} training rows")
    return scaler

def apply_scaler(df, scaler):
    df = df.copy()
    df[FEATURE_VECTOR] = scaler.transform(df[FEATURE_VECTOR])
    return df

def save_scaler(scaler, save_dir):
    path = Path(save_dir) / "scaler.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved → {path}")

def load_scaler(save_dir):
    path = Path(save_dir) / "scaler.pkl"
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    print(f"  Scaler loaded ← {path}")
    return scaler


# ─────────────────────────────────────────
# Splitting logic and Stat aggregation
# ─────────────────────────────────────────
# me: this is for npy data
def build_windows(df, window_length, stride):
    X_list = []
    y_list = []

    values = df[FEATURE_VECTOR].values
    labels = df["label"].values
    T = len(values)

    for start in range(0, T - window_length + 1, stride):
        end = start + window_length

        window_x = values[start:end]
        window_y = labels[start:end]
        label = int(np.bincount(window_y.astype(int)).argmax())

        X_list.append(window_x)
        y_list.append(label)

    if not X_list:
        return (
            np.empty((0, window_length, len(FEATURE_VECTOR)), dtype=np.float32),
            np.empty((0,), dtype=np.int64)
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y


def build_all_windows(session_dfs, window_length, stride):
    X_all = []
    y_all = []

    for df in session_dfs:
        X, y = build_windows(df, window_length, stride)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        return (
            np.empty((0, window_length, len(FEATURE_VECTOR)), dtype=np.float32),
            np.empty((0,), dtype=np.int64)
        )

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def compute_window_stats(X_3d):
    """
    Parameters
    ----------
    X_3d : numpy array shape (N, window_length, len(FEATURE_VECTOR))

    Returns
    -------
    X_stats : numpy array shape (N, len(STAT_FEATURES) * num of stats)
        
    """
    indices  = [FEATURE_VECTOR.index(f) for f in STAT_FEATURES]
    X_target = X_3d[:, :, indices]

    means = np.mean(X_target, axis=1)
    stds  = np.std(X_target, axis=1)
    # those are removed
    mins = np.min(X_target, axis=1)
    maxs = np.max(X_target, axis=1)
    ranges = maxs - mins

    from scipy.stats import skew, kurtosis

    skewness = skew(X_target, axis=1)
    kurt     = kurtosis(X_target, axis=1)

    X_stats = np.concatenate(
        [means, stds,mins,maxs,ranges,skewness,kurt],
        axis=1
    )

    return X_stats

