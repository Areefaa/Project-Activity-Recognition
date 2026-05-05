import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os

# ========================
# CONFIG
# ========================
INPUT_FILE = "data/raw/master_dataset.csv"
OUTPUT_FILE = "data/processed/windowed_data.csv"

FS = 50
WINDOW_SIZE = 100   # 2 detik
STEP_SIZE = 50      # 50% overlap

# ========================
# FILTER
# ========================
def butter_lowpass_filter(data, cutoff=10, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

# ========================
# LOAD DATA
# ========================
def load_data():
    df = pd.read_csv(INPUT_FILE)

    print("Kolom asli:", df.columns)

    # rename kolom biar konsisten
    df.columns = [
        "time",
        "acc_x", "acc_y", "acc_z", "acc_abs",
        "gyro_x", "gyro_y", "gyro_z", "gyro_abs",
        "label", "subject_id"
    ]

    return df

# ========================
# PREPROCESS
# ========================
def preprocess(df):
    axes = [
        "gyro_x", "gyro_y", "gyro_z",
        "acc_x", "acc_y", "acc_z"
    ]

    # FILTER
    for col in axes:
        df[col] = butter_lowpass_filter(df[col])

    # MAGNITUDE (ganti yang NaN)
    df["gyro_mag"] = np.sqrt(df["gyro_x"]**2 +
                             df["gyro_y"]**2 +
                             df["gyro_z"]**2)

    df["acc_mag"] = np.sqrt(df["acc_x"]**2 +
                            df["acc_y"]**2 +
                            df["acc_z"]**2)

    return df

# ========================
# WINDOWING
# ========================
def create_windows(df):
    windows = []

    for i in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        window = df.iloc[i:i+WINDOW_SIZE]

        # ambil hanya window dengan label konsisten
        if window["label"].nunique() == 1:
            w = window.copy()
            w["window_id"] = i
            windows.append(w)

    return pd.concat(windows, ignore_index=True)

# ========================
# MAIN
# ========================
def main():
    df = load_data()
    print("Data loaded:", df.shape)

    df = preprocess(df)
    print("Filtering selesai")

    df_windowed = create_windows(df)
    print("Windowing selesai:", df_windowed.shape)

    os.makedirs("data/processed", exist_ok=True)
    df_windowed.to_csv(OUTPUT_FILE, index=False)

    print("Saved ke:", OUTPUT_FILE)

if __name__ == "__main__":
    main()