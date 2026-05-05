import pandas as pd
import numpy as np
from scipy.fft import fft
import os

# ========================
# CONFIG
# ========================
INPUT_FILE = "data/processed/windowed_data.csv"
OUTPUT_FILE = "data/processed/features.csv"

# ========================
# FEATURE EXTRACTION
# ========================
def extract_features(df):

    feature_list = []

    # group berdasarkan window
    grouped = df.groupby("window_id")

    for window_id, window in grouped:

        feat = {}

        # ========================
        # LABEL
        # ========================
        feat["label"] = window["label"].iloc[0]

        # ========================
        # TIME DOMAIN FEATURES
        # ========================
        axes = [
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z"
        ]

        for col in axes:
            feat[f"{col}_mean"] = window[col].mean()
            feat[f"{col}_std"] = window[col].std()

        # ========================
        # MAGNITUDE FEATURES
        # ========================
        feat["acc_mag_mean"] = window["acc_mag"].mean()
        feat["acc_mag_std"] = window["acc_mag"].std()

        feat["gyro_mag_mean"] = window["gyro_mag"].mean()
        feat["gyro_mag_std"] = window["gyro_mag"].std()

        # ========================
        # SMA (Signal Magnitude Area)
        # ========================
        feat["SMA_acc"] = np.sum(
            np.abs(window["acc_x"]) +
            np.abs(window["acc_y"]) +
            np.abs(window["acc_z"])
        ) / len(window)

        feat["SMA_gyro"] = np.sum(
            np.abs(window["gyro_x"]) +
            np.abs(window["gyro_y"]) +
            np.abs(window["gyro_z"])
        ) / len(window)

        # ========================
        # FFT (Dominant Frequency)
        # ========================
        def dominant_freq(signal):
            fft_vals = np.abs(fft(signal))
            return np.argmax(fft_vals)

        feat["acc_dom_freq"] = dominant_freq(window["acc_mag"])
        feat["gyro_dom_freq"] = dominant_freq(window["gyro_mag"])

        feature_list.append(feat)

    return pd.DataFrame(feature_list)

# ========================
# MAIN
# ========================
def main():
    df = pd.read_csv(INPUT_FILE)

    print("Windowed data loaded:", df.shape)

    df_features = extract_features(df)

    print("Feature extraction selesai:", df_features.shape)

    os.makedirs("data/processed", exist_ok=True)
    df_features.to_csv(OUTPUT_FILE, index=False)

    print("Saved ke:", OUTPUT_FILE)

if __name__ == "__main__":
    main()