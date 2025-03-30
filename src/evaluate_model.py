import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from math import sqrt
from sklearn.metrics import mean_squared_error
from pathlib import Path
import sys

from src.feature_engineering import add_rolling_features
from src.model import build_lstm_model

MODEL_PATH = "rul_lstm_model.h5"
SCALER_PATH = "scaler.pkl"
COLS_FILE = "sensor_columns.txt"
SEQ_LENGTH = 30

def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(COLS_FILE, "r") as f:
        final_cols = f.read().splitlines()
    return model, scaler, final_cols

def preprocess_test_data(df, scaler, base_cols):
    df = df.copy()
    df[base_cols] = scaler.transform(df[base_cols])
    df = add_rolling_features(df, base_cols, window=3)
    return df.sort_values(['engine_id', 'time_cycle'])

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.evaluate_model data/test_FD00X.txt data/RUL_FD00X.txt")
        return

    test_path, rul_path = sys.argv[1], sys.argv[2]
    col_names = ['engine_id', 'time_cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                [f'sensor_{i}' for i in range(1, 22)]

    df_test = pd.read_csv(test_path, delim_whitespace=True, header=None, names=col_names)
    true_rul = pd.read_csv(rul_path, header=None).values.flatten()

    model, scaler, final_cols = load_artifacts()
    base_cols = [c for c in final_cols if c.startswith("sensor_") and not c.endswith("rollmean3")]
    df_test = preprocess_test_data(df_test, scaler, base_cols)

    predicted = []
    for engine_id in df_test["engine_id"].unique():
        sub_df = df_test[df_test["engine_id"] == engine_id]
        if len(sub_df) < SEQ_LENGTH:
            continue
        last_30 = sub_df.tail(SEQ_LENGTH)[final_cols].values.astype(np.float32)
        input_seq = np.expand_dims(last_30, axis=0)
        pred_rul = float(model.predict(input_seq)[0][0])
        predicted.append(pred_rul)

    if len(predicted) != len(true_rul):
        print("❌ Mismatch between predictions and RUL ground truth!")
    else:
        rmse = sqrt(mean_squared_error(true_rul, predicted))
        print(f"✅ RMSE on {len(true_rul)} test engines: {rmse:.2f}")

if __name__ == "__main__":
    main()
