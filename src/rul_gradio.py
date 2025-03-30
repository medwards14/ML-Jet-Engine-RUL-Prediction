import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error
from src.feature_engineering import add_rolling_features

MODEL_PATH = "rul_lstm_model.h5"
SCALER_PATH = "scaler.pkl"
COLS_FILE = "sensor_columns.txt"
SEQ_LENGTH = 30

model = None
scaler = None
final_cols = None

def load_artifacts():
    global model, scaler, final_cols
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
    if final_cols is None:
        with open(COLS_FILE, "r") as f:
            final_cols = f.read().splitlines()

def preprocess_nasa_file(csv_path: str) -> pd.DataFrame:
    col_names = ['engine_id', 'time_cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                [f'sensor_{i}' for i in range(1, 22)]
    df_raw = pd.read_csv(csv_path, sep='\s+', header=None, names=col_names)
    load_artifacts()
    base_sensors = [c for c in final_cols if c.startswith("sensor_") and not c.endswith("_rollmean3")]
    df_raw[base_sensors] = scaler.transform(df_raw[base_sensors].astype(float))
    df_processed = add_rolling_features(df_raw, base_sensors, window=3)
    return df_processed.sort_values(["engine_id", "time_cycle"])

def largest_sensor_change_info(df_30: pd.DataFrame) -> str:
    base_sensors = [c for c in final_cols if c.startswith("sensor_") and not c.endswith("_rollmean3")]
    first_row, last_row = df_30.iloc[0], df_30.iloc[-1]
    diffs = {s: abs(float(last_row[s]) - float(first_row[s])) for s in base_sensors}
    sensor_name = max(diffs, key=diffs.get)
    v0, v1 = float(first_row[sensor_name]), float(last_row[sensor_name])
    return f"Sensor with biggest change: {sensor_name} changed from {v0:.2f} to {v1:.2f} (Δ={v1 - v0:.2f})"

def get_actual_rul(engine_id: int, rul_path: str) -> float:
    rul_df = pd.read_csv(rul_path, header=None)
    return float(rul_df.iloc[engine_id - 1][0])

def compute_overall_rmse(df: pd.DataFrame, rul_path: str):
    full_true_rul = pd.read_csv(rul_path, header=None).values.flatten()
    predicted = []
    used_rul = []  # Actual RULs for engines that were used

    for i, engine_id in enumerate(df["engine_id"].unique()):
        engine_df = df[df["engine_id"] == engine_id]
        if len(engine_df) < SEQ_LENGTH:
            continue
        last_30 = engine_df.tail(SEQ_LENGTH)[final_cols].values.astype(np.float32)
        pred = float(model.predict(np.expand_dims(last_30, axis=0))[0][0])
        predicted.append(pred)
        used_rul.append(full_true_rul[i])  # Only include matching actual RUL

    if len(predicted) != len(used_rul):
        return "Error - Prediction/Truth mismatch", None

    rmse = np.sqrt(mean_squared_error(used_rul, predicted))
    return f"Overall RMSE: {rmse:.2f}", rmse

def run_inference(df: pd.DataFrame, engine_id: int, rul_path: str):
    engine_df = df[df["engine_id"] == engine_id]
    if len(engine_df) < SEQ_LENGTH:
        raise ValueError(f"Engine {engine_id} has only {len(engine_df)} rows, need ≥ {SEQ_LENGTH}.")
    df_30 = engine_df.tail(SEQ_LENGTH)
    input_seq = np.expand_dims(df_30[final_cols].values.astype(np.float32), axis=0)
    predicted = float(model.predict(input_seq)[0][0])
    actual = get_actual_rul(engine_id, rul_path)
    sensor_info = largest_sensor_change_info(df_30)
    return predicted, actual, sensor_info

def plot_rul_vs_actual(predicted: float, actual: float):
    fig, ax = plt.subplots()
    ax.bar(["Predicted RUL", "Actual RUL"], [max(0, predicted), max(0, actual)], color=["blue", "green"])
    ax.set_ylim([0, 160])
    ax.set_ylabel("Cycles")
    ax.set_title("Predicted vs Actual RUL")
    return fig

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# NASA C-MAPSS RUL Predictor")
        gr.Markdown("""
Upload a **test_FD00X.txt** and its matching **RUL_FD00X.txt** file from NASA's CMAPSS Jet Engine Simulated Data set.  
Then enter an engine ID from that test set (usually 1–100 for FD001).  
This GUI will use my trained LSTM to:
- Predict RUL for the selected engine
- Compare it with actual RUL
- Show the most degraded sensor
- Plot predicted vs actual RUL
- Display RMSE across the entire test file
""")
        test_file_input = gr.File(label="Upload test_FD00X.txt")
        rul_file_input = gr.File(label="Upload RUL_FD00X.txt")
        engine_id_input = gr.Number(label="Engine ID", precision=0)
        predict_button = gr.Button("Predict")

        output_text = gr.Textbox(label="Prediction Summary", lines=8)
        output_plot = gr.Plot(label="Predicted vs Actual RUL")

        def on_click(test_file, rul_file, engine_id):
            try:
                engine_id = int(engine_id)
                if engine_id < 1:
                    return "Invalid engine ID", None
                df = preprocess_nasa_file(test_file.name)
                predicted, actual, info = run_inference(df, engine_id, rul_file.name)
                rmse_msg, _ = compute_overall_rmse(df, rul_file.name)
                result = (
                    f"Engine {engine_id}\n"
                    f"Predicted RUL: {predicted:.2f} cycles\n"
                    f"Actual RUL:    {actual:.2f} cycles\n"
                    f"{info}\n\n"
                    f"{rmse_msg}"
                )
                return result, plot_rul_vs_actual(predicted, actual)
            except Exception as e:
                return f"Error: {e}", None

        predict_button.click(
            fn=on_click,
            inputs=[test_file_input, rul_file_input, engine_id_input],
            outputs=[output_text, output_plot]
        )

    demo.launch()

if __name__ == "__main__":
    main()
