import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from src.feature_engineering import (
    generate_rul_labels,
    drop_constant_sensors,
    scale_sensors,
    add_rolling_features
)
from src.dataset_utils import create_sequences

def build_lstm_model(seq_length, num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, num_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    data_dir = Path("data")
    seq_length = 30

    col_names = ['engine_id', 'time_cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                [f'sensor_{i}' for i in range(1, 22)]

    dfs = []
    for i in range(1, 5):
        train_file = data_dir / f"train_FD00{i}.txt"
        df = pd.read_csv(train_file, sep='\s+', header=None, names=col_names)
        df["dataset_id"] = f"FD00{i}"  # Optional tag
        dfs.append(df)
    train_df = pd.concat(dfs, ignore_index=True)

    train_df = generate_rul_labels(train_df)
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    non_const_sensors = drop_constant_sensors(train_df, sensor_cols)

    train_df = train_df[['engine_id', 'time_cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
                        + non_const_sensors + ['RUL']]

    scaler, train_df, _ = scale_sensors(train_df, None, non_const_sensors)
    train_df = add_rolling_features(train_df, non_const_sensors, window=3)

    updated_sensor_cols = non_const_sensors + [f'{col}_rollmean3' for col in non_const_sensors]
    train_df = train_df.sort_values(['engine_id', 'time_cycle'])

    X_train, y_train = create_sequences(
        train_df, sensor_cols=updated_sensor_cols, seq_length=seq_length, is_train=True
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    model = build_lstm_model(seq_length, X_train.shape[2])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("rul_lstm_model.h5", save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    model.save("rul_lstm_model.h5")
    joblib.dump(scaler, "scaler.pkl")
    with open("sensor_columns.txt", "w") as f:
        for col in updated_sensor_cols:
            f.write(col + "\n")

    print("Artifacts saved: model, scaler, sensor columns")

if __name__ == "__main__":
    main()
