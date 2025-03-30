import numpy as np
import pandas as pd

# function to transform sensor data DataFrame into sequences for neural network.
def create_sequences(
    df: pd.DataFrame,
    sensor_cols: list,
    seq_length: int = 30,
    is_train: bool = True
):
    """
    Create 3D sequences (X) for NN input.
      - If is_train=True, also create y (RUL at last step in each sequence).
      - If is_train=False, returns only X (no labels).
    The DataFrame must have columns: ['engine_id', 'time_cycle'] + sensor_cols.
    If is_train=True, we expect an 'RUL' column as well.
    """
    X, y = [], []

    for eng_id in df['engine_id'].unique():
        eng_subset = df[df['engine_id'] == eng_id].sort_values('time_cycle')
        sensor_values = eng_subset[sensor_cols].values

        if is_train:
            rul_values = eng_subset['RUL'].values

        # Sliding window across each engine's timeline
        for i in range(len(sensor_values) - seq_length + 1):
            seq_x = sensor_values[i : i+seq_length]
            X.append(seq_x)

            if is_train:
                # Use RUL at the last step of the sequence as the label
                seq_y = rul_values[i + seq_length - 1]
                y.append(seq_y)

    X = np.array(X)
    if is_train:
        y = np.array(y)
        return X, y
    else:
        return X
