import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle_df = df.groupby('engine_id')['time_cycle'].max().reset_index()
    max_cycle_df.rename(columns={'time_cycle': 'max_cycle'}, inplace=True)
    merged = df.merge(max_cycle_df, on='engine_id', how='left')
    merged['RUL'] = merged['max_cycle'] - merged['time_cycle']
    merged.drop(columns='max_cycle', inplace=True)
    return merged

def drop_constant_sensors(df: pd.DataFrame, sensor_cols: list) -> list:
    return [col for col in sensor_cols if df[col].std() > 1e-5]

def scale_sensors(train_df: pd.DataFrame, test_df: pd.DataFrame, sensor_cols: list):
    scaler = MinMaxScaler()
    train_vals = train_df[sensor_cols].values
    scaler.fit(train_vals)
    train_df[sensor_cols] = scaler.transform(train_vals)

    if test_df is not None:
        test_vals = test_df[sensor_cols].values
        test_df[sensor_cols] = scaler.transform(test_vals)
        return scaler, train_df, test_df
    else:
        return scaler, train_df, None

def add_rolling_features(df: pd.DataFrame, sensor_cols: list, window=3) -> pd.DataFrame:
    df = df.copy()
    for col in sensor_cols:
        rolled = df.groupby('engine_id')[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_rollmean{window}'] = rolled.reset_index(level=0, drop=True)
    return df
