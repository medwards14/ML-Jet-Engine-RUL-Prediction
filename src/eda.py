import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# utility functions for exploratory data analysis
def plot_sensor_distribution(df: pd.DataFrame, sensor_col: str):
    """
    Plot the distribution of a single sensor column.
    """
    plt.figure()
    sns.histplot(df[sensor_col], kde=True)
    plt.title(f'Distribution of {sensor_col}')
    plt.xlabel(sensor_col)
    plt.ylabel("Count")
    plt.show()

def plot_time_series(df: pd.DataFrame, engine_id: int, sensor_col: str):
    """
    Plot how a single sensor's values change over time for a single engine.
    """
    subset = df[df['engine_id'] == engine_id].sort_values('time_cycle')
    plt.figure()
    plt.plot(subset['time_cycle'], subset[sensor_col], marker='o')
    plt.title(f'{sensor_col} over Time (Engine {engine_id})')
    plt.xlabel("Time Cycle")
    plt.ylabel(sensor_col)
    plt.show()
