# RUL Prediction with NASA C-MAPSS Dataset

This project demonstrates how to predict the **Remaining Useful Life (RUL)** of aircraft engines using NASA’s C-MAPSS dataset. It covers the end-to-end pipeline:
- Data loading and exploratory analysis
- Preprocessing and feature engineering
- Model building (LSTM neural networks in TensorFlow/Keras)
- Training and saving the trained model

By following these steps, you’ll be able to reproduce the results and adapt them for your own predictive-maintenance or prognostics projects.

--------------------------------------------------------------------------------
## Table of Contents
--------------------------------------------------------------------------------

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Acquire and Place the Data](#1-acquire-and-place-the-data)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Training the Model](#4-training-the-model)
- [Data Description](#data-description)
- [Key Files and Scripts](#key-files-and-scripts)
- [Future Improvements](#future-improvements)

--------------------------------------------------------------------------------
## Project Structure
--------------------------------------------------------------------------------

rul_prediction_project/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── ... (other FD00x if available)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing.ipynb
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── dataset_utils.py
│   └── model.py
├── requirements.txt  (optional)
└── README.md         (this file)

--------------------------------------------------------------------------------
## Installation
--------------------------------------------------------------------------------

1. **Clone** or **download** this repository to your local machine (or WSL if you’re on Windows).

2. **Create and activate** a Python virtual environment (recommended):
   cd rul_prediction_project
   python -m venv venv
   source venv/bin/activate
   # (On Windows, venv\Scripts\activate)

3. **Install dependencies**:
   pip install --upgrade pip
   pip install numpy pandas matplotlib scikit-learn tensorflow
   # or
   pip install -r requirements.txt   # if you have this file

--------------------------------------------------------------------------------
## Usage
--------------------------------------------------------------------------------

### 1. Acquire and Place the Data

- Download the NASA C-MAPSS dataset files (e.g., train_FD001.txt, test_FD001.txt, RUL_FD001.txt) and place them in the data/ folder.
- Ensure your directory looks like this:

rul_prediction_project/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── ...
├── notebooks/
├── src/
└── ...

### 2. Exploratory Data Analysis

1. Launch Jupyter Notebook:
   jupyter notebook

2. Open notebooks/01_eda.ipynb:
   - Reads the training file into a pandas DataFrame
   - Checks for missing values, prints distributions, correlation heatmaps, etc.

### 3. Data Preprocessing

1. Open notebooks/02_preprocessing.ipynb:
   - Demonstrates how to generate RUL labels, drop low-variance sensors, scale data, and create sequences for an RNN.
   - Produces arrays (X_train, y_train) ready for model training.

### 4. Training the Model

1. From a terminal, enter the src/ directory:
   cd src

2. Run the model.py script:
   python model.py
   - Loads the raw data
   - Preprocesses (RUL label creation, scaling, etc.)
   - Builds and trains an LSTM model
   - Saves the trained model to rul_lstm_model.h5

3. (Optional) Evaluate:
   - model.py can be adapted to evaluate on test_FD001.txt using the partial run data and RUL_FD001.txt file. This typically involves generating sequences for each test engine’s final cycles and comparing predicted vs. actual RUL.

--------------------------------------------------------------------------------
## Data Description
--------------------------------------------------------------------------------

The NASA C-MAPSS dataset simulates turbofan engines with various fault modes and operating conditions. Each row in the data:
- Engine ID (engine_id)
- Time cycle (time_cycle)
- Operational settings (op_setting_1, op_setting_2, op_setting_3)
- Sensor measurements (sensor_1 ... sensor_21)

**Train Data:** Each engine is run to failure.  
**Test Data:** Engines stop before failure; RUL_FD00x.txt gives the true Remaining Useful Life for each engine's last recorded cycle.

--------------------------------------------------------------------------------
## Key Files and Scripts
--------------------------------------------------------------------------------

- notebooks/01_eda.ipynb
  - Performs exploratory analysis on the training set
  - Visualizes sensor correlations and time-series trends

- notebooks/02_preprocessing.ipynb
  - Generates RUL labels
  - Drops constant/low-variance sensors
  - Scales data
  - Creates sequences for an LSTM

- src/feature_engineering.py
  - generate_rul_labels(df): Adds an RUL column to the train DataFrame
  - drop_constant_sensors(df, sensor_cols): Filters out sensors with near-zero variance
  - scale_sensors(train_df, test_df, sensor_cols): Fits scaler on train, applies to test
  - add_rolling_features(df, sensor_cols): Creates rolling averages or other stats

- src/dataset_utils.py
  - create_sequences(df, sensor_cols, seq_length, is_train): Builds 3D arrays (samples, seq_length, features) for RNN input; if is_train=True,
