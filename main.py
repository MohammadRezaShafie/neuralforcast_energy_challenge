import time
import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.tsdataset import TimeSeriesDataset
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from utilsforecast.plotting import plot_series
from neuralforecast.models import NBEATS, NHITS, LSTM, NHITS, RNN, TFT, TimeLLM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import (
    load_and_preprocess,
    apply_fourier,
    apply_trend,
    apply_time_features,
    apply_future_exog_to_historic,
    apply_pipeline,
    get_next_plot_dir,
    get_next_plot_filename
)
from config import MODEL_CONFIGS 

print('imported all the packages')

# Suppress PyTorch Lightning logs
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='NeuralForecast for Energy challenge')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--models', nargs='+', default=['LSTM'], help='List of models to run')
parser.add_argument('--fe', type=str, default=None,
                    choices=['fourier', 'trend', 'time_features', 'future_exog', 'pipeline'],
                    help='Feature engineering method to apply to the data')
args = parser.parse_args()

# Load data

X_train_df = load_and_preprocess('data/training.csv')
X_test_df = load_and_preprocess('data/testing.csv')

# Split into train/validation
mid_point = len(X_train_df) // 3
train_df = X_train_df.iloc[:2*mid_point]
val_df = X_train_df.iloc[2*mid_point:]
horizon = len(val_df)
input_size = horizon
max_steps = args.epochs


freq = 'H'  # Or infer based on your dataset
if args.fe:
    print(f"Applying feature engineering method: {args.fe}")
    if args.fe == 'fourier':
        train_df, _ = apply_fourier(train_df, freq=freq, season_length=24, k=2, h=horizon)
        val_df, _ = apply_fourier(val_df, freq=freq, season_length=24, k=2, h=horizon)
    elif args.fe == 'trend':
        train_df, _ = apply_trend(train_df, freq=freq, h=horizon)
        val_df, _ = apply_trend(val_df, freq=freq, h=horizon)
    elif args.fe == 'time_features':
        train_df, _ = apply_time_features(train_df, freq=freq, h=horizon)
        val_df, _ = apply_time_features(val_df, freq=freq, h=horizon)
    elif args.fe == 'future_exog':
        train_df, _ = apply_future_exog_to_historic(train_df, freq=freq, features=[], h=horizon)
        val_df, _ = apply_future_exog_to_historic(val_df, freq=freq, features=[], h=horizon)
    elif args.fe == 'pipeline':
        train_df, _ = apply_pipeline(train_df, freq=freq, h=horizon)
        val_df, _ = apply_pipeline(val_df, freq=freq, h=horizon)


results_dir = "results/train"
current_plot_dir = get_next_plot_dir(base_dir=results_dir)
print(f"Created new plot directory: {current_plot_dir}")

selected_models = args.models
print(f"Selected models: {selected_models}")

for name in selected_models:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_CONFIGS.keys())}")

# Loop through selected models
for model_name in selected_models:
    
    model_conf = MODEL_CONFIGS[model_name]
    model_class = model_conf['model_class']
    config = model_conf['params'].copy()
    
    if config['h'] is None:
        config['h'] = horizon
        config['input_size'] = input_size
    else:
        horizon = config['h']
        input_size= config['input_size']
        
    print(f'{horizon=}')
        
    config['max_steps'] = max_steps
    config['val_check_steps'] = max_steps
    model = model_class(**config)
    model_name = model.__class__.__name__

    print(f"\n--- Running Model: {model_name} ---")
    
    nf = NeuralForecast(models=[model], freq='H')
    nf.fit(df=train_df)
    
    Y_hat_df = nf.predict(futr_df=val_df)
    
    # Evaluate
    if model_name == "NBEATSx":
        model_name = "NBEATSx-hi-80"
    merged_df = val_df[['ds', 'y']].copy().merge(Y_hat_df[['ds', model_name]], on='ds', how='inner')
    # Drop any rows with missing predictions or ground truth
    merged_df = merged_df.dropna(subset=['y', model_name])
    y_true = merged_df['y']
    y_pred = merged_df[model_name]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"MAE      : {mae:.2f}")

    # Plot
    fig = plot_series(val_df, Y_hat_df)
    
    metrics_text = f"R²: {r2:.3f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"
    config_text = f"Model: {model_name}\n" + '\n'.join([f"{k}: {v}" for k, v in config.items()])

    fig.text(0.75, 0.85, metrics_text, fontsize=10, bbox=dict(facecolor='black', alpha=0.8), color='white')
    fig.text(0.02, 0.05, config_text, fontsize=9, va='bottom', bbox=dict(facecolor='black', alpha=0.8), color='white')

    # Count existing files for this model in the results directory
    filename = get_next_plot_filename(current_plot_dir, model_name)

    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Plot saved as {filename}")

