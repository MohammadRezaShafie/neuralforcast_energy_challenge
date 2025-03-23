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
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic, augment_calendar_df, AirPassengersDF
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from utilsforecast.plotting import plot_series
from neuralforecast.models import NBEATS, NHITS, LSTM, NHITS, RNN, TFT, TimeLLM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import load_and_preprocess
from config import MODEL_CONFIGS 

print('imported all the packages')

# Suppress PyTorch Lightning logs
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='NeuralForecast for Energy challenge')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--new-epochs', type=int, default=5)
parser.add_argument('--models', nargs='+', default=['LSTM'], help='List of models to run')
parser.add_argument('--use-preds', action='store_true',
                    help='Use predicted values instead of ground truth for continued training')

args = parser.parse_args()

# Load data

Y_train_df = load_and_preprocess('data/training_energy.csv')
Y_test_df = load_and_preprocess('data/test_energy.csv')

# Split into train/validation
mid_point = len(Y_train_df) // 3
train_df = Y_train_df.iloc[:2*mid_point]
val_df = Y_train_df.iloc[2*mid_point:]
horizon = len(val_df)
input_size = horizon
max_steps = args.epochs
new_max_steps = args.new_epochs

# Output dir
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

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
        
    config['max_steps'] = max_steps
    config['val_check_steps'] = max_steps

    model = model_class(**config)
    model_name = model.__class__.__name__

    print(f"\n--- Running Model: {model_name} ---")
    
    print(f'{model.max_steps=}')
    # Train + Predict
    nf = NeuralForecast(models=[model], freq='H')
    nf.fit(df=train_df)
    print(f'{horizon=}')
    # model.max_steps = new_max_steps
    for m in nf.models:
        m.max_steps = new_max_steps
        m.trainer_kwargs['max_steps'] = 5   
        m.val_check_steps = new_max_steps
        # Reset training state to enforce new max_steps
        m.trainer = None  # Force reinitialization of the Trainer
        m._fit_called = False  # Reset internal flag if exists
    print(f'Actual model max_steps: {nf.models[0].max_steps}')  # Verify change

    all_preds = []
    start_idx = 0
    
    while start_idx + horizon <= len(val_df):
        window_df = val_df.iloc[start_idx : start_idx + horizon].copy()
        preds = nf.predict(futr_df=window_df)
        all_preds.append(preds)    
        if args.use_preds:
            window_df = window_df.merge(preds[['ds', model_name]], on='ds', how='left')
            window_df['y'] = window_df[model_name]
            window_df.drop(columns=[model_name], inplace=True)
        start_idx += horizon  # move window
        df_combined = pd.concat([train_df, window_df], ignore_index=True)
        nf.fit(df=df_combined, use_init_models = False)
        
    # Y_hat_df = nf.predict(futr_df=val_df, step_size=horizon)
    
    Y_hat_df = pd.concat(all_preds).sort_values(['ds'])
    # Y_hat_df = Y_hat_df.reset_index()
    # print(f'{Y_hat_df.head(200)=}')
    # Evaluate
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
      
    if args.use_preds:
        metrics_text = f"R²: {r2:.3f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nUsed Prediction"
    else:
        metrics_text = f"R²: {r2:.3f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"
    config_text = f"Model: {model_name}\n" + '\n'.join([f"{k}: {v}" for k, v in config.items()])

    fig.text(0.75, 0.85, metrics_text, fontsize=10, bbox=dict(facecolor='black', alpha=0.8), color='white')
    fig.text(0.02, 0.05, config_text, fontsize=9, va='bottom', bbox=dict(facecolor='black', alpha=0.8), color='white')

    # Count existing files for this model in the results directory
    existing_files = [
        f for f in os.listdir(results_dir)
        if f.startswith(f"forecast_{model_name}_idx") and f.endswith(".png")
    ]
    existing_indices = [
        int(f.split("_idx")[1].split(".")[0]) for f in existing_files
        if f.split("_idx")[1].split(".")[0].isdigit()
    ]
    next_index = max(existing_indices, default=0) + 1

    filename = f"{results_dir}/forecast_{model_name}_idx{next_index}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Plot saved as {filename}")
