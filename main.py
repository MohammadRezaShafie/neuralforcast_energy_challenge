import time
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

print('imported all the packages')

# Suppress PyTorch Lightning logs
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# Load data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"date": "ds", "Load": "y"})
    df["unique_id"] = 1
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(by=["unique_id", "ds"])
    return df

Y_train_df = load_and_preprocess('data/training_energy.csv')
Y_test_df = load_and_preprocess('data/test_energy.csv')


futr_exog = ['Site-1 Temp', 'Site-1 GHI']

# Iterate through models
print(f"\n--- Running Model: LSTM ---")

# Split into first and second half of training data
mid_point = len(Y_train_df) // 3
train_df = Y_train_df.iloc[:2*mid_point]
val_df = Y_train_df.iloc[2*mid_point:]

# Dynamically set horizon based on prediction target (val_df)
horizon = len(val_df)
input_size = horizon  # 1 month input
max_steps=15
# Define the model with correct horizon

config = {
    'h': horizon,
    'input_size': input_size,
    'inference_input_size': 168,
    'futr_exog_list': futr_exog,
    'max_steps': max_steps,
    'scaler_type': 'standard',
    'encoder_hidden_size': 256,
    'decoder_hidden_size': 256,
    'learning_rate': 1e-3,
}

model = LSTM(**config)

model_name = model.__class__.__name__

# Fit the model on the first half
nf = NeuralForecast(models=[model], freq='H')
nf.fit(df=train_df)

# Predict the second half
Y_hat_df = nf.predict(futr_df=val_df)

# Merge true and predicted values for evaluation
merged_df = val_df[['ds', 'y']].copy()
merged_df = merged_df.merge(Y_hat_df, on='ds', how='left')

y_true = merged_df['y']
y_pred = merged_df[model_name]  # Dynamically access by model name

# Calculate metrics
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# Plotting with metrics and model settings
fig = plot_series(val_df, Y_hat_df)

# Format metrics
metrics_text = f"R²: {r2:.3f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"

# Format model config for annotation
model_settings_text = f"Model: {model_name}\n"
for k, v in config.items():
    model_settings_text += f"{k}: {v}\n"
    
# Add annotations directly to the figure
fig.text(0.75, 0.85, metrics_text, fontsize=10, bbox=dict(facecolor='black', alpha=0.8), color='white')
fig.text(0.02, 0.05, model_settings_text, fontsize=9, va='bottom', bbox=dict(facecolor='black', alpha=0.8), color='white')
# Save plot
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
# 5. Generate indexed timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
index = len(os.listdir(results_dir)) + 1
filename = f"{results_dir}/forecast_{model_name}_idx{index}_{timestamp}_r2_{r2:.3f}.png"


fig.savefig(filename, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved as {filename}")
