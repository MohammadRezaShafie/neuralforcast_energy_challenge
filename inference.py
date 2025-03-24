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
from utils import load_and_preprocess
from config import MODEL_CONFIGS 

print('imported all the packages')

# Suppress PyTorch Lightning logs
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='NeuralForecast for Energy challenge')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--model', type=str, default='LSTM', help='name of model for test')
parser.add_argument('--save', action='store_true', help='save model')

args = parser.parse_args()

# Load data

X_train_df = load_and_preprocess('data/training.csv')
X_test_df = load_and_preprocess('data/testing.csv')


# Output dir
results_dir = "results/test"
os.makedirs(results_dir, exist_ok=True)
# Create a new unique subdirectory for this run's plots
existing_plot_dirs = [
    d for d in os.listdir(results_dir)
    if d.startswith("plots_idx_") and os.path.isdir(os.path.join(results_dir, d))
]
existing_indices = [
    int(d.split("_")[-1]) for d in existing_plot_dirs
    if d.split("_")[-1].isdigit()
]
next_run_index = max(existing_indices, default=-1) + 1
current_plot_dir = os.path.join(results_dir, f"plots_idx_{next_run_index}")
os.makedirs(current_plot_dir, exist_ok=True)

print(f"Created new plot directory: {current_plot_dir}")

selected_model = args.model

model_path = f"./checkpoints/{selected_model}"
nf= NeuralForecast.load(path=model_path)

for m in nf.models:
    horizon = m.h
    model_name = m.__class__.__name__

print(f'{horizon=}')
print(f"\n--- Running Model: {model_name} ---")


Y_hat_df = nf.predict(futr_df=X_test_df)

fig = plot_series(X_train_df, Y_hat_df)
existing_files = [
    f for f in os.listdir(current_plot_dir)
    if f.startswith(f"forecast_{model_name}_idx") and f.endswith(".png")
]
existing_indices = [
    int(f.split("_idx")[1].split(".")[0]) for f in existing_files
    if f.split("_idx")[1].split(".")[0].isdigit()
]
next_index = max(existing_indices, default=0) + 1

filename = os.path.join(current_plot_dir, f"forecast_{model_name}_idx{next_index}.png")
fig.savefig(filename, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved as {filename}")




