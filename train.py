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
parser.add_argument('--models', nargs='+', default=['LSTM'], help='List of models to run')
args = parser.parse_args()

# Load data

X_train_df = load_and_preprocess('data/training.csv')
X_test_df = load_and_preprocess('data/testing.csv')

horizon = len(X_test_df)
input_size = horizon
max_steps = args.epochs


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
    nf.fit(df=X_train_df)
    
    checkpoints_dir = f"checkpoints/{model_name}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    nf.save(path=checkpoints_dir,
        model_index=None, 
        overwrite=True,
        save_dataset=True)