import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM 
from neuralforecast.losses.pytorch import MAE
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic, augment_calendar_df
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
import pandas as pd
import logging
from utilsforecast.plotting import plot_series
from neuralforecast.models import NBEATS, NHITS, LSTM, NHITS, RNN, TFT
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF


print('imported all the packages')

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

Y_train_df = pd.read_csv('training_energy.csv')
Y_test_df = pd.read_csv('test_energy.csv')
Y_train_df = Y_train_df.rename(columns={"date": "ds", "Load": "y"})
Y_train_df["unique_id"] = 1
Y_train_df["ds"] = pd.to_datetime(Y_train_df["ds"])
Y_train_df = Y_train_df.sort_values(by=["unique_id", "ds"])

Y_test_df = Y_test_df.rename(columns={"date": "ds", "Load": "y"})
Y_test_df["unique_id"] = 1
Y_test_df["ds"] = pd.to_datetime(Y_test_df["ds"])
Y_test_df = Y_test_df.sort_values(by=["unique_id", "ds"])

futr_exog = ['Site-1 Temp', 'Site-1 GHI']
horizon = len(Y_test_df)
input_size = 168
# horizon = 100
# Try different hyperparmeters to improve accuracy.

models = [
    LSTM(
        h=horizon,
        input_size=horizon,
        futr_exog_list=futr_exog,
        max_steps=1500,
        scaler_type='standard',
        encoder_hidden_size=128,
        decoder_hidden_size=128,
        learning_rate=1e-3,
      #   early_stop_patience_steps=5,
    )
]

# nf = NeuralForecast(models=models, freq='H')
# nf.fit(df=Y_train_df)

# Y_hat_df = nf.predict(df= Y_train_df, futr_df=Y_test_df)
# # Plot forecast

# fig = plot_series(Y_train_df, Y_hat_df)
# fig.savefig("forecast_plot.png", dpi=300, bbox_inches='tight')
# plt.close(fig)


# Split into train/validation
train_size = int(0.8 * len(Y_train_df))
train_df = Y_train_df.iloc[:train_size]
val_df = Y_train_df.iloc[train_size:]

# Train with validation
nf = NeuralForecast(models=models, freq='H')
# nf.fit(df=train_df,val_size = input_size)

nf.fit(df=Y_train_df)
# Predict
Y_hat_df = nf.predict(futr_df=Y_test_df)

# Plot results
fig = plot_series(Y_train_df, Y_hat_df)
fig.savefig("forecast_plot3.png", dpi=300, bbox_inches='tight')
plt.close(fig)