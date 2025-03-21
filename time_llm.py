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

print('imported all the packages')

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




gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
gpt2 = GPT2Model.from_pretrained('openai-community/gpt2',config=gpt2_config)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

prompt_prefix = '''
This dataset contains 2 years of hourly energy load data from California sites,
with site temperature and global horizontal irradiance (GHI) as key features.
The primary target is accurate load forecasting to enhance grid reliability,
demand response, and energy optimization.
The structured train/val/test split ensures robust model evaluation for long-term forecasting."
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timellm = TimeLLM(h=len(Y_test_df),
                 input_size=168,
                 llm=gpt2,
                 llm_config=gpt2_config,
                 llm_tokenizer=gpt2_tokenizer,
                 prompt_prefix=prompt_prefix,
                 max_steps=100,
                 batch_size=24,
                 windows_batch_size=24,
                 accelerator= 'cuda')

nf = NeuralForecast(
    models=[timellm],
    freq='H'
)

nf.fit(df=Y_train_df)
print('fitting')
forecasts = nf.predict(df=Y_train_df, futr_df=Y_test_df)
# Merge predictions with test data
Y_test_df = Y_test_df.rename(columns={"ds": "date"})  # Rename 'ds' back to 'date'
forecasts = forecasts.rename(columns={"ds": "date"})  # Rename 'ds' back to 'date'
results = Y_test_df.merge(forecasts, on=["unique_id", "date"], how="left")

# Save results to a new CSV file
results.to_csv('predictions_with_test_data.csv', index=False)
print('Predictions saved to predictions_with_test_data.csv')

# Print results
print(results)