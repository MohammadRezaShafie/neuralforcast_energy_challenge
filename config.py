# model_configs.py

from neuralforecast.models import LSTM, NHITS, TFT

futr_exog = ['Site-1 Temp', 'Site-1 GHI']

MODEL_CONFIGS = [
    {
        'model_class': LSTM,
        'params': {
            'h': None,  # Will be set dynamically
            'input_size': None,  # Will be set dynamically
            'futr_exog_list': futr_exog,
            'max_steps': None,  # Will be set from args
            'scaler_type': 'standard',
            'encoder_hidden_size': 256,
            'decoder_hidden_size': 256,
            'learning_rate': 1e-3,
        }
    },
    {
        'model_class': NHITS,
        'params': {
            'h': None,
            'input_size': None,
            'futr_exog_list': futr_exog,
            'max_steps': None,
            'scaler_type': 'standard',
            'learning_rate': 1e-3,
        }
    }
]
