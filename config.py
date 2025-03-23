# model_configs.py

from neuralforecast.models import LSTM, NHITS, TFT

futr_exog = ['Site-1 Temp', 'Site-1 GHI']

MODEL_CONFIGS = {
    'LSTM': {
        'model_class': LSTM,
        'params': {
            'h': None,
            'input_size': None,
            'futr_exog_list': futr_exog,
            'max_steps': None,
            'scaler_type': 'standard',
            'encoder_hidden_size': 128,
            'decoder_hidden_size': 128,
            'learning_rate': 1e-3,
        }
    },
    'NHITS': {
        'model_class': NHITS,
        'params': {
            'h': None,
            'input_size': None,
            'futr_exog_list': futr_exog,
            'max_steps': None,
            'scaler_type': 'standard',
            'learning_rate': 1e-3,
        }
    },
    'TFT': {
        'model_class': TFT,
        'params': {
            'h': None,
            'input_size': None,
            'futr_exog_list': futr_exog,
            'max_steps': None,
            'scaler_type': 'standard',
            'hidden_size': 128,
            'learning_rate': 1e-3,
            'batch_size': 2,
        }
    }
}
