# model_configs.py

from neuralforecast.models import LSTM, NHITS, TFT, KAN, VanillaTransformer ,NBEATSx

from neuralforecast.losses.pytorch import MAE, MQLoss

# futr_exog = ['Site-1 Temp', 'Site-1 GHI']

futr_exog = ["Site-1 Temp","Site-2 Temp","Site-3 Temp","Site-4 Temp","Site-5 Temp","Site-1 GHI","Site-2 GHI","Site-3 GHI","Site-4 GHI","Site-5 GHI"]

MODEL_CONFIGS = {
    'LSTM': {
        'model_class': LSTM,
        'params': {
            'h': None,
            'input_size': None,
            'futr_exog_list': futr_exog,
            'max_steps': None,
            'scaler_type': 'standard',
            'encoder_hidden_size': 256,
            'decoder_hidden_size': 256,
            'learning_rate': 1e-3,
            'val_check_steps': None,
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
            'val_check_steps': None,
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
            'hidden_size': 64,
            'learning_rate': 1e-3,
            'batch_size': 8,
            'val_check_steps': None,
        }
    },
        'VanillaTransformer': {
        'model_class': VanillaTransformer,
        'params': {
            'h': None,                      # to be filled at runtime
            'input_size': None,             # to be filled at runtime
            'futr_exog_list': futr_exog,
            'exclude_insample_y': False,
            'decoder_input_size_multiplier': 0.5,
            'hidden_size': 128,
            'dropout': 0.05,
            'n_head': 4,
            'conv_hidden_size': 32,
            'activation': 'gelu',
            'encoder_layers': 2,
            'decoder_layers': 1,
            'loss': MAE(),
            'max_steps': None,              # to be filled at runtime from CLI
            'learning_rate': 1e-3,
            'val_check_steps': None,
            'batch_size': 32,
            'scaler_type': 'robust',
            'random_seed': 1,
        }
    },
        'NBEATSx': {
        'model_class': NBEATSx,
        'params': {
            'h': None,
            'input_size': None,
            'loss': MQLoss(level=[80, 90]),
            'scaler_type': 'robust',
            'dropout_prob_theta': 0.5,
            'max_steps': None,
            'val_check_steps': None,
        }
    }
}
