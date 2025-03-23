# model_configs.py

from neuralforecast.models import LSTM, NHITS, TFT, KAN, VanillaTransformer

from neuralforecast.losses.pytorch import MAE

futr_exog = ['Site-1 Temp', 'Site-1 GHI']

MODEL_CONFIGS = {
    'LSTM': {
        'model_class': LSTM,
        'params': {
            'h': 720,
            'input_size': 1440,
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
            'h': 720,
            'input_size': 1440,
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
            'h': 168,
            'input_size': 336,
            'futr_exog_list': futr_exog,
            'max_steps': None,
            'scaler_type': 'standard',
            'hidden_size': 64,
            'learning_rate': 1e-3,
            'batch_size': 16,
            'val_check_steps': None,
        }
    },
        'KAN': {
        'model_class': KAN,
        'params': {
            'h': 168,                      # Set dynamically at runtime
            'input_size': 336,             # Set dynamically at runtime
            'futr_exog_list': futr_exog,
            'hist_exog_list': None,
            'stat_exog_list': None,
            'exclude_insample_y': False,
            'grid_size': 5,
            'spline_order': 3,
            'scale_noise': 0.1,
            'scale_base': 1.0,
            'scale_spline': 1.0,
            'enable_standalone_scale_spline': True,
            'grid_eps': 0.02,
            'grid_range': [-1, 1],
            'n_hidden_layers': 1,
            'hidden_size': 512,
            'loss': MAE(),
            'max_steps': None,              # Set from CLI
            'learning_rate': 1e-3,
            # 'early_stop_patience_steps': 5,
            'batch_size': 32,
            # 'windows_batch_size': 1024,
            # 'inference_windows_batch_size': -1,
            # 'step_size': 1,
            'start_padding_enabled': False,
            'scaler_type': 'standard',
            'random_seed': 1,
            'val_check_steps': None,
        }
    }, 
        'VanillaTransformer': {
        'model_class': VanillaTransformer,
        'params': {
            'h': 168,                      # to be filled at runtime
            'input_size': 336,             # to be filled at runtime
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
            # 'inference_windows_batch_size': 1024,
            # 'windows_batch_size': 1024,
            'scaler_type': 'robust',
            # 'step_size': 1,
            'random_seed': 1,
        }
    }
}
