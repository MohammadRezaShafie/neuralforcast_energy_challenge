# model_configs.py

from neuralforecast.models import (
    LSTM,
    NHITS,
    TFT,
    DeepAR,
    NBEATSx,
    DeepNPTS,
    DilatedRNN,
    DLinear,
    FEDformer,
    TCN,
    StemGNN,
    Informer,
    MLP,
    TiDE,
    iTransformer,
    NLinear,
    BiTCN)
from neuralforecast.losses.pytorch import GMM
from neuralforecast.losses.pytorch import MAE, MQLoss, DistributionLoss, MSE

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
        'NBEATSx': {
        'model_class': NBEATSx,
        'params': {
            'h': None,
            'input_size': None,
            'futr_exog_list': futr_exog,
            'loss': MQLoss(level=[80, 90]),
            'scaler_type': 'robust',
            'dropout_prob_theta': 0.5,
            'max_steps': None,
            'val_check_steps': None,
        }
    },
        'BiTCN': {
        'model_class': BiTCN,
        'params': {
            'h': None,  # to be set dynamically
            'input_size': None,  # to be set dynamically
            'loss': GMM(n_components=7, level=[80, 90]),
            'max_steps': None,  # to be set dynamically
            'scaler_type': 'standard',
            'futr_exog_list': futr_exog,
            'windows_batch_size': 256,
            'val_check_steps': None,
            'early_stop_patience_steps': -1,
        }
    },
        'DeepAR': {
        'model_class': DeepAR,
        'params': {
            'h': None,  # set dynamically
            'input_size': None,  # set dynamically
            'lstm_n_layers': 1,
            'trajectory_samples': 100,
            'loss': DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
            'valid_loss': MQLoss(level=[80, 90]),
            'learning_rate': 0.005,
            'futr_exog_list': futr_exog,
            'max_steps': None,  # set dynamically
            'val_check_steps': None,  # set dynamically
            'scaler_type': 'standard',
            'enable_progress_bar': True,
        }
    },
        'DeepNPTS': {
        'model_class': DeepNPTS,
        'params': {
            'h': None,                # to be set dynamically
            'input_size': None,       # to be set dynamically
            'futr_exog_list': futr_exog,
            'max_steps': None,        # to be set dynamically
            'val_check_steps': None,  # to be set dynamically
            'scaler_type': 'robust',
            'enable_progress_bar': True,
        }
    },
        'DilatedRNN': {
        'model_class': DilatedRNN,
        'params': {
            'h': None,                # to be set dynamically
            'input_size': None,       # to be set dynamically
            'loss': DistributionLoss(distribution='Normal', level=[80, 90]),
            'scaler_type': 'robust',
            'encoder_hidden_size': 100,
            'max_steps': None,        # to be set dynamically
            'val_check_steps': None,  # to be set dynamically
            'futr_exog_list': futr_exog,
        }
    },
        'DLinear': {
        'model_class': DLinear,
        'params': {
            'h': None,                # to be set dynamically
            'input_size': None,       # to be set dynamically
            'loss': MAE(),
            'scaler_type': 'robust',
            'learning_rate': 1e-3,
            'max_steps': None,        # to be set dynamically
            'val_check_steps': None,  # to be set dynamically
            'futr_exog_list': futr_exog,
        }
    },
        'FEDformer': {
        'model_class': FEDformer,
        'params': {
            'h': None,                  # to be set dynamically
            'input_size': None,         # to be set dynamically
            'modes': 64,
            'hidden_size': 64,
            'conv_hidden_size': 128,
            'n_head': 8,
            'loss': MAE(),
            'futr_exog_list': futr_exog,
            'scaler_type': 'robust',
            'learning_rate': 1e-3,
            'max_steps': None,          # to be set dynamically
            'val_check_steps': None,    # to be set dynamically
            'batch_size': 2,
            'windows_batch_size': 32,
        }
    },
        'Informer': {
        'model_class': Informer,
        'params': {
            'h': None,                # to be set dynamically
            'input_size': None,       # to be set dynamically
            'hidden_size': 16,
            'conv_hidden_size': 32,
            'n_head': 2,
            'loss': MAE(),
            'futr_exog_list': futr_exog,
            'scaler_type': 'robust',
            'learning_rate': 1e-3,
            'max_steps': None,        # to be set dynamically
            'val_check_steps': None,  # to be set dynamically
        }
    },
        'iTransformer': {
        'model_class': iTransformer,
        'params': {
            'h': None,                # to be set dynamically
            'input_size': None,       # to be set dynamically
            'n_series': 2,
            'hidden_size': 128,
            'n_heads': 2,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 4,
            'factor': 1,
            'dropout': 0.1,
            'use_norm': True,
            'loss': MSE(),
            'valid_loss': MAE(),
            'batch_size': 32,
            'futr_exog_list': futr_exog,
            'max_steps': None,        # to be set dynamically
            'val_check_steps': None,  # to be set dynamically
        }
    },
        'MLP': {
        'model_class': MLP,
        'params': {
            'h': None,                # to be set dynamically
            'input_size': None,       # to be set dynamically
            'loss': DistributionLoss(distribution='Normal', level=[80, 90]),
            'scaler_type': 'robust',
            'learning_rate': 1e-3,
            'max_steps': None,        # to be set dynamically
            'val_check_steps': None,  # to be set dynamically
            'futr_exog_list': futr_exog,
        }
    },
        'NLinear': {
        'model_class': NLinear,
        'params': {
            'h': None,                # to be set dynamically
            'input_size': None,       # to be set dynamically
            'loss': DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
            'scaler_type': 'robust',
            'learning_rate': 1e-3,
            'max_steps': None,        # to be set dynamically
            'val_check_steps': None,  # to be set dynamically
            'futr_exog_list': futr_exog,
        }
    },
        'StemGNN': {
        'model_class': StemGNN,
        'params': {
            'h': None,                # to be set dynamically
            'input_size': None,       # to be set dynamically
            'n_series': 2,
            'scaler_type': 'standard',
            'learning_rate': 1e-3,
            'loss': MAE(),
            'valid_loss': MAE(),
            'batch_size': 32,
            'futr_exog_list': futr_exog,
            'max_steps': None,        # to be set dynamically
            'val_check_steps': None,  # to be set dynamically
        }
    },
        'TCN': {
        'model_class': TCN,
        'params': {
            'h': None,                  # to be set dynamically
            'input_size': None,         # to be set dynamically
            'loss': DistributionLoss(distribution='Normal', level=[80, 90]),
            'learning_rate': 5e-4,
            'kernel_size': 2,
            'dilations': [1, 2, 4, 8, 16],
            'encoder_hidden_size': 128,
            'context_size': 10,
            'decoder_hidden_size': 128,
            'decoder_layers': 2,
            'max_steps': None,          # to be set dynamically
            'val_check_steps': None,    # to be set dynamically
            'scaler_type': 'robust',
            'futr_exog_list': futr_exog,
        }
    },
        'TiDE': {
        'model_class': TiDE,
        'params': {
            'h': None,                  # to be set dynamically
            'input_size': None,         # to be set dynamically
            'loss': GMM(n_components=7, return_params=True, level=[80, 90], weighted=True),
            'max_steps': None,          # to be set dynamically
            'val_check_steps': None,    # to be set dynamically
            'scaler_type': 'standard',
            'futr_exog_list': futr_exog,
        }
    }
        
}
