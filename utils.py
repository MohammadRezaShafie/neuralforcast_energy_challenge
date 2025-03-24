import pandas as pd
import numpy as np
from typing import Tuple, List, Callable, Union
from utilsforecast.feature_engineering import fourier, trend, time_features, future_exog_to_historic, pipeline
from functools import partial
import os

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"date": "ds", "Load": "y"})
    df["unique_id"] = 1
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(by=["unique_id", "ds"])
    return df

DFType = pd.DataFrame

def apply_fourier(df: DFType, freq: Union[str, int], h: int = 0,
                  season_length: int = 24, k: int = 2,
                  id_col: str = 'unique_id', time_col: str = 'ds') -> Tuple[DFType, DFType]:
    return fourier(df=df, freq=freq, season_length=season_length, k=k, h=h, id_col=id_col, time_col=time_col)

def apply_trend(df: DFType, freq: Union[str, int], h: int = 0,
                id_col: str = 'unique_id', time_col: str = 'ds') -> Tuple[DFType, DFType]:
    return trend(df=df, freq=freq, h=h, id_col=id_col, time_col=time_col)

def apply_time_features(df: DFType, freq: Union[str, int], h: int = 0,
                         features: List[Union[str, Callable]] = ['month', 'day', 'week'],
                         id_col: str = 'unique_id', time_col: str = 'ds') -> Tuple[DFType, DFType]:
    return time_features(df=df, freq=freq, features=features, h=h, id_col=id_col, time_col=time_col)

def apply_future_exog_to_historic(df: DFType, freq: Union[str, int], h: int = 0,
                                   features: List[str] = [],
                                   id_col: str = 'unique_id', time_col: str = 'ds') -> Tuple[DFType, DFType]:
    return future_exog_to_historic(df=df, freq=freq, features=features, h=h, id_col=id_col, time_col=time_col)

def apply_pipeline(df: DFType, freq: Union[str, int], h: int = 0,
                   id_col: str = 'unique_id', time_col: str = 'ds') -> Tuple[DFType, DFType]:
    def is_weekend(times):
        if isinstance(times, pd.Index):
            return times.weekday + 1 >= 6
        else:
            return times.dt.weekday() >= 6

    def even_days_and_months(times):
        if isinstance(times, pd.Index):
            return pd.DataFrame({
                'even_day': (times.weekday + 1) % 2 == 0,
                'even_month': times.month % 2 == 0,
            })
        else:
            return [
                (times.dt.weekday() % 2 == 0).alias('even_day'),
                (times.dt.month() % 2 == 0).alias('even_month'),
            ]

    features = [
        trend,
        partial(fourier, season_length=7, k=1),
        partial(fourier, season_length=28, k=1),
        partial(time_features, features=['day', is_weekend, even_days_and_months])
    ]
    return pipeline(df=df, features=features, freq=freq, h=h, id_col=id_col, time_col=time_col)


def get_next_plot_dir(base_dir="results/train", prefix="plots_idx_"):
    os.makedirs(base_dir, exist_ok=True)
    existing_plot_dirs = [
        d for d in os.listdir(base_dir)
        if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d))
    ]
    existing_indices = [
        int(d.split("_")[-1]) for d in existing_plot_dirs
        if d.split("_")[-1].isdigit()
    ]
    next_run_index = max(existing_indices, default=-1) + 1
    plot_dir = os.path.join(base_dir, f"{prefix}{next_run_index}")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def get_next_plot_filename(plot_dir, model_name):
    existing_files = [
        f for f in os.listdir(plot_dir)
        if f.startswith(f"forecast_{model_name}_idx") and f.endswith(".png")
    ]
    existing_indices = [
        int(f.split("_idx")[1].split(".")[0]) for f in existing_files
        if f.split("_idx")[1].split(".")[0].isdigit()
    ]
    next_index = max(existing_indices, default=0) + 1
    return os.path.join(plot_dir, f"forecast_{model_name}_idx{next_index}.png")
