import pandas as pd

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"date": "ds", "Load": "y"})
    df["unique_id"] = 1
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(by=["unique_id", "ds"])
    return df
