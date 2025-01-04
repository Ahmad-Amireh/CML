import shutil
from pathlib import Path
import pandas as pd

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["Date"]
TARGET_COLUMN = "RainTomorrow"
RAW_DATASET = "raw_dataset/weather.csv"
PROCESSED_DATASET_TRAIN = "processed_dataset/weather_train.csv"
PROCESSED_DATASET_TEST = "processed_dataset/weather_test.csv"


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)



def read_csv(filename: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        raise Exception(f"File {filename} not found. Please check the path.")

def save_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, index=False)

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y