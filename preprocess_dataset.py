from typing import List
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from utils_and_constant import (
    DROP_COLNAMES,
    PROCESSED_DATASET,
    RAW_DATASET,
    TARGET_COLUMN,
)

def read_dataset(filename: str, drop_columns: List[str], target_column: str) -> pd.DataFrame:
    df = pd.read_csv(filename).drop(columns=drop_columns)
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})
    return df

def target_encode_categorical_features(df: pd.DataFrame, categorical_columns: List[str], target_column: str) -> pd.DataFrame:
    encoded_data = df.copy()
    for col in categorical_columns:
        encoding_map = df.groupby(col)[target_column].mean().to_dict()
        encoded_data[col] = encoded_data[col].map(encoding_map)
    return encoded_data

def train_test_splitting(encoded_data):
    train, test = sklearn_train_test_split(encoded_data, test_size=0.25, random_state=42)
    return train, test

def impute_and_scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)
    X_train_preprocessed = imputer.transform(X_train.values)
    X_test_preprocessed = imputer.transform(X_test.values)

    scaler = StandardScaler()
    scaler.fit(X_train_preprocessed)
    X_train_scaled = scaler.transform(X_train_preprocessed)
    X_test_scaled = scaler.transform(X_test_preprocessed)

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)

def main():
    # Read data
    weather = read_dataset(filename=RAW_DATASET, drop_columns=DROP_COLNAMES, target_column=TARGET_COLUMN)

    # Target encode categorical columns
    categorical_columns = weather.select_dtypes(include=[object]).columns.to_list()
    weather = target_encode_categorical_features(weather, categorical_columns, TARGET_COLUMN)

    # Split data
    train_data, test_data = train_test_splitting(weather)

    # Impute and scale features
    X_train = train_data.drop(columns=TARGET_COLUMN)
    X_test = test_data.drop(columns=TARGET_COLUMN)
    X_train_pre, X_test_pre = impute_and_scale_data(X_train, X_test)

    # Concatenate features and labels
    y_train = train_data[TARGET_COLUMN].reset_index(drop=True)
    y_test = test_data[TARGET_COLUMN].reset_index(drop=True)
    weather_train = pd.concat([X_train_pre, y_train], axis=1)
    weather_test = pd.concat([X_test_pre, y_test], axis=1)

    # Save processed datasets
    weather_train.to_csv(PROCESSED_DATASET.replace(".csv", "_train.csv"), index=False)
    weather_test.to_csv(PROCESSED_DATASET.replace(".csv", "_test.csv"), index=False)

    return weather_train, weather_test

if __name__ == "__main__":
    main()
