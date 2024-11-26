from _params_models import *
import io
import sys
import sympy as sp
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)

def nmae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    nmae_value = mae / np.mean(y_true)
    return nmae_value


def get_datasets(dataset_name="sinusoid") -> Tuple[pd.DataFrame, pd.DataFrame]:
    source_data = "sinusoid_8h"

    if dataset_name == "mix":
        source_data = "mix_5h"
    if dataset_name == "flashcrowd":
        source_data = "flashcrowd_6h"

    data_log = pd.read_csv(f"assets/data/log_INT_{source_data}.txt", delimiter=",")

    data_log.columns = data_log.columns.str.replace(" ", "")

    data_dash = pd.read_csv(f"assets/data/dash_{source_data}.log", sep=",")

    return data_log, data_dash


def merge_and_clean_data(
    data_log: pd.DataFrame, data_dash: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    for column in data_log.columns:
        if data_log[column].std() == 0.0:
            data_log = data_log.drop(columns=[column])

    merged_data = pd.merge(data_log, data_dash, on="timestamp", how="right")

    merged_data.fillna(merged_data.mean(), inplace=True)

    z_scores = np.abs(stats.zscore(merged_data["framesDisplayedCalc"]))
    threshold = 60
    merged_data = merged_data[z_scores < threshold]

    labels = merged_data["framesDisplayedCalc"]
    columns_to_remove = list(data_dash.columns)
    merged_data = merged_data.drop(columns=columns_to_remove)

    return merged_data, labels


def alert_end():
    from plyer import notification

    notification.notify(
        title="Treino do Modelo Concluído",
        message="O teste foi finalizado com sucesso!",
        app_icon="alert.ico",
        timeout=3,
        ticker="Treino do Modelo Concluído",
        toast=True,
    )


def remove_useless_attribute(dataset):
    dataset.drop(columns=dataset.columns[dataset.nunique() == 1], inplace=True)
    return dataset

def remove_outlier_IQR(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_final = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]
    return df_final

def change_NaN_to_mean(dataset):
    dataset = dataset.fillna(dataset.mean())
    return dataset


def normalization(X):
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return X

def merge_dataset(data_log, data_dash):
    data_log = remove_useless_attribute(data_log)

    data_dash['timestamp'] = data_dash['timestamp'].astype(str).str[:10].astype(int)
    total = data_log.merge(data_dash, on=['timestamp', 'timestamp'], how='left')
    
    total = remove_outlier_IQR(total)
    total = change_NaN_to_mean(total)
    features = total.iloc[:,1:len(data_log.columns)].values
    labels = total['framesDisplayedCalc'].values

    features = normalization(features)

    return features, labels
