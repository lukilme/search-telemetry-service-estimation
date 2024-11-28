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
    y_true = np.array(y_true, dtype=float) 
    y_pred = np.array(y_pred, dtype=float)  
    return np.mean(np.abs(y_true - y_pred) / y_true)

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

    total = data_log.merge(data_dash, on=['timestamp', 'timestamp'], how='left')
    total = total.dropna()
    
    #total = remove_outlier_IQR(total)
    features = total.iloc[:,1:len(data_log.columns)].values
    
    labels = total['framesDisplayedCalc'].values

    return normalization(features), labels

