import pandas as pd
from typing import Tuple

def nmae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    nmae_value = mae / np.mean(y_true)
    return nmae_value

def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse_value = rmse / np.mean(y_true)
    return nrmse_value

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

def merge_dataset(data_log, data_dash):

    total = data_log.merge(data_dash, on=['timestamp', 'timestamp'], how='left')
    total = total.dropna()
    
    features = total.iloc[:,1:len(data_log.columns)].values
    
    labels = total['framesDisplayedCalc'].values

    return normalization(features), labels