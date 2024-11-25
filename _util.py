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


def decompose_dataframe(df: pd.DataFrame):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format)

    print("\n" + "=" * 40 + " General DataFrame Information " + "=" * 40)

    print("\n" + "-" * 30 + " Statistical maxmary " + "-" * 30)
    print(df.describe())

    print("\n" + "-" * 30 + " First Rows of DataFrame " + "-" * 30)
    print(df.head())

    print("\n" + "-" * 30 + " Format (rows, columns) " + "-" * 30)
    print(f"{df.shape[0]} rows, {df.shape[1]} columns")

    pd.reset_option("display.width")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.float_format")


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


def get_data_concated(
    data_log: pd.DataFrame, data_dash: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:

    data_log = data_log.loc[:, data_log.nunique() > 1]

    data_log = (
        data_log.groupby("timestamp")
        .agg(
            {
            "ingress_global_timestamp3": "max",
            "ingress_global_timestamp2": "max",
            "ingress_global_timestamp1": "max",
            "egress_global_timestamp3": "max",
            "egress_global_timestamp2": "max",
            "egress_global_timestamp1": "max",
            "enq_timestamp3": "max",
            "enq_timestamp2": "max",
            "enq_timestamp1": "max",
            "enq_qdepth3": "sum",
            "deq_qdepth3": "sum",
            "deq_timedelta3": "mean",
            "enq_qdepth2": "sum",
            "deq_qdepth2": "sum",
            "deq_timedelta2": "mean",
            "enq_qdepth1": "sum",
            "deq_qdepth1": "sum",
            "deq_timedelta1": "mean",
        }
        )
        .reset_index()
    )



    result_total = pd.concat(
        [data_log.set_index("timestamp"), data_dash.set_index("timestamp")],
        axis=1,
        join="outer",
    )

    #result_total = result_total.fillna(result_total.mean())

    # z_scores = np.abs(stats.zscore(result_total["framesDisplayedCalc"]))
    # threshold = 60
    # result_total = result_total[z_scores < threshold]

    labels = result_total["framesDisplayedCalc"]

    labels = labels.reset_index()
    result_total = result_total.reset_index()
    columns_to_remove = list(data_dash.columns)
    # columns_to_remove.remove("timestamp")
    result_total = result_total.drop(columns=columns_to_remove)
    labels = labels.drop(columns=["timestamp"])
    timestamp_columns = [
        "ingress_global_timestamp3",
        "ingress_global_timestamp2",
        "ingress_global_timestamp1",
        "egress_global_timestamp3",
        "egress_global_timestamp2",
        "egress_global_timestamp1",
        "enq_timestamp3",
        "enq_timestamp2",
        "enq_timestamp1",
    ]

 

    return result_total, labels




def get_data_concated_v2(
    data_log: pd.DataFrame, data_dash: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:

    for column in data_log.columns:
        if data_log[column].std() == 0.0:
            data_log = data_log.drop(columns=[column])

    result_total = pd.concat(
        [data_log.set_index("timestamp"), data_dash.set_index("timestamp")],
        axis=1,
        join="outer",
    )

    result_total = result_total.fillna(result_total.mean())

    z_scores = np.abs(stats.zscore(result_total["framesDisplayedCalc"]))
    threshold = 60
    result_total = result_total[z_scores < threshold]

    labels = result_total["framesDisplayedCalc"]

    labels = labels.reset_index()
    result_total = result_total.reset_index()
    columns_to_remove = list(data_dash.columns)
    # columns_to_remove.remove("timestamp")
    result_total = result_total.drop(columns=columns_to_remove)
    labels = labels.drop(columns=["timestamp"])
    timestamp_columns = [
        "ingress_global_timestamp3",
        "ingress_global_timestamp2",
        "ingress_global_timestamp1",
        "egress_global_timestamp3",
        "egress_global_timestamp2",
        "egress_global_timestamp1",
        "enq_timestamp3",
        "enq_timestamp2",
        "enq_timestamp1",
    ]

    for column in timestamp_columns:
        if column in result_total.columns:
            result_total[column] = result_total[column].astype(int)

    return result_total, labels


def nmae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    nmae_value = mae / np.mean(y_true)
    return nmae_value * 100


def default_random_forest_model(
    features: pd.DataFrame, labels: pd.Series, model_params: rf_model_params
):

    X_train, X_validation, y_train, y_validation = train_test_split(
        features,
        np.ravel(labels),
        test_size=model_params.test_size,
        random_state=model_params.random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    rf_model = RandomForestRegressor(
        n_estimators=model_params.n_estimators,
        max_depth=model_params.max_depth,
        min_samples_split=model_params.min_samples_split,
        min_samples_leaf=model_params.min_samples_leaf,
        bootstrap=model_params.bootstrap,
        verbose=model_params.verbose,
        max_features=model_params.max_features,
        n_jobs=model_params.n_jobs,
        random_state=model_params.random_state,
    )

    mae_scorer = make_scorer(nmae, greater_is_better=False)

    kf = KFold(
        n_splits=model_params.n_splits,
        shuffle=model_params.shuffle,
        random_state=model_params.random_state,
    )

    cross_val_scores = cross_val_score(
        rf_model, X_train_scaled, y_train, cv=kf, scoring=mae_scorer
    )

    avg_cross_val_score = np.mean(cross_val_scores)

    """output = io.StringIO()
    sys.stdout = output"""

    rf_model.fit(X_train_scaled, y_train)

    """verbose_output = output.getvalue()
    with open("assets/logs/train_log_rf.txt", "w") as f:
        f.write(verbose_output)"""

    predictions = rf_model.predict(X_validation_scaled)
    mae_rf = mean_absolute_error(y_validation, predictions)
    nmae_rf = nmae(y_validation, predictions)

    return mae_rf, nmae_rf, rf_model




def default_rf_model_randomsearch(
    features_train: pd.DataFrame, labels_train: pd.Series, iter_n = 50
) -> Tuple[float, float, RandomForestRegressor, np.ndarray]:

    X_train, X_validation, y_train, y_validation = train_test_split(
        features_train, np.ravel(labels_train), test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    rf_model = RandomForestRegressor(random_state=42, n_jobs=2)
    nmae_scorer = make_scorer(nmae, greater_is_better=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    random_grid = {
        "n_estimators": [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        "max_features": ["log2", "sqrt"],
        "max_depth": [28, 37, 46, 55, 64, 73, 82, 91, 100, None],
        "max_samples_split": [2, 5, 10],
        "max_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=random_grid,
        scoring=nmae_scorer,
        cv=kf,
        n_iter=iter_n,
        verbose=3,
        n_jobs=2,
    )

    # Redireciona o output para uma variável
    """output = io.StringIO()
    sys.stdout = output"""

    random_search.fit(X_train_scaled, y_train) 

    # Guarda o output do verbose em uma variável ou salva em um arquivo
    """verbose_output = output.getvalue()
    with open("assets/logs/train_randomsearch_output2.txt", "w") as f:
        f.write(verbose_output)"""

    best_rf_model = random_search.best_estimator_
    best_params = random_search.best_params_
    print(f"Melhores hiperparâmetros: {best_params}")

    y_pred_rf = best_rf_model.predict(X_validation_scaled)
    mae_rf = mean_absolute_error(y_validation, y_pred_rf)
    nmae_rf = nmae(y_validation, y_pred_rf)

    feature_importances = best_rf_model.feature_importances_
    print(f"Importâncias das features: {feature_importances}")

    return mae_rf, nmae_rf, best_rf_model, best_params


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

def visualize_results(feature_importances, feature_names):

    feature_importances_df = pd.DataFrame({
        'Features': feature_names,
        'Importância': feature_importances
    }).sort_values(by='Importância', ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importância', y='Features', data=feature_importances_df)
    plt.title('Importância das Features')
    plt.xlabel('Importância')
    plt.ylabel('Features')
    plt.show()

def nmae(y_true, y_pred):
    nmae_value = mean_absolute_error(y_true, y_pred) / np.mean(y_true)
    return nmae_value

def default_random_forest_model(
    features: pd.DataFrame, labels: pd.Series, model_params: rf_model_params
):

    X_train, X_validation, y_train, y_validation = train_test_split(
        features,
        labels,
        test_size=model_params.test_size,
        random_state=model_params.random_state,
        shuffle=model_params.shuffle
    )

    X_train_scaled = X_train
    X_validation_scaled = X_validation

    rf_model = RandomForestRegressor(
        n_estimators=model_params.n_estimators,
        max_depth=model_params.max_depth,
        min_samples_split=model_params.min_samples_split,
        min_samples_leaf=model_params.min_samples_leaf,
        bootstrap=model_params.bootstrap,
        verbose=model_params.verbose,
        max_features=model_params.max_features,
        n_jobs=model_params.n_jobs,
        random_state=model_params.random_state,
    )

    mae_scorer = make_scorer(nmae, greater_is_better=False)

    kf = KFold(
        n_splits=model_params.n_splits,
        shuffle=model_params.shuffle,
    )

    cross_val_scores = cross_val_score(
        rf_model, X_train_scaled, y_train, cv=kf, scoring=mae_scorer
    )

    avg_cross_val_score = np.mean(cross_val_scores)

    rf_model.fit(X_train_scaled, y_train)
    
    predictions = rf_model.predict(X_validation_scaled)
    mae_rf = mean_absolute_error(y_validation, predictions)
    nmae_rf = nmae(y_validation, predictions)

    return mae_rf, nmae_rf, rf_model