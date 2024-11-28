from _params_models import *
from _util import nmae
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


def default_random_forest_model(
    features: pd.DataFrame, labels: pd.Series, model_params
):
    # Divisão de treino e validação
    X_train, X_validation, y_train, y_validation = train_test_split(
        features,
        labels,
        test_size=model_params.test_size,
        random_state=model_params.random_state,
        shuffle=model_params.shuffle,
    )

    # Padronização (se necessário)
    X_train_scaled = X_train
    X_validation_scaled = X_validation

    # Definição do modelo
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

    # Definição do scorer para NMAE
    nmae_scorer = make_scorer(nmae, greater_is_better=False)

    # K-Fold
    if(model_params.shuffle):
        kf = KFold(
            n_splits=model_params.n_splits,
            shuffle=model_params.shuffle,
            random_state=model_params.random_state,
        )
    else:
        kf = KFold(
            n_splits=model_params.n_splits,
            shuffle=model_params.shuffle,
        )
    # Cross-Validation
    cross_val_scores = cross_val_score(
        rf_model, X_train_scaled, y_train, cv=kf, scoring=nmae_scorer
    )

    avg_cross_val_score = np.mean(cross_val_scores)


    rf_model.fit(X_train_scaled, y_train)

    # Previsões
    predictions = rf_model.predict(X_validation_scaled)
    mae_rf = mean_absolute_error(y_validation, predictions)
    nmae_rf = nmae(y_validation, predictions)

    return mae_rf, nmae_rf, rf_model



def default_rf_model_randomsearch(
    features_train: pd.DataFrame, labels_train: pd.Series, iter_n=50
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
        "n_estimators": [70, 80, 90, 100, 110, 120, 130, 140, 150],
        "max_features": ["log2", "sqrt"],
        "max_depth": [28, 37, 46, 55, 64, 73, 82, 91, 100, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
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


    random_search.fit(X_train_scaled, y_train)

    best_rf_model = random_search.best_estimator_
    best_params = random_search.best_params_
    print(f"Melhores hiperparâmetros: {best_params}")

    y_pred_rf = best_rf_model.predict(X_validation_scaled)
    mae_rf = mean_absolute_error(y_validation, y_pred_rf)
    nmae_rf = nmae(y_validation, y_pred_rf)

    feature_importances = best_rf_model.feature_importances_
    print(f"Importâncias das features: {feature_importances}")

    return nmae_rf, best_rf_model, best_params