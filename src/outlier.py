import pandas as pd
import numpy as np
from typing import Literal
from sklearn.ensemble import IsolationForest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
        Remove outliers usando o método IQR (Interquartile Range)
    """
    initial_rows = len(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        logger.warning("Nenhuma coluna numérica encontrada para remoção de outliers")
        return df
    
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    mask = ~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)
    df_clean = df[mask]
    
    removed_rows = initial_rows - len(df_clean)
    logger.info(f"Outliers IQR removidos: {removed_rows} linhas ({removed_rows/initial_rows*100:.1f}%)")
    
    return df_clean


def remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers usando Z-Score.
    """
    initial_rows = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        logger.warning("Nenhuma coluna numérica encontrada para remoção de outliers")
        return df

    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    
    mask = (z_scores < threshold).all(axis=1)
    df_clean = df[mask]
    
    removed_rows = initial_rows - len(df_clean)
    logger.info(f"Outliers Z-Score removidos: {removed_rows} linhas ({removed_rows/initial_rows*100:.1f}%)")
    
    return df_clean


def remove_outliers_modified_zscore(df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    Remove outliers usando Modified Z-Score (baseado na mediana).
    """
    initial_rows = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        logger.warning("Nenhuma coluna numérica encontrada para remoção de outliers")
        return df
    
    def modified_z_score(data):
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return 0.6745 * (data - median) / mad

    modified_z_scores = df[numeric_cols].apply(lambda x: np.abs(modified_z_score(x)))
    
    mask = (modified_z_scores < threshold).all(axis=1)
    df_clean = df[mask]
    
    removed_rows = initial_rows - len(df_clean)
    logger.info(f"Outliers Modified Z-Score removidos: {removed_rows} linhas ({removed_rows/initial_rows*100:.1f}%)")
    
    return df_clean


def remove_outliers_isolation_forest(
    df: pd.DataFrame, 
    contamination: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Remove outliers usando Isolation Forest.
    """
    initial_rows = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        logger.warning("Nenhuma coluna numérica encontrada para remoção de outliers")
        return df
    
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(df[numeric_cols])

    mask = outlier_labels == 1
    df_clean = df[mask]
    
    removed_rows = initial_rows - len(df_clean)
    logger.info(f"Outliers Isolation Forest removidos: {removed_rows} linhas ({removed_rows/initial_rows*100:.1f}%)")
    
    return df_clean


def remove_outliers_lof(
    df: pd.DataFrame, 
    n_neighbors: int = 20,
    contamination: float = 0.1
) -> pd.DataFrame:
    """
    Remove outliers usando Local Outlier Factor (LOF).
    """
    initial_rows = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        logger.warning("Nenhuma coluna numérica encontrada para remoção de outliers")
        return df

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(df[numeric_cols])

    mask = outlier_labels == 1
    df_clean = df[mask]
    
    removed_rows = initial_rows - len(df_clean)
    logger.info(f"Outliers LOF removidos: {removed_rows} linhas ({removed_rows/initial_rows*100:.1f}%)")
    
    return df_clean


def remove_outliers_percentile(df: pd.DataFrame, lower_percentile: float = 5, upper_percentile: float = 95) -> pd.DataFrame:
    """
    Remove outliers baseado em percentis.
    """
    initial_rows = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        logger.warning("Nenhuma coluna numérica encontrada para remoção de outliers")
        return df
    
    lower_bounds = df[numeric_cols].quantile(lower_percentile / 100)
    upper_bounds = df[numeric_cols].quantile(upper_percentile / 100)
  
    mask = ((df[numeric_cols] >= lower_bounds) & (df[numeric_cols] <= upper_bounds)).all(axis=1)
    df_clean = df[mask]
    
    removed_rows = initial_rows - len(df_clean)
    logger.info(f"Outliers Percentil removidos: {removed_rows} linhas ({removed_rows/initial_rows*100:.1f}%)")
    
    return df_clean


def remove_outliers(
    df: pd.DataFrame,
    method: Literal["iqr", "zscore", "modified_zscore", "isolation_forest", "lof", "percentile"] = "iqr",
    **kwargs
) -> pd.DataFrame:
    """
    Remove outliers usando o método especificado.
    """
    method_functions = {
        "iqr": remove_outliers_iqr,
        "zscore": remove_outliers_zscore,
        "modified_zscore": remove_outliers_modified_zscore,
        "isolation_forest": remove_outliers_isolation_forest,
        "lof": remove_outliers_lof,
        "percentile": remove_outliers_percentile
    }
    
    if method not in method_functions:
        raise ValueError(f"Método '{method}' não suportado. Opções: {list(method_functions.keys())}")
    
    return method_functions[method](df, **kwargs)