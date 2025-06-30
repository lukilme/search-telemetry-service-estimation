import pandas as pd
import numpy as np
from typing import Literal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_missing_values(
    dataset: pd.DataFrame, 
    strategy: Literal["mean", "median", "drop","moda"] = "mean"
) -> pd.DataFrame:
    """
    Trata valores ausentes no dataset.
    """
    missing_count = dataset.isnull().sum().sum()
    
    if missing_count == 0:
        logger.info("Nenhum valor ausente encontrado")
        return dataset
    
    logger.info(f"Valores ausentes encontrados: {missing_count}")
    
    if strategy == "drop":
        dataset_clean = dataset.dropna()
        logger.info(f"Linhas removidas: {len(dataset) - len(dataset_clean)}")
    elif strategy == "mean":
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        dataset_clean = dataset.copy()
        dataset_clean[numeric_cols] = dataset_clean[numeric_cols].fillna(
            dataset_clean[numeric_cols].mean()
        )
        logger.info("Valores ausentes preenchidos com a média")
    elif strategy == "median":
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        dataset_clean = dataset.copy()
        dataset_clean[numeric_cols] = dataset_clean[numeric_cols].fillna(
            dataset_clean[numeric_cols].median()
        )
        logger.info("Valores ausentes preenchidos com a mediana")
    elif strategy == "moda":
        raise ValueError("Estratégia não implementada ainda")
    else:
        raise ValueError(f"Estratégia não suportada: {strategy}")
    
    return dataset_clean
