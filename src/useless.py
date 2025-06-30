import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_constant_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas com valores constantes (sem variabilidade).
    """
    initial_cols = len(dataset.columns)
    constant_cols = dataset.columns[dataset.nunique() == 1]
    
    dataset_clean = dataset.drop(columns=constant_cols)
    
    if len(constant_cols) > 0:
        logger.info(f"Removidas {len(constant_cols)} colunas constantes: {list(constant_cols)}")
    
    logger.info(f"Colunas: {initial_cols} → {len(dataset_clean.columns)}")
    return dataset_clean