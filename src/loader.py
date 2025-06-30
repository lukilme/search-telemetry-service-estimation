import pandas as pd
import numpy as np
from typing import Tuple, Optional, Literal, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from scipy import stats
import logging
from src.normalize import normalize_features
# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_datasets(
    dataset_name: Literal["sinusoid", "mix", "flashcrowd"] = "sinusoid",
    sample_fraction: Optional[float] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os datasets de telemetria de rede e DASH com opção de amostragem.
    """
    dataset_configs = {
        "sinusoid": "sinusoid_8h",
        "mix": "mix_5h", 
        "flashcrowd": "flashcrowd_6h"
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset '{dataset_name}' não suportado. "
                        f"Opções: {list(dataset_configs.keys())}")
    
    if sample_fraction is not None and (sample_fraction <= 0 or sample_fraction > 1):
        raise ValueError("sample_fraction deve estar entre 0 e 1")
    
    source_data = dataset_configs[dataset_name]
    
    try:
        log_path = f"assets/data/log_INT_{source_data}.txt"
        data_log = pd.read_csv(log_path, delimiter=",")
        data_log.columns = data_log.columns.str.replace(" ", "")
        

        dash_path = f"assets/data/dash_{source_data}.log"
        data_dash = pd.read_csv(dash_path, sep=",")

        if sample_fraction is not None:
            original_log_size = len(data_log)
            original_dash_size = len(data_dash)
            
            data_log = data_log.sample(frac=sample_fraction, random_state=random_state)
            data_dash = data_dash.sample(frac=sample_fraction, random_state=random_state)
            
            logger.info(f"Amostragem aplicada ({sample_fraction*100:.1f}%):")
            logger.info(f"  Log: {original_log_size} → {len(data_log)} linhas")
            logger.info(f"  DASH: {original_dash_size} → {len(data_dash)} linhas")
        
        logger.info(f"Datasets carregados: {dataset_name}")
        logger.info(f"Log shape: {data_log.shape}, DASH shape: {data_dash.shape}")
        
        return data_log, data_dash
        
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        
        
def merge_datasets(
    data_log: pd.DataFrame,
    data_dash: pd.DataFrame,
    join_type: Literal["inner", "left", "right", "outer"] = "inner",
    normalization_method: Literal["standard", "minmax", "robust", "quantile", "power"] = "standard"
) -> Tuple[np.ndarray, np.ndarray, Union[StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer]]:
    """
    Combina datasets de telemetria e DASH, preparando para ML.
    """
    required_cols = ['timestamp']
    for col in required_cols:
        if col not in data_log.columns:
            raise KeyError(f"Coluna '{col}' não encontrada em data_log")
        if col not in data_dash.columns:
            raise KeyError(f"Coluna '{col}' não encontrada em data_dash")
    
    if 'framesDisplayedCalc' not in data_dash.columns:
        raise KeyError("Coluna 'framesDisplayedCalc' não encontrada em data_dash")
    
    logger.info(f"Realizando merge ({join_type}) dos datasets...")
    total = data_log.merge(data_dash, on='timestamp', how=join_type)
    
    if len(total) == 0:
        raise ValueError("Merge resultou em dataset vazio")
    
 
    initial_rows = len(total)
    total = total.dropna()
    final_rows = len(total)
    
    if final_rows < initial_rows:
        logger.info(f"Removidas {initial_rows - final_rows} linhas com NaN após merge")
    

    feature_end_idx = len(data_log.columns) - 1  
    features = total.iloc[:, 1:feature_end_idx + 1].values
    labels = total['framesDisplayedCalc'].values
    
    features_normalized, scaler = normalize_features(features, method=normalization_method)
    
    logger.info(f"Dataset final: {len(total)} amostras, {features.shape[1]} features")
    logger.info(f"Normalização aplicada: {normalization_method}")
    
    return features_normalized, labels, scaler