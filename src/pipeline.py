import pandas as pd
import numpy as np
from src.outlier import remove_outliers
from src.loader import load_datasets
from src.normalize import normalize_features
from src.loader import merge_datasets
from src.useless import remove_constant_features
from src.missing import handle_missing_values
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class INTDataProcessor:
    """Classe para processamento completo dos dados INT."""
    
    def __init__(self):
        self.scaler = None
        self.processed_data = None
    
    def process_pipeline(
        self,
        dataset_name: str = "sinusoid",
        remove_outliers_strategy: str = "iqr",
        missing_strategy: str = "mean",
        join_type: str = "inner"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pipeline completo de processamento dos dados.
        """
        logger.info("Iniciando pipeline de processamento...")

        data_log, data_dash = load_datasets(dataset_name)
        

        data_log = remove_constant_features(data_log)
        data_dash = remove_constant_features(data_dash)
        
        data_log = handle_missing_values(data_log, missing_strategy)
        data_dash = handle_missing_values(data_dash, missing_strategy)
        
        data_dash = remove_outliers(data_dash, remove_outliers_strategy)
        data_log = remove_outliers(data_log, remove_outliers_strategy)

        features, labels, self.scaler = merge_datasets(data_log, data_dash, join_type)
        
        self.processed_data = (features, labels)
        logger.info("Pipeline concluído com sucesso!")
        
        return features, labels