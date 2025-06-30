import numpy as np
from typing import Tuple, Optional, Literal, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_power(
            features: np.ndarray,
            scaler: Optional[PowerTransformer] = None) -> Tuple[np.ndarray, PowerTransformer]:
    """
    Normalização Power (Box-Cox) - transforma para distribuição mais normal.
    """
    if scaler is None:
        if np.any(features <= 0):
            logger.warning(
                "Box-Cox requer valores positivos. Aplicando shift para tornar dados positivos.")
            features = features + np.abs(features.min()) + 1

        scaler = PowerTransformer(method='box-cox')
        features_normalized = scaler.fit_transform(features)
        logger.info("PowerTransformer (Box-Cox) criado e ajustado")
    else:
        features_normalized = scaler.transform(features)
        logger.info("PowerTransformer (Box-Cox) existente aplicado")

    return features_normalized, scaler

def normalize_standard(features: np.ndarray, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalização Z-Score (StandardScaler) - média 0, desvio padrão 1.
    """
    if scaler is None:
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        logger.info("StandardScaler criado e ajustado")
    else:
        features_normalized = scaler.transform(features)
        logger.info("StandardScaler existente aplicado")
    
    return features_normalized, scaler


def normalize_minmax(features: np.ndarray, scaler: Optional[MinMaxScaler] = None, feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalização Min-Max - escala para um intervalo específico.
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
        features_normalized = scaler.fit_transform(features)
        logger.info(f"MinMaxScaler criado e ajustado para intervalo {feature_range}")
    else:
        features_normalized = scaler.transform(features)
        logger.info("MinMaxScaler existente aplicado")
    
    return features_normalized, scaler


def normalize_robust(features: np.ndarray, scaler: Optional[RobustScaler] = None) -> Tuple[np.ndarray, RobustScaler]:
    """
    Normalização Robusta - usa mediana e IQR, menos sensível a outliers.
    """
    if scaler is None:
        scaler = RobustScaler()
        features_normalized = scaler.fit_transform(features)
        logger.info("RobustScaler criado e ajustado")
    else:
        features_normalized = scaler.transform(features)
        logger.info("RobustScaler existente aplicado")
    
    return features_normalized, scaler


def normalize_quantile(features: np.ndarray, scaler: Optional[PowerTransformer] = None) -> Tuple[np.ndarray, PowerTransformer]:
    """
    Normalização Quantile - transforma para distribuição uniforme.
    """
    if scaler is None:
        scaler = PowerTransformer(method='quantile')
        features_normalized = scaler.fit_transform(features)
        logger.info("QuantileTransformer criado e ajustado")
    else:
        features_normalized = scaler.transform(features)
        logger.info("QuantileTransformer existente aplicado")
    
    return features_normalized, scaler

def normalize_features(
    features: np.ndarray, 
    method: Literal["standard", "minmax", "robust", "quantile", "power"] = "standard",
    scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer]] = None
) -> Tuple[np.ndarray, Union[StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer]]:
    """
    Normaliza as features usando diferentes métodos de escalonamento.
    """
    scaler_classes = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "quantile": lambda: PowerTransformer(method='quantile'),
        "power": lambda: PowerTransformer(method='box-cox')
    }
    
    if method not in scaler_classes:
        raise ValueError(f"Método '{method}' não suportado. Opções: {list(scaler_classes.keys())}")
    
    if scaler is None:
        if method in ["quantile", "power"]:
            scaler = scaler_classes[method]()
        else:
            scaler = scaler_classes[method]()
        
        if method == "power":
            if np.any(features <= 0):
                logger.warning("Box-Cox requer valores positivos. Aplicando shift para tornar dados positivos.")
                features = features + np.abs(features.min()) + 1
        
        features_normalized = scaler.fit_transform(features)
        logger.info(f"Novo scaler {method} criado e ajustado")
    else:
        features_normalized = scaler.transform(features)
        logger.info(f"Scaler {method} existente aplicado")
    
    return features_normalized, scaler
