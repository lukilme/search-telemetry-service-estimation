import pandas as pd
from typing import Tuple

class rf_model_params:
    def __init__(self, 
                 test_size: float,
                 random_state: int = None,
                 n_estimators: int = 100,
                 max_features: str = 'sqrt',
                 max_depth: int = None,
                 min_samples_leaf: int = 1,
                 min_samples_split: int = 2,
                 bootstrap: bool = True,
                 verbose: int = 0,
                 n_jobs: int = 2,
                 n_splits: int = 5,
                 shuffle: bool = False):
        '''
        Inicializa os parâmetros para o modelo Random Forest.

        Parâmetros:
        - test_size (float): Proporção do conjunto de dados a ser usada como teste.
        - random_state (int): Semente para o gerador de números aleatórios.
        - n_estimators (int): Número de árvores na floresta.
        - max_features (str): O número de features a serem consideradas ao procurar a melhor divisão.
        - max_depth (int): Profundidade máxima da árvore. Se None, os nós serão expandidos até que todas as folhas sejam puras.
        - min_samples_leaf (int): Número mínimo de amostras que devem estar presentes em um nó folha.
        - min_samples_split (int): Número mínimo de amostras necessárias para dividir um nó.
        - bootstrap (bool): Se True, as amostras são extraídas com substituição.
        - verbose (int): Controle de verbosidade. O valor 0 significa nenhuma saída.
        - n_jobs (int): Número de jobs a serem executados em paralelo (-1 significa usar todos os processadores).
        - n_splits (int): Número de splits para validação cruzada.
        - shuffle (bool): Se True, embaralha os dados antes de dividi-los.
        '''
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.shuffle = shuffle
    
    def __str__(self):
        return (f"rf_model_params("
                f"test_size={self.test_size}, "
                f"random_state={self.random_state}, "
                f"n_estimators={self.n_estimators}, "
                f"max_features='{self.max_features}', "
                f"max_depth={self.max_depth}, "
                f"min_samples_leaf={self.min_samples_leaf}, "
                f"min_samples_split={self.min_samples_split}, "
                f"bootstrap={self.bootstrap}, "
                f"verbose={self.verbose}, "
                f"n_jobs={self.n_jobs}, "
                f"n_splits={self.n_splits}, "
                f"shuffle={self.shuffle})")

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Literal
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InBandNetworkTelemetry:
    """
    Schema do dataset INT (In-Band Network Telemetry)
    
    Representa os campos de telemetria de rede coletados pelos switches P4.
    """
    FIELDS = {
        'switchID_t': 31,           # ID do switch
        'ingress_port': 9,          # Porta de entrada
        'egress_port': 9,           # Porta de saída
        'egress_spec': 9,           # Especificação de saída
        'ingress_global_timestamp': 48,  # Timestamp global de entrada
        'egress_global_timestamp': 48,   # Timestamp global de saída
        'enq_timestamp': 32,        # Timestamp de enfileiramento
        'enq_qdepth': 19,          # Profundidade da fila no enfileiramento
        'deq_timedelta': 32,       # Delta de tempo no desenfileiramento
        'deq_qdepth': 19           # Profundidade da fila no desenfileiramento
    }

def normalized_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    if np.mean(y_true) == 0:
        raise ValueError("Divisão por zero: média dos valores verdadeiros é zero")
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae / np.mean(y_true)


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.mean(y_true) == 0:
        raise ValueError("Divisão por zero: média dos valores verdadeiros é zero")
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / np.mean(y_true)


def load_datasets(
    dataset_name: Literal["sinusoid", "mix", "flashcrowd"] = "sinusoid"
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    dataset_configs = {
        "sinusoid": "sinusoid_8h",
        "mix": "mix_5h", 
        "flashcrowd": "flashcrowd_6h"
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset '{dataset_name}' não suportado. "
                        f"Opções: {list(dataset_configs.keys())}")
    
    source_data = dataset_configs[dataset_name]
    
    try:

        log_path = f"assets/data/log_INT_{source_data}.txt"
        data_log = pd.read_csv(log_path, delimiter=",")
        data_log.columns = data_log.columns.str.replace(" ", "")
 
        dash_path = f"assets/data/dash_{source_data}.log"
        data_dash = pd.read_csv(dash_path, sep=",")
        
        logger.info(f"Datasets carregados: {dataset_name}")
        logger.info(f"Log shape: {data_log.shape}, DASH shape: {data_dash.shape}")
        
        return data_log, data_dash
        
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        raise


def remove_constant_features(dataset: pd.DataFrame) -> pd.DataFrame:

    initial_cols = len(dataset.columns)
    constant_cols = dataset.columns[dataset.nunique() == 1]
    
    dataset_clean = dataset.drop(columns=constant_cols)
    
    if len(constant_cols) > 0:
        logger.info(f"Removidas {len(constant_cols)} colunas constantes: {list(constant_cols)}")
    
    logger.info(f"Colunas: {initial_cols} → {len(dataset_clean.columns)}")
    return dataset_clean


def remove_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:

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
    logger.info(f"Outliers removidos: {removed_rows} linhas ({removed_rows/initial_rows*100:.1f}%)")
    
    return df_clean


def handle_missing_values(
    dataset: pd.DataFrame, 
    strategy: Literal["mean", "median", "drop"] = "mean"
) -> pd.DataFrame:

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
    else:
        raise ValueError(f"Estratégia não suportada: {strategy}")
    
    return dataset_clean


def normalize_features(features: np.ndarray, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:

    if scaler is None:
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        logger.info("Novo scaler criado e ajustado")
    else:
        features_normalized = scaler.transform(features)
        logger.info("Scaler existente aplicado")
    
    return features_normalized, scaler


def merge_datasets(
    data_log: pd.DataFrame,
    data_dash: pd.DataFrame,
    join_type: Literal["inner", "left", "right", "outer"] = "inner"
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:

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
    
    # Extrai features (exclui timestamp e target)
    feature_end_idx = len(data_log.columns) - 1  # -1 para excluir timestamp
    features = total.iloc[:, 1:feature_end_idx + 1].values
    labels = total['framesDisplayedCalc'].values
    
    # Normaliza features
    features_normalized, scaler = normalize_features(features)
    
    logger.info(f"Dataset final: {len(total)} amostras, {features.shape[1]} features")
    
    return features_normalized, labels, scaler


# Classe utilitária para pipeline completo
class INTDataProcessor:
    """Classe para processamento completo dos dados INT."""
    
    def __init__(self):
        self.scaler = None
        self.processed_data = None
    
    def process_pipeline(
        self,
        dataset_name: str = "sinusoid",
        remove_outliers: bool = True,
        missing_strategy: str = "mean",
        join_type: str = "inner"
    ) -> Tuple[np.ndarray, np.ndarray]:

        logger.info("Iniciando pipeline de processamento...")
        
        data_log, data_dash = load_datasets(dataset_name)
        
        data_log = remove_constant_features(data_log)
        data_dash = remove_constant_features(data_dash)
        
        data_log = handle_missing_values(data_log, missing_strategy)
        data_dash = handle_missing_values(data_dash, missing_strategy)
        
        if remove_outliers:
            data_log = remove_outliers_iqr(data_log)
            data_dash = remove_outliers_iqr(data_dash)
        
        # 3. Merge e normalização
        features, labels, self.scaler = merge_datasets(data_log, data_dash, join_type)
        
        self.processed_data = (features, labels)
        logger.info("Pipeline concluído com sucesso!")
        
        return features, labels