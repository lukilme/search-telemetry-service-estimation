import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def normalized_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Erro Absoluto Médio Normalizado (NMAE).
    """
    if np.mean(y_true) == 0:
        raise ValueError("Divisão por zero: média dos valores verdadeiros é zero")
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae / np.mean(y_true)

def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula a Raiz do Erro Quadrático Médio Normalizado (NRMSE).
    """
    if np.mean(y_true) == 0:
        raise ValueError("Divisão por zero: média dos valores verdadeiros é zero")
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / np.mean(y_true)