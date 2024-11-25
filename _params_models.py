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



class xgb_model_params:
    def __init__(self, 
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_child_weight: int = 1,
                 gamma: float = 0,
                 subsample: float = 1,
                 colsample_bytree: float = 1,
                 colsample_bylevel: float = 1,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 scale_pos_weight: float = 1,
                 random_state: int = None,
                 verbose: int = 0,
                 n_jobs: int = 2
                 ):
        '''
        Inicializa os parâmetros para o modelo XGBoost.

        Parâmetros:
        - learning_rate (float): Taxa de aprendizado do modelo.
        - n_estimators (int): Número de árvores a serem construídas.
        - max_depth (int): Profundidade máxima da árvore.
        - min_child_weight (int): Peso mínimo da criança; pode ser usado para regularização.
        - gamma (float): Redução mínima da perda para fazer uma divisão adicional.
        - subsample (float): Fração de amostras a serem usadas para treinar cada árvore.
        - colsample_bytree (float): Fração de colunas a serem usadas para cada árvore.
        - colsample_bylevel (float): Fração de colunas a serem usadas em cada nível.
        - reg_alpha (float): Termo de regularização L1.
        - reg_lambda (float): Termo de regularização L2.
        - scale_pos_weight (float): Controla o balanço entre classes em dados desbalanceados.
        - random_state (int): Semente para o gerador de números aleatórios.
        - verbose (int): Controle de verbosidade. O valor 0 significa nenhuma saída.
        '''
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
