class EarlyStop:
    def __init__(self,  patience: int = 10, delta: float = 0.0001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric: float):
        if self.best_score is None:
            self.best_score = metric
        elif metric > self.best_score - self.delta:
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.counter = 0
