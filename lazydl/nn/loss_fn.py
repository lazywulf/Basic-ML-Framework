import numpy as np
from .module import Module
from lazydl.func import f


class _Loss(Module):
    def __init__(self, target_module):
        super().__init__()
        self.target_module: Module = target_module
        self.loss = None

    def __call__(self, predictions: np.ndarray, targets:np.ndarray) -> np.ndarray:
        self.loss = self.derivative(predictions, targets)
        return self.forward(predictions, targets)

    def backward(self):
        self.target_module._backprop(self.loss)
    

class SimpleLoss(_Loss):
    def __init__(self, target_module):
        super().__init__(target_module)

    def forward(self, predictions: np.ndarray, targets:np.ndarray) -> np.ndarray:
        return np.mean((predictions - targets) ** 2)
    
    def derivative(self, predictions: np.ndarray, targets:np.ndarray) -> np.ndarray:
        tmp = predictions - targets
        return 2 * tmp

class RMSELoss(_Loss):
    def __init__(self, target_module):
        super().__init__(target_module)

    def forward(self, predictions: np.ndarray, targets:np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean((predictions - targets) ** 2))
    
    def derivative(self, predictions: np.ndarray, targets:np.ndarray) -> np.ndarray:
        tmp, m = predictions - targets, targets.shape[0]
        # an attempt to fix the invalid number problem
        return 2 * (1 / np.sqrt(m)) *  np.divide(tmp, np.abs(tmp) + 1e-10)


class MSELoss(_Loss):
    def __init__(self, target_module):
        super().__init__(target_module)

    def forward(self, predictions: np.ndarray, targets:np.ndarray) -> np.ndarray:
        return np.mean((predictions - targets) ** 2)
    
    def derivative(self, predictions: np.ndarray, targets:np.ndarray) -> np.ndarray:
        tmp, m = predictions - targets, targets.shape[0]
        return 2 * tmp / m


class CrossEntropyLoss(_Loss):
    def __init__(self, target_module):
        super().__init__(target_module)

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        m, x, t = targets.shape[0], f.softmax(predictions), f.onehot(targets, np.arange(predictions.shape[1]))
        return -np.sum(t * np.log(x + 1e-12)) / m
    
    def derivative(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        m, x, t = targets.shape[0], f.softmax(predictions), f.onehot(targets, np.arange(predictions.shape[1]))
        return (x - t) / m
