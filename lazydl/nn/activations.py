import numpy as np
from .module import Module
from lazydl.func import f

class _Activation(Module):
    def __init__(self):
        super().__init__()
    
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return self.derivative(grad_out) * grad_out


class ReLU(_Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return f.relu(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        x[x <= 0] = 0.0
        x[x > 0] = 1.0
        return x


class Sigmoid(_Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return f.sigmoid(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        tmp = f.sigmoid(x)
        return tmp * (1 - tmp)


class LogSigmoid(_Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return f.logsigmoid(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        tmp = f.sigmoid(x)
        return 1 - tmp
    

class Softmax(_Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return f.softmax(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        # we don't need the whole Jacobian
        # we only need the output wrt its corresponding input
        # which mean... only the i == j case is considered
        s = f.softmax(x)
        return s * (np.ones(s.shape) - s)