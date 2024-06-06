import numpy as np
from .module import Module

class Linear(Module):
    def __init__(self, in_feat: int, out_feat: int, bias: bool =True, zero_init: bool =False):
        super().__init__()
        self._parameters['weight'] = np.zeros((in_feat, out_feat)) if zero_init else np.random.randn(in_feat, out_feat)
        self._gradients['weight'] = np.zeros((in_feat, out_feat))
        if bias:
            self._parameters['bias'] = np.zeros((1, out_feat)) if zero_init else np.random.randn(1, out_feat)
            self._gradients['bias'] = np.zeros((1, out_feat))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if 'bias' in self._parameters.keys():
            return np.dot(x, self._parameters['weight']) + self._parameters['bias']
        else:
            return np.dot(x, self._parameters['weight'])
        
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        m = grad_out.shape[0]
        self._gradients['weight'] = np.dot(self._buffer.T, grad_out) / m
        self._gradients['bias'] = np.average(grad_out, axis=0, keepdims=True)
        grad_out = np.dot(grad_out, self._parameters['weight'].T)
        return grad_out
        
