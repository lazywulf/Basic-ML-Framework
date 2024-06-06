import numpy as np
from lazydl.nn import Module
from typing import Tuple


class _Optimizer:
    def __init__(self, model: Module):
        self.model = model
        
    def zero_grad(self) -> None:
        self.model.zero_grad()

    def step(self) -> None:
        raise NotImplementedError
    

class SGD(_Optimizer):
    def __init__(self, model: Module, lr: float = 1e-3):
        super().__init__(model)
        self.lr = lr

    def step(self) -> None:
        for layer in self.model.get_module_iter():
            for param_name in layer._parameters.keys():
                layer._parameters[param_name] -= self.lr * layer._gradients[param_name]


class MSGD(_Optimizer):
    def __init__(self, model: Module, lr: float = 1e-3, beta: float = 0.9):
        super().__init__(model)
        self.lr = lr
        self.beta = beta
        self.m_iter = model.get_module_iter()

        self.m = [{} for _ in range(len(self.m_iter))]
        for i, layer in enumerate(self.m_iter):
            for param_name in layer._parameters.keys():
                self.m[i][param_name] = np.zeros_like(layer._parameters[param_name])

    def step(self) -> None:
        for i, layer in enumerate(self.m_iter):
            for param_name in layer._parameters.keys():
                grad = layer._gradients[param_name]
                self.m[i][param_name] = self.beta * self.m[i][param_name] + (1 - self.beta) * grad
                m_hat = self.m[i][param_name] / (1 - self.beta)
                layer._parameters[param_name] -= self.lr * m_hat


class AdaGrad(_Optimizer):
    def __init__(self, model: Module, lr: float = 1e-3, eps: float = 1e-8):
        super().__init__(model)
        self.lr = lr
        self.eps = eps
        self.m_iter = model.get_module_iter()
        self.cnt = 0
        
        self.acc_grad = [{} for _ in range(len(self.m_iter))]

        for i, layer in enumerate(self.m_iter):
            for param_name in layer._parameters.keys():
                self.acc_grad[i][param_name] = np.zeros_like(layer._parameters[param_name])

    def step(self) -> None:
        for i, layer in enumerate(self.m_iter):
            for param_name in layer._parameters.keys():
                grad = layer._gradients[param_name]
                self.acc_grad[i][param_name] += grad ** 2
                self.cnt += 1
                adj_lr = self.lr / (np.sqrt(sum(self.acc_grad[i][param_name]) / self.cnt) + self.eps)
                layer._parameters[param_name] -= adj_lr * grad


class RMSProp(_Optimizer):
    def __init__(self, model: Module, lr: float = 1e-3, beta: float = 0.99, eps: float = 1e-8):
        super().__init__(model)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.m_iter = model.get_module_iter()
        
        self.v = [{} for _ in range(len(self.m_iter))]
        for i, layer in enumerate(self.m_iter):
            for param_name in layer._parameters.keys():
                self.v[i][param_name] = np.zeros_like(layer._parameters[param_name])

    def step(self) -> None:
        for i, layer in enumerate(self.m_iter):
            for param_name in layer._parameters.keys():
                grad = layer._gradients[param_name]
                self.v[i][param_name] = self.beta * self.v[i][param_name] + (1 - self.beta) * (grad ** 2)
                v_hat = self.lr / (np.sqrt(self.v[i][param_name]) + self.eps)
                layer._parameters[param_name] -= v_hat * grad


class Adam(_Optimizer):
    def __init__(self, model: Module, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(model)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m_iter = model.get_module_iter()
        
        self.m = [{} for _ in range(len(self.m_iter))]
        self.v = [{} for _ in range(len(self.m_iter))]

        for i, layer in enumerate(self.m_iter):
            for param_name in layer._parameters.keys():
                self.m[i][param_name] = np.zeros_like(layer._parameters[param_name])
                self.v[i][param_name] = np.zeros_like(layer._parameters[param_name])

    def step(self) -> None:
        for i, layer in enumerate(self.m_iter):
            for param_name in layer._parameters.keys():
                grad = layer._gradients[param_name]
                self.m[i][param_name] = self.beta1 * self.m[i][param_name] + (1 - self.beta1) * grad
                self.v[i][param_name] = self.beta2 * self.v[i][param_name] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.m[i][param_name] / (1 - self.beta1)
                v_hat = self.v[i][param_name] / (1 - self.beta2)
                layer._parameters[param_name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

