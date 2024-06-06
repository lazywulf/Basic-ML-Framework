from __future__ import annotations
from typing import Any, Dict, Union, List, Iterable
import numpy as np
import pickle


class Module:
    def __init__(self) -> None:
        self._parameters: Dict[str, np.ndarray] = {}
        self._gradients: Dict[str, np.ndarray] = {}
        self._modules: Union[Dict[str, Module], List[Module]] = []
        self._buffer: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._buffer = x
        if isinstance(self._modules, dict):
            for l in self._modules.values():
                x = l(x)
        elif isinstance(self._modules, list):
            for l in self._modules:
                x = l(x)
        else: 
            pass
        x = self.forward(x)
        return x

    def _backprop(self, grad_out: np.ndarray) -> np.ndarray:
        grad_out = self.backward(grad_out)
        if isinstance(self._modules, dict):
            for l in reversed(self._modules.values()):
                grad_out = l.backward(grad_out)
        elif isinstance(self._modules, list):
            for l in reversed(self._modules):
                grad_out = l.backward(grad_out)
        else: 
            pass
        return grad_out

    def zero_grad(self) -> None: 
        for grad in self._gradients.values():
            grad.fill(0)

        for m in self.get_module_iter()[1:]:
            m.zero_grad()

    def has_grad(self) -> bool:
        return self._gradients.items() != 0
    
    def get_module_iter(self) -> Iterable[Module]:
        m_iter = [self]
        if isinstance(self._modules, dict):
            m_iter += self._modules.values()
        elif isinstance(self._modules, list):
            m_iter += self._modules
        else:
            raise TypeError(f'Expect dict or list, got "{self._modules.__class__}"')
        
        return m_iter

    def save_model(self, save_file: str) -> None:
        with open(save_file, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(param_file: str) -> Module:
        with open(param_file, 'rb') as inp:
            return pickle.load(inp)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out