from __future__ import annotations
import numpy as np
import pickle


class MaxMinNorm:
    def __init__(self):
        self._max: float = ...
        self._min: float = ...

    def fit(self, X: np.ndarray):
        self._max, self._min = X.max(axis=0), X.min(axis=0)
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return (X - self._min) / (self._max - self._min)
    
    def transform(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self._min) / (self._max - self._min)
    
    def save_model(self, save_file: str) -> None:
        with open(save_file, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(param_file: str) -> MaxMinNorm:
        with open(param_file, 'rb') as inp:
            return pickle.load(inp)

