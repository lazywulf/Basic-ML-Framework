import numpy as np
from typing import Optional, List, Union, Sequence

class f:
    @staticmethod
    def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
        # overflowed... again
        # the following is equiv to the original softmax function
        tmp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return tmp / np.sum(tmp, axis=axis, keepdims=True)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # fix to the exp overflow problem
        # the two functions are the same, but the second one avoids kaboom-ing np.exp()
        y = x.copy()
        y[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
        y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
        return y

    @staticmethod
    def logsigmoid(x: np.ndarray) -> np.ndarray:
        return np.log(f.sigmoid(x))

    @staticmethod
    def logsoftmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
        return np.log(f.softmax(x, axis=axis))
    
    @staticmethod
    def onehot(y: np.ndarray, categories: Optional[Sequence] = None) -> np.ndarray:
        y = y.flatten()
        if categories is not None:
            cat_dict = {x: i for i, x in enumerate(categories)}
            num = len(cat_dict)
        else:
            labels = set(y)
            cat_dict = {label: label for label in labels}
            num = max(labels) + 1

        oh = np.zeros((y.shape[0], num), dtype=np.float64)
        indices = np.array([cat_dict[label] for label in y])
        oh[np.arange(y.shape[0]), indices] = 1.0
        return oh
    
    @staticmethod
    def minmaxnorm(x: np.ndarray, axis: int = 0) -> np.ndarray:
        min_vals = np.min(x, axis=axis, keepdims=True)
        max_vals = np.max(x, axis=axis, keepdims=True)
        normalized = (x - min_vals) / (max_vals - min_vals)
        return normalized
