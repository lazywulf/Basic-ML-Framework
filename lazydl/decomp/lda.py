from __future__ import annotations
import numpy as np
from numpy.linalg import svd
import pickle


class LDA:
    def __init__(self, n_comp: int = 1):
        self.component_count = n_comp
        self.scalings: np.ndarray  = None

    def fit(self, X: np.ndarray, T: np.ndarray):
        T = np.asarray(T).flatten()

        unique_cls = np.unique(T)
        feat_cnt = X.shape[1]

        mean_all = np.mean(X, axis=0)
        sw = np.zeros((feat_cnt, feat_cnt))
        sb = np.zeros((feat_cnt, feat_cnt))

        for cls in unique_cls:
            sample_from_cls = X[T == cls]
            mean_cls = np.mean(sample_from_cls, axis=0)
            
            sw += np.dot((sample_from_cls - mean_cls).T, (sample_from_cls - mean_cls))

            cls_sample_count = sample_from_cls.shape[0]
            mean_diff = (mean_cls - mean_all).reshape((-1, 1))
            sb += cls_sample_count * np.dot(mean_diff, mean_diff.T)

        sw_inv = np.linalg.inv(sw)
        U, S, Vt= svd(sw_inv @ sb)
        U, Vt = self._svd_flip(U, Vt)
        self.scalings = U[:, : self.component_count]
        self.means = {cls: np.mean(X[T == cls], axis=0) for cls in unique_cls}
        return self
        
    def fit_transform(self, X, T):
        self.fit(X, T)
        return np.dot(X, self.scalings)

    def transform(self, X):
        return np.dot(X, self.scalings)
    
    def _svd_flip(self, u, v):
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = -np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u, v
    
    def predict(self, X: np.ndarray):
        X_transformed = self.transform(X)
        distances = []

        # for cls, mean in self.means.items():
        #     mean_transformed = mean @ self.scalings
        #     distances.append(np.linalg.norm(X_transformed - mean_transformed, axis=1))
        # distances = np.array(distances)
        # predictions = np.argmin(distances, axis=0)
        
        for cls, mean in self.means.items():
            mean_transformed = mean @ self.scalings
            distance = np.sqrt(((X_transformed - mean_transformed) ** 2).sum(axis=1))
            distances.append(distance)
        distances = np.array(distances)
        predictions = [min(list(range(len(distances))), key=lambda x: d[x]) for d in distances.T]
        
        unique_classes = list(self.means.keys())
        predicted_labels = np.array([unique_classes[p] for p in predictions])
        return predicted_labels
    
    def save_model(self, save_file: str) -> None:
        with open(save_file, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(param_file: str) -> LDA:
        with open(param_file, 'rb') as inp:
            return pickle.load(inp)
        

