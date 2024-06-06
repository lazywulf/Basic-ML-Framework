from __future__ import annotations
import numpy as np
from numpy.linalg import svd
import pickle
from typing import Union


class PCA:
    def __init__(self, n_components: Union[int, float] = 1):
        self._n_comp = n_components
        self.component_count: int = ...
        self.sample_count: int = ...
        self.feature_count: int = ...
        self.covariance: np.ndarray = ...
        self.components: np.ndarray = ...
        self.singular_values: np.ndarray = ...
        self.explained_variance: np.ndarray = ...
        self.explained_variance_ratio: np.ndarray = ...

    def fit(self, X: np.ndarray):
        self.sample_count = X.shape[0]
        self.feature_count = X.shape[1]

        """
        Steps to do PCA
        1. Move data and make the mean == 0
        2. Calculate the covariance matrix
        3. Get eigenvector and eigenvalue of the covariance matrix
        4. Project data onto PCs

        Because we do SVD using the convariance matrix, not the data, so the singular value we get are the explained variances.
        (Unlike what sklearn did in their library: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/decomposition/_pca.py#L430)

        So, to retrieve the sigular values of the original data, we'll need to reverse what they did.

        $C = (1/n) X^T X = (1/n) \cdot (V \Sigma U^T)(U \Sigma V^T) = V ((1/n) \cdot \Sigma^2) V^T$  
        """

        self.covariance = np.cov(X, rowvar=False)
        V, sing, Vt = svd(self.covariance)

        ## why this?
        ## we need to make the result of svd deterministic
        V, Vt = self._svd_flip(V, Vt)
        
        self.explained_variance = sing
        self.explained_variance_ratio = self.explained_variance / np.sum(self.explained_variance)

        if isinstance(self._n_comp, int):
            self.component_count = self._n_comp
        elif isinstance(self._n_comp, float): 
            n, c = 0, 0
            for p in self.explained_variance_ratio:
                c += p
                n += 1
                if c > self._n_comp:
                    break
            self.component_count = n
        else:
            raise TypeError(f'Expect type "int" or "float", got {self._n_comp.__class__}')    

        self.explained_variance = self.explained_variance[:self.component_count]
        self.explained_variance_ratio = self.explained_variance_ratio[:self.component_count]
        self.singular_values = np.sqrt(sing * (self.sample_count - 1))[:self.component_count]
        self.components = V[:, :self.component_count]
        
        return self
    
    def fit_transform(self, X):
        self.fit(X)
        return np.dot(X, self.components)

    def transform(self, X):
        return np.dot(X, self.components)
    
    def _svd_flip(self, u, v):
        ## I don't really know how to explain this. :(
        ## It just works.
        ## A modification of sklearn.decomposition.PCA._svd_flip()
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = -np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u, v
    
    def save_model(self, save_file: str) -> None:
        with open(save_file, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(param_file: str) -> PCA:
        with open(param_file, 'rb') as inp:
            return pickle.load(inp)

