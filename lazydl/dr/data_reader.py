import os
import pandas as pd
import numpy as np
import cv2 
from typing import Union, List, Iterable, Optional, Any

class Dataset:
    def __init__(self, 
                 labels: Union[List, np.ndarray, pd.DataFrame], 
                 targets: Union[List, np.ndarray, pd.DataFrame] = None):
        self.labels = pd.DataFrame(labels)
        self.targets = pd.DataFrame(targets) if targets is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.targets is not None:
            return np.asarray(self.labels.iloc[idx]), np.asarray(self.targets.iloc[idx])
        else:
            return np.asarray(self.labels.iloc[idx])


class IRISDataset(Dataset):
    def __init__(self, label_file: str, target_file: str = None):
        labels = pd.read_csv(label_file, header=None)
        targets = pd.read_csv(target_file, header=None) if target_file else None
        super().__init__(labels, targets)
        self.targets -= 1 # ['1', '2', '3'] => [0, 1, 2]

class ORLDataset(Dataset):
    def __init__(self, root_folder: str):
        self.labels = pd.DataFrame()
        self.targets = []

        for dir in os.listdir(root_folder):
            tmp = []
            for file in os.listdir(os.path.join(root_folder, dir)):
                img = cv2.imread(os.path.join(root_folder, dir, file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    tmp.append(pd.Series(img.flatten()))
                    self.targets += [int(dir) - 1] # into zero-based class index
            self.labels = pd.concat([self.labels, pd.DataFrame(tmp)], ignore_index=True)
        self.targets = pd.DataFrame(self.targets)
            
