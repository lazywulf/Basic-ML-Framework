import numpy as np
from .data_reader import Dataset

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 0, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size if 0 < batch_size < len(dataset) else len(dataset)
        self.shuffle = shuffle
        self.idx = 0
        self.indices = np.arange(len(dataset))
        if shuffle:
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            self.dataset.labels = self.dataset.labels.iloc[indices]
            if self.dataset.targets is not None:
                self.dataset.targets = self.dataset.targets.iloc[indices]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration
        else:
            if self.dataset.targets is not None:
                x, y = self.dataset[self.idx: self.idx + self.batch_size]
            else:
                x = self.dataset[self.idx: self.idx + self.batch_size]
            self.idx += self.batch_size

        return (x, y) if y is not None else x
        

class DataSplitter:
    def __init__(self, dataset: Dataset, batch_size: int = 0, shuffle: bool = False, train_ratio: float = 0.8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_ratio = train_ratio
        self.train_loader = None
        self.test_loader = None

        if shuffle:
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            self.dataset.labels = self.dataset.labels.iloc[indices]
            if self.dataset.targets is not None:
                self.dataset.targets =self.dataset.targets.iloc[indices]

        train_size = int(len(self.dataset) * train_ratio)
        train_labels, test_labels = self.dataset.labels[:train_size], self.dataset.labels[train_size:]
        train_targets, test_targets = self.dataset.targets[:train_size], self.dataset.targets[train_size:] if self.dataset.targets is not None else None

        train_dataset = Dataset(train_labels, train_targets)
        test_dataset = Dataset(test_labels, test_targets)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)

    def get_loader(self):
        return self.train_loader, self.test_loader
