from lazydl import nn, dr, optim, decomp, preprocess
import os
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from lab1 import Lab1
from lab2 import Lab2
from lab3 import Lab3
from lab4 import Lab4


def acc(test_loader, model):
    correct, total = 0, 0
    for x, y in test_loader:
        y_pred = model.predict(x)
        correct += np.sum(y_pred == y)
        total += len(y)
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy: .6f} %")
    return accuracy


def top_n_acc(test_loader, model, n=1):
    correct, total = 0, 0
    for x, y in test_loader:
        y_pred_probs = model.predict_prop(x)
        
        top_n_preds = np.argsort(y_pred_probs, axis=1)[:, -n:]

        correct += np.sum([y[i] in top_n_preds[i] for i in range(len(y))])
        total += len(y)
        
    accuracy = correct / total * 100
    return accuracy


def acc_lab5(y_pred, y_true):
    return np.average(np.where(y_pred == y_true, 1, 0))


def load_iris_model(model_cls, model_file, label_file, target_file):
    m = model_cls.load_model(model_file)
    data = dr.IRISDataset(label_file, target_file)
    test_loader = dr.DataLoader(data, 0, shuffle=True)
    acc(test_loader, m)


def load_ORL_lab4(model_cls, model_dir, test_dir):
    def split_path(path_str):
        path = pathlib.Path(path_str)
        return list(path.parts)

    data = dr.ORLDataset(f'{test_dir}')
    pca = decomp.PCA.load_model(os.path.join(model_dir, f'{split_path(model_dir)[-1]}.pca.pkl'))
    mm = preprocess.MaxMinNorm.load_model(os.path.join(model_dir, f'{split_path(model_dir)[-1]}.mm.pkl'))

    tedata = pca.transform(data.labels)
    tedata = mm.transform(tedata)
    data.labels = pd.DataFrame(tedata)

    loader = dr.DataLoader(data, shuffle=True)
    m = model_cls.load_model(os.path.join(model_dir, f'{split_path(model_dir)[-1]}.pkl'))
    print(f"Top-1 Accuracy: {top_n_acc(loader, m, ): .2f}%")
    print(f"Top-3 Accuracy: {top_n_acc(loader, m, 3): .2f}%")


def load_ORL_lab5(model_dir, test_dir):
    def split_path(path_str):
        path = pathlib.Path(path_str)
        return list(path.parts)

    data = dr.ORLDataset(f'{test_dir}')
    pca = decomp.PCA.load_model(os.path.join(model_dir, f'{split_path(model_dir)[-1]}.pca.pkl'))
    lda = decomp.LDA.load_model(os.path.join(model_dir, f'{split_path(model_dir)[-1]}.lda.pkl'))
    tedata = pca.transform(data.labels)
    y_pred = lda.predict(tedata)

    print(f"Accuracy: {acc_lab5(y_pred, np.asarray(data.targets).flatten()) * 100: .2f}%")


if __name__ == '__main__':
    load_iris_model(Lab1, './saved_model/lab1/lab1.pkl', './data/iris/iris_in.csv', './data/iris/iris_out.csv')
    print('---------------')
    load_iris_model(Lab2, './saved_model/lab2/lab2.pkl', './data/iris/iris_in.csv', './data/iris/iris_out.csv')
    print('---------------')
    load_iris_model(Lab3, './saved_model/lab3/lab3.pkl', './data/iris/iris_in.csv', './data/iris/iris_out.csv')
    print('---------------')
    load_ORL_lab4(Lab4, './saved_model/lab4/lab4', './data/ORL_dat/test')
    print('---------------')
    load_ORL_lab5('./saved_model/lab5/lab5', './data/ORL_dat/test')

