from lazydl import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("error")


PCA_N_COMP = .97
LDA_N_COMP = 18
PARAM_FILE_NAME = 'lab5'
DATA_PATH = './data/ORL_dat'


def acc(y_pred, y_true):
    return np.average(np.where(y_pred == y_true, 1, 0))


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main():
    create_dir(f'./saved_model/lab5/{PARAM_FILE_NAME}/')
    train_data, test_data = dr.ORLDataset(f'{DATA_PATH}/train'), dr.ORLDataset(f'{DATA_PATH}/test')

    pca = decomp.PCA(PCA_N_COMP).fit(train_data.labels)
    trdata = pca.transform(train_data.labels)
    tedata = pca.transform(test_data.labels)
    lda = decomp.LDA(LDA_N_COMP).fit(trdata, train_data.targets)
    y_pred = lda.predict(tedata)

    print(f'Accuracy: {acc(y_pred, np.asarray(test_data.targets).flatten()) * 100: .2f}%')

    pca.save_model(f'./saved_model/lab5/{PARAM_FILE_NAME}/{PARAM_FILE_NAME}.pca.pkl')
    lda.save_model(f'./saved_model/lab5/{PARAM_FILE_NAME}/{PARAM_FILE_NAME}.lda.pkl')


if __name__ == '__main__':
    main()
