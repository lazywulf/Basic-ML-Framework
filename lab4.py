from lazydl import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("error")


EPOCH = 12500
N_COMP = .97
BS = 64
LR = 1e-3
PARAM_FILE_NAME = 'lab4'

DATA_PATH = './data/ORL_dat'

    
class Lab4(nn.Module):
    def __init__(self, comp_count):
        super().__init__()
        self._modules = [
            nn.Linear(comp_count, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 40),
        ]
    
    def predict(self, x: np.ndarray):
        return np.argmax(f.softmax(self(x)), axis=1, keepdims=True)
    
    def predict_prop(self, x: np.ndarray):
        return f.softmax(self(x), axis=1)
    

def top_n_acc(test_loader, model, n=1):
    correct, total = 0, 0
    for x, y in test_loader:
        y_pred_probs = model.predict_prop(x)
        
        top_n_preds = np.argsort(y_pred_probs, axis=1)[:, -n:]

        correct += np.sum([y[i] in top_n_preds[i] for i in range(len(y))])
        total += len(y)
        
    accuracy = correct / total * 100
    return accuracy


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def transform():
    train_data, test_data = dr.ORLDataset(f'{DATA_PATH}/train'), dr.ORLDataset(f'{DATA_PATH}/test')

    pca = decomp.PCA(N_COMP).fit(train_data.labels)
    print(f'Component count: {pca.component_count}')
    print(f'Variance explained: {np.sum(pca.explained_variance_ratio): .4f}%')
    trdata = pca.transform(train_data.labels)
    tedata = pca.transform(test_data.labels)
    mm = preprocess.MaxMinNorm().fit(trdata)
    trdata = mm.transform(trdata)
    tedata = mm.transform(tedata)

    pca.save_model(f'./saved_model/lab4/{PARAM_FILE_NAME}/{PARAM_FILE_NAME}.pca.pkl')
    mm.save_model(f'./saved_model/lab4/{PARAM_FILE_NAME}/{PARAM_FILE_NAME}.mm.pkl')
    
    train_data.labels = pd.DataFrame(trdata)
    test_data.labels = pd.DataFrame(tedata)
    
    return train_data, test_data, pca.component_count


def train(train_data, test_data, comp_count):
    m = Lab4(comp_count)
    train_loader, test_loader = dr.DataLoader(train_data, batch_size=BS, shuffle=True), dr.DataLoader(test_data, shuffle=True)
    loss, opt = nn.CrossEntropyLoss(m),  optim.Adam(m, LR)
    l_list, highest_acc = [], 0
    tracc_list = []
    teacc_list_top_1, teacc_list_top_3 = [], []


    for e in tqdm(range(EPOCH)):
        tmp = []
        for x, y in train_loader:
            y_pred = m(x)
            tmp.append(loss(y_pred, y))
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        l_list.append(sum(tmp) / len(tmp))
        tmp.clear()

        aaa = top_n_acc(test_loader, m)
        teacc_list_top_3.append(top_n_acc(test_loader, m, 3))
        tracc_list.append(top_n_acc(train_loader, m, 3))
        teacc_list_top_1.append(aaa)
        if aaa > highest_acc:
            highest_acc = aaa
            m.save_model(f'./saved_model/lab4/{PARAM_FILE_NAME}/{PARAM_FILE_NAME}.pkl')

    print(f'Testing acc (Top-1): {highest_acc}%')


    epochs = range(1, len(l_list) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, l_list, 'b-', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, tracc_list, 'g-', label='Training Accuracy')
    plt.plot(epochs, teacc_list_top_3, 'r-', label='Top-3 Accuracy')
    plt.plot(epochs, teacc_list_top_1, 'b-', label='Top-1 Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    create_dir(f'./saved_model/lab4/{PARAM_FILE_NAME}/')
    a, b, c = transform()
    train(a, b, c)


if __name__ == '__main__':
    main()
