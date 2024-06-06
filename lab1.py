from lazydl import nn, dr, optim, f
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("default")

EPOCH = 20
LR = .01
BATCH_SIZE = 1
PARAM_FILE_NAME = 'lab1.pkl'

class Lab1(nn.Module):
    def __init__(self, zero_init=True):
        super().__init__()
        self._modules = [
            nn.Linear(4, 1, zero_init=zero_init),
        ]
    
    def predict(self, x: np.ndarray):
        y = self(x)
        y[y <= 1.5] = 1
        y[np.abs(y - 2) < 0.5] = 2
        y[2.5 <= y] = 3
        return y


def acc(test_loader, model):
    correct, total = 0, 0
    for x, y in test_loader:
        y_pred = model.predict(x)
        correct += np.sum(y_pred == y)
        total += len(y)
    accuracy = correct / total * 100
    return accuracy


def main():
    m = Lab1()
    data = dr.IRISDataset('./data/iris/iris_in.csv', './data/iris/iris_out.csv')
    train_loader, test_loader = dr.DataSplitter(data, BATCH_SIZE, shuffle=True, train_ratio=.5).get_loader()
    loss, opt = nn.RMSELoss(m),  optim.SGD(m, LR)

    l_list, highest_acc = [], 0
    tracc_list = []
    teacc_list = []

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

        aaa = acc(test_loader, m)
        tracc_list.append(acc(train_loader, m))
        teacc_list.append(aaa)
        if aaa > highest_acc:
            highest_acc = aaa
            m.save_model(f'./saved_model/lab1/{PARAM_FILE_NAME}')

    print(f'Testing acc: {highest_acc}', end='')
    
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
    plt.plot(epochs, teacc_list, 'r-', label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    main()
