from lazydl import nn, dr, optim
from lazydl.func import f
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("error")

EPOCH = 100
LR = .04
BATCH_SIZE = 25
PARAM_FILE_NAME = 'lab2.pkl'
CAT = [0, 1, 2]

class Lab2(nn.Module):
    def __init__(self, zero_init=False):
        super().__init__()
        self._modules = {
            'l1': nn.Linear(4, 12, zero_init=zero_init),
            'a1': nn.Sigmoid(),
            'l2': nn.Linear(12, 3, zero_init=zero_init),
            'a2': nn.Softmax(),
        }
    
    def predict(self, x: np.ndarray):
        return np.argmax(self(x), axis=1, keepdims=True)


def acc(test_loader, model):
    correct, total = 0, 0
    for x, y in test_loader:
        y_pred = model.predict(x)
        correct += np.sum(y_pred == y)
        total += len(y)
    accuracy = correct / total * 100
    return accuracy


def main():
    m = Lab2()
    data = dr.IRISDataset('./data/iris/iris_in.csv', './data/iris/iris_out.csv')
    train_loader, test_loader = dr.DataSplitter(data, BATCH_SIZE, shuffle=True, train_ratio=.5).get_loader()
    loss, opt = nn.MSELoss(m),  optim.Adam(m, LR)

    l_list, highest_acc = [], 0
    tracc_list = []
    teacc_list = []

    for e in tqdm(range(EPOCH)):
        tmp = []
        for x, y in train_loader:
            y_pred = m(x)
            y = f.onehot(y, CAT)
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
            m.save_model(f'./saved_model/lab2/{PARAM_FILE_NAME}')

    print(f'Testing acc: {highest_acc}')

    epochs = range(1, len(l_list) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, l_list, 'bo-', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, tracc_list, 'go-', label='Training Accuracy')
    plt.plot(epochs, teacc_list, 'ro-', label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()
