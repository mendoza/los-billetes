import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from Perito import Net
from Billetes import BilletesDataset
import json
import sys
import torch
from torch.utils.data.sampler import SubsetRandomSampler


class TraindataSet(Dataset):
    def __init__(self, train_features, train_labels):
        self.x_data = train_features
        self.y_data = train_labels
        self.len = len(train_labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def argumentExist():
    try:
        in_features = sys.argv[1]
        in_etiquetas = sys.argv[2]
        in_folds = int(sys.argv[3])
        return in_features, in_etiquetas, in_folds
    except IndexError:
        print(
            "Por favor proporcione ambos archivos, ingresando el grafo primero luego la entrada del problema"
        )
        sys.exit(1)


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, data, num_epochs=3, learning_rate=0.001, batch_size=5):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    X = []
    Y = []
    for i in range(len(data)):
        x, y = data[i]
        X.append(x)
        Y.append(y)

    for i in range(k):
        # Get k-fold cross-validation training and verification data
        X_train, y_train, X_valid, y_valid = get_k_fold_data(
            k, i, torch.stack(X), torch.stack(Y))
        net = Net()  # Instantiate the model
        ### Each piece of data is trained, reflecting step three####
        train_ls, valid_ls = train(
            net,  X_train, y_train, X_valid, y_valid, num_epochs, learning_rate, batch_size)
        print('*'*25, 'section', f'{i+1}/{k}', 'fold', '*'*25)
        print('train_loss:%.6f' % train_ls[-1][0], 'train_acc:%.4f\n' % valid_ls[-1][1],
              'valid loss:%.6f' % valid_ls[-1][0], 'valid_acc:%.4f' % valid_ls[-1][1])
        train_loss_sum += train_ls[-1][0]
        valid_loss_sum += valid_ls[-1][0]
        train_acc_sum += train_ls[-1][1]
        valid_acc_sum += valid_ls[-1][1]
        print('#'*10, 'Final k-fold cross-validation result', '#'*10)
        #### ######
    print('train_loss_sum:%.4f' % (train_loss_sum/k), 'train_acc_sum:%.4f\n' % (train_acc_sum/k),
          'valid_loss_sum:%.4f' % (valid_loss_sum/k), 'valid_acc_sum:%.4f' % (valid_acc_sum/k))


def train(net, X_train, y_train, X_valid, y_valid,  num_epochs, learning_rate, batch_size):
    train_ls, test_ls = [], []
    # Finding indices for validation set
    dataset = TraindataSet(X_train, y_train)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(
        params=net.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for _, (X, y) in enumerate(train_iter):  # Batch training
            output = net(X)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Get the loss and accuracy of each epoch

        train_ls.append(log_rmse(0, net, criterion, X_train, y_train))
        if y_valid is not None:
            test_ls.append(log_rmse(1, net, criterion, X_valid, y_valid))

        return train_ls, test_ls


def log_rmse(flag, net, loss_func, x, y):
    if flag == 1:
        net.eval()
    output = net(x)
    result = torch.max(output, 1)[1].view(y.size())
    corrects = (result.data == y.data).sum().item()
    accuracy = corrects*100.0/len(y)  # 5 is batch_size
    loss = loss_func(output, y)
    net.train()
    return (loss.data.item(), accuracy)


def main():
    in_features, in_etiquetas, in_folds = argumentExist()
    etiquetas = json.load(open(in_etiquetas, 'r'))
    data = BilletesDataset(csv_file=in_features,
                           root_dir='', etiquetas=etiquetas)

    k_fold(in_folds, data, batch_size=100,
           learning_rate=0.1e-4, num_epochs=100)


if __name__ == '__main__':
    main()
