import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
from Perito import Net
from Billetes import BilletesDataset

import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim


def argumentExist():
    try:
        in_features = sys.argv[1]
        in_etiquetas = sys.argv[2]
        out_classifier = sys.argv[3]
        return in_features, in_etiquetas, out_classifier
    except IndexError:
        print(
            "Por favor proporcione los parametros, ingresando el archivo de caracteristicas primero luego la entrada del etiquetas y como ultimo la salida del clasificador"
        )
        sys.exit(1)


def main():
    in_features, in_etiquetas, out_classifier = argumentExist()
    etiquetas = json.load(open(in_etiquetas, 'r'))

    num_workers = 0
    batch_size = 36
    valid_size = 0.02  # Data augmentation for train data + conversion to tensor

    train_data = BilletesDataset(
        csv_file=in_features, root_dir='', etiquetas=etiquetas)

    # Finding indices for validation set
    num_train = len(train_data)
    indices = list(range(num_train))
    # Randomize indices
    np.random.shuffle(indices)
    split = int(np.floor(num_train*valid_size))

    # Making samplers for training and validation batches
    train_index, test_index = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(test_index)

    # Creating data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=batch_size, num_workers=num_workers)

    model = Net()
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Total Trainable Parameters in Network: " + str(pytorch_total_params))

    # loss function (cross entropy loss)
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # epochs to train for
    epochs = 25

    # tracks validation loss change after each epoch
    minimum_validation_loss = np.inf

    for epoch in range(1, epochs+1):
        train_loss = 0
        valid_loss = 0

        # training steps
        model.train()
        for batch_index, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        # validation steps
        model.eval()
        for batch_index, (data, target) in enumerate(valid_loader):
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)

        print(
            f'Epoch {epoch}\t Training Loss: {train_loss/len(train_loader)}\t Validation Loss:{valid_loss/len(valid_loader)}')
        # Saving model every time validation loss decreases
        if valid_loss <= minimum_validation_loss:
            print(
                f'Validation loss decreased from {round(minimum_validation_loss, 6)} to {round(valid_loss, 6)}')
            torch.save(model.state_dict(), 'trained_model.pt')
            minimum_validation_loss = valid_loss
            print('Saving New Model')


if __name__ == '__main__':
    main()
