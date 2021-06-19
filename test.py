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
import torch.functional as TF


def argumentExist():
    try:
        in_features = sys.argv[1]
        in_etiquetas = sys.argv[2]
        return in_features, in_etiquetas
    except IndexError:
        print(
            "Por favor proporcione los parametros, ingresando el archivo de caracteristicas primero luego la entrada del etiquetas y como ultimo la salida del clasificador"
        )
        sys.exit(1)


def stable_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax


def main():
    in_features, in_etiquetas = argumentExist()
    etiquetas = json.load(open(in_etiquetas, 'r'))

    num_workers = 0
    batch_size = 1
    valid_size = 0.2  # Data augmentation for train data + conversion to tensor

    train_data = BilletesDataset(
        csv_file=in_features, root_dir='', etiquetas=etiquetas)

    # Creating data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers)

    model = Net()
    model.load_state_dict(torch.load('./trained_model.pt'))
    model.eval()
    classes = [
        "1",
        "2",
        "5",
        "10",
        "20",
        "50",
        "100",
        "500"
    ]

    for batch_index, (data, target) in enumerate(train_loader):
        target = target.detach().numpy()[0]
        output = model(data)
        m = torch.nn.LogSoftmax()
        output = m(output).detach().numpy()
        classified = np.argmax(output)
        print(f'era {classes[target]} y predijo {classes[classified]}')


if __name__ == '__main__':
    main()
