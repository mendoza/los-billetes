import torch
from torch.utils.data import DataLoader
import sys
import json
import os
import numpy as np
from Billetes import BilletesDataset
from Perito import Net


def argumentExist():
    try:
        in_features = sys.argv[1]
        in_clasificador = sys.argv[2]
        out_etiquetas = sys.argv[3]
        return in_features, in_clasificador, out_etiquetas
    except IndexError:
        print(
            "Por favor proporcione los parametros, ingresando el archivo de caracteristicas primero luego la entrada del etiquetas y como ultimo la salida del clasificador"
        )
        sys.exit(1)


def main():
    in_features, in_clasificador, out_etiquetas = argumentExist()
    data = BilletesDataset(
        csv_file=in_features, root_dir='', etiquetas={})

    model = Net()
    model.load_state_dict(torch.load(in_clasificador))
    loader = DataLoader(data, batch_size=1, shuffle=False)
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

    output_dict = {}
    for (x, name) in loader:
        img_name = os.path.basename(name[0])
        output_dict[img_name] = {}

        output_dict[img_name]["denominacion"] = classes[np.argmax(
            model(x).detach().numpy())]

    json.dump(output_dict, open(out_etiquetas, 'w'), indent=4)


if __name__ == '__main__':
    main()
