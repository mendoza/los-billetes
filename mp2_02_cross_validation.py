import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math


def argumentExist():
    try:
        in_features = sys.argv[1]
        in_etiquetas = sys.argv[2]
        in_folds = sys.argv[3]
        return in_features, in_etiquetas, in_folds
    except IndexError:
        print(
            "Por favor proporcione ambos archivos, ingresando el grafo primero luego la entrada del problema"
        )
        sys.exit(1)


def main():
    in_features, in_etiquetas, in_folds = argumentExist()


if __name__ == '__main__':
    main()
