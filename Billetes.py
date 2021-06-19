import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2 as cv
from PIL import Image
import torchvision.transforms.functional as TF


class ToTensor(object):

    def __call__(self, sample):
        print(sample)
        # sample = {'image': sample['image'], 'r': sample['r'], 'g': sample['g'], 'b': sample['b'],}
        return sample


class BilletesDataset(Dataset):

    def prepareImage(self, path, x1, y1, x2, y2):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        (h, w) = img.shape[:2]
        while h > w:
            img = np.rot90(img)
            (h, w) = img.shape[:2]

        img = img[int(y1):int(y2), int(x1):int(x2)]
        resized = cv.resize(img, (320, 192))
        pil_img = Image.fromarray(resized)
        return os.path.basename(path), pil_img

    def __init__(self, csv_file, root_dir, etiquetas, transform=None):
        self.images = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.etiquetas = etiquetas
        self.classes = [
            "1",
            "2",
            "5",
            "10",
            "20",
            "50",
            "100",
            "500"
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # img,r,g,b,x1,y1,x2,y2
        img_name = self.images.iloc[idx, 0]
        r = self.images.iloc[idx, 1]
        g = self.images.iloc[idx, 2]
        b = self.images.iloc[idx, 3]
        x1 = self.images.iloc[idx, 4]
        y1 = self.images.iloc[idx, 5]
        x2 = self.images.iloc[idx, 6]
        y2 = self.images.iloc[idx, 7]

        name, image = self.prepareImage(img_name, x1, y1, x2, y2)

        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(
            img_tensor, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.etiquetas != {}:
            return img_tensor, torch.tensor(self.classes.index(self.etiquetas[name]['denominacion']))
        else:
            return img_tensor, img_name
